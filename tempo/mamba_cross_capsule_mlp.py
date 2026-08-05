"""
MambaCapsuleCross-DTI  (Morgan FP drug encoder)
=================================================
Drug-Target Interaction prediction via:
    Protein : Mamba SSM    →  CrossAttention  →  CapsuleBranch  →  (B, 128)
    Drug    : Morgan FP MLP (2048 → 256 → 128)
              →  CrossAttention(Q=drug, K/V=protein)  →  (B, 128)
    Fusion  : concat (B, 256)  →  MLP  →  logit  (BCEWithLogitsLoss)

Drug side uses fixed-length Morgan fingerprint (B, 2048) — no sequence,
no padding.  Drug embedding is unsqueezed to (B, 1, 128) for cross-attention,
then squeezed back.  No capsules on the drug side.

Cross-attention sits between the encoders and the protein capsule branch so
the capsule routing can operate on context-aware representations:
    protein (Q=64)  attends to drug    (K=V=128)  — no mask needed
    drug    (Q=128) attends to protein (K=V=64)   — length-1 query

Both directions are residual + LayerNorm.

NaN fixes applied
─────────────────
1. squash()       : sq_norm clamped to min=1e-8 before sqrt in unit branch.
                    Prevents 1/(2*sqrt(0))=inf in backward pass.
2. PrimaryCaps    : GroupNorm replaces BatchNorm1d.
3. DigitCaps W    : orthogonal init; routing logits clamped [-10, 10].
4. ProteinEncoder : post-Mamba LayerNorm.
5. CapsuleBranch  : output LayerNorm before pooling.
6. Classifier     : xavier_uniform_ gain=0.5; bias → 0.
7. run_epoch      : per-batch NaN/Inf guard.
8. load_protein_features: z-score normalisation.
"""

import copy

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from mamba_ssm import Mamba

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold


# ─────────────────────────────────────────────────────────────────────────────
#  Shared utility
# ─────────────────────────────────────────────────────────────────────────────

def squash(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    Squash activation (Sabour et al. 2017).

    Backward-safe: sq_norm clamped to min=1e-8 only for the unit vector
    branch to prevent 1/(2*sqrt(0))=inf in autograd. scale uses unclamped
    sq_norm so inactive capsules correctly produce output ≈ 0.
    """
    sq_norm = (x ** 2).sum(dim=dim, keepdim=True)
    scale   = sq_norm / (1.0 + sq_norm)
    unit    = x / (sq_norm.clamp(min=1e-8).sqrt())
    return scale * unit


# ─────────────────────────────────────────────────────────────────────────────
#  Protein encoder   (B, 220) → (B, 220, d_model)
# ─────────────────────────────────────────────────────────────────────────────

class ProteinEncoder(nn.Module):
    """
    Input  : (B, 220)
    Output : (B, 220, d_model)
    """

    def __init__(self, d_model: int = 64):
        super().__init__()
        self.project   = nn.Linear(1, d_model)
        self.pre_norm  = nn.LayerNorm(d_model)
        self.mamba     = Mamba(d_model=d_model, d_state=16, d_conv=4, expand=2)
        self.post_norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(-1)
        x = self.pre_norm(self.project(x))
        x = self.mamba(x)
        x = self.post_norm(x)
        return x


# ─────────────────────────────────────────────────────────────────────────────
#  Drug encoder  (B, 2048) → (B, 128)  — Morgan fingerprint MLP
# ─────────────────────────────────────────────────────────────────────────────

class DrugEncoder(nn.Module):
    """
    Two-layer MLP that maps a fixed-length Morgan fingerprint to an
    embedding vector.

    Input  : (B, 2048)
    Output : (B, 128)
    """

    def __init__(self, in_dim: int = 2048, hidden_dim: int = 256, out_dim: int = 128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)   # (B, 128)


# ─────────────────────────────────────────────────────────────────────────────
#  Cross-attention block
# ─────────────────────────────────────────────────────────────────────────────

class CrossAttention(nn.Module):
    """
    Single-direction cross-attention: query attends to context.

    Output shape matches query (residual-compatible).
    PyTorch MultiheadAttention supports q_dim ≠ c_dim via kdim/vdim args.

    Args
    ────
    q_dim     : query (and output) dimension
    c_dim     : context key/value dimension
    num_heads : must divide q_dim evenly

    Input  : query (B, L_q, q_dim), context (B, L_c, c_dim)
    Output : (B, L_q, q_dim)
    """

    def __init__(
        self,
        q_dim     : int,
        c_dim     : int,
        num_heads : int = 4,
        dropout   : float = 0.1,
    ):
        super().__init__()
        assert q_dim % num_heads == 0, (
            f"q_dim={q_dim} must be divisible by num_heads={num_heads}"
        )
        self.attn    = nn.MultiheadAttention(
            embed_dim = q_dim,
            num_heads = num_heads,
            kdim      = c_dim,
            vdim      = c_dim,
            dropout   = dropout,
            batch_first = True,
        )
        self.norm    = nn.LayerNorm(q_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query              : torch.Tensor,
        context            : torch.Tensor,
        context_pad_mask   : torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        query            : (B, L_q, q_dim)
        context          : (B, L_c, c_dim)
        context_pad_mask : (B, L_c) bool — True = PAD, ignored in attention

        returns : (B, L_q, q_dim)
        """
        out, _ = self.attn(
            query, context, context,
            key_padding_mask = context_pad_mask,   # True = ignore
        )
        return self.norm(query + self.dropout(out))


# ─────────────────────────────────────────────────────────────────────────────
#  Primary Capsule layer
# ─────────────────────────────────────────────────────────────────────────────

class PrimaryCaps(nn.Module):
    """
    Convolutional primary capsule layer.
    GroupNorm replaces BatchNorm1d for sequence-length-independent stability.

    Input  : (B, L_in, in_channels)
    Output : (B, num_caps, L_out, cap_dim)
    """

    def __init__(
        self,
        in_channels : int,
        num_caps    : int = 8,
        cap_dim     : int = 16,
        kernel_size : int = 9,
        stride      : int = 2,
    ):
        super().__init__()
        self.num_caps = num_caps
        self.cap_dim  = cap_dim
        out_channels  = num_caps * cap_dim

        self.conv = nn.Conv1d(
            in_channels  = in_channels,
            out_channels = out_channels,
            kernel_size  = kernel_size,
            stride       = stride,
            padding      = 0,
        )
        self.norm = nn.GroupNorm(num_groups=num_caps, num_channels=out_channels)

        nn.init.kaiming_normal_(self.conv.weight, mode="fan_out", nonlinearity="relu")
        nn.init.zeros_(self.conv.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        x = self.conv(x)
        x = self.norm(x)
        B, _, L_out = x.shape
        x = x.view(B, self.num_caps, self.cap_dim, L_out)
        x = x.permute(0, 1, 3, 2)
        return squash(x, dim=-1)


# ─────────────────────────────────────────────────────────────────────────────
#  Digit Capsule layer  (dynamic routing)
# ─────────────────────────────────────────────────────────────────────────────

class DigitCaps(nn.Module):
    """
    Sequence-aware DigitCaps with dynamic routing (Sabour et al. 2017).

    Input  : (B, in_caps,  L, dim_in)
    Output : (B, out_caps, L, dim_out)
    """

    def __init__(
        self,
        in_caps      : int = 8,
        dim_in       : int = 16,
        out_caps     : int = 8,
        dim_out      : int = 16,
        routing_iters: int = 3,
    ):
        super().__init__()
        self.out_caps      = out_caps
        self.routing_iters = routing_iters

        self.W = nn.Parameter(torch.empty(in_caps, out_caps, dim_out, dim_in))
        for i in range(in_caps):
            for j in range(out_caps):
                nn.init.orthogonal_(self.W[i, j])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, in_caps, L, _ = x.shape

        u_hat    = torch.einsum("i j d c, b i l c -> b i l j d", self.W, x)
        u_hat_sg = u_hat.detach()

        b_ij = torch.zeros(B, in_caps, L, self.out_caps, device=x.device)

        v = None
        for t in range(self.routing_iters):
            c = F.softmax(b_ij.clamp(-10, 10), dim=-1)
            u = u_hat_sg if t < self.routing_iters - 1 else u_hat
            s = (c.unsqueeze(-1) * u).sum(dim=1)
            v = squash(s, dim=-1)

            if t < self.routing_iters - 1:
                agreement = (u_hat_sg * v.unsqueeze(1)).sum(dim=-1)
                b_ij = b_ij + agreement

        return v.permute(0, 2, 1, 3).contiguous()


# ─────────────────────────────────────────────────────────────────────────────
#  Capsule branch wrapper
# ─────────────────────────────────────────────────────────────────────────────

class CapsuleBranch(nn.Module):
    """
    PrimaryCaps → DigitCaps → project → LayerNorm → masked mean pool.

    Input  : (B, L, encoder_dim)
    Output : (B, embed_dim)
    """

    def __init__(
        self,
        encoder_dim   : int,
        num_caps      : int = 8,
        cap_dim       : int = 16,
        out_caps      : int = 8,
        out_cap_dim   : int = 16,
        embed_dim     : int = 128,
        kernel_size   : int = 9,
        stride        : int = 2,
        routing_iters : int = 3,
    ):
        super().__init__()
        self.primary = PrimaryCaps(
            in_channels = encoder_dim,
            num_caps    = num_caps,
            cap_dim     = cap_dim,
            kernel_size = kernel_size,
            stride      = stride,
        )
        self.digit = DigitCaps(
            in_caps       = num_caps,
            dim_in        = cap_dim,
            out_caps      = out_caps,
            dim_out       = out_cap_dim,
            routing_iters = routing_iters,
        )
        self.proj = nn.Linear(out_cap_dim, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(
        self,
        x            : torch.Tensor,
        padding_mask : torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = self.primary(x)
        x = self.digit(x)
        x = self.proj(x)
        x = self.norm(x)

        if padding_mask is not None:
            k       = self.primary.conv.kernel_size[0]
            s       = self.primary.conv.stride[0]
            L_prime = x.shape[2]
            indices = torch.arange(L_prime, device=x.device) * s + (k - 1)
            indices = indices.clamp(max=padding_mask.shape[1] - 1)
            mask_Lp = padding_mask[:, indices]
            valid   = (~mask_Lp).float().unsqueeze(1).unsqueeze(-1)
            x = (x * valid).sum(dim=2) / (valid.sum(dim=2) + 1e-8)
        else:
            x = x.mean(dim=2)

        return x.mean(dim=1)


# ─────────────────────────────────────────────────────────────────────────────
#  Full model
# ─────────────────────────────────────────────────────────────────────────────

class MambaCapsuleCrossDTI(nn.Module):
    """
    Drug-Target Interaction predictor — cross-attention + capsule model.

    Protein : Mamba SSM    →  CrossAttn(Q=protein, K/V=drug)
                           →  CapsuleBranch  →  (B, 128)
    Drug    : Morgan FP MLP (2048 → 256 → 128)
              →  unsqueeze(1) → CrossAttn(Q=drug, K/V=protein)
              →  squeeze(1)  →  (B, 128)
    Fusion  : concat  →  MLP  →  logit

    Cross-attention sits between encoder and protein capsule branch so
    routing operates on context-enriched representations.
    Drug side has no capsules — just MLP + cross-attention.

    Forward args
    ────────────
    protein : (B, 220)     PsePSSM feature vectors (z-score normalised)
    drug    : (B, 2048)    Morgan fingerprint (fixed-length)

    Returns
    ───────
    logits : (B,)  raw logits for BCEWithLogitsLoss
    """

    def __init__(
        self,
        drug_in_dim   : int = 2048,
        d_model       : int = 64,
        drug_proj_dim : int = 128,
        num_caps      : int = 8,
        cap_dim       : int = 16,
        out_caps      : int = 8,
        out_cap_dim   : int = 16,
        embed_dim     : int = 128,
        routing_iters : int = 3,
        num_heads     : int = 4,
        attn_dropout  : float = 0.1,
    ):
        super().__init__()

        # ── Encoders ──────────────────────────────────────────────────────────
        self.protein_encoder = ProteinEncoder(d_model=d_model)
        self.drug_encoder    = DrugEncoder(
            in_dim=drug_in_dim, hidden_dim=256, out_dim=drug_proj_dim,
        )

        # ── Cross-attention (bidirectional) ───────────────────────────────────
        # protein (Q=d_model=64) attends to drug (K/V=drug_proj_dim=128)
        self.protein_cross = CrossAttention(
            q_dim     = d_model,
            c_dim     = drug_proj_dim,
            num_heads = num_heads,
            dropout   = attn_dropout,
        )
        # drug (Q=drug_proj_dim=128) attends to protein (K/V=d_model=64)
        self.drug_cross = CrossAttention(
            q_dim     = drug_proj_dim,
            c_dim     = d_model,
            num_heads = num_heads,
            dropout   = attn_dropout,
        )

        # ── Protein capsule branch (drug has no capsules) ────────────────────
        self.protein_caps = CapsuleBranch(
            encoder_dim   = d_model,
            num_caps      = num_caps,
            cap_dim       = cap_dim,
            out_caps      = out_caps,
            out_cap_dim   = out_cap_dim,
            embed_dim     = embed_dim,
            kernel_size   = 9,
            stride        = 2,
            routing_iters = routing_iters,
        )

        # ── Fusion MLP ────────────────────────────────────────────────────────
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim * 2, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
        )
        self._init_classifier()

    def _init_classifier(self):
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                nn.init.zeros_(m.bias)

    def forward(
        self,
        protein : torch.Tensor,   # (B, 220)
        drug    : torch.Tensor,   # (B, 2048)
    ) -> torch.Tensor:

        # ── Encode ────────────────────────────────────────────────────────────
        p = self.protein_encoder(protein)   # (B, 220, d_model)
        d = self.drug_encoder(drug)         # (B, drug_proj_dim)

        # ── Cross-attend ──────────────────────────────────────────────────────
        # drug needs to be (B, 1, drug_proj_dim) for cross-attention
        d = d.unsqueeze(1)
        
        # protein attends to drug; no padding mask since drug is length 1
        p = self.protein_cross(p, d, context_pad_mask=None)            # (B, 220, d_model)
        # drug attends to protein; protein has no padding
        d = self.drug_cross(d, p)                                      # (B, 1, drug_proj_dim)
        
        d = d.squeeze(1)                                               # (B, drug_proj_dim)

        # ── Capsule branches ──────────────────────────────────────────────────
        p = self.protein_caps(p)                        # (B, embed_dim)

        # ── Fusion ────────────────────────────────────────────────────────────
        x = torch.cat([p, d], dim=-1)
        return self.classifier(x).squeeze(-1)


# ─────────────────────────────────────────────────────────────────────────────
#  Dataset / DataLoader
# ─────────────────────────────────────────────────────────────────────────────

class DTIDataset(Dataset):
    def __init__(self, interactions, protein_features, drug_embeddings):
        self.df               = interactions.reset_index(drop=True)
        self.protein_features = protein_features
        self.drug_embeddings  = drug_embeddings

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row     = self.df.iloc[idx]
        protein = torch.tensor(
            self.protein_features[row["protein_id"]], dtype=torch.float32
        )
        drug = self.drug_embeddings[row["drug_id"]]
        if not isinstance(drug, torch.Tensor):
            drug = torch.from_numpy(drug)
        label = torch.tensor(row["label"], dtype=torch.float32)
        return protein, drug.float(), label


def collate_fn(batch):
    proteins = torch.stack([x[0] for x in batch])
    drugs    = torch.stack([x[1] for x in batch])
    labels   = torch.stack([x[2] for x in batch])
    return proteins, drugs, labels


# ─────────────────────────────────────────────────────────────────────────────
#  Metrics
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(labels, probs):
    preds  = (np.array(probs) >= 0.5).astype(int)
    labels = np.array(labels, dtype=int)
    tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
    return {
        "accuracy"   : accuracy_score(labels, preds),
        "precision"  : precision_score(labels, preds, zero_division=0),
        "recall"     : recall_score(labels, preds, zero_division=0),
        "specificity": tn / (tn + fp + 1e-8),
        "mcc"        : matthews_corrcoef(labels, preds),
        "roc_auc"    : roc_auc_score(labels, probs),
        "pr_auc"     : average_precision_score(labels, probs),
    }


# ─────────────────────────────────────────────────────────────────────────────
#  Epoch runner
# ─────────────────────────────────────────────────────────────────────────────

def run_epoch(model, loader, criterion, device, optimizer=None):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss = 0.0
    all_labels, all_probs = [], []

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for batch_idx, (protein, drug, label) in enumerate(loader):
            protein, drug, label = (
                protein.to(device), drug.to(device), label.to(device)
            )
            logits = model(protein, drug)
            loss   = criterion(logits, label)

            if torch.isnan(loss) or torch.isinf(loss):
                raise ValueError(
                    f"{'NaN' if torch.isnan(loss) else 'Inf'} loss at "
                    f"batch {batch_idx}.\n"
                    f"  logits : min={logits.min():.4f}  max={logits.max():.4f}  "
                    f"nan={torch.isnan(logits).any()}\n"
                    f"  protein: min={protein.min():.4f}  max={protein.max():.4f}  "
                    f"nan={torch.isnan(protein).any()}\n"
                    f"  drug   : min={drug.min():.4f}  max={drug.max():.4f}  "
                    f"nan={torch.isnan(drug).any()}"
                )

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            total_loss += loss.item() * label.size(0)
            probs = torch.sigmoid(logits).detach().cpu().numpy()
            all_probs.extend(probs.tolist())
            all_labels.extend(label.cpu().numpy().tolist())

    epoch_loss = total_loss / len(loader.dataset)
    metrics    = compute_metrics(all_labels, all_probs)
    return epoch_loss, metrics


def _print_metrics(split: str, loss: float, m: dict):
    print(f"\n  [{split}]  loss={loss:.4f}")
    for k, v in m.items():
        print(f"    {k:<14}: {v:.4f}")


# ─────────────────────────────────────────────────────────────────────────────
#  Data helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_protein_features(protein_csv: str) -> dict:
    """
    Load PsePSSM CSV, z-score normalise across all proteins, return dict.

    Returns
    ───────
    dict: {protein_id: np.ndarray of shape (220,), dtype float32}
    """
    df       = pd.read_csv(protein_csv)
    features = {
        row["protein_id"]: row.iloc[1:].values.astype(np.float32)
        for _, row in df.iterrows()
    }
    all_feats = np.stack(list(features.values()))
    mean      = all_feats.mean(axis=0)
    std       = all_feats.std(axis=0) + 1e-8
    return {pid: (v - mean) / std for pid, v in features.items()}


# ─────────────────────────────────────────────────────────────────────────────
#  Training  (5-fold cross-validation)
# ─────────────────────────────────────────────────────────────────────────────

def train(
    interactions,
    protein_features,
    drug_embeddings,
    *,
    num_epochs   : int   = 15,
    batch_size   : int   = 16,
    lr           : float = 1e-3,
    weight_decay : float = 1e-4,
    num_folds    : int   = 5,
    patience     : int   = 5,
    save_prefix  : str   = "best_model",
    device       = None,
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    criterion    = nn.BCEWithLogitsLoss()
    labels_array = interactions["label"].values
    skf          = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=42)
    fold_summaries = []

    for fold, (train_idx, val_idx) in enumerate(
        skf.split(np.zeros(len(labels_array)), labels_array), start=1
    ):
        print(f"\n{'='*60}\n  Fold {fold}\n{'='*60}")

        train_loader = DataLoader(
            DTIDataset(interactions.iloc[train_idx], protein_features, drug_embeddings),
            batch_size  = batch_size,
            shuffle     = True,
            collate_fn  = collate_fn,
            num_workers = 0,
        )
        val_loader = DataLoader(
            DTIDataset(interactions.iloc[val_idx], protein_features, drug_embeddings),
            batch_size  = batch_size,
            shuffle     = False,
            collate_fn  = collate_fn,
            num_workers = 0,
        )

        model     = MambaCapsuleCrossDTI().to(device)
        optimizer = torch.optim.Adam(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=3, verbose=True
        )

        best_roc     = -1.0
        best_metrics = None
        wait         = 0

        for epoch in range(1, num_epochs + 1):
            t_loss, t_m = run_epoch(model, train_loader, criterion, device, optimizer)
            v_loss, v_m = run_epoch(model, val_loader,   criterion, device)
            scheduler.step(v_m["roc_auc"])

            print(f"\n  Epoch {epoch}/{num_epochs}")
            _print_metrics("train", t_loss, t_m)
            _print_metrics("val",   v_loss, v_m)

            if v_m["roc_auc"] > best_roc:
                best_roc     = v_m["roc_auc"]
                best_metrics = copy.deepcopy(v_m)
                torch.save(model.state_dict(), f"{save_prefix}_fold{fold}.pt")
                wait = 0
                print(f"  ✓  Saved  (ROC-AUC={best_roc:.4f})")
            else:
                wait += 1
                if wait >= patience:
                    print(f"  Early stopping at epoch {epoch}")
                    break

        fold_summaries.append(best_metrics)

    metric_keys = ["accuracy", "precision", "recall",
                   "specificity", "mcc", "roc_auc", "pr_auc"]
    print(f"\n{'='*60}\n  5-Fold CV Summary\n{'='*60}")
    for key in metric_keys:
        vals  = [s[key] for s in fold_summaries]
        label = key.replace("_", " ").title()
        print(f"  {label:<16}: {np.mean(vals):.4f}  ±  {np.std(vals):.4f}")

    return fold_summaries


# ─────────────────────────────────────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Sanity check ─────────────────────────────────────────────────────────
    model = MambaCapsuleCrossDTI().to(device)
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {total:,}")

    B = 4
    protein = torch.randn(B, 220, device=device)
    drug    = torch.randn(B, 2048, device=device)
    logits  = model(protein, drug)
    assert not torch.isnan(logits).any(), "NaN in sanity-check logits!"
    print(f"Output shape : {logits.shape}")
    print(f"Logit range  : [{logits.min():.3f}, {logits.max():.3f}]")
    print("Sanity check passed.\n")

    # ── Load data ─────────────────────────────────────────────────────────────
    protein_features = load_protein_features(    
    "/mnt/c/Users/Harshini J/Engineering/Projects/Nimisha Ma'am - Project/"
    "Cross Attention/HINMTDTI/Mamba_Capsule/Enzyme/code/psepssm_features.csv"
    )

    drug_embeddings = torch.load(
        "/mnt/c/Users/Harshini J/Engineering/Projects/Nimisha Ma'am - Project/"
        "Cross Attention/HINMTDTI/Enzyme/Embeddings/drugs_enzyme_embeddings.pt"
    )

    interactions = pd.read_csv(
        "/mnt/c/Users/Harshini J/Engineering/Projects/Nimisha Ma'am - Project/"
        "Cross Attention/HINMTDTI/Enzyme/Data/Drug_Target_Pair_Enzyme_COMPLETE.csv"
    )

    interactions = interactions[
        interactions["protein_id"].isin(protein_features.keys())
        & interactions["drug_id"].isin(drug_embeddings.keys())
    ].reset_index(drop=True)

    print(f"Interactions after filtering : {len(interactions):,}")
    print(f"Label distribution:\n{interactions['label'].value_counts()}\n")

    assert len(interactions) > 0, (
        "Filtered interactions is empty. Check that protein_id / drug_id values "
        "in your CSV match keys in psepssm_features.csv and drug_embeddings.pt.\n"
        f"  Sample protein_ids (features) : {list(protein_features.keys())[:3]}\n"
        f"  Sample drug_ids    (embeddings): {list(drug_embeddings.keys())[:3]}"
    )

    # ── Train ─────────────────────────────────────────────────────────────────
    fold_summaries = train(
        interactions,
        protein_features,
        drug_embeddings,
        num_epochs   = 50,
        batch_size   = 16,
        lr           = 1e-3,
        weight_decay = 1e-4,
        num_folds    = 5,
        patience     = 5,
        device       = device,
    )