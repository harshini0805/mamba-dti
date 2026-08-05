"""
MambaCapsule-DTI  (NaN-hardened v2, Morgan FP drug encoder)
===========================================================
Drug-Target Interaction prediction via:
    Protein : Mamba SSM  →  PrimaryCaps  →  DigitCaps  →  pool  →  (B, 128)
    Drug    : Morgan FP MLP (2048 → 256 → 128)                 →  (B, 128)
    Fusion  : concat (B, 256)  →  MLP  →  logit  (BCEWithLogitsLoss)

No cross-attention. No transformers.
Drug side uses fixed-length Morgan fingerprint (B, 2048) — no sequence, no padding.

NaN fixes (v2 additions marked with ★)
────────────────────────────────────────
★ 1. squash()      : sq_norm clamped to min=1e-8 BEFORE sqrt in unit vector only.
                     Prevents 1/(2*sqrt(0))=inf in backward pass — the root cause
                     of deferred NaN at batch 229.  scale still uses the unclamped
                     sq_norm so inactive capsules (s≈0) correctly produce ≈0 output.
★ 2. PrimaryCaps   : BatchNorm1d → GroupNorm(num_groups=num_caps, num_channels=...).
                     GroupNorm normalises per-sample over capsule-type channel groups,
                     immune to variable-length padded drug sequences and small effective
                     batch sizes that destabilise BN statistics.
  3. DigitCaps W   : orthogonal init instead of xavier (stabler for routing chains)
  4. DigitCaps     : routing logits clamped to [-10, 10] before softmax
  5. ProteinEncoder: post-Mamba LayerNorm (Mamba can shift output scale significantly)
  6. CapsuleBranch : output LayerNorm before pooling (prevents unbounded capsule norms)
  7. Classifier    : xavier_uniform_ with gain=0.5; bias → 0 (logits near 0 at init)
  8. run_epoch     : per-batch NaN/Inf guard with diagnostic context
  9. load_protein_features: z-score normalisation enforced before training
★10. __main__      : protein_features loaded via load_protein_features(), not
                     pd.read_csv() — fixes the empty-interactions bug where
                     DataFrame.keys() returns column names, not protein IDs.
"""

import copy
import math

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
    Maps any vector to length in [0, 1) while preserving direction.

    ★ NaN fix v2 — backward-pass stability:
    The unit vector computation x / sqrt(sq_norm) has gradient
    1 / (2 * sqrt(sq_norm)) w.r.t. sq_norm.  When sq_norm → 0
    this gradient → inf.  scale = sq_norm / (1 + sq_norm) → 0
    at the same time, giving 0 * inf = NaN in fp32 autograd.

    Fix: clamp sq_norm to min=1e-8 ONLY for the unit vector branch.
    scale deliberately uses the unclamped value so inactive capsules
    correctly produce output ≈ 0 rather than a spurious unit vector.
    """
    sq_norm      = (x ** 2).sum(dim=dim, keepdim=True)   # ‖v‖²  (unclamped)
    scale        = sq_norm / (1.0 + sq_norm)              # ∈ [0, 1)
    unit         = x / (sq_norm.clamp(min=1e-8).sqrt())   # ★ clamp before sqrt
    return scale * unit


# ─────────────────────────────────────────────────────────────────────────────
#  Protein encoder   (B, 220) → (B, 220, d_model)
# ─────────────────────────────────────────────────────────────────────────────

class ProteinEncoder(nn.Module):
    """
    Lifts scalar PsePSSM features to d_model-dimensional sequence,
    applies Mamba SSM, then re-normalises output.

    NaN fix: post-Mamba LayerNorm — Mamba's internal SSM state
             can shift the output distribution substantially on early steps.

    Input  : (B, 220)
    Output : (B, 220, d_model)
    """

    def __init__(self, d_model: int = 64):
        super().__init__()
        self.project   = nn.Linear(1, d_model)
        self.pre_norm  = nn.LayerNorm(d_model)
        self.mamba     = Mamba(d_model=d_model, d_state=16, d_conv=4, expand=2)
        self.post_norm = nn.LayerNorm(d_model)   # stabilises Mamba output

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x : (B, 220)
        x = x.unsqueeze(-1)    # (B, 220, 1)
        x = self.project(x)    # (B, 220, d_model)
        x = self.pre_norm(x)   # (B, 220, d_model)
        x = self.mamba(x)      # (B, 220, d_model)
        x = self.post_norm(x)  # (B, 220, d_model)
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
#  Primary Capsule layer
# ─────────────────────────────────────────────────────────────────────────────

class PrimaryCaps(nn.Module):
    """
    Convolutional primary capsule layer.

    ★ NaN fix v2 — GroupNorm replaces BatchNorm1d:
    BatchNorm1d normalises over (B, L_out) per channel.  Drug sequences
    are zero-padded to the longest sequence in each batch, so L_out varies
    between batches.  Short-drug batches give few BN statistics → unstable
    running_mean / running_var under gradient flow.
    GroupNorm(num_groups=num_caps) normalises per sample over each capsule
    type's channels — independent of batch size and sequence length.

    Input  : (B, L_in, in_channels)
    Output : (B, num_caps, L_out, cap_dim)
             L_out = floor((L_in - kernel_size) / stride) + 1
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

        out_channels = num_caps * cap_dim

        self.conv = nn.Conv1d(
            in_channels  = in_channels,
            out_channels = out_channels,
            kernel_size  = kernel_size,
            stride       = stride,
            padding      = 0,
        )

        # ★ GroupNorm: num_groups=num_caps means each group = one capsule type
        #   (cap_dim channels per group).  Semantically natural and batch-stable.
        self.norm = nn.GroupNorm(
            num_groups   = num_caps,
            num_channels = out_channels,
        )

        nn.init.kaiming_normal_(self.conv.weight, mode="fan_out", nonlinearity="relu")
        nn.init.zeros_(self.conv.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x : (B, L_in, in_channels)
        x = x.transpose(1, 2)                                  # (B, in_channels, L_in)
        x = self.conv(x)                                       # (B, num_caps*cap_dim, L_out)
        x = self.norm(x)                                       # ★ GroupNorm (not BatchNorm)

        B, _, L_out = x.shape
        x = x.view(B, self.num_caps, self.cap_dim, L_out)     # (B, num_caps, cap_dim, L_out)
        x = x.permute(0, 1, 3, 2)                             # (B, num_caps, L_out, cap_dim)
        return squash(x, dim=-1)                               # (B, num_caps, L_out, cap_dim)


# ─────────────────────────────────────────────────────────────────────────────
#  Digit Capsule layer  (dynamic routing)
# ─────────────────────────────────────────────────────────────────────────────

class DigitCaps(nn.Module):
    """
    Sequence-aware DigitCaps with dynamic routing (Sabour et al. 2017).
    Each position L is routed independently, preserving spatial structure.

    NaN fixes
    ─────────
    • W initialised with nn.init.orthogonal_ (stabler than xavier for
      matrix-multiply chains in routing)
    • Routing logits b_ij clamped to [-10, 10] before softmax to prevent
      exp() overflow when routing diverges early in training
    • squash() now has backward-safe sq_norm clamping (see squash docstring)

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

        # W : (in_caps, out_caps, dim_out, dim_in)
        self.W = nn.Parameter(
            torch.empty(in_caps, out_caps, dim_out, dim_in)
        )
        # Orthogonal init: each (dim_out × dim_in) block is an isometry.
        # Keeps prediction-vector norms ≈ 1 at init → routing logits stable.
        for i in range(in_caps):
            for j in range(out_caps):
                nn.init.orthogonal_(self.W[i, j])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (B, in_caps, L, dim_in)
        """
        B, in_caps, L, _ = x.shape

        # 1. Prediction vectors
        #    u_hat[b,i,l,j,d] = W[i,j,d,:] · x[b,i,l,:]
        u_hat = torch.einsum("i j d c, b i l c -> b i l j d", self.W, x)
        # u_hat : (B, in_caps, L, out_caps, dim_out)

        u_hat_sg = u_hat.detach()   # stop-gradient copy for routing updates

        # 2. Routing logits — initialised to zero each forward pass
        b_ij = torch.zeros(B, in_caps, L, self.out_caps, device=x.device)

        v = None
        for t in range(self.routing_iters):

            # Clamp before softmax to prevent exp() overflow
            c = F.softmax(b_ij.clamp(-10, 10), dim=-1)   # (B, in_caps, L, out_caps)

            u = u_hat_sg if t < self.routing_iters - 1 else u_hat
            s = (c.unsqueeze(-1) * u).sum(dim=1)          # (B, L, out_caps, dim_out)
            v = squash(s, dim=-1)                          # (B, L, out_caps, dim_out)

            if t < self.routing_iters - 1:
                agreement = (u_hat_sg * v.unsqueeze(1)).sum(dim=-1)
                b_ij = b_ij + agreement

        v = v.permute(0, 2, 1, 3).contiguous()            # (B, out_caps, L, dim_out)
        return v


# ─────────────────────────────────────────────────────────────────────────────
#  Capsule branch wrapper
# ─────────────────────────────────────────────────────────────────────────────

class CapsuleBranch(nn.Module):
    """
    PrimaryCaps → DigitCaps → project → LayerNorm → pool.

    NaN fix: output LayerNorm applied before pooling to prevent
             capsule norms from growing unboundedly through routing.

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
        self.norm = nn.LayerNorm(embed_dim)   # stabilises pre-pooling activations

    def forward(
        self,
        x            : torch.Tensor,
        padding_mask : torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        x            : (B, L, encoder_dim)
        padding_mask : (B, L) bool — True for PAD positions (optional)

        returns : (B, embed_dim)
        """
        x = self.primary(x)   # (B, num_caps, L', cap_dim)
        x = self.digit(x)     # (B, out_caps, L', out_cap_dim)
        x = self.proj(x)      # (B, out_caps, L', embed_dim)
        x = self.norm(x)      # (B, out_caps, L', embed_dim)

        # ── Masked mean pooling over sequence dim (L') ───────────────────────
        # padding_mask refers to original L; after Conv1d with kernel k, stride s:
        # L' = (L - k) // s + 1.  We map the mask to L' resolution.
        if padding_mask is not None:
            k      = self.primary.conv.kernel_size[0]
            s      = self.primary.conv.stride[0]
            L_prime = x.shape[2]
            # A position l' is valid iff the end of its receptive field (l'*s + k-1)
            # is not a padding position in the original sequence.
            indices = torch.arange(L_prime, device=x.device) * s + (k - 1)
            indices = indices.clamp(max=padding_mask.shape[1] - 1)
            mask_Lp = padding_mask[:, indices]                               # (B, L')
            valid   = (~mask_Lp).float().unsqueeze(1).unsqueeze(-1)          # (B, 1, L', 1)
            x = (x * valid).sum(dim=2) / (valid.sum(dim=2) + 1e-8)          # (B, out_caps, embed_dim)
        else:
            x = x.mean(dim=2)   # (B, out_caps, embed_dim)

        x = x.mean(dim=1)       # (B, embed_dim) — pool over capsule slots
        return x


# ─────────────────────────────────────────────────────────────────────────────
#  Full model
# ─────────────────────────────────────────────────────────────────────────────

class MambaCapsuleDTI(nn.Module):
    """
    Drug-Target Interaction predictor.

    Protein : Mamba SSM  →  CapsuleBranch (PrimaryCaps→DigitCaps)  →  (B, 128)
    Drug    : Morgan FP MLP (2048 → 256 → 128)                    →  (B, 128)
    Fusion  : concat  →  MLP  →  logit

    No attention. No transformers.
    Drug side has no capsules — just a fixed-length fingerprint MLP.

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
    ):
        super().__init__()

        # ── Protein branch ────────────────────────────────────────────────────
        self.protein_encoder = ProteinEncoder(d_model=d_model)
        self.protein_caps    = CapsuleBranch(
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

        # ── Drug branch (MLP only — no capsules) ──────────────────────────────
        self.drug_encoder = DrugEncoder(
            in_dim=drug_in_dim, hidden_dim=256, out_dim=drug_proj_dim,
        )

        # ── Fusion MLP — small init gain keeps logits near 0 at start ─────────
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

        # Protein  (B, 220) → (B, 128)
        p = self.protein_encoder(protein)               # (B, 220, 64)
        p = self.protein_caps(p)                        # (B, 128)

        # Drug  (B, 2048) → (B, 128)
        d = self.drug_encoder(drug)                     # (B, 128)

        # Fusion  (B, 256) → (B,)
        x = torch.cat([p, d], dim=-1)                  # (B, 256)
        return self.classifier(x).squeeze(-1)           # (B,)


# ─────────────────────────────────────────────────────────────────────────────
#  Dataset / DataLoader
# ─────────────────────────────────────────────────────────────────────────────

class DTIDataset(Dataset):
    def __init__(self, interactions, protein_features, drug_embeddings):
        self.df               = interactions.reset_index(drop=True)
        self.protein_features = protein_features   # dict: {protein_id: np.ndarray}
        self.drug_embeddings  = drug_embeddings     # dict: {drug_id: Tensor/ndarray}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
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

            # ── NaN / Inf guard with diagnostic context ───────────────────────
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

    Z-scoring is required — raw PsePSSM values span orders of magnitude
    and will cause NaN in Mamba's SSM state if left unnormalised.

    ★ This function must be used in __main__ instead of pd.read_csv().
      DataFrame.keys() returns column names, not protein IDs, which
      causes the filter to produce an empty interactions DataFrame.
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

        model     = MambaCapsuleDTI().to(device)
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
    model = MambaCapsuleDTI().to(device)
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
    # ★ Use load_protein_features() — NOT pd.read_csv().
    #   pd.read_csv() returns a DataFrame; DataFrame.keys() returns column
    #   names, so the isin() filter below produces 0 matching rows.
    #   load_protein_features() returns a dict {protein_id: np.ndarray},
    #   whose .keys() correctly returns protein IDs.
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

    # ── Filter to matched IDs ─────────────────────────────────────────────────
    interactions = interactions[
        interactions["protein_id"].isin(protein_features.keys())
        & interactions["drug_id"].isin(drug_embeddings.keys())
    ].reset_index(drop=True)

    print(f"Interactions after filtering : {len(interactions):,}")
    print(f"Label distribution:\n{interactions['label'].value_counts()}\n")

    assert len(interactions) > 0, (
        "Filtered interactions is empty.\n"
        "Check that protein_id / drug_id values in your CSV match the keys "
        "in psepssm_features.csv and drugs_drugbank_embeddings.pt.\n"
        f"  Sample protein_ids (interactions) : {interactions['protein_id'].iloc[:3].tolist() if len(interactions) else 'N/A'}\n"
        f"  Sample protein_ids (features dict): {list(protein_features.keys())[:3]}\n"
        f"  Sample drug_ids    (interactions) : {interactions['drug_id'].iloc[:3].tolist() if len(interactions) else 'N/A'}\n"
        f"  Sample drug_ids    (embeddings)   : {list(drug_embeddings.keys())[:3]}"
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