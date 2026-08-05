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

import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba


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


class DrugEncoder(nn.Module):
    """
    Drug encoder: Morgan fingerprints → MLP → embedding

    Input: (B, 2048)
    Output: (B, 128)
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
        return self.mlp(x)


class CrossAttention(nn.Module):
    """
    Single-direction cross-attention: query attends to context.

    Input: query (B, L_q, q_dim), context (B, L_c, c_dim)
    Output: (B, L_q, q_dim)
    """

    def __init__(
        self,
        q_dim: int,
        c_dim: int,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        assert q_dim % num_heads == 0, f"q_dim={q_dim} must be divisible by num_heads={num_heads}"

        self.attn = nn.MultiheadAttention(
            embed_dim=q_dim,
            num_heads=num_heads,
            kdim=c_dim,
            vdim=c_dim,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(q_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        context: torch.Tensor,
        context_pad_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        out, _ = self.attn(query, context, context, key_padding_mask=context_pad_mask)
        return self.norm(query + self.dropout(out))


class PrimaryCaps(nn.Module):
    """
    Primary capsule layer using GroupNorm for stability.

    Input: (B, L_in, in_channels)
    Output: (B, num_caps, L_out, cap_dim)
    """

    def __init__(
        self,
        in_channels: int,
        num_caps: int = 8,
        cap_dim: int = 16,
        kernel_size: int = 9,
        stride: int = 2,
    ):
        super().__init__()
        self.num_caps = num_caps
        self.cap_dim = cap_dim

        out_channels = num_caps * cap_dim

        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=0,
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
