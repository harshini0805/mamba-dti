"""
MambaCapsuleCross-DTI  (Capsule → Cross-Attention order)
=========================================================
Drug-Target Interaction prediction via:
    Protein : Mamba SSM → CapsuleBranch → CrossAttention(Q=protein_caps, K/V=drug) → (B, 128)
    Drug    : Morgan FP MLP (2048 → 256 → 128) → CrossAttention(Q=drug, K/V=protein_caps) → (B, 128)
    Fusion  : concat (B, 256) → MLP → logit (BCEWithLogitsLoss)

Capsule layer applies to protein BEFORE cross-attention, allowing attention
to operate on capsule-encoded protein representations.

NaN fixes applied
─────────────────
1. squash()       : sq_norm clamped to min=1e-8 before sqrt in unit branch.
2. PrimaryCaps    : GroupNorm replaces BatchNorm1d.
3. DigitCaps W    : orthogonal init; routing logits clamped [-10, 10].
4. ProteinEncoder : post-Mamba LayerNorm.
5. CapsuleBranch  : output LayerNorm before pooling.
6. Classifier     : xavier_uniform_ gain=0.5; bias → 0.
7. run_epoch      : per-batch NaN/Inf guard.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba


def squash(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    Squash activation (Sabour et al. 2017).

    Backward-safe: sq_norm clamped to min=1e-8 only for the unit vector
    branch to prevent 1/(2*sqrt(0))=inf in autograd.
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
        return self.mlp(x)


class CrossAttention(nn.Module):
    """
    Single-direction cross-attention: query attends to context.

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
        assert q_dim % num_heads == 0
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
        out, _ = self.attn(
            query, context, context,
            key_padding_mask = context_pad_mask,
        )
        return self.norm(query + self.dropout(out))


class PrimaryCaps(nn.Module):
    """
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


class DigitCaps(nn.Module):
    """
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
    PrimaryCaps → DigitCaps → project → LayerNorm.
    Returns sequence of capsule features (NOT pooled) for use with attention.

    Input  : (B, L, encoder_dim)
    Output : (B, L_out, embed_dim)  where L_out < L due to conv stride
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns: (B, L_out, embed_dim)
        Sequence of capsule features for cross-attention (no pooling).
        """
        x = self.primary(x)           # (B, num_caps, L_out, cap_dim)
        x = self.digit(x)             # (B, out_caps, L_out, out_cap_dim) after permute
        # x shape after digit: (B, out_caps, L_out, out_cap_dim)
        # Transpose to (B, L_out, out_caps, out_cap_dim) for processing
        x = x.permute(0, 2, 1, 3)     # (B, L_out, out_caps, out_cap_dim)
        # Average over capsule dimension (dim=2)
        x = x.mean(dim=2)             # (B, L_out, out_cap_dim)
        x = self.proj(x)              # (B, L_out, embed_dim)
        x = self.norm(x)
        return x


class MambaCapsuleCrossDTI(nn.Module):
    """
    Drug-Target Interaction predictor — capsule THEN cross-attention.

    Protein : Mamba SSM → CapsuleBranch → CrossAttn(Q=caps, K/V=drug) → (B, 128)
    Drug    : Morgan FP MLP → CrossAttn(Q=drug, K/V=caps) → (B, 128)
    Fusion  : concat → MLP → logit

    Capsule layer operates on Mamba output, then cross-attention operates
    on capsule-encoded representations.

    Forward args
    ────────────
    protein : (B, 220)
    drug    : (B, 2048)

    Returns
    ───────
    logits : (B,)
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

        # Encoders
        self.protein_encoder = ProteinEncoder(d_model=d_model)
        self.drug_encoder    = DrugEncoder(
            in_dim=drug_in_dim, hidden_dim=256, out_dim=drug_proj_dim,
        )

        # Capsule branch (on protein FIRST)
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

        # Cross-attention (AFTER capsules)
        # protein_caps output is (B, embed_dim), need to unsqueeze for attention
        self.protein_cross = CrossAttention(
            q_dim     = embed_dim,
            c_dim     = drug_proj_dim,
            num_heads = num_heads,
            dropout   = attn_dropout,
        )
        self.drug_cross = CrossAttention(
            q_dim     = drug_proj_dim,
            c_dim     = embed_dim,
            num_heads = num_heads,
            dropout   = attn_dropout,
        )

        # Fusion MLP
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
        protein : torch.Tensor,
        drug    : torch.Tensor,
    ) -> torch.Tensor:

        # Encode
        p = self.protein_encoder(protein)   # (B, 220, d_model)
        d = self.drug_encoder(drug)         # (B, drug_proj_dim)

        # Capsule branch FIRST (on protein) - outputs sequence
        p = self.protein_caps(p)            # (B, L_out, embed_dim)

        # Cross-attention AFTER capsules on sequence
        d_expanded = d.unsqueeze(1)         # (B, 1, drug_proj_dim) for cross-attention

        # protein sequence attends to drug (context)
        p = self.protein_cross(p, d_expanded, context_pad_mask=None)  # (B, L_out, embed_dim)

        # drug attends to protein sequence (context)
        d_expanded = self.drug_cross(d_expanded, p, context_pad_mask=None)  # (B, 1, drug_proj_dim)
        d = d_expanded.squeeze(1)           # (B, drug_proj_dim)

        # Pool protein capsule sequence to get final embedding
        p = p.mean(dim=1)                   # (B, embed_dim)

        # Fusion & Decode
        x = torch.cat([p, d], dim=-1)
        return self.classifier(x).squeeze(-1)
