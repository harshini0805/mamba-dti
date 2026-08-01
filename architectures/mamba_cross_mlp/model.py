"""
Mamba+CrossAttention+MLP DTI Architecture

Drug-Target Interaction prediction via:
- Protein: Mamba SSM → CrossAttention (query) → mean pool → (B, 128)
- Drug: MLP → unsqueeze(1) → CrossAttention (query) → squeeze → (B, 128)
- Fusion: concat (B, 256) → MLP → binary logit

Bidirectional cross-attention:
  - Drug attends to Protein (drug=query, protein=key/value)
  - Protein attends to Drug (protein=query, drug=key/value)

Mamba is on protein side ONLY (✓)
"""

import torch
import torch.nn as nn
from mamba_ssm import Mamba


class ProteinEncoder(nn.Module):
    """
    Maps PsePSSM vectors to contextual sequence via Mamba.

    Operations:
      1. Unsqueeze: (B, 220) → (B, 220, 1)
      2. Project: (B, 220, 1) → (B, 220, d_model)
      3. Mamba: (B, 220, d_model) → (B, 220, d_model) [long-range deps]
      4. LayerNorm: Stabilize training

    Output kept as full sequence (no pooling yet) so cross-attention
    can access all 220 positions when drug attends to it.

    Input: (B, 220) PsePSSM features
    Output: (B, 220, d_model) Mamba-encoded sequence
    """

    def __init__(self, d_model: int = 128):
        super().__init__()
        self.d_model = d_model

        self.project = nn.Linear(1, d_model)
        self.mamba = Mamba(
            d_model=d_model,
            d_state=16,
            d_conv=4,
            expand=2,
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 220) PsePSSM features
        Returns:
            (B, 220, d_model) Mamba-encoded sequence
        """
        x = x.unsqueeze(-1)  # (B, 220, 1)
        x = self.project(x)  # (B, 220, d_model)
        x = self.mamba(x)    # (B, 220, d_model)
        x = self.norm(x)     # (B, 220, d_model) — stabilize scale
        return x


class DrugEncoder(nn.Module):
    """
    Projects Morgan fingerprints to embedding, unsqueezed for cross-attention.

    Fixed-length fingerprints (2048) → MLP → (B, out_dim) → unsqueeze(1) → (B, 1, out_dim)

    The unsqueeze creates a degenerate sequence of length T=1 so it can
    feed into cross-attention layers with standard (B, T, dim) shape.

    Input: (B, 2048) Morgan fingerprints
    Output: (B, 1, out_dim) single-position sequence
    """

    def __init__(
        self,
        input_dim: int = 2048,
        hidden_dim: int = 256,
        out_dim: int = 128,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 2048) Morgan fingerprints
        Returns:
            (B, 1, out_dim) single-position sequence
        """
        x = self.net(x)  # (B, out_dim)
        return x.unsqueeze(1)  # (B, 1, out_dim)


class MambaCrossAttentionDTI(nn.Module):
    """
    Binary DTI classifier with bidirectional cross-attention fusion.

    Architecture:
      1. ProteinEncoder: PsePSSM (220) → Mamba → (B, 220, 128)
      2. DrugEncoder: Morgan FP (2048) → MLP → unsqueeze → (B, 1, 128)
      3. CrossAttention (bidirectional):
         - Drug (Q) attends to Protein (K, V) → (B, 1, 128)
         - Protein (Q) attends to Drug (K, V) → (B, 220, 128)
      4. Post-residual layer norms
      5. Pooling:
         - Protein: mean over 220 positions → (B, 128)
         - Drug: squeeze dimension 1 → (B, 128)
      6. Decoder: MLP (256 → 128 → 1)
      7. Output: (B,) binary logits

    Loss: BCEWithLogitsLoss

    Key points:
    - Mamba is protein-side only (✓)
    - Cross-attention is bidirectional (both ways)
    - Post-residual layer norm (numerically stable)
    - No padding masks needed (fixed-length sequences)
    """

    def __init__(self, drug_input_dim: int = 2048):
        super().__init__()

        # ── Encoders ────────────────────────────────────────────────────────
        self.protein_encoder = ProteinEncoder(d_model=128)
        self.drug_encoder = DrugEncoder(input_dim=drug_input_dim, out_dim=128)

        # ── Cross-Attention Layers ──────────────────────────────────────────
        # Drug (Q) attends to Protein (K, V)
        self.cross_dp = nn.MultiheadAttention(
            embed_dim=128,
            num_heads=4,
            batch_first=True,
        )
        # Protein (Q) attends to Drug (K, V)
        self.cross_pd = nn.MultiheadAttention(
            embed_dim=128,
            num_heads=4,
            batch_first=True,
        )

        # ── Post-Residual Layer Norms ───────────────────────────────────────
        self.norm_d = nn.LayerNorm(128)
        self.norm_p = nn.LayerNorm(128)

        # ── Decoder MLP ─────────────────────────────────────────────────────
        self.decoder = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
        )

    def forward(self, protein: torch.Tensor, drug: torch.Tensor) -> torch.Tensor:
        """
        Args:
            protein: (B, 220) PsePSSM features
            drug: (B, 2048) Morgan fingerprints

        Returns:
            (B,) binary logits (pre-sigmoid)
        """
        # ── Encode ────────────────────────────────────────────────────────
        P = self.protein_encoder(protein)  # (B, 220, 128)
        D = self.drug_encoder(drug)  # (B, 1, 128)

        # ── Cross-Attention (Bidirectional) ───────────────────────────────
        # Drug (Q) attends to Protein (K, V)
        # Protein has no padding → no key_padding_mask needed
        D_attn, _ = self.cross_dp(query=D, key=P, value=P)  # (B, 1, 128)

        # Protein (Q) attends to Drug (K, V)
        # Drug is T=1 with no padding → no key_padding_mask needed
        P_attn, _ = self.cross_pd(query=P, key=D, value=D)  # (B, 220, 128)

        # ── Residual + Post-LayerNorm ──────────────────────────────────────
        D = self.norm_d(D + D_attn)  # (B, 1, 128) post-residual LN
        P = self.norm_p(P + P_attn)  # (B, 220, 128) post-residual LN

        # ── Pooling ───────────────────────────────────────────────────────
        # Protein: mean pooling over 220 positions
        P_vec = P.mean(dim=1)  # (B, 128)

        # Drug: squeeze the degenerate T=1 dimension
        D_vec = D.squeeze(1)  # (B, 128)

        # ── Fusion & Decode ────────────────────────────────────────────────
        x = torch.cat([P_vec, D_vec], dim=-1)  # (B, 256)
        logits = self.decoder(x)  # (B, 1)
        return logits.squeeze(-1)  # (B,)
