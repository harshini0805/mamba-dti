"""
Mamba+AttentionPool+MLP DTI Architecture

Drug-Target Interaction prediction via:
- Protein: Mamba SSM → AttentionPool → (B, 128)
- Drug: MLP → unsqueeze(1) → AttentionPool → (B, 128)
- Fusion: concat (B, 256) → MLP → binary logit

Mamba is on protein side ONLY. Drug uses simple MLP projection + attention pooling.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba


class AttentionPool(nn.Module):
    """
    Learned attention pooling over a sequence (fixed or variable length).

    Mechanism:
      1. Linear layer scores each position → (B, L, 1)
      2. Softmax over positions → (B, L, 1) attention weights
      3. Weighted sum across positions → (B, in_dim)
      4. Linear projection → (B, embed_dim)
      5. LayerNorm → (B, embed_dim)

    NaN safety: Masking uses -1e9 instead of -inf so softmax returns finite values
    even if all positions are masked (edge case that shouldn't occur with real data).

    Input: (B, L, in_dim) sequence or (B, 1, in_dim) single vector
    Output: (B, embed_dim) pooled embedding
    """

    def __init__(self, in_dim: int, embed_dim: int):
        super().__init__()
        # Score layer: output 1 scalar per position
        self.score = nn.Linear(in_dim, 1)
        # Projection to embedding dimension
        self.proj = nn.Linear(in_dim, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)

        # Small gain on score layer keeps initial attention close to uniform
        nn.init.xavier_uniform_(self.score.weight, gain=0.1)
        nn.init.zeros_(self.score.bias)
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(
        self,
        x: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (B, L, in_dim) sequence
            padding_mask: (B, L) bool, True for positions to mask (optional)

        Returns:
            (B, embed_dim) pooled embedding
        """
        # Score each position
        scores = self.score(x)  # (B, L, 1)

        # Apply padding mask if provided
        if padding_mask is not None:
            # -1e9 instead of -inf prevents NaN in softmax
            scores = scores.masked_fill(padding_mask.unsqueeze(-1), -1e9)

        # Attention weights via softmax
        weights = F.softmax(scores, dim=1)  # (B, L, 1)

        # Weighted sum across sequence
        pooled = (weights * x).sum(dim=1)  # (B, in_dim)

        # Project and normalize
        return self.norm(self.proj(pooled))  # (B, embed_dim)


class ProteinEncoder(nn.Module):
    """
    Maps PsePSSM vectors to fixed-size embedding via Mamba + LayerNorm.

    Input: (B, 220) PsePSSM features
    Output: (B, 220, d_model) Mamba-encoded sequence
    """

    def __init__(self, d_model: int = 64):
        super().__init__()
        self.d_model = d_model

        # Project scalars to d_model dimension
        self.project = nn.Linear(1, d_model)
        self.pre_norm = nn.LayerNorm(d_model)

        # Mamba SSM for sequence modeling
        self.mamba = Mamba(
            d_model=d_model,
            d_state=16,
            d_conv=4,
            expand=2,
        )

        # Stabilize Mamba output scale
        self.post_norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 220) PsePSSM features
        Returns:
            (B, 220, d_model) Mamba output sequence
        """
        # Project each feature independently
        x = x.unsqueeze(-1)  # (B, 220, 1)
        x = self.project(x)  # (B, 220, d_model)

        # Pre-normalization before Mamba
        x = self.pre_norm(x)

        # Mamba SSM processes sequence
        x = self.mamba(x)  # (B, 220, d_model)

        # Normalize output for stability
        x = self.post_norm(x)
        return x


class DrugEncoder(nn.Module):
    """
    Projects Morgan fingerprints to embedding, then unsqueezes to sequence format.

    Fixed-length fingerprints (2048 bits) → MLP → (B, 1, out_dim)

    The unsqueeze(1) creates a "sequence" of length 1 so AttentionPool can
    accept it in the same (B, L, dim) format as the protein.

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
            (B, 1, out_dim) single-position sequence for AttentionPool
        """
        x = self.net(x)  # (B, out_dim)
        return x.unsqueeze(1)  # (B, 1, out_dim)


class MambaAttnDTI(nn.Module):
    """
    Binary DTI classifier with Mamba protein encoder and attention pooling.

    Architecture:
      1. ProteinEncoder: PsePSSM (220) → Mamba → (B, 220, d_model)
      2. ProteinAttentionPool: (B, 220, d_model) → (B, 128)
      3. DrugEncoder: Morgan FP (2048) → MLP → (B, 1, 128)
      4. DrugAttentionPool: (B, 1, 128) → (B, 128)
      5. Decoder: MLP (256 → 128 → 1)
      6. Output: (B,) binary logits

    Loss: BCEWithLogitsLoss

    Key points:
    - Mamba is protein-side only (✓)
    - Drug uses simple MLP + attention (no sequence modeling)
    - AttentionPool is learned weighted mean with projection
    """

    def __init__(
        self,
        drug_input_dim: int = 2048,
        d_model: int = 64,
        embed_dim: int = 128,
    ) -> None:
        super().__init__()

        # ── Protein branch (Mamba + AttentionPool) ──────────────────────────
        self.protein_encoder = ProteinEncoder(d_model=d_model)
        self.protein_pool = AttentionPool(in_dim=d_model, embed_dim=embed_dim)

        # ── Drug branch (MLP + AttentionPool) ────────────────────────────────
        self.drug_encoder = DrugEncoder(
            input_dim=drug_input_dim,
            hidden_dim=256,
            out_dim=embed_dim,
        )
        self.drug_pool = AttentionPool(in_dim=embed_dim, embed_dim=embed_dim)

        # ── Fusion decoder (MLP) ─────────────────────────────────────────────
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim * 2, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
        )

        # Initialize classifier weights
        self._init_classifier()

    def _init_classifier(self) -> None:
        """Initialize classifier with Xavier uniform (gain=0.5) and zero bias."""
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, protein: torch.Tensor, drug: torch.Tensor) -> torch.Tensor:
        """
        Args:
            protein: (B, 220) PsePSSM features
            drug: (B, 2048) Morgan fingerprints

        Returns:
            (B,) binary logits (pre-sigmoid)
        """
        # Protein: Mamba → AttentionPool
        p = self.protein_encoder(protein)  # (B, 220, d_model)
        p = self.protein_pool(p)  # (B, embed_dim)

        # Drug: MLP → unsqueeze → AttentionPool
        d = self.drug_encoder(drug)  # (B, 1, embed_dim)
        d = self.drug_pool(d)  # (B, embed_dim)

        # Fusion: concatenate and decode
        x = torch.cat([p, d], dim=-1)  # (B, embed_dim * 2)
        logits = self.classifier(x)  # (B, 1)
        return logits.squeeze(-1)  # (B,)
