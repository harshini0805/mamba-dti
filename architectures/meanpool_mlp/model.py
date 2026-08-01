"""
MeanPool+MLP DTI Architecture (Baseline)

Drug-Target Interaction prediction via:
- Protein: Linear proj → LayerNorm → mean pool → proj → LayerNorm → (B, 128)
- Drug: MLP encoder → (B, 128)
- Fusion: concat (B, 256) → MLP → binary logit

NO sequential modeling. NO Mamba. NO capsules. NO attention.
This is the floor baseline for ablation series.

Rationale: If any sequential/attention/routing version fails to beat this,
its added complexity is unjustified on this task.
"""

import torch
import torch.nn as nn


class ProteinBranch(nn.Module):
    """
    Baseline protein branch: no sequential modeling whatsoever.

    Each of the 220 PsePSSM positions is treated independently;
    order is discarded by mean pooling.

    Operations:
      1. Unsqueeze: (B, 220) → (B, 220, 1)
      2. Project: (B, 220, 1) → (B, 220, d_model)
      3. LayerNorm: Normalize
      4. Mean pool: (B, 220, d_model) → (B, d_model)
      5. Project to embed_dim: (B, d_model) → (B, embed_dim)
      6. LayerNorm: Final normalization

    Input: (B, 220) PsePSSM features
    Output: (B, embed_dim) fixed representation
    """

    def __init__(self, d_model: int = 64, embed_dim: int = 128):
        super().__init__()
        self.d_model = d_model

        # Initial projection and norm
        self.proj = nn.Linear(1, d_model)
        self.norm = nn.LayerNorm(d_model)

        # Final projection and norm
        self.out_proj = nn.Linear(d_model, embed_dim)
        self.out_norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 220) PsePSSM features
        Returns:
            (B, embed_dim) baseline protein representation
        """
        # Project and normalize each position independently
        x = x.unsqueeze(-1)  # (B, 220, 1)
        x = self.proj(x)  # (B, 220, d_model)
        x = self.norm(x)  # (B, 220, d_model)

        # Mean pooling over all positions (discard order)
        x = x.mean(dim=1)  # (B, d_model)

        # Final projection and normalization
        x = self.out_proj(x)  # (B, embed_dim)
        x = self.out_norm(x)  # (B, embed_dim)
        return x


class DrugBranch(nn.Module):
    """
    Projects Morgan fingerprints to dense embedding via MLP.

    Identical to other architectures - fingerprints are fixed-length,
    no sequential structure, MLP is appropriate.

    Input: (B, 2048) Morgan fingerprints
    Output: (B, embed_dim) drug representation
    """

    def __init__(
        self,
        input_dim: int = 2048,
        hidden_dim: int = 256,
        embed_dim: int = 128,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 2048) Morgan fingerprints
        Returns:
            (B, embed_dim) drug representation
        """
        return self.net(x)


class MeanPoolDTI(nn.Module):
    """
    Baseline DTI classifier with no sequential modeling.

    Protein Branch:
      - Linear projection of each position independently
      - Mean pooling (discards order)
      - No sequential modeling whatsoever

    Drug Branch:
      - Simple MLP on fixed-length fingerprints

    Fusion:
      - Concatenate: (B, 256)
      - MLP decoder: (256 → 128 → 1)

    Loss: BCEWithLogitsLoss

    Purpose:
    Baseline for ablation study. If sequence-based versions (BiLSTM, Mamba, etc.)
    don't beat this, the added complexity is unjustified.

    Matched architecture:
    - Same d_model=64 as Mamba versions (same capacity per position)
    - Same embed_dim=128 output
    - Same classifier MLP head
    - Parameter count comparable to other versions
    """

    def __init__(
        self,
        drug_input_dim: int = 2048,
        d_model: int = 64,
        embed_dim: int = 128,
    ):
        super().__init__()

        # Protein branch: project + mean pool (no seq modeling)
        self.protein_branch = ProteinBranch(
            d_model=d_model,
            embed_dim=embed_dim,
        )

        # Drug branch: MLP (identical to other versions)
        self.drug_branch = DrugBranch(
            input_dim=drug_input_dim,
            hidden_dim=256,
            embed_dim=embed_dim,
        )

        # Classifier MLP (same as other versions)
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim * 2, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize weights: Xavier uniform (gain=0.5), zero bias."""
        for m in self.modules():
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
        # Encode independently
        p = self.protein_branch(protein)  # (B, embed_dim)
        d = self.drug_branch(drug)  # (B, embed_dim)

        # Concatenate and decode
        x = torch.cat([p, d], dim=-1)  # (B, embed_dim * 2)
        return self.classifier(x).squeeze(-1)  # (B,)
