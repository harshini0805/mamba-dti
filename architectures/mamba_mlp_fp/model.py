"""
Mamba+MLP DTI Architecture
- Mamba-based protein encoder (PsePSSM → dense embedding)
- MLP drug encoder (Morgan fingerprints → dense embedding)
- MLP decoder (concatenated embeddings → binary prediction)
"""

import torch
import torch.nn as nn
from mamba_ssm import Mamba


class ProteinEncoder(nn.Module):
    """
    Maps PsePSSM vectors to a fixed-size protein embedding via Mamba.

    Rationale:
      - PsePSSM is a sequence-like feature (220 pseudo-position-specific
        scoring matrix values)
      - Mamba captures sequential dependencies before mean-pooling
      - Final output: dense 128-d embedding

    Input  : (B, 220)
    Output : (B, 128)
    """

    def __init__(self, d_model: int = 128) -> None:
        super().__init__()
        self.d_model = d_model
        # Project scalar features to model dimension
        self.project = nn.Linear(1, d_model)
        # Sequence modeling via Mamba
        self.mamba = Mamba(
            d_model=d_model,
            d_state=16,
            d_conv=4,
            expand=2,
        )
        # Normalize after sequence processing
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 220) PsePSSM features
        Returns:
            (B, 128) protein embedding
        """
        # Reshape for projection: each of 220 values is projected independently
        x = x.unsqueeze(-1)          # (B, 220, 1)
        x = self.project(x)          # (B, 220, 128)

        # Mamba learns dependencies across the 220 pseudo-positions
        x = self.mamba(x)            # (B, 220, 128)
        x = self.norm(x)             # (B, 220, 128)

        # Mean pooling over sequence dimension -> fixed embedding
        x = x.mean(dim=1)            # (B, 128)
        return x


class DrugEncoder(nn.Module):
    """
    Projects Morgan fingerprints (fixed-length bit-vectors) to dense embedding.

    Rationale:
      - Morgan fingerprints are fixed-length (e.g., 2048 bits), already vectorized
      - No sequence structure, so MLP suffices (vs. Mamba/Transformer)
      - Two-layer MLP allows learning feature interactions before compression

    Input  : (B, input_dim) binary {0., 1.} vectors
    Output : (B, 128)
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
            x: (B, input_dim) Morgan fingerprints
        Returns:
            (B, 128) drug embedding
        """
        return self.net(x)


class MambaMLPDTI(nn.Module):
    """
    Binary DTI classifier combining Mamba protein encoding and MLP drug encoding.

    Architecture:
      1. ProteinEncoder: PsePSSM (220) -> Mamba -> (128)
      2. DrugEncoder: Morgan FP (2048) -> MLP -> (128)
      3. Concatenate: (256)
      4. Decoder: MLP (256 -> 256 -> 128 -> 1)
      5. Output: binary logit (raw, not sigmoid-ed)

    Loss: BCEWithLogitsLoss (applies sigmoid + BCE internally)
    """

    def __init__(self, drug_input_dim: int = 2048) -> None:
        super().__init__()
        protein_dim = 128
        drug_dim = 128
        concat_dim = protein_dim + drug_dim

        self.protein_encoder = ProteinEncoder(d_model=protein_dim)
        self.drug_encoder = DrugEncoder(input_dim=drug_input_dim, out_dim=drug_dim)

        # MLP decoder: maps concatenated embedding to binary logit
        self.decoder = nn.Sequential(
            nn.Linear(concat_dim, 256),
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
            drug:    (B, 2048) Morgan fingerprints
        Returns:
            (B,) binary logits (pre-sigmoid)
        """
        p_vec = self.protein_encoder(protein)         # (B, 128)
        d_vec = self.drug_encoder(drug)               # (B, 128)
        x = torch.cat([p_vec, d_vec], dim=-1)         # (B, 256)
        logits = self.decoder(x)                      # (B, 1)
        return logits.squeeze(-1)                     # (B,)
