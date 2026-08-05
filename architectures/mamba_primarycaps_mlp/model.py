"""
Mamba+PrimaryCaps DTI Architecture
- Mamba-based protein encoder with capsule layer
- MLP drug encoder (Morgan fingerprints)
- PrimaryCaps for protein sequence modeling
- MLP decoder for binary DTI prediction
"""

import torch
import torch.nn as nn
from mamba_ssm import Mamba


def squash(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    Squash activation for capsule networks.
    Maps vectors to length in [0, 1) while preserving direction.
    """
    sq_norm = (x ** 2).sum(dim=dim, keepdim=True)
    scale = sq_norm / (1.0 + sq_norm)
    unit = x / (sq_norm.clamp(min=1e-8).sqrt())
    return scale * unit


class ProteinEncoder(nn.Module):
    """
    Protein encoder: PsePSSM → projection → Mamba → LayerNorm

    Input: (B, 220)
    Output: (B, 220, d_model)
    """

    def __init__(self, d_model: int = 64):
        super().__init__()
        self.project = nn.Linear(1, d_model)
        self.pre_norm = nn.LayerNorm(d_model)
        self.mamba = Mamba(d_model=d_model, d_state=16, d_conv=4, expand=2)
        self.post_norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(-1)  # (B, 220, 1)
        x = self.project(x)  # (B, 220, d_model)
        x = self.pre_norm(x)
        x = self.mamba(x)
        x = self.post_norm(x)
        return x


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
        x = x.transpose(1, 2)  # (B, in_channels, L_in)
        x = self.conv(x)  # (B, num_caps*cap_dim, L_out)
        x = self.norm(x)

        B, _, L_out = x.shape
        x = x.view(B, self.num_caps, self.cap_dim, L_out)
        x = x.permute(0, 1, 3, 2)  # (B, num_caps, L_out, cap_dim)
        return squash(x, dim=-1)


class CapsuleBranch(nn.Module):
    """
    PrimaryCaps → project → LayerNorm → masked mean pool

    Input: (B, L, encoder_dim)
    Output: (B, embed_dim)
    """

    def __init__(
        self,
        encoder_dim: int,
        num_caps: int = 8,
        cap_dim: int = 16,
        embed_dim: int = 128,
        kernel_size: int = 9,
        stride: int = 2,
    ):
        super().__init__()
        self.primary = PrimaryCaps(
            in_channels=encoder_dim,
            num_caps=num_caps,
            cap_dim=cap_dim,
            kernel_size=kernel_size,
            stride=stride,
        )
        self.proj = nn.Linear(cap_dim, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, L, encoder_dim)
        returns: (B, embed_dim)
        """
        x = self.primary(x)  # (B, num_caps, L', cap_dim)
        x = self.proj(x)  # (B, num_caps, L', embed_dim)
        x = self.norm(x)

        # Mean pooling over sequence and capsule dimensions
        x = x.mean(dim=2)  # (B, num_caps, embed_dim)
        x = x.mean(dim=1)  # (B, embed_dim)
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


class MambaPrimaryCapsMLPDTI(nn.Module):
    """
    Binary DTI classifier with Mamba protein encoder and PrimaryCaps.

    Architecture:
      1. ProteinEncoder: PsePSSM (220) → Mamba → (B, 220, 64)
      2. CapsuleBranch: PrimaryCaps → projection → (B, 128)
      3. DrugEncoder: Morgan FP (2048) → MLP → (B, 128)
      4. Concatenate: (256)
      5. Decoder: MLP (256 → 256 → 128 → 1)
      6. Output: (B,) binary logits

    Loss: BCEWithLogitsLoss
    """

    def __init__(
        self,
        drug_input_dim: int = 2048,
        d_model: int = 64,
        num_caps: int = 8,
        cap_dim: int = 16,
        embed_dim: int = 128,
    ):
        super().__init__()

        # Protein branch
        self.protein_encoder = ProteinEncoder(d_model=d_model)
        self.protein_caps = CapsuleBranch(
            encoder_dim=d_model,
            num_caps=num_caps,
            cap_dim=cap_dim,
            embed_dim=embed_dim,
            kernel_size=9,
            stride=2,
        )

        # Drug branch
        self.drug_encoder = DrugEncoder(in_dim=drug_input_dim, out_dim=embed_dim)

        # Fusion decoder
        self.decoder = nn.Sequential(
            nn.Linear(embed_dim * 2, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.decoder.modules():
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
        # Protein: Mamba → CapsuleBranch → (B, 128)
        p = self.protein_encoder(protein)
        p = self.protein_caps(p)

        # Drug: MLP → (B, 128)
        d = self.drug_encoder(drug)

        # Fusion & Decode
        x = torch.cat([p, d], dim=-1)
        logits = self.decoder(x)
        return logits.squeeze(-1)
