import torch
import torch.nn as nn

from mamba_ssm import Mamba


class ProteinEncoder(nn.Module):
    """
    Protein encoder using Mamba over PsePSSM features.

    Input:
        (B, 220)

    Output:
        (B, 128)
    """

    def __init__(
        self,
        input_dim: int = 220,
        d_model: int = 128,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
    ):
        super().__init__()

        self.project = nn.Linear(1, d_model)

        self.mamba = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )

        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):

        # (B,220)
        x = x.unsqueeze(-1)

        # (B,220,128)
        x = self.project(x)

        # (B,220,128)
        x = self.mamba(x)

        # (B,220,128)
        x = self.norm(x)

        # Global Average Pooling
        x = x.mean(dim=1)

        return x