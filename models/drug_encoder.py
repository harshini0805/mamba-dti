import torch
import torch.nn as nn


class DrugEncoder(nn.Module):
    """
    MLP encoder for Morgan fingerprints.

    Input:
        (B, 2048)

    Output:
        (B, 128)
    """

    def __init__(
        self,
        input_dim: int = 2048,
        hidden_dim: int = 256,
        output_dim: int = 128,
        dropout: float = 0.3,
    ):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.encoder(x)