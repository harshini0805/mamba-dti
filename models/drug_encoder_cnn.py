import torch
import torch.nn as nn


class CNNDrugEncoder(nn.Module):
    """
    1D-CNN encoder for Morgan fingerprints — drop-in alternative to
    DrugEncoder (models/drug_encoder.py, plain MLP), same (B, 2048) -> (B, 128)
    interface so it can be swapped into MambaDTI's drug-encoder slot without
    changing anything else (see models/mamba_cnn_dti.py).

    Treats the 2048-bit fingerprint as a single-channel 1D signal and stacks
    Conv1d -> ReLU -> MaxPool1d blocks over it to learn local bit
    co-occurrence patterns (adjacent Morgan bits are NOT spatially
    meaningful the way pixels or sequence positions are, so this is a
    heuristic inductive bias, not a physically motivated one — same
    trade-off any CNN-over-fingerprint DTI model makes).

    Deliberately uses no BatchNorm anywhere in this file. Every other model
    in this project (ProteinEncoder's LayerNorm, MambaDTI's plain
    Linear/Dropout) avoids BatchNorm specifically so checkpoint averaging
    (SWA, see train_swa.py) works with a plain per-parameter weight mean
    and no running-stats recalibration pass. Adding BatchNorm here would
    silently break that property for this variant if SWA is ever tried on
    it — so it's left out on purpose, not an oversight.

    Input:
        (B, 2048)

    Output:
        (B, 128)
    """

    def __init__(
        self,
        input_dim: int = 2048,
        output_dim: int = 128,
        channels=(32, 64, 128),
        kernel_size: int = 7,
        dropout: float = 0.3,
    ):
        super().__init__()

        assert kernel_size % 2 == 1, (
            "kernel_size must be odd so padding=kernel_size//2 preserves "
            "the sequence length exactly (stride=1, dilation=1)."
        )

        layers = []
        in_channels = 1

        for out_channels in channels:

            layers.append(
                nn.Conv1d(
                    in_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    padding=kernel_size // 2,
                )
            )
            layers.append(nn.ReLU())
            layers.append(nn.MaxPool1d(kernel_size=2))
            layers.append(nn.Dropout(dropout))

            in_channels = out_channels

        self.conv = nn.Sequential(*layers)

        # Global average pool collapses whatever sequence length remains
        # after pooling down to 1, so the projection head below doesn't
        # need to hardcode a length that would change if `channels` or
        # `kernel_size` are tuned later.
        self.pool = nn.AdaptiveAvgPool1d(1)

        self.project = nn.Linear(in_channels, output_dim)

    def forward(self, x):

        # (B, 2048) -> (B, 1, 2048): one input channel
        x = x.unsqueeze(1)

        x = self.conv(x)            # (B, C_last, L_last)
        x = self.pool(x)            # (B, C_last, 1)
        x = x.squeeze(-1)           # (B, C_last)

        return self.project(x)      # (B, output_dim)
