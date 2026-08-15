"""
Configuration for EMM-DTI (FCS + Mamba-SSM + CNN + MLP) Architecture.

Faithful replication of EMM-DTI from paper while maintaining standardized
hyperparameters consistent with all other mamba-dti architectures.
"""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class EMMDTICONFIG:
    """Training and model configuration for EMM-DTI."""

    # ─── Model Architecture ────────────────────────────────────────────────
    # FCS Fragment Encoding
    fcs_embedding_dim: int = 128
    fcs_min_support: float = 0.3
    fcs_max_k: int = 3  # Max k-mer size (1,2,3-mers)

    # Bidirectional Mamba-SSM Encoder
    mamba_hidden_dim: int = 256
    mamba_n_layers: int = 2
    mamba_d_state: int = 16
    mamba_expand_factor: int = 2

    # CNN Feature Extraction
    cnn_out_channels: int = 3
    cnn_kernel_size: int = 3
    cnn_stride: int = 1
    cnn_padding: int = 1

    # MLP Decoder
    mlp_hidden_dim: int = 128
    mlp_dropout: float = 0.1
    dropout: float = 0.1

    # ─── Training Hyperparameters ───────────────────────────────────────
    num_epochs: int = 200
    batch_size: int = 16
    learning_rate: float = 3e-4  # Same across all architectures for fair comparison
    weight_decay: float = 1e-4
    gradient_clip_norm: float = 1.0
    dropout_rate: float = 0.1

    # ─── Cross-Validation ───────────────────────────────────────────────
    num_folds: int = 5
    random_state: int = 42
    patience: int = 30  # Early stopping patience

    # ─── Loss & Metrics ──────────────────────────────────────────────────
    loss_fn: str = "bce_with_logits"  # Binary classification
    metric_keys: list = field(
        default_factory=lambda: [
            "accuracy",
            "precision",
            "recall",
            "specificity",
            "mcc",
            "roc_auc",
            "pr_auc",
        ]
    )

    # ─── Data Paths (relative to project root) ──────────────────────────
    # Will be overridden by dataset config at runtime
    data_dir: Path = Path("data")
    results_dir: Path = Path("architectures/emm_dti/results")
    logs_dir: Path = Path("architectures/emm_dti/logs")
    checkpoints_dir: Path = Path("architectures/emm_dti/checkpoints")

    def __post_init__(self):
        """Ensure all path fields are Path objects."""
        if isinstance(self.data_dir, str):
            self.data_dir = Path(self.data_dir)
        if isinstance(self.results_dir, str):
            self.results_dir = Path(self.results_dir)
        if isinstance(self.logs_dir, str):
            self.logs_dir = Path(self.logs_dir)
        if isinstance(self.checkpoints_dir, str):
            self.checkpoints_dir = Path(self.checkpoints_dir)


# Default config instance
default_config = EMMDTICONFIG()
