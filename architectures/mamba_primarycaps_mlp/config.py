"""
Configuration for Mamba+PrimaryCaps+MLP DTI architecture.
"""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class MambaPrimaryCapsMLPConfig:
    """Training and model configuration."""

    # ─── Model Architecture ────────────────────────────────────────────────
    protein_d_model: int = 64
    protein_embedding_dim: int = 128
    drug_input_dim: int = 2048
    drug_embedding_dim: int = 128

    # ─── PrimaryCaps ────────────────────────────────────────────────────
    num_caps: int = 8
    cap_dim: int = 16
    kernel_size: int = 9
    stride: int = 2

    # ─── Decoder MLP ────────────────────────────────────────────────────
    decoder_hidden_dim: int = 256
    decoder_dropout: float = 0.3

    # ─── Training Hyperparameters ───────────────────────────────────────
    num_epochs: int = 200
    batch_size: int = 16
    learning_rate: float = 3e-4  # Same across all architectures for fair comparison
    weight_decay: float = 1e-4
    gradient_clip_norm: float = 1.0
    dropout_rate: float = 0.3

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
    results_dir: Path = Path("architectures/mamba_primarycaps_mlp/results")
    logs_dir: Path = Path("architectures/mamba_primarycaps_mlp/logs")
    checkpoints_dir: Path = Path("architectures/mamba_primarycaps_mlp/checkpoints")

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
default_config = MambaPrimaryCapsMLPConfig()
