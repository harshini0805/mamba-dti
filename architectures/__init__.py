"""
Architectures module.

Each subdirectory (mamba_mlp_fp, attention_mlp, etc.) contains a complete,
self-contained architecture implementation with its own:
  - model.py: Architecture definition
  - config.py: Hyperparameters
  - train.py: Training script
  - results/, logs/, checkpoints/: Per-dataset outputs

Usage:
    python architectures/{arch_name}/train.py --dataset {dataset_name}
"""
