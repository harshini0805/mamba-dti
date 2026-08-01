"""
Training script for Mamba+MLP DTI architecture.

Usage:
    python train.py --dataset bindingdb
    python train.py --dataset humans --epochs 100 --batch_size 32

All config is loaded from datasets/{dataset}.py and this module's config.py.
Hyperparameters can be overridden via CLI.
"""

import argparse
import copy
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Add parent directories to path for imports
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from common.dataset_loader import DTIDataset, collate_fn
from common.metrics import compute_metrics
from config import default_config as arch_config
from model import MambaMLPDTI

# Import dataset config dynamically
def load_dataset_config(dataset_name: str):
    """Dynamically import dataset config."""
    try:
        dataset_module = __import__(
            f"datasets.{dataset_name}",
            fromlist=["config"],
        )
        return dataset_module.config
    except ImportError as e:
        raise ValueError(
            f"Dataset '{dataset_name}' not found. "
            f"Ensure datasets/{dataset_name}.py exists with a 'config' object."
        ) from e


# ─── Hyperparameter Defaults ────────────────────────────────────────────

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[float, dict]:
    """
    Run one epoch of training or validation.

    Args:
        model: DTI model
        loader: Data loader
        criterion: Loss function
        optimizer: Optimizer (None for validation/eval)

    Returns:
        (epoch_loss, metrics_dict)
    """
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss = 0.0
    all_labels = []
    all_probs = []

    ctx = torch.enable_grad() if is_train else torch.no_grad()

    with ctx:
        for protein, drug, label in loader:
            protein = protein.to(DEVICE)
            drug = drug.to(DEVICE)
            label = label.to(DEVICE)

            logits = model(protein, drug)
            loss = criterion(logits, label)

            if torch.isnan(loss):
                raise ValueError("NaN loss detected — check inputs and model weights.")

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=arch_config.gradient_clip_norm)
                optimizer.step()

            total_loss += loss.item() * label.size(0)
            probs = torch.sigmoid(logits).detach().cpu().numpy()
            all_probs.extend(probs.tolist())
            all_labels.extend(label.cpu().numpy().tolist())

    epoch_loss = total_loss / len(loader.dataset)
    metrics = compute_metrics(all_labels, all_probs)
    return epoch_loss, metrics


def train_fold(
    fold: int,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config_arch,
    config_data,
    checkpoint_dir: Path,
) -> dict:
    """
    Train one fold of cross-validation.

    Returns:
        best_val_metrics (dict)
    """
    model = MambaMLPDTI(drug_input_dim=config_data.drug_input_dim).to(DEVICE)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config_arch.learning_rate,
        weight_decay=config_arch.weight_decay,
    )
    criterion = nn.BCEWithLogitsLoss()

    best_val_roc_auc = -1.0
    best_val_metrics = None
    wait = 0

    print(f"\n{'='*70}")
    print(f"  Fold {fold} / {config_arch.num_folds}")
    print(f"{'='*70}")

    for epoch in range(1, config_arch.num_epochs + 1):
        train_loss, train_m = run_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_m = run_epoch(model, val_loader, criterion)

        # ─── Per-epoch table ───────────────────────────────────────────────
        if epoch == 1:
            header = f"  {'Metric':<16}  {'Train':>12}  {'Val':>12}"
            sep = f"  {'─'*16}  {'─'*12}  {'─'*12}"
            print(sep)
            print(header)
            print(sep)

        if epoch % 5 == 0 or epoch == 1:  # Print every 5 epochs + first epoch
            for key in config_arch.metric_keys:
                label = key.replace("_", " ").title()
                print(
                    f"  {label:<16}  {train_m[key]:>12.4f}  {val_m[key]:>12.4f}"
                )
            print(
                f"  {'Loss':<16}  {train_loss:>12.4f}  {val_loss:>12.4f}"
            )
            print(sep)

        # ─── Checkpoint & Early Stopping ────────────────────────────────
        if val_m["roc_auc"] > best_val_roc_auc:
            best_val_roc_auc = val_m["roc_auc"]
            best_val_metrics = copy.deepcopy(val_m)
            checkpoint_path = checkpoint_dir / f"best_model_fold_{fold}.pt"
            torch.save(model.state_dict(), checkpoint_path)
            print(
                f"  ✓ Saved {checkpoint_path.name} "
                f"(ROC-AUC = {best_val_roc_auc:.4f})"
            )
            wait = 0
        else:
            wait += 1
            if wait >= config_arch.patience:
                print(f"\n  Early stopping triggered at epoch {epoch}.")
                break

    return best_val_metrics


def main():
    parser = argparse.ArgumentParser(
        description="Train Mamba+MLP DTI model on specified dataset."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Dataset name (e.g., 'bindingdb', 'humans')",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        help="Override number of epochs",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        help="Override batch size",
    )
    parser.add_argument(
        "--lr",
        type=float,
        help="Override learning rate",
    )

    args = parser.parse_args()

    # Load configs
    config_data = load_dataset_config(args.dataset)
    config = arch_config

    # Override with CLI arguments
    if args.epochs:
        config.num_epochs = args.epochs
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.lr:
        config.learning_rate = args.lr

    # Create output directories
    results_dir = config.results_dir / args.dataset
    logs_dir = config.logs_dir / args.dataset
    checkpoint_dir = config.checkpoints_dir / args.dataset
    results_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nTraining Mamba+MLP on {args.dataset}")
    print(f"Results dir: {results_dir}")
    print(f"Logs dir: {logs_dir}")
    print(f"Checkpoint dir: {checkpoint_dir}")

    # Load data (pre-split into train/valid/test)
    print(f"\nLoading data from {config_data.data_dir}...")
    protein_features, drug_embeddings, train_df, val_df, test_df = config_data.load_data()

    print(f"\nLoaded {len(protein_features):,} proteins")
    print(f"Loaded {len(drug_embeddings):,} drugs")
    print(f"\nData split sizes:")
    print(f"  Train: {len(train_df):,} interactions")
    print(f"  Valid: {len(val_df):,} interactions")
    print(f"  Test:  {len(test_df):,} interactions")

    # Verify drug fingerprint dimension
    sample_drug_id = next(iter(drug_embeddings.keys()))
    fingerprint_dim = len(drug_embeddings[sample_drug_id])
    if fingerprint_dim != config_data.drug_input_dim:
        print(
            f"Warning: Expected fingerprint dim {config_data.drug_input_dim}, "
            f"got {fingerprint_dim}"
        )
        config_data.drug_input_dim = fingerprint_dim

    # Print model info
    model_info = MambaMLPDTI(drug_input_dim=config_data.drug_input_dim).to(DEVICE)
    num_params = sum(p.numel() for p in model_info.parameters() if p.requires_grad)
    print(f"\nTrainable parameters: {num_params:,}")
    del model_info

    # 5 random seeds, 5 independent runs (same train/valid split, different model init)
    SEEDS = [42, 123, 2024, 456, 789]
    run_summaries = []

    for run_idx, seed in enumerate(SEEDS, start=1):
        print(f"\n{'='*70}")
        print(f"  Run {run_idx}/5 (seed={seed})")
        print(f"{'='*70}")

        # Set random seeds for reproducibility
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

        # Use same pre-split train/valid data
        train_loader = DataLoader(
            DTIDataset(train_df, protein_features, drug_embeddings),
            batch_size=config.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=0,
        )
        val_loader = DataLoader(
            DTIDataset(val_df, protein_features, drug_embeddings),
            batch_size=config.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=0,
        )

        best_metrics = train_fold(
            fold=run_idx,
            train_loader=train_loader,
            val_loader=val_loader,
            config_arch=config,
            config_data=config_data,
            checkpoint_dir=checkpoint_dir,
        )
        run_summaries.append(best_metrics)

    # Final summary across 5 runs
    print(f"\n{'='*70}")
    print("  Summary: 5 Runs with Different Seeds (Same Train/Val Split)")
    print(f"{'='*70}")
    print(f"  {'Metric':<16}  {'Mean':>12}  {'Std':>12}")
    print(f"  {'─'*16}  {'─'*12}  {'─'*12}")
    for key in config.metric_keys:
        values = [s[key] for s in run_summaries]
        label = key.replace("_", " ").title()
        print(
            f"  {label:<16}  {np.mean(values):>12.4f}  {np.std(values):>12.4f}"
        )
    print(f"  {'─'*16}  {'─'*12}  {'─'*12}\n")


if __name__ == "__main__":
    main()
