"""
Training script for Mamba+CrossAttention+MLP DTI architecture.

Usage:
    python train.py --dataset humans
    python train.py --dataset bindingdb --epochs 100 --batch_size 32 --lr 1e-4

All config is loaded from datasets/{dataset}.py and this module's config.py.
Hyperparameters can be overridden via CLI.
"""

import argparse
import copy
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Add parent directories to path for imports
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).parent))  # Add architecture config path

from common.dataset_loader import DTIDataset, collate_fn
from common.metrics import compute_metrics
from config import default_config as arch_config
from model import MambaCrossAttentionDTI

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


# ─── Device & Loss ─────────────────────────────────────────────────────────

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
        for batch_idx, (protein, drug, label) in enumerate(loader):
            protein = protein.to(DEVICE)
            drug = drug.to(DEVICE)
            label = label.to(DEVICE)

            logits = model(protein, drug)
            loss = criterion(logits, label)

            # Detect NaN/Inf with diagnostics
            if torch.isnan(loss) or torch.isinf(loss):
                raise ValueError(
                    f"{'NaN' if torch.isnan(loss) else 'Inf'} loss at batch {batch_idx}.\n"
                    f"  logits : min={logits.min():.4f}  max={logits.max():.4f}  "
                    f"nan={torch.isnan(logits).any()}\n"
                    f"  protein: min={protein.min():.4f}  max={protein.max():.4f}  "
                    f"nan={torch.isnan(protein).any()}\n"
                    f"  drug   : min={drug.min():.4f}  max={drug.max():.4f}  "
                    f"nan={torch.isnan(drug).any()}"
                )

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
    test_loader: DataLoader,
    config_arch,
    config_data,
    checkpoint_dir: Path,
    logger=None,
) -> dict:
    """
    Train one independent run and evaluate on test set.

    Returns:
        dict with keys: val_{metric}, test_{metric} for all metrics
    """
    model = MambaCrossAttentionDTI(
        drug_input_dim=config_data.drug_input_dim,
    ).to(DEVICE)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config_arch.learning_rate,
        weight_decay=config_arch.weight_decay,
    )
    criterion = nn.BCEWithLogitsLoss()

    best_val_pr_auc = -1.0
    best_val_loss = None
    best_val_metrics = None
    wait = 0

    for epoch in range(1, config_arch.num_epochs + 1):
        train_loss, train_m = run_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_m = run_epoch(model, val_loader, criterion)

        # ─── Per-epoch table ───────────────────────────────────────────────
        sep = "  " + "─" * 50
        print(sep)
        print(f"  Epoch {epoch}")
        print(sep)

        header = f"  {'Metric':<16}  {'Train':>12}  {'Val':>12}"
        header_sep = f"  {'─'*16}  {'─'*12}  {'─'*12}"
        print(header_sep)
        print(header)
        print(header_sep)

        if logger:
            logger.info(sep)
            logger.info(f"Epoch {epoch}")
            logger.info(sep)
            logger.info(header_sep)
            logger.info(header)
            logger.info(header_sep)

        # Print all metrics
        for key in config_arch.metric_keys:
            label = key.replace("_", " ").title()
            line = f"  {label:<16}  {train_m[key]:>12.4f}  {val_m[key]:>12.4f}"
            print(line)
            if logger:
                logger.info(line)

        # Print loss
        loss_line = f"  {'Loss':<16}  {train_loss:>12.4f}  {val_loss:>12.4f}"
        print(loss_line)
        print(sep)
        if logger:
            logger.info(loss_line)
            logger.info(sep)

        # ─── Checkpoint & Early Stopping (based on PR-AUC) ──────────────────
        if val_m["pr_auc"] > best_val_pr_auc:
            best_val_pr_auc = val_m["pr_auc"]
            best_val_metrics = copy.deepcopy(val_m)
            best_val_loss = val_loss
            checkpoint_path = checkpoint_dir / f"best_model_run_{fold}.pt"
            torch.save(model.state_dict(), checkpoint_path)
            msg = f"  ✓ Saved {checkpoint_path.name} (PR-AUC = {best_val_pr_auc:.4f})"
            print(msg)
            if logger:
                logger.info(msg)
            wait = 0
        else:
            wait += 1
            if wait >= config_arch.patience:
                msg = f"\n  Early stopping triggered at epoch {epoch}."
                print(msg)
                if logger:
                    logger.info(msg)
                break

    # ─── Evaluate on test set using best model ───────────────────────────
    checkpoint_path = checkpoint_dir / f"best_model_run_{fold}.pt"
    if checkpoint_path.exists():
        model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
        print(f"\n  Evaluating best model on test set...")
        test_loss, test_metrics = run_epoch(model, test_loader, criterion)

        # Combine val and test metrics
        combined_metrics = {}
        if best_val_metrics:
            for key in best_val_metrics.keys():
                combined_metrics[f"val_{key}"] = best_val_metrics[key]
        for key in test_metrics.keys():
            combined_metrics[f"test_{key}"] = test_metrics[key]
        combined_metrics["test_loss"] = test_loss
        if best_val_loss is not None:
            combined_metrics["val_loss"] = best_val_loss

        # Print test results
        print(f"  {'─'*16}  {'─'*12}  {'─'*12}")
        print(f"  {'Metric':<16}  {'Val':>12}  {'Test':>12}")
        print(f"  {'─'*16}  {'─'*12}  {'─'*12}")
        for key in config_arch.metric_keys:
            val_key = f"val_{key}"
            test_key = f"test_{key}"
            if val_key in combined_metrics and test_key in combined_metrics:
                print(
                    f"  {key.replace('_', ' ').title():<16}  "
                    f"{combined_metrics[val_key]:>12.4f}  "
                    f"{combined_metrics[test_key]:>12.4f}"
                )
        print(f"  {'Loss':<16}  {combined_metrics.get('val_loss', 'N/A'):>12}  {test_loss:>12.4f}")
        print(f"  {'─'*16}  {'─'*12}  {'─'*12}\n")

        return combined_metrics
    else:
        print(f"  Warning: Checkpoint not found at {checkpoint_path}")
        return best_val_metrics if best_val_metrics else {}


def main():
    parser = argparse.ArgumentParser(
        description="Train Mamba+CrossAttention+MLP DTI model on specified dataset."
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

    # Setup logging to file
    log_file = logs_dir / f"{args.dataset}_training.log"
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(log_file)
    formatter = logging.Formatter('%(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    print(f"\nTraining Mamba+CrossAttention+MLP on {args.dataset}")
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
    model_info = MambaCrossAttentionDTI(drug_input_dim=config_data.drug_input_dim).to(DEVICE)
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

        # Load test data
        test_loader = DataLoader(
            DTIDataset(test_df, protein_features, drug_embeddings),
            batch_size=config.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=0,
        )

        best_metrics = train_fold(
            fold=run_idx,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            config_arch=config,
            config_data=config_data,
            checkpoint_dir=checkpoint_dir,
            logger=logger,
        )
        run_summaries.append(best_metrics)

    # Final summary across 5 runs
    print(f"\n{'='*70}")
    print("  Summary: 5 Runs with Different Seeds (Same Train/Val Split)")
    print(f"{'='*70}")
    print(f"  {'Metric':<16}  {'Val Mean':>12}  {'Val Std':>12}  {'Test Mean':>12}  {'Test Std':>12}")
    print(f"  {'─'*16}  {'─'*12}  {'─'*12}  {'─'*12}  {'─'*12}")

    # Save results to CSV
    results_data = []
    for run_idx, metrics in enumerate(run_summaries, start=1):
        row = {"run": run_idx}
        for key in config.metric_keys:
            val_key = f"val_{key}"
            test_key = f"test_{key}"
            if val_key in metrics:
                row[f"val_{key}"] = metrics[val_key]
            if test_key in metrics:
                row[f"test_{key}"] = metrics[test_key]
        results_data.append(row)

    # Compute and print summary statistics
    for key in config.metric_keys:
        val_values = [s.get(f"val_{key}", np.nan) for s in run_summaries]
        test_values = [s.get(f"test_{key}", np.nan) for s in run_summaries]
        label = key.replace("_", " ").title()
        val_mean = np.nanmean(val_values) if not np.all(np.isnan(val_values)) else 0.0
        val_std = np.nanstd(val_values) if not np.all(np.isnan(val_values)) else 0.0
        test_mean = np.nanmean(test_values) if not np.all(np.isnan(test_values)) else 0.0
        test_std = np.nanstd(test_values) if not np.all(np.isnan(test_values)) else 0.0
        print(
            f"  {label:<16}  {val_mean:>12.4f}  {val_std:>12.4f}  "
            f"{test_mean:>12.4f}  {test_std:>12.4f}"
        )

    print(f"  {'─'*16}  {'─'*12}  {'─'*12}  {'─'*12}  {'─'*12}\n")

    # Save to CSV and JSON
    results_csv_path = results_dir / "results.csv"
    results_json_path = results_dir / "results.json"

    results_df = pd.DataFrame(results_data)
    results_df.to_csv(results_csv_path, index=False)
    print(f"  ✓ Saved results to {results_csv_path}")

    # Also save summary statistics
    summary_data = {
        "dataset": args.dataset,
        "num_runs": len(run_summaries),
        "metrics": {}
    }
    for key in config.metric_keys:
        val_values = [s.get(f"val_{key}", np.nan) for s in run_summaries]
        test_values = [s.get(f"test_{key}", np.nan) for s in run_summaries]
        summary_data["metrics"][key] = {
            "val": {
                "mean": float(np.nanmean(val_values)) if not np.all(np.isnan(val_values)) else None,
                "std": float(np.nanstd(val_values)) if not np.all(np.isnan(val_values)) else None,
            },
            "test": {
                "mean": float(np.nanmean(test_values)) if not np.all(np.isnan(test_values)) else None,
                "std": float(np.nanstd(test_values)) if not np.all(np.isnan(test_values)) else None,
            }
        }

    with open(results_json_path, "w") as f:
        json.dump(summary_data, f, indent=2)
    print(f"  ✓ Saved summary to {results_json_path}\n")


if __name__ == "__main__":
    main()
