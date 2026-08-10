"""
5-Fold Cross-Validation training script for Mamba+MLP DTI architecture.

Runs stratified 5-fold CV with 3 different random seeds.

Usage:
    python train_cv.py --dataset drugbank
    python train_cv.py --dataset enzyme --epochs 200
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
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).parent))

from common.dataset_loader import DTIDataset, collate_fn
from common.metrics import compute_metrics
from config import default_config as arch_config
from model import MambaMLPDTI


def load_dataset_config(dataset_name: str):
    """Dynamically import dataset config."""
    try:
        dataset_module = __import__(
            f"datasets.{dataset_name}",
            fromlist=["config"],
        )
        return dataset_module.config
    except ImportError as e:
        raise ValueError(f"Dataset '{dataset_name}' not found.") from e


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[float, dict]:
    """Run one epoch of training or validation."""
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

            if torch.isnan(loss) or torch.isinf(loss):
                raise ValueError(f"NaN/Inf loss at batch")

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
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    protein_features: dict,
    drug_embeddings: dict,
    config_arch,
    config_data,
    checkpoint_dir: Path,
) -> dict:
    """Train one fold and evaluate on validation set."""
    model = MambaMLPDTI(drug_in_dim=config_data.drug_input_dim).to(DEVICE)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config_arch.learning_rate,
        weight_decay=config_arch.weight_decay,
    )
    criterion = nn.BCEWithLogitsLoss()

    best_val_pr_auc = -1.0
    best_val_metrics = None
    best_val_loss = None
    wait = 0

    for epoch in range(1, config_arch.num_epochs + 1):
        train_loader = DataLoader(
            DTIDataset(train_df, protein_features, drug_embeddings),
            batch_size=config_arch.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=0,
        )
        val_loader = DataLoader(
            DTIDataset(val_df, protein_features, drug_embeddings),
            batch_size=config_arch.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=0,
        )

        train_loss, train_m = run_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_m = run_epoch(model, val_loader, criterion)

        if epoch % 10 == 0:
            print(f"  Fold {fold} | Epoch {epoch:3d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val PR-AUC: {val_m['pr_auc']:.4f}")

        if val_m["pr_auc"] > best_val_pr_auc:
            best_val_pr_auc = val_m["pr_auc"]
            best_val_metrics = copy.deepcopy(val_m)
            best_val_loss = val_loss
            checkpoint_path = checkpoint_dir / f"best_model_fold_{fold}.pt"
            torch.save(model.state_dict(), checkpoint_path)
            wait = 0
        else:
            wait += 1
            if wait >= config_arch.patience:
                print(f"  Fold {fold} | Early stopping at epoch {epoch}")
                break

    # Load best model and return metrics
    checkpoint_path = checkpoint_dir / f"best_model_fold_{fold}.pt"
    if checkpoint_path.exists():
        model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))

    combined_metrics = {}
    if best_val_metrics:
        for key in best_val_metrics.keys():
            combined_metrics[f"val_{key}"] = best_val_metrics[key]
    if best_val_loss is not None:
        combined_metrics["val_loss"] = best_val_loss

    return combined_metrics


def main():
    parser = argparse.ArgumentParser(description="Train Mamba+MLP with 5-fold CV")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name (drugbank or enzyme)")
    parser.add_argument("--epochs", type=int, help="Override number of epochs")
    parser.add_argument("--batch_size", type=int, help="Override batch size")
    parser.add_argument("--lr", type=float, help="Override learning rate")

    args = parser.parse_args()

    config_data = load_dataset_config(args.dataset)
    config = arch_config

    if args.epochs:
        config.num_epochs = args.epochs
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.lr:
        config.learning_rate = args.lr

    results_dir = Path("results") / args.dataset
    checkpoint_dir = Path("checkpoints") / args.dataset
    results_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nLoading {args.dataset} dataset for 5-fold CV...")
    protein_features, drug_embeddings, interactions = config_data.load_data()

    print(f"Total interactions: {len(interactions):,}")

    # 3 different seeds for 5-fold CV
    CV_SEEDS = [42, 123, 2024]
    all_results = []

    for seed_idx, cv_seed in enumerate(CV_SEEDS, start=1):
        print(f"\n{'='*70}\n  CV Run {seed_idx}/3 (seed={cv_seed})\n{'='*70}")

        np.random.seed(cv_seed)
        torch.manual_seed(cv_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(cv_seed)

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=cv_seed)
        fold_results = []

        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(interactions, interactions["label"]), start=1):
            train_df = interactions.iloc[train_idx].reset_index(drop=True)
            val_df = interactions.iloc[val_idx].reset_index(drop=True)

            print(f"\n  Fold {fold_idx}/5 | Train: {len(train_df):,} | Val: {len(val_df):,}")

            fold_metrics = train_fold(
                fold=fold_idx,
                train_df=train_df,
                val_df=val_df,
                protein_features=protein_features,
                drug_embeddings=drug_embeddings,
                config_arch=config,
                config_data=config_data,
                checkpoint_dir=checkpoint_dir,
            )

            fold_results.append(fold_metrics)

        all_results.append(fold_results)

    # Aggregate results across 3 CV runs and 5 folds
    print(f"\n{'='*70}")
    print("  SUMMARY: 3 CV Runs × 5 Folds")
    print(f"{'='*70}")

    summary_metrics = {}
    for metric_key in ["val_pr_auc", "val_roc_auc", "val_accuracy", "val_precision", "val_recall", "val_specificity", "val_mcc"]:
        all_vals = []
        for cv_results in all_results:
            for fold_metrics in cv_results:
                key = metric_key.replace("val_", "")
                if key in fold_metrics:
                    all_vals.append(fold_metrics[key])

        if all_vals:
            summary_metrics[metric_key] = {
                "mean": float(np.mean(all_vals)),
                "std": float(np.std(all_vals)),
            }
            print(f"  {metric_key:<20}: {np.mean(all_vals):.4f} ± {np.std(all_vals):.4f}")

    # Save summary
    summary_file = results_dir / "cv_summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary_metrics, f, indent=2)
    print(f"\n  ✓ Summary saved: {summary_file}")


if __name__ == "__main__":
    main()
