"""
5-Fold CV training for EMM-DTI architecture with FCS mining.

This script:
- Loads data from EMM-DTI format (drugs.csv, proteins.csv, interactions.csv)
- Implements 5-fold stratified CV with proper FCS mining on each fold's training data only
- Uses 3 random seeds [42, 123, 2024] for robust evaluation
- Saves results and checkpoints to architectures/emm_dti/results/{dataset}/

CRITICAL: Data must be converted from mamba-dti format first:
  python convert_to_emm_dti.py
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

# Setup paths
PROJECT_ROOT = Path(__file__).parent  # architectures/emm_dti
MAMBA_ROOT = PROJECT_ROOT.parent.parent  # mamba-dti
EMM_DTI_ROOT = MAMBA_ROOT.parent / "EMM_DTI_Replication"  # d:\Projects\EMM_DTI_Replication

sys.path.insert(0, str(MAMBA_ROOT))
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(EMM_DTI_ROOT))

# Import from mamba-dti common
from common.metrics import compute_metrics

# Import from EMM-DTI package
from emm_dti.data.loaders import DTIDataset
from emm_dti.models.fcs import FCSModule, FragmentVocabulary
from emm_dti.models.emm_dti import EMMDTI

# Setup logging
logger = None
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")


def load_raw_data(data_dir, split=None):
    """Load raw data from EMM-DTI format CSV files.

    Args:
        data_dir: Path to dataset directory
        split: If None, load all data. If 'train'/'valid'/'test', load from mamba-dti raw format

    Returns:
        (drugs, proteins, interactions_df)
    """
    # Load drugs
    drugs_df = pd.read_csv(data_dir / "drugs.csv")
    drugs = drugs_df[["drug_id", "smiles"]].set_index("drug_id").to_dict()["smiles"]

    # Load proteins
    proteins_df = pd.read_csv(data_dir / "proteins.csv")
    proteins = proteins_df[["protein_id", "sequence"]].set_index("protein_id").to_dict()["sequence"]

    # Load interactions (already converted, includes all data)
    interactions_df = pd.read_csv(data_dir / "interactions.csv")

    return drugs, proteins, interactions_df


def create_fcs_vocabulary(train_interactions_df, drugs, proteins):
    """Mine FCS patterns from training data only."""
    train_drug_ids = train_interactions_df["drug_id"].unique()
    train_protein_ids = train_interactions_df["protein_id"].unique()

    train_sequences = (
        [drugs[drug_id] for drug_id in train_drug_ids] +
        [proteins[protein_id] for protein_id in train_protein_ids]
    )

    # Mine frequent patterns
    fcs = FCSModule(min_support=0.3)
    patterns = fcs.mine(train_sequences, max_k=3)

    # Build vocabulary from mined patterns
    fcs_vocab = FragmentVocabulary()
    fcs_vocab.build_from_fcs(fcs)

    return fcs_vocab, fcs


def run_epoch(model, loader, criterion, optimizer=None, device=DEVICE):
    """Run one epoch of training or evaluation."""
    is_train = optimizer is not None
    model.train() if is_train else model.eval()
    total_loss, all_labels, all_probs = 0.0, [], []

    with (torch.enable_grad() if is_train else torch.no_grad()):
        for batch in loader:
            # Unpack batch: (drug_indices, protein_indices, label)
            drug_indices, protein_indices, labels = batch

            drug_indices = drug_indices.to(device)
            protein_indices = protein_indices.to(device)
            labels = labels.to(device).float()

            # Forward pass
            logits = model(drug_indices, protein_indices)
            if logits.dim() > 1:
                logits = logits.squeeze(-1)

            loss = criterion(logits, labels)

            if torch.isnan(loss) or torch.isinf(loss):
                raise ValueError(f"NaN/Inf loss detected: {loss}")

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            total_loss += loss.item() * labels.size(0)
            all_probs.extend(torch.sigmoid(logits).detach().cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())

    return total_loss / len(loader.dataset), compute_metrics(all_labels, all_probs)


def train_fold(
    fold, train_indices, val_indices, drugs, proteins, interactions_df,
    config_training, checkpoint_dir, device=DEVICE
):
    """Train and validate a single fold with fold-specific FCS mining."""
    # Get fold data
    train_interactions = interactions_df.iloc[train_indices].reset_index(drop=True)
    val_interactions = interactions_df.iloc[val_indices].reset_index(drop=True)

    # Mine FCS from this fold's training data only (prevent data leakage)
    logger.info(f"  Mining FCS patterns from fold {fold} training data...")
    fcs_vocab, fcs = create_fcs_vocabulary(train_interactions, drugs, proteins)
    logger.info(f"  FCS vocabulary size: {len(fcs_vocab)}")

    # Create datasets
    train_drug_seqs = [drugs[drug_id] for drug_id in train_interactions["drug_id"]]
    train_protein_seqs = [proteins[protein_id] for protein_id in train_interactions["protein_id"]]
    train_labels = train_interactions["label"].tolist()

    val_drug_seqs = [drugs[drug_id] for drug_id in val_interactions["drug_id"]]
    val_protein_seqs = [proteins[protein_id] for protein_id in val_interactions["protein_id"]]
    val_labels = val_interactions["label"].tolist()

    train_dataset = DTIDataset(
        drug_sequences=train_drug_seqs,
        protein_sequences=train_protein_seqs,
        interactions=train_labels,
        fcs_vocab=fcs_vocab,
        fcs_patterns=fcs.get_patterns(),
    )

    val_dataset = DTIDataset(
        drug_sequences=val_drug_seqs,
        protein_sequences=val_protein_seqs,
        interactions=val_labels,
        fcs_vocab=fcs_vocab,
        fcs_patterns=fcs.get_patterns(),
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config_training["batch_size"],
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config_training["batch_size"],
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )

    # Create model with fold-specific FCS vocab
    model = EMMDTI(
        vocab_size=len(fcs_vocab),
        fcs_embedding_dim=128,
        mamba_hidden_dim=256,
        mamba_n_layers=2,
        mamba_state_size=16,
        cnn_out_channels=3,
        dropout=0.1,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config_training["learning_rate"],
        weight_decay=config_training["weight_decay"]
    )
    criterion = nn.BCEWithLogitsLoss()

    best_val_pr_auc, best_val_metrics, best_val_loss, wait = -1.0, None, None, 0

    # Training loop
    for epoch in range(1, config_training["num_epochs"] + 1):
        train_loss, train_m = run_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_m = run_epoch(model, val_loader, criterion, device=device)

        # Print per-epoch metrics
        sep = "    " + "─" * 50
        print(sep)
        print(f"    Fold {fold} | Epoch {epoch}")
        print(sep)

        header = f"    {'Metric':<16}  {'Train':>12}  {'Val':>12}"
        header_sep = f"    {'─'*16}  {'─'*12}  {'─'*12}"
        print(header_sep)
        print(header)
        print(header_sep)

        # Print all metrics
        metric_keys = ["accuracy", "precision", "recall", "specificity", "mcc", "roc_auc", "pr_auc"]
        for key in metric_keys:
            if key in train_m and key in val_m:
                label = key.replace("_", " ").title()
                line = f"    {label:<16}  {train_m[key]:>12.4f}  {val_m[key]:>12.4f}"
                print(line)

        # Print loss
        loss_line = f"    {'Loss':<16}  {train_loss:>12.4f}  {val_loss:>12.4f}"
        print(loss_line)
        print(sep)

        # Early stopping on PR-AUC
        if val_m["pr_auc"] > best_val_pr_auc:
            best_val_pr_auc = val_m["pr_auc"]
            best_val_metrics = copy.deepcopy(val_m)
            best_val_loss = val_loss
            torch.save(model.state_dict(), checkpoint_dir / f"best_model_fold_{fold}.pt")
            wait = 0
        else:
            wait += 1
            if wait >= config_training["patience"]:
                logger.info(f"  Early stopping at epoch {epoch} (patience={config_training['patience']})")
                break

    # Load best model
    if (checkpoint_dir / f"best_model_fold_{fold}.pt").exists():
        model.load_state_dict(torch.load(checkpoint_dir / f"best_model_fold_{fold}.pt", map_location=device))

    return {f"val_{k}": v for k, v in (best_val_metrics or {}).items()} | ({"val_loss": best_val_loss} if best_val_loss else {})


def main():
    parser = argparse.ArgumentParser(description="5-Fold CV training for EMM-DTI")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name (human, enzyme, biosnap, bindingdb, celegans, drugbank)")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    args = parser.parse_args()

    # Setup data paths - converted data is stored in mamba-dti/data/emm_dti/
    data_dir = MAMBA_ROOT / "data" / "emm_dti" / args.dataset.lower()
    if not data_dir.exists():
        raise ValueError(
            f"Dataset '{args.dataset}' not found at {data_dir}\n"
            f"Available datasets: human, enzyme, biosnap, bindingdb, celegans, drugbank\n"
            f"First, run: python convert_to_emm_dti.py"
        )

    # Setup output dirs
    results_dir = PROJECT_ROOT / "results" / args.dataset.lower()
    checkpoint_dir = PROJECT_ROOT / "checkpoints" / args.dataset.lower()
    logs_dir = PROJECT_ROOT / "logs" / args.dataset.lower()

    for d in [results_dir, checkpoint_dir, logs_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Setup logging
    global logger
    log_file = logs_dir / f"{args.dataset}_cv_training.log"
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(log_file)
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    logger.info("="*70)
    logger.info(f"EMM-DTI 5-Fold CV Training")
    logger.info("="*70)
    logger.info(f"Dataset: {args.dataset}")
    logger.info(f"Data dir: {data_dir}")
    logger.info(f"Results dir: {results_dir}")
    logger.info(f"Device: {DEVICE}")
    logger.info("")

    # Config
    config_training = {
        "num_epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "weight_decay": 1e-4,
        "patience": 30,
    }

    logger.info(f"Training config: {config_training}")
    logger.info("")

    # Load raw data
    logger.info(f"Loading data from {data_dir}...")
    drugs, proteins, interactions_df = load_raw_data(data_dir)
    logger.info(f"Loaded: {len(drugs)} drugs, {len(proteins)} proteins, {len(interactions_df)} interactions")
    logger.info("")

    # Check if this dataset has pre-made splits (valid/test)
    has_splits = "split" in interactions_df.columns and set(interactions_df["split"].unique()) > {"train"}
    if has_splits:
        raise ValueError(
            f"Dataset {args.dataset} has pre-made train/valid/test splits.\n"
            f"Use train.py for this dataset instead (5 independent runs).\n"
            f"train_cv.py is only for datasets without pre-made splits (enzyme, drugbank)."
        )

    # 5-fold stratified CV with 3 seeds (for enzyme and drugbank only)
    CV_SEEDS = [42, 123, 2024]
    labels = interactions_df["label"].values
    print(f"\n{'='*70}\n  Using 5-Fold Stratified CV\n{'='*70}")
    logger.info(f"Using 5-Fold Stratified CV with {len(CV_SEEDS)} seeds")
    all_results = []

    for seed_idx, cv_seed in enumerate(CV_SEEDS, 1):
        print(f"\n{'='*70}\n  CV Run {seed_idx}/{len(CV_SEEDS)} (seed={cv_seed})\n{'='*70}")
        logger.info(f"CV Run {seed_idx}/{len(CV_SEEDS)} (seed={cv_seed})")

        np.random.seed(cv_seed)
        torch.manual_seed(cv_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(cv_seed)

        fold_results = []
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=cv_seed)

        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(interactions_df, labels), 1):
            print(f"  Fold {fold_idx}/5 | Train: {len(train_idx):,} | Val: {len(val_idx):,}")
            logger.info(f"Fold {fold_idx}/5 | Train: {len(train_idx):,} | Val: {len(val_idx):,}")

            fold_metrics = train_fold(
                fold_idx,
                train_idx,
                val_idx,
                drugs,
                proteins,
                interactions_df,
                config_training,
                checkpoint_dir,
            )
            fold_results.append(fold_metrics)

        all_results.append(fold_results)

    # Aggregate results
    print(f"\n{'='*70}\n  SUMMARY: {len(CV_SEEDS)} CV Runs × 5 Folds\n{'='*70}")
    logger.info("")
    logger.info(f"{'='*70}")
    logger.info(f"SUMMARY: {len(CV_SEEDS)} CV Runs × 5 Folds")
    logger.info(f"{'='*70}")

    results_data = []

    # Aggregate across seeds and folds
    for seed_idx, cv_seed in enumerate(CV_SEEDS, 1):
        cv_fold_results = all_results[seed_idx - 1]
        for fold_idx, fold_metrics in enumerate(cv_fold_results, 1):
            row = {"seed": cv_seed, "fold": fold_idx}
            for key, val in fold_metrics.items():
                row[key] = val
            results_data.append(row)

    # Compute summary statistics
    metric_keys = ["val_pr_auc", "val_roc_auc", "val_accuracy", "val_precision", "val_recall", "val_specificity", "val_mcc"]
    summary_metrics = {}

    for metric_key in metric_keys:
        all_vals = [row[metric_key] for row in results_data if metric_key in row]
        if all_vals:
            mean_val = float(np.mean(all_vals))
            std_val = float(np.std(all_vals))
            summary_metrics[metric_key] = {"mean": mean_val, "std": std_val}
            msg = f"  {metric_key:<20}: {mean_val:.4f} ± {std_val:.4f}"
            print(msg)
            logger.info(msg)

    # Save results
    results_csv = results_dir / "results.csv"
    pd.DataFrame(results_data).to_csv(results_csv, index=False)
    print(f"\n  ✓ Saved fold results to {results_csv}")
    logger.info(f"✓ Saved fold results to {results_csv}")

    summary_json = results_dir / "results.json"
    summary_data = {
        "dataset": args.dataset.lower(),
        "num_seeds": len(CV_SEEDS),
        "num_folds": 5,
        "total_folds": len(results_data),
        "metrics": summary_metrics,
        "config": config_training,
    }
    with open(summary_json, "w") as f:
        json.dump(summary_data, f, indent=2)
    print(f"  ✓ Saved summary to {summary_json}")
    logger.info(f"✓ Saved summary to {summary_json}")

    logger.info(f"{'='*70}")
    logger.info("Training complete!")


if __name__ == "__main__":
    main()
