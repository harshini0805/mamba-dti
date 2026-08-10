"""5-Fold CV training for Mamba+AttentionPool+MLP architecture."""
import argparse, copy, json, sys
from pathlib import Path
import numpy as np, pandas as pd, torch, torch.nn as nn
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).parent))
from common.dataset_loader import DTIDataset, collate_fn
from common.metrics import compute_metrics
from config import default_config as arch_config
from model import MambaAttnDTI
def load_dataset_config(dataset_name: str):
    try: return __import__(f"datasets.{dataset_name}", fromlist=["config"]).config
    except ImportError as e: raise ValueError(f"Dataset '{dataset_name}' not found.") from e
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")
def run_epoch(model, loader, criterion, optimizer=None):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()
    total_loss, all_labels, all_probs = 0.0, [], []
    with (torch.enable_grad() if is_train else torch.no_grad()):
        for protein, drug, label in loader:
            protein, drug, label = protein.to(DEVICE), drug.to(DEVICE), label.to(DEVICE)
            logits = model(protein, drug)
            loss = criterion(logits, label)
            if torch.isnan(loss) or torch.isinf(loss): raise ValueError("NaN/Inf loss")
            if is_train:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=arch_config.gradient_clip_norm)
                optimizer.step()
            total_loss += loss.item() * label.size(0)
            all_probs.extend(torch.sigmoid(logits).detach().cpu().numpy().tolist())
            all_labels.extend(label.cpu().numpy().tolist())
    return total_loss / len(loader.dataset), compute_metrics(all_labels, all_probs)
def train_fold(fold, train_df, val_df, protein_features, drug_embeddings, config_arch, config_data, checkpoint_dir):
    model = MambaAttnDTI(drug_input_dim=config_data.drug_input_dim).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=config_arch.learning_rate, weight_decay=config_arch.weight_decay)
    best_val_pr_auc, best_val_metrics, best_val_loss, wait = -1.0, None, None, 0
    for epoch in range(1, config_arch.num_epochs + 1):
        train_loader = DataLoader(DTIDataset(train_df, protein_features, drug_embeddings), batch_size=config_arch.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=0)
        val_loader = DataLoader(DTIDataset(val_df, protein_features, drug_embeddings), batch_size=config_arch.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=0)
        train_loss, train_m = run_epoch(model, train_loader, nn.BCEWithLogitsLoss(), optimizer)
        val_loss, val_m = run_epoch(model, val_loader, nn.BCEWithLogitsLoss())

        # ─── Per-epoch table ───────────────────────────────────────────────
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
        for key in arch_config.metric_keys:
            label = key.replace("_", " ").title()
            line = f"    {label:<16}  {train_m[key]:>12.4f}  {val_m[key]:>12.4f}"
            print(line)

        # Print loss
        loss_line = f"    {'Loss':<16}  {train_loss:>12.4f}  {val_loss:>12.4f}"
        print(loss_line)
        print(sep)
        if val_m["pr_auc"] > best_val_pr_auc:
            best_val_pr_auc, best_val_metrics, best_val_loss = val_m["pr_auc"], copy.deepcopy(val_m), val_loss
            torch.save(model.state_dict(), checkpoint_dir / f"best_model_fold_{fold}.pt")
            wait = 0
        else:
            wait += 1
            if wait >= config_arch.patience: break
    if (checkpoint_dir / f"best_model_fold_{fold}.pt").exists(): model.load_state_dict(torch.load(checkpoint_dir / f"best_model_fold_{fold}.pt", map_location=DEVICE))
    return {f"val_{k}": v for k, v in (best_val_metrics or {}).items()} | ({"val_loss": best_val_loss} if best_val_loss else {})
def main():
    parser = argparse.ArgumentParser()
    for arg in ["dataset", "epochs", "batch_size", "lr"]:
        parser.add_argument(f"--{arg}", type=int if arg != "dataset" else str, required=(arg == "dataset"))
    args = parser.parse_args()
    config = arch_config
    for attr in ["num_epochs", "batch_size", "learning_rate"]:
        val = getattr(args, attr.split("_")[0], None) if attr != "num_epochs" else args.epochs
        if val: setattr(config, attr, val)
    config_data = load_dataset_config(args.dataset)
    results_dir, checkpoint_dir = config.results_dir / args.dataset, config.checkpoints_dir / args.dataset
    results_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    protein_features, drug_embeddings, interactions = config_data.load_data()
    CV_SEEDS = [42, 123, 2024]
    all_results = []
    for seed_idx, cv_seed in enumerate(CV_SEEDS, 1):
        print(f"\n{'='*70}\n  CV Run {seed_idx}/3 (seed={cv_seed})\n{'='*70}")
        np.random.seed(cv_seed)
        torch.manual_seed(cv_seed)
        if torch.cuda.is_available(): torch.cuda.manual_seed(cv_seed)
        fold_results = []
        for fold_idx, (train_idx, val_idx) in enumerate(StratifiedKFold(n_splits=5, shuffle=True, random_state=cv_seed).split(interactions, interactions["label"]), 1):
            train_df, val_df = interactions.iloc[train_idx].reset_index(drop=True), interactions.iloc[val_idx].reset_index(drop=True)
            print(f"  Fold {fold_idx}/5 | Train: {len(train_df):,} | Val: {len(val_df):,}")
            fold_metrics = train_fold(fold_idx, train_df, val_df, protein_features, drug_embeddings, config, config_data, checkpoint_dir)
            fold_results.append(fold_metrics)
        all_results.append(fold_results)
    print(f"\n{'='*70}\n  SUMMARY: 3 CV Runs × 5 Folds\n{'='*70}")
    results_data = []
    for seed_idx, cv_seed in enumerate(CV_SEEDS, 1):
        cv_fold_results = all_results[seed_idx - 1]
        for fold_idx, fold_metrics in enumerate(cv_fold_results, 1):
            row = {"seed": cv_seed, "fold": fold_idx}
            for key, val in fold_metrics.items():
                row[key] = val
            results_data.append(row)
    summary_metrics = {}
    metric_keys = ["val_pr_auc", "val_roc_auc", "val_accuracy", "val_precision", "val_recall", "val_specificity", "val_mcc"]
    for metric_key in metric_keys:
        all_vals = [row[metric_key] for row in results_data if metric_key in row]
        if all_vals:
            summary_metrics[metric_key] = {"mean": float(np.mean(all_vals)), "std": float(np.std(all_vals))}
            print(f"  {metric_key:<20}: {np.mean(all_vals):.4f} ± {np.std(all_vals):.4f}")
    results_csv_path = results_dir / "results.csv"
    pd.DataFrame(results_data).to_csv(results_csv_path, index=False)
    print(f"\n  ✓ Saved fold results to {results_csv_path}")
    summary_data = {"dataset": args.dataset, "num_seeds": len(CV_SEEDS), "num_folds": 5, "total_folds": len(results_data), "metrics": summary_metrics}
    results_json_path = results_dir / "results.json"
    with open(results_json_path, "w") as f: json.dump(summary_data, f, indent=2)
    print(f"  ✓ Saved summary to {results_json_path}")
    cv_summary_file = results_dir / "cv_summary.json"
    with open(cv_summary_file, "w") as f: json.dump(summary_metrics, f, indent=2)
    print(f"  ✓ Saved CV summary to {cv_summary_file}")
if __name__ == "__main__": main()
