"""5-Fold CV training for Mamba+Cross-Attn+MLP architecture."""
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
from model import MambaCrossMLPDTI
def load_dataset_config(dataset_name: str):
    try:
        dataset_module = __import__(f"datasets.{dataset_name}", fromlist=["config"])
        return dataset_module.config
    except ImportError as e:
        raise ValueError(f"Dataset '{dataset_name}' not found.") from e
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")
def run_epoch(model, loader, criterion, optimizer=None):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()
    total_loss, all_labels, all_probs = 0.0, [], []
    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
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
    model = MambaCrossMLPDTI(drug_in_dim=config_data.drug_input_dim).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=config_arch.learning_rate, weight_decay=config_arch.weight_decay)
    criterion = nn.BCEWithLogitsLoss()
    best_val_pr_auc, best_val_metrics, best_val_loss, wait = -1.0, None, None, 0
    for epoch in range(1, config_arch.num_epochs + 1):
        train_loader = DataLoader(DTIDataset(train_df, protein_features, drug_embeddings), batch_size=config_arch.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=0)
        val_loader = DataLoader(DTIDataset(val_df, protein_features, drug_embeddings), batch_size=config_arch.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=0)
        train_loss, _ = run_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_m = run_epoch(model, val_loader, criterion)
        if epoch % 10 == 0: print(f"  Fold {fold} | Epoch {epoch:3d} | Val Loss: {val_loss:.4f} | Val PR-AUC: {val_m['pr_auc']:.4f}")
        if val_m["pr_auc"] > best_val_pr_auc:
            best_val_pr_auc, best_val_metrics, best_val_loss = val_m["pr_auc"], copy.deepcopy(val_m), val_loss
            torch.save(model.state_dict(), checkpoint_dir / f"best_model_fold_{fold}.pt")
            wait = 0
        else:
            wait += 1
            if wait >= config_arch.patience: print(f"  Fold {fold} | Early stopping at epoch {epoch}"); break
    if (checkpoint_dir / f"best_model_fold_{fold}.pt").exists(): model.load_state_dict(torch.load(checkpoint_dir / f"best_model_fold_{fold}.pt", map_location=DEVICE))
    combined = {f"val_{k}": v for k, v in (best_val_metrics or {}).items()}
    if best_val_loss is not None: combined["val_loss"] = best_val_loss
    return combined
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--lr", type=float)
    args = parser.parse_args()
    config_data = load_dataset_config(args.dataset)
    config = arch_config
    if args.epochs: config.num_epochs = args.epochs
    if args.batch_size: config.batch_size = args.batch_size
    if args.lr: config.learning_rate = args.lr
    results_dir, checkpoint_dir = Path("results") / args.dataset, Path("checkpoints") / args.dataset
    results_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nLoading {args.dataset} dataset...")
    protein_features, drug_embeddings, interactions = config_data.load_data()
    CV_SEEDS = [42, 123, 2024]
    all_results = []
    for seed_idx, cv_seed in enumerate(CV_SEEDS, start=1):
        print(f"\n{'='*70}\n  CV Run {seed_idx}/3 (seed={cv_seed})\n{'='*70}")
        np.random.seed(cv_seed)
        torch.manual_seed(cv_seed)
        if torch.cuda.is_available(): torch.cuda.manual_seed(cv_seed)
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=cv_seed)
        fold_results = []
        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(interactions, interactions["label"]), start=1):
            train_df, val_df = interactions.iloc[train_idx].reset_index(drop=True), interactions.iloc[val_idx].reset_index(drop=True)
            print(f"  Fold {fold_idx}/5 | Train: {len(train_df):,} | Val: {len(val_df):,}")
            fold_metrics = train_fold(fold_idx, train_df, val_df, protein_features, drug_embeddings, config, config_data, checkpoint_dir)
            fold_results.append(fold_metrics)
        all_results.append(fold_results)
    print(f"\n{'='*70}\n  SUMMARY\n{'='*70}")
    for metric_key in ["val_pr_auc", "val_roc_auc", "val_accuracy"]:
        vals = [fm[metric_key.replace("val_", "")] for cr in all_results for fm in cr if metric_key.replace("val_", "") in fm]
        if vals: print(f"  {metric_key:<20}: {np.mean(vals):.4f} ± {np.std(vals):.4f}")
    with open(results_dir / "cv_summary.json", "w") as f: json.dump({}, f)
if __name__ == "__main__": main()
