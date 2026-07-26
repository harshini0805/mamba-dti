"""
Checkpoint averaging (SWA-style) experiment.

STANDALONE — does not import from or modify train.py or config.py's
behavior for existing scripts. Run this instead of train.py to try it;
train.py is untouched and still works exactly as before.

Why this exists: across five runs (see conversation/commit history),
several epochs within a single run land within noise-distance of each
other on valid MCC (sometimes within 0.0003) while differing meaningfully
in precision/recall balance and even ROC-AUC. Picking a single "best"
epoch by raw MCC means the reported result is effectively decided by
whichever epoch wins that near-tie — a different roll of the dice each
run. Averaging the WEIGHTS of the top-K such epochs (Stochastic Weight
Averaging-style) is a standard, low-risk way to land in the middle of
that noisy cluster instead of at one arbitrary corner of it.

Safe to do a plain parameter-wise average here because nothing in this
model uses BatchNorm (which would need a running-stats recalibration
pass after averaging) — ProteinEncoder uses LayerNorm, which has no
running buffers, and the rest of the model is Linear/ReLU/Dropout/Mamba.

This trains a FRESH run from scratch (it can't retroactively average
checkpoints from a previous run's process — those only ever existed in
memory during that run, and only the single best one was ever written to
disk as checkpoints/best_model.pt). With config.SEED applied the same
way train.py applies it, this should follow a closely similar trajectory
to your last run, but is not guaranteed bit-identical across different
GPU/driver stacks.

Saves the averaged model to checkpoints/swa_model.pt — a new file,
separate from checkpoints/best_model.pt, so nothing train.py produced is
overwritten.
"""

import copy
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from datasets.dti_dataset import DTIDataset
from models.mamba_dti import MambaDTI
from utils.trainer import run_epoch, predict
from utils.metrics import compute_metrics, find_best_threshold

from config import *

# ============================================================
# EXPERIMENT-SPECIFIC SETTING (not in config.py — this script is meant
# to be tweakable without touching the shared config used by train.py)
# ============================================================

TOP_K = 5  # number of near-best checkpoints (by raw valid MCC) to average

# ============================================================
# REPRODUCIBILITY (same approach as train.py)
# ============================================================

def set_seed(seed):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(SEED)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"Using device : {DEVICE}")

# ============================================================
# DATASETS / DATALOADERS (identical to train.py)
# ============================================================

train_dataset = DTIDataset(TRAIN_DIR / "samples.csv", PSEPSSM_FILE, MORGAN_FILE)
valid_dataset = DTIDataset(VALID_DIR / "samples.csv", PSEPSSM_FILE, MORGAN_FILE)
test_dataset = DTIDataset(TEST_DIR / "samples.csv", PSEPSSM_FILE, MORGAN_FILE)

train_loader_generator = torch.Generator()
train_loader_generator.manual_seed(SEED)

train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True,
    num_workers=0, generator=train_loader_generator,
)
valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

print(f"Train : {len(train_dataset)}")
print(f"Valid : {len(valid_dataset)}")
print(f"Test  : {len(test_dataset)}")

# ============================================================
# MODEL
# ============================================================

model = MambaDTI(dropout=DROPOUT).to(DEVICE)

print(
    f"Trainable Parameters : "
    f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,}"
)

criterion = nn.BCEWithLogitsLoss()

optimizer = torch.optim.Adam(
    model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY,
)

# ============================================================
# TRAINING
#
# Same decoupled checkpoint/early-stop logic as train.py, EXTENDED to
# keep the top TOP_K checkpoints (by raw valid MCC) instead of just 1.
# Kept in memory (CPU tensors) rather than written to disk individually
# — this model is small (see param count above), so this costs nothing
# meaningful, and avoids littering checkpoints/ with K new files.
# ============================================================

top_checkpoints = []  # list of (valid_mcc, state_dict_on_cpu), kept sorted descending, len <= TOP_K

mcc_history = []
best_smoothed_mcc = -1.0
wait = 0

for epoch in range(NUM_EPOCHS):

    train_loss, train_metrics = run_epoch(model, train_loader, criterion, optimizer)
    valid_loss, valid_metrics = run_epoch(model, valid_loader, criterion)

    print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")
    print(f"Train Loss : {train_loss:.4f} | Val Loss : {valid_loss:.4f}")

    metric_keys = ["accuracy", "precision", "recall", "specificity", "mcc", "roc_auc", "pr_auc"]

    print("Train  : " + " | ".join(f"{k} {train_metrics[k]:.4f}" for k in metric_keys))
    print("Valid  : " + " | ".join(f"{k} {valid_metrics[k]:.4f}" for k in metric_keys))

    valid_mcc = valid_metrics["mcc"]

    # ---- Maintain top-K checkpoints by raw valid MCC ----

    cpu_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    top_checkpoints.append((valid_mcc, cpu_state))
    top_checkpoints.sort(key=lambda pair: pair[0], reverse=True)

    if len(top_checkpoints) > TOP_K:
        top_checkpoints = top_checkpoints[:TOP_K]

    if valid_mcc == top_checkpoints[0][0]:
        print(f"New top checkpoint (valid MCC = {valid_mcc:.4f}).")

    # ---- Early stopping on smoothed (rolling-average) valid MCC — identical to train.py ----

    mcc_history.append(valid_mcc)
    smoothed_window = mcc_history[-SMOOTHING_WINDOW:]
    smoothed_mcc = sum(smoothed_window) / len(smoothed_window)

    if smoothed_mcc > best_smoothed_mcc:
        best_smoothed_mcc = smoothed_mcc
        wait = 0
    else:
        wait += 1
        if wait >= PATIENCE:
            print(
                f"\nEarly stopping (smoothed valid MCC over last "
                f"{SMOOTHING_WINDOW} epochs stalled for {PATIENCE} epochs)."
            )
            break

print(f"\nTop {len(top_checkpoints)} checkpoints kept (by valid MCC):")
for mcc, _ in top_checkpoints:
    print(f"  valid MCC = {mcc:.4f}")

# ============================================================
# BASELINE: single best checkpoint alone (same as train.py would report)
# ============================================================

best_mcc, best_state = top_checkpoints[0]

model.load_state_dict(best_state)

test_loss, test_metrics_single = run_epoch(model, test_loader, criterion)

print("\n" + "=" * 50)
print(f"SINGLE-BEST CHECKPOINT (valid MCC = {best_mcc:.4f}) — TEST RESULTS (threshold = 0.50)")
print("=" * 50)
for key, value in test_metrics_single.items():
    print(f"{key:<15}: {value:.4f}")
print("=" * 50)

# ============================================================
# AVERAGED MODEL: mean of the top-K checkpoints' weights
# ============================================================

averaged_state = {}
for key in best_state.keys():
    stacked = torch.stack([state[key].float() for _, state in top_checkpoints], dim=0)
    averaged_state[key] = stacked.mean(dim=0)

model.load_state_dict(averaged_state)

CHECKPOINT_DIR.mkdir(exist_ok=True)
torch.save(averaged_state, CHECKPOINT_DIR / "swa_model.pt")
print(f"\nAveraged model saved to {CHECKPOINT_DIR / 'swa_model.pt'} (does not overwrite best_model.pt).")

test_loss, test_metrics_avg = run_epoch(model, test_loader, criterion)

print("\n" + "=" * 50)
print(f"AVERAGED MODEL (mean of top {len(top_checkpoints)} checkpoints) — TEST RESULTS (threshold = 0.50)")
print("=" * 50)
for key, value in test_metrics_avg.items():
    print(f"{key:<15}: {value:.4f}")
print("=" * 50)

# ---- Threshold tuning on the averaged model, same methodology as train.py ----

valid_labels, valid_probs = predict(model, valid_loader)

best_threshold, best_valid_mcc = find_best_threshold(valid_labels, valid_probs, metric="mcc")

print(
    f"\n[Averaged model] Threshold tuned on validation set (maximizing MCC): "
    f"{best_threshold:.4f}  (validation MCC at this threshold = {best_valid_mcc:.4f})"
)

test_labels, test_probs = predict(model, test_loader)

test_metrics_avg_tuned = compute_metrics(test_labels, test_probs, threshold=best_threshold)

print("\n" + "=" * 50)
print(f"AVERAGED MODEL — TEST RESULTS (threshold = {best_threshold:.4f}, tuned on valid MCC)")
print("=" * 50)
for key, value in test_metrics_avg_tuned.items():
    print(f"{key:<15}: {value:.4f}")
print("=" * 50)
