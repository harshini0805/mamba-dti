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
# REPRODUCIBILITY
#
# Previously config.SEED was defined but never actually used anywhere,
# so every run got a different random weight init, dropout mask sequence,
# and train-loader shuffle order. That made it impossible to tell whether
# a change in results (e.g. from an early-stopping config change) was a
# real effect or just a different random draw. Seeding everything here
# makes reruns with unchanged config reproducible, and makes A/B
# comparisons across config changes actually attributable to that change.
# ============================================================

def set_seed(seed):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    torch.cuda.manual_seed_all(seed)

    # Trade a little speed for determinism on CUDA convolution/kernel
    # selection; without this, cuDNN can pick different (faster but
    # non-deterministic) algorithms run to run even with a fixed seed.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(SEED)

# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"Using device : {DEVICE}")

# ============================================================
# DATASETS
# ============================================================

train_dataset = DTIDataset(
    TRAIN_DIR / "samples.csv",
    PSEPSSM_FILE,
    MORGAN_FILE,
)

valid_dataset = DTIDataset(
    VALID_DIR / "samples.csv",
    PSEPSSM_FILE,
    MORGAN_FILE,
)

test_dataset = DTIDataset(
    TEST_DIR / "samples.csv",
    PSEPSSM_FILE,
    MORGAN_FILE,
)

# ============================================================
# DATALOADERS
# ============================================================

train_loader_generator = torch.Generator()
train_loader_generator.manual_seed(SEED)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    generator=train_loader_generator,
)

valid_loader = DataLoader(
    valid_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
)

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
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
)

# ============================================================
# TRAINING
#
# Checkpoint selection and early-stop triggering are intentionally
# DECOUPLED, based on inspecting a prior full 50-epoch run of this model:
#
#   - CHECKPOINT: saved whenever a single epoch's raw valid MCC beats the
#     best raw valid MCC seen so far. (Switched from ROC-AUC: on the prior
#     run, valid ROC-AUC was saturated/flat within noise (std ~0.004) from
#     epoch ~4 onward, so it couldn't distinguish real MCC improvements
#     from noise, and picked a checkpoint ~0.009 MCC below the true best.)
#
#   - EARLY STOP: triggered from a rolling average of valid MCC over the
#     last SMOOTHING_WINDOW epochs, not the raw per-epoch value. On the
#     prior run, raw per-epoch MCC has enough epoch-to-epoch noise
#     (std ~0.01-0.012 on this validation split size) that any patience
#     from 5-20 on the raw value stalled on an early local peak and never
#     reached the true best epoch, which came 27 epochs later. Smoothing
#     filters that noise so a much smaller patience is reliable.
# ============================================================

best_checkpoint_mcc = -1.0
best_weights = None

mcc_history = []
best_smoothed_mcc = -1.0
wait = 0

for epoch in range(NUM_EPOCHS):

    train_loss, train_metrics = run_epoch(
        model,
        train_loader,
        criterion,
        optimizer,
    )

    valid_loss, valid_metrics = run_epoch(
        model,
        valid_loader,
        criterion,
    )

    print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")

    print(
        f"Train Loss : {train_loss:.4f} | "
        f"Val Loss : {valid_loss:.4f}"
    )

    metric_keys = [
        "accuracy",
        "precision",
        "recall",
        "specificity",
        "mcc",
        "roc_auc",
        "pr_auc",
    ]

    print(
        "Train  : " + " | ".join(
            f"{key} {train_metrics[key]:.4f}"
            for key in metric_keys
        )
    )

    print(
        "Valid  : " + " | ".join(
            f"{key} {valid_metrics[key]:.4f}"
            for key in metric_keys
        )
    )

    valid_mcc = valid_metrics["mcc"]

    # ---- Checkpoint on raw best-so-far valid MCC ----

    if valid_mcc > best_checkpoint_mcc:

        best_checkpoint_mcc = valid_mcc

        best_weights = copy.deepcopy(model.state_dict())

        torch.save(
            best_weights,
            CHECKPOINT_DIR / "best_model.pt",
        )

        print(f"Best model saved (valid MCC = {valid_mcc:.4f}).")

    # ---- Early stopping on smoothed (rolling-average) valid MCC ----

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

# ============================================================
# TEST
# ============================================================

print("\nLoading best model...")

model.load_state_dict(best_weights)

test_loss, test_metrics = run_epoch(
    model,
    test_loader,
    criterion,
)

print("\n" + "=" * 50)
print("TEST RESULTS (threshold = 0.50, default)")
print("=" * 50)

for key, value in test_metrics.items():

    print(f"{key:<15}: {value:.4f}")

print("=" * 50)

# ============================================================
# THRESHOLD OPTIMIZATION
#
# The 0.5 decision threshold is arbitrary; it is not guaranteed to
# maximize MCC/F1. Instead we find the threshold that maximizes MCC on
# the VALIDATION set only, then apply that fixed threshold to the test
# set. This never touches test labels during threshold selection, so
# there is no test-set leakage. ROC-AUC/PR-AUC are threshold-independent
# and are unaffected by this step.
# ============================================================

valid_labels, valid_probs = predict(model, valid_loader)

best_threshold, best_valid_mcc = find_best_threshold(
    valid_labels,
    valid_probs,
    metric="mcc",
)

print(
    f"\nThreshold tuned on validation set (maximizing MCC): "
    f"{best_threshold:.4f}  (validation MCC at this threshold = {best_valid_mcc:.4f})"
)

test_labels, test_probs = predict(model, test_loader)

test_metrics_tuned = compute_metrics(
    test_labels,
    test_probs,
    threshold=best_threshold,
)

print("\n" + "=" * 50)
print(f"TEST RESULTS (threshold = {best_threshold:.4f}, tuned on valid MCC)")
print("=" * 50)

for key, value in test_metrics_tuned.items():

    print(f"{key:<15}: {value:.4f}")

print("=" * 50)
