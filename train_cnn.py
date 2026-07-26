"""
Trains MambaCNNDTI (Mamba protein encoder + CNN drug encoder + MLP head,
see models/mamba_cnn_dti.py) to compare against the Mamba+MLP baseline
(train.py).

STANDALONE — does not import from or modify train.py or config.py's
behavior for existing scripts. Uses the exact same seeding, decoupled
checkpoint/early-stop logic, and validation-threshold-tuning-then-applied-
to-test protocol as train.py, so any difference in results between this
script and train.py isolates the effect of the drug-encoder architecture
(CNN vs MLP) rather than being confounded by a different training recipe.

Saves to checkpoints/<DATASET>/best_model_cnn.pt — a new filename, does not
touch best_model.pt / swa_model.pt / ensemble_seed_*.pt.
"""

import copy
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from datasets.dti_dataset import DTIDataset
from models.mamba_cnn_dti import MambaCNNDTI
from utils.trainer import run_epoch, predict
from utils.metrics import compute_metrics, find_best_threshold

from config import *

# ============================================================
# REPRODUCIBILITY (identical to train.py)
# ============================================================

def set_seed(seed):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    torch.cuda.manual_seed_all(seed)

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
print(f"Model        : MambaCNNDTI (Mamba protein encoder + CNN drug encoder + MLP head)")

# ============================================================
# DATASETS
# ============================================================

train_dataset = DTIDataset(TRAIN_DIR / "samples.csv", PSEPSSM_FILE, MORGAN_FILE)
valid_dataset = DTIDataset(VALID_DIR / "samples.csv", PSEPSSM_FILE, MORGAN_FILE)
test_dataset = DTIDataset(TEST_DIR / "samples.csv", PSEPSSM_FILE, MORGAN_FILE)

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

model = MambaCNNDTI(dropout=DROPOUT).to(DEVICE)

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
# TRAINING (identical checkpoint/early-stop logic to train.py)
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
            CHECKPOINT_DIR / "best_model_cnn.pt",
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
# THRESHOLD OPTIMIZATION (identical methodology to train.py)
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
