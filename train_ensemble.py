"""
Multi-seed prediction ensembling, built on top of the checkpoint-averaging
(SWA) approach from train_swa.py.

STANDALONE — does not import from or modify train.py, config.py, or
train_swa.py. Run this to try it; none of those are touched.

Why this exists (continuing directly from the SWA result): weight
averaging within ONE run's trajectory (train_swa.py) cancels noise from
*where in that trajectory* you stopped, and it gave a real, large MCC
gain (0.84 -> 0.88) over a single checkpoint. This script targets a
different, larger noise source: which random init / data order a given
run happened to get. It trains SEEDS (below) independent runs, each
internally SWA-averaged exactly like train_swa.py, then averages their
PREDICTED PROBABILITIES (not weights) on validation and test.

Weight-averaging across independent runs is NOT done here on purpose —
independently initialized networks generally don't share a linearly
connected loss basin (permutation symmetry), so naively averaging their
weights can land in a bad region even if each model individually is
good. Averaging PREDICTIONS has no such requirement and is the standard,
low-risk way to combine genuinely independent models.

This also gives something we didn't have before: a mean +/- std for SWA
test MCC across independent seeds, i.e. an actual, honest error bar
instead of a single number.

Trains SEEDS full runs (up to NUM_EPOCHS each, early-stopping applies
per run same as before) — expect roughly SEEDS x the wall-clock time of
one train_swa.py run.

Saves each seed's averaged model to checkpoints/ensemble_seed_<seed>.pt
— new files, does not touch best_model.pt or swa_model.pt.
"""

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
# EXPERIMENT-SPECIFIC SETTINGS (not in config.py, same reasoning as
# train_swa.py's TOP_K — kept here so this experiment is tweakable
# without touching the shared config train.py relies on)
# ============================================================

SEEDS = [42, 123, 2024]  # independent runs to train and ensemble
TOP_K = 5                # per-run checkpoint averaging, same as train_swa.py

# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"Using device : {DEVICE}")

# ============================================================
# DATASETS — built once, reused for every seed (no randomness in how
# these are constructed, so there's nothing seed-dependent about them)
# ============================================================

train_dataset = DTIDataset(TRAIN_DIR / "samples.csv", PSEPSSM_FILE, MORGAN_FILE)
valid_dataset = DTIDataset(VALID_DIR / "samples.csv", PSEPSSM_FILE, MORGAN_FILE)
test_dataset = DTIDataset(TEST_DIR / "samples.csv", PSEPSSM_FILE, MORGAN_FILE)

valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

print(f"Train : {len(train_dataset)}")
print(f"Valid : {len(valid_dataset)}")
print(f"Test  : {len(test_dataset)}")


def train_one_seeded_swa_run(seed):
    """
    Train one full run from scratch under `seed`, keeping the top TOP_K
    checkpoints by raw valid MCC and averaging their weights at the end
    — identical methodology to train_swa.py, just factored into a
    function so it can be repeated per seed.

    Returns the averaged model (weights loaded into a fresh MambaDTI).
    """

    set_seed(seed)

    train_loader_generator = torch.Generator()
    train_loader_generator.manual_seed(seed)

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=0, generator=train_loader_generator,
    )

    model = MambaDTI(dropout=DROPOUT).to(DEVICE)

    criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY,
    )

    top_checkpoints = []  # (valid_mcc, cpu_state_dict), sorted desc, len <= TOP_K

    mcc_history = []
    best_smoothed_mcc = -1.0
    wait = 0

    for epoch in range(NUM_EPOCHS):

        train_loss, train_metrics = run_epoch(model, train_loader, criterion, optimizer)
        valid_loss, valid_metrics = run_epoch(model, valid_loader, criterion)

        print(
            f"  [seed {seed}] epoch {epoch+1}/{NUM_EPOCHS}  "
            f"train_loss={train_loss:.4f} val_loss={valid_loss:.4f} "
            f"train_mcc={train_metrics['mcc']:.4f} valid_mcc={valid_metrics['mcc']:.4f}"
        )

        valid_mcc = valid_metrics["mcc"]

        cpu_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        top_checkpoints.append((valid_mcc, cpu_state))
        top_checkpoints.sort(key=lambda pair: pair[0], reverse=True)

        if len(top_checkpoints) > TOP_K:
            top_checkpoints = top_checkpoints[:TOP_K]

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
                    f"  [seed {seed}] early stopping at epoch {epoch+1} "
                    f"(smoothed valid MCC stalled for {PATIENCE} epochs)."
                )
                break

    print(f"  [seed {seed}] top {len(top_checkpoints)} checkpoint MCCs: "
          f"{[round(m, 4) for m, _ in top_checkpoints]}")

    averaged_state = {}
    reference_keys = top_checkpoints[0][1].keys()
    for key in reference_keys:
        stacked = torch.stack([state[key].float() for _, state in top_checkpoints], dim=0)
        averaged_state[key] = stacked.mean(dim=0)

    model.load_state_dict(averaged_state)

    CHECKPOINT_DIR.mkdir(exist_ok=True)
    save_path = CHECKPOINT_DIR / f"ensemble_seed_{seed}.pt"
    torch.save(averaged_state, save_path)
    print(f"  [seed {seed}] averaged model saved to {save_path}")

    return model


# ============================================================
# RUN ALL SEEDS
# ============================================================

per_seed_valid_probs = []
per_seed_test_probs = []

reference_valid_labels = None
reference_test_labels = None

per_seed_test_mcc = []

for seed in SEEDS:

    print(f"\n{'#'*60}\n# SEED {seed}\n{'#'*60}")

    model = train_one_seeded_swa_run(seed)

    valid_labels, valid_probs = predict(model, valid_loader)
    test_labels, test_probs = predict(model, test_loader)

    # Sanity check: valid/test loaders use shuffle=False, so label order
    # must be identical across seeds. If this ever fires, something is
    # wrong with dataset construction, not just "different random draw".
    if reference_valid_labels is None:
        reference_valid_labels = valid_labels
        reference_test_labels = test_labels
    else:
        assert np.allclose(valid_labels, reference_valid_labels), (
            "Validation label order differs between seeds — loaders "
            "should be deterministic (shuffle=False). Aborting."
        )
        assert np.allclose(test_labels, reference_test_labels), (
            "Test label order differs between seeds — loaders should "
            "be deterministic (shuffle=False). Aborting."
        )

    per_seed_valid_probs.append(np.asarray(valid_probs))
    per_seed_test_probs.append(np.asarray(test_probs))

    seed_test_metrics = compute_metrics(test_labels, test_probs, threshold=0.5)
    per_seed_test_mcc.append(seed_test_metrics["mcc"])

    print(f"\n  [seed {seed}] SWA model alone — TEST (threshold=0.50):")
    for key, value in seed_test_metrics.items():
        print(f"    {key:<15}: {value:.4f}")

# ============================================================
# PER-SEED SUMMARY (an honest mean +/- std, not a single lucky number)
# ============================================================

print("\n" + "=" * 60)
print("PER-SEED SWA TEST MCC (threshold=0.50)")
print("=" * 60)
for seed, mcc in zip(SEEDS, per_seed_test_mcc):
    print(f"  seed {seed:<6}: mcc = {mcc:.4f}")
print(f"  mean = {np.mean(per_seed_test_mcc):.4f}   std = {np.std(per_seed_test_mcc):.4f}")

# ============================================================
# ENSEMBLE: average PREDICTED PROBABILITIES across seeds
# (validation-only for threshold selection, no test leakage)
# ============================================================

ensemble_valid_probs = np.mean(np.stack(per_seed_valid_probs, axis=0), axis=0)
ensemble_test_probs = np.mean(np.stack(per_seed_test_probs, axis=0), axis=0)

ensemble_metrics_default = compute_metrics(reference_test_labels, ensemble_test_probs, threshold=0.5)

print("\n" + "=" * 60)
print(f"ENSEMBLE OF {len(SEEDS)} SEEDS — TEST RESULTS (threshold = 0.50)")
print("=" * 60)
for key, value in ensemble_metrics_default.items():
    print(f"{key:<15}: {value:.4f}")
print("=" * 60)

best_threshold, best_valid_mcc = find_best_threshold(
    reference_valid_labels, ensemble_valid_probs, metric="mcc",
)

print(
    f"\n[Ensemble] Threshold tuned on validation set (maximizing MCC): "
    f"{best_threshold:.4f}  (validation MCC at this threshold = {best_valid_mcc:.4f})"
)

ensemble_metrics_tuned = compute_metrics(reference_test_labels, ensemble_test_probs, threshold=best_threshold)

print("\n" + "=" * 60)
print(f"ENSEMBLE OF {len(SEEDS)} SEEDS — TEST RESULTS (threshold = {best_threshold:.4f}, tuned on valid MCC)")
print("=" * 60)
for key, value in ensemble_metrics_tuned.items():
    print(f"{key:<15}: {value:.4f}")
print("=" * 60)
