from pathlib import Path

# =============================================================================
# DATASET SELECTION
#
# Change DATASET to switch which dataset the ENTIRE pipeline operates on —
# scripts/01_split_fasta.py, 02_generate_pssm.py, 03_generate_morgan_fingerprints.py,
# 04_generate_psepssm.py, train.py, train_swa.py, and train_ensemble.py all
# derive their paths from this one value, so nothing below this line should
# ever hardcode "human_random" (or any other dataset name) again.
#
# Every dataset gets its own subfolder under each of data/raw, data/processed,
# data/pssm, data/features, and checkpoints/, keyed by this name — so running
# the pipeline on bindingdb_random or biosnap_random can never overwrite
# human_random's raw data, generated features, or trained checkpoints (this
# used to all be one shared, unparameterized set of paths — see git history
# for the migration that split human_random's existing files into
# data/*/human_random/ to match this scheme).
#
# To add a new dataset "<name>", drop its raw split files at:
#   data/raw/<name>/train/{samples.csv, smiles.csv, sequence.csv}
#   data/raw/<name>/valid/{samples.csv, smiles.csv, sequence.csv}
#   data/raw/<name>/test/{samples.csv,  smiles.csv, sequence.csv}
# (same schema as human_random: samples.csv holds per-split-local integer
# indices into that split's own smiles.csv/sequence.csv — never a global ID,
# see datasets/dti_dataset.py) plus a data/processed/<name>/protein_lookup.csv
# (protein_id -> sequence, deduplicated by sequence) to seed scripts/01-04,
# then set DATASET below and run 01 -> 02 -> 03 -> 04 -> train.py in order.
# =============================================================================

import os

DATASET = os.getenv("DTI_DATASET", "human_random")

# =============================================================================
# PATHS (all derived from DATASET — do not hardcode a dataset name below here)
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent

DATA_DIR = PROJECT_ROOT / "data"

RAW_DIR = DATA_DIR / "raw" / DATASET
PROCESSED_DIR = DATA_DIR / "processed" / DATASET
FASTA_DIR = PROCESSED_DIR / "fasta"
LOOKUP_FILE = PROCESSED_DIR / "protein_lookup.csv"

PSSM_DIR = DATA_DIR / "pssm" / DATASET

FEATURE_DIR = DATA_DIR / "features" / DATASET

CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints" / DATASET

for _dir in (PROCESSED_DIR, FASTA_DIR, PSSM_DIR, FEATURE_DIR, CHECKPOINT_DIR):
    _dir.mkdir(parents=True, exist_ok=True)

# =============================================================================
# DATA
# =============================================================================

TRAIN_DIR = RAW_DIR / "train"
VALID_DIR = RAW_DIR / "valid"
TEST_DIR = RAW_DIR / "test"

PSEPSSM_FILE = FEATURE_DIR / "psepssm_features.npz"
MORGAN_FILE = FEATURE_DIR / "morgan_fingerprints.npz"

# =============================================================================
# MODEL
#
# NOTE ON THIS SECTION: PROTEIN_DIM, DRUG_DIM, EMBED_DIM, MAMBA_STATE_DIM,
# MAMBA_CONV_DIM, and MAMBA_EXPAND are currently DECORATIVE — models/
# mamba_dti.py constructs ProteinEncoder() and previously DrugEncoder()
# with no arguments, so those classes silently fall back to their own
# hardcoded __init__ defaults (which currently happen to match these
# values, so nothing looked broken). Changing any of these six numbers
# here would do nothing until MambaDTI is updated to accept and thread
# them through, the way DROPOUT was just fixed below. Flagging this now
# rather than leaving it to be rediscovered the way the DROPOUT and SEED
# no-ops were — fix it the same way if/when capacity tuning is wanted.
# =============================================================================

PROTEIN_DIM = 220
DRUG_DIM = 2048

EMBED_DIM = 128
MAMBA_STATE_DIM = 16
MAMBA_CONV_DIM = 4
MAMBA_EXPAND = 2

# DROPOUT: previously also dead config (models/mamba_dti.py hardcoded
# nn.Dropout(0.3) twice and never passed dropout to DrugEncoder either)
# — fixed in this pass, MambaDTI(dropout=DROPOUT) now actually uses it.
# Bumped 0.3 -> 0.5: every run so far (all under the old, unwired 0.3)
# shows a large, consistent train/valid MCC gap (~0.09-0.11) regardless
# of checkpoint/threshold changes, i.e. real overfitting, not noise.
DROPOUT = 0.5

# =============================================================================
# TRAINING
# =============================================================================

BATCH_SIZE = 16

# NUM_EPOCHS/PATIENCE: bumped to match the E1 setting requested by
# faculty (200 epochs, early exit at patience 30) — was 60/25. Split
# ratio and dedup were NOT changed for this run (kept the existing
# 8:1:1 human_random split as-is, per explicit decision to just try
# the new epoch/patience budget first and look at results before
# deciding whether to re-split to 7:1:2).
NUM_EPOCHS = 200

LEARNING_RATE = 3e-4

# WEIGHT_DECAY: unlike DROPOUT, this one WAS already wired correctly
# (used directly in the optimizer in train.py) in every prior run.
# Bumped 1e-4 -> 5e-4 (5x) alongside the dropout increase, as the second
# half of the regularization sweep targeting the same overfitting gap.
WEIGHT_DECAY = 5e-4

DEVICE = "cuda"

SEED = 42

# =============================================================================
# EARLY STOPPING
#
# Checkpoint selection and early-stop triggering are DECOUPLED (see
# train.py):
#   - the saved checkpoint tracks the single best raw valid MCC seen so far
#   - the stop/continue decision tracks a rolling average of valid MCC over
#     the last SMOOTHING_WINDOW epochs, to avoid stopping on ordinary
#     epoch-to-epoch noise (empirically std ~0.01-0.012 on this validation
#     split size).
#
# PATIENCE=18/SMOOTHING_WINDOW=5 were originally chosen by REPLAYING the
# stop/checkpoint logic against a real prior 50-epoch valid-MCC sequence
# (see git history / prior comment). That replay confirmed those values
# correctly ride out that run's noise and reach its true best epoch.
#
# HOWEVER: a real run using PATIENCE=18 stopped at epoch 26, checkpointing
# epoch 7 (valid MCC 0.8810) — worse on test accuracy/recall/MCC than
# both prior runs. Root cause: no random seed was actually being applied
# anywhere in the codebase (SEED was defined but never consumed by
# torch/numpy/random), so that run was a genuinely different, unseeded
# noise trajectory from the one the replay validated against, not a
# repeat of it. train.py now applies SEED for real, so results are
# reproducible going forward.
#
# PATIENCE is bumped 25 -> 30 to match the E1 setting ("200 epochs with
# early exit at 30") explicitly requested by faculty. This is a larger
# epoch budget (60 -> 200) with proportionally similar patience, so the
# smoothed-MCC early-stop logic should behave the same way qualitatively
# — just with far more headroom to keep training if valid MCC is still
# improving late. SMOOTHING_WINDOW left at 5 (untouched, not part of the
# E1 spec); revisit only if the longer run shows the same noise profile
# doesn't hold at this scale.
# =============================================================================

PATIENCE = 30
SMOOTHING_WINDOW = 5
