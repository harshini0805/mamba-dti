"""
Build data/processed/<DATASET>/protein_lookup.csv from the raw split
sequence.csv files, for whichever dataset config.DATASET currently points
at. This is the one upstream input scripts/01-04 all depend on and none of
them generate — human_random's copy already existed; new datasets need it
built once before running 01 -> 04.

Pools train/valid/test sequence.csv together and deduplicates by the
SEQUENCE STRING (not by whatever local index each split file uses), then
assigns fresh global protein_ids 0..N-1. This works the same way whether a
dataset's sequence.csv is split-local (a different subset per split, e.g.
biosnap_random) or a single global table copied into every split folder
(e.g. bindingdb_random) — either way, deduplicating by sequence string
across all three collapses down to the same correct global set.

Output schema matches human_random/protein_lookup.csv exactly:
    protein_id,sequence
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import RAW_DIR, LOOKUP_FILE, DATASET

sequence_frames = []

for split in ("train", "valid", "test"):
    split_file = RAW_DIR / split / "sequence.csv"
    df = pd.read_csv(split_file)

    if "sequence" not in df.columns:
        raise ValueError(
            f"{split_file} has no 'sequence' column (columns found: "
            f"{list(df.columns)}). Fix the raw file before continuing."
        )

    sequence_frames.append(df[["sequence"]])

all_sequences = pd.concat(sequence_frames, ignore_index=True)

unique_sequences = (
    all_sequences
    .drop_duplicates(subset=["sequence"])
    .reset_index(drop=True)
)

unique_sequences.insert(0, "protein_id", range(len(unique_sequences)))

assert unique_sequences["sequence"].duplicated().sum() == 0, (
    "Duplicate sequences survived deduplication — should be impossible."
)

LOOKUP_FILE.parent.mkdir(parents=True, exist_ok=True)

unique_sequences.to_csv(LOOKUP_FILE, index=False)

print(f"[DATASET={DATASET}] wrote {len(unique_sequences)} unique proteins -> {LOOKUP_FILE}")
