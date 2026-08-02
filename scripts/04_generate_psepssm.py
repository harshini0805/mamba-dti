"""
Generate PsePSSM features from ASCII PSSM files.

Feature Dimension:
------------------
AAC Features      : 20
PsePSSM Features  : 20 × λ

Total = 20 + (20 × λ)

For λ = 10
Feature Size = 220
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import PSSM_DIR, LOOKUP_FILE, FEATURE_DIR, DATASET

# protein_lookup.csv is the per-dataset, sequence-deduplicated table
# (protein_id -> sequence) that data/processed/<DATASET>/fasta/*.fasta and
# data/pssm/<DATASET>/*.pssm were generated from (see scripts/01_split_fasta.py).
# We reuse it here purely to recover the sequence string for each
# protein_id — no PSI-BLAST is rerun.

OUTPUT_DIR = FEATURE_DIR
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"[DATASET={DATASET}] reading PSSMs from {PSSM_DIR} -> writing PsePSSM features to {OUTPUT_DIR}")

LAMBDA = 10

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))



def read_pssm(pssm_file):

    matrix = []

    with open(pssm_file, "r") as f:
        lines = f.readlines()

    for line in lines:

        parts = line.strip().split()

        # Standard PSSM rows contain at least 22 columns
        if len(parts) < 22:
            continue

        # First column must be residue index
        if not parts[0].isdigit():
            continue

        try:
            scores = list(map(int, parts[2:22]))
            matrix.append(scores)
        except ValueError:
            continue

    return np.asarray(matrix, dtype=np.float32)



def compute_psepssm(pssm_matrix, lamda=10):

    norm_pssm = sigmoid(pssm_matrix)

    L = norm_pssm.shape[0]

    aac = np.mean(norm_pssm, axis=0)


    pse_features = []

    for l in range(1, lamda + 1):

        if L <= l:
            pse_features.extend([0.0] * 20)
            continue

        theta = []

        for j in range(20):

            diff = (
                norm_pssm[:-l, j]
                - norm_pssm[l:, j]
            ) ** 2

            theta.append(np.mean(diff))

        pse_features.extend(theta)

    feature_vector = np.concatenate(
        [
            aac,
            np.asarray(pse_features)
        ]
    )

    return feature_vector.astype(np.float32)



pssm_files = sorted(PSSM_DIR.glob("*.pssm"))

print(f"Found {len(pssm_files)} PSSM files.\n")

# protein_id -> sequence (global, sequence-deduplicated lookup table).
protein_lookup_df = pd.read_csv(LOOKUP_FILE)
protein_id_to_sequence = dict(
    zip(protein_lookup_df["protein_id"], protein_lookup_df["sequence"])
)



protein_ids = []
sequences = []
features = []

failed = []

for file in tqdm(pssm_files, desc="Generating PsePSSM"):

    protein_name = file.stem

    try:
        protein_id = int(protein_name.split("_")[1])
    except Exception:
        failed.append(protein_name)
        continue

    sequence = protein_id_to_sequence.get(protein_id)

    if sequence is None:
        # No entry in protein_lookup.csv for this protein_id — cannot
        # resolve a sequence string, so this PSSM can't be safely keyed.
        failed.append(protein_name)
        continue

    pssm = read_pssm(file)

    if pssm.shape[0] == 0:
        failed.append(protein_name)
        continue

    feature_vector = compute_psepssm(
        pssm,
        lamda=LAMBDA
    )

    assert feature_vector.shape[0] == 20 + 20 * LAMBDA

    protein_ids.append(protein_id)
    sequences.append(sequence)
    features.append(feature_vector)


features = np.asarray(features, dtype=np.float32)

print("\nFeature Matrix Shape:", features.shape)

# Sanity check: the dataset loader keys these features by sequence string,
# so there must be exactly one feature vector per unique sequence.
assert len(sequences) == len(set(sequences)), (
    "Duplicate protein sequences detected — the dataset loader would "
    "silently overwrite PsePSSM features for these proteins. Aborting "
    "instead of writing a corrupt feature file."
)

np.savez_compressed(
    OUTPUT_DIR / "psepssm_features.npz",
    protein_id=np.asarray(protein_ids),
    sequence=np.array(sequences, dtype=object),
    features=features
)


feature_df = pd.DataFrame(
    features,
    columns=[f"f_{i}" for i in range(features.shape[1])]
)

feature_df.insert(0, "sequence", sequences)
feature_df.insert(0, "protein_id", protein_ids)

feature_df.to_csv(
    OUTPUT_DIR / "psepssm_features.csv",
    index=False
)



failed_df = pd.DataFrame(
    {
        "protein": failed
    }
)

failed_df.to_csv(
    OUTPUT_DIR / "failed_pssm.csv",
    index=False
)


print("\n")
print("SUMMARY")

print(f"PSSM files found        : {len(pssm_files)}")
print(f"Successful             : {len(protein_ids)}")
print(f"Failed                 : {len(failed)}")
print(f"Feature dimension      : {features.shape[1]}")

print("\nSaved:")

print(OUTPUT_DIR / "psepssm_features.npz")
print(OUTPUT_DIR / "psepssm_features.csv")
print(OUTPUT_DIR / "failed_pssm.csv")