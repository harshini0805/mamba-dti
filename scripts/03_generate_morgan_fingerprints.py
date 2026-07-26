"""
Generate Morgan fingerprints for all unique SMILES in config.DATASET.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from rdkit import RDLogger

# Silence RDKit warnings
RDLogger.DisableLog("rdApp.*")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import RAW_DIR, FEATURE_DIR, DATASET

DATA_DIR = RAW_DIR

OUTPUT_DIR = FEATURE_DIR
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"[DATASET={DATASET}] reading SMILES from {DATA_DIR} -> writing fingerprints to {OUTPUT_DIR}")



RADIUS = 2          # ECFP4
N_BITS = 2048



train = pd.read_csv(DATA_DIR / "train" / "smiles.csv")
valid = pd.read_csv(DATA_DIR / "valid" / "smiles.csv")
test = pd.read_csv(DATA_DIR / "test" / "smiles.csv")

smiles_df = pd.concat([train, valid, test], ignore_index=True)

smiles_df = (
    smiles_df
    .drop_duplicates(subset=["smiles"])
    .reset_index(drop=True)
)

smiles_df = smiles_df.rename(columns={"index": "smiles_id"})

# NOTE: "smiles_id" here is only the row's original per-split index and is
# kept purely for traceability/debugging. It is NOT a global identifier
# (train/valid/test each restart their "index" column at 0), so it must
# never be used as a lookup key downstream. The canonical, collision-free
# key for this feature file is the "smiles" string itself.
assert smiles_df["smiles"].duplicated().sum() == 0, (
    "Duplicate SMILES strings survived deduplication — this should be "
    "impossible and indicates a bug in the drop_duplicates step above."
)

print(f"Unique molecules: {len(smiles_df)}")



fingerprints = []
valid_ids = []
valid_smiles = []

invalid_rows = []

for _, row in tqdm(smiles_df.iterrows(),
                   total=len(smiles_df),
                   desc="Generating fingerprints"):

    smiles_id = row["smiles_id"]
    smiles = row["smiles"]

    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        invalid_rows.append(
            {
                "smiles_id": smiles_id,
                "smiles": smiles
            }
        )
        continue

    fp = AllChem.GetMorganFingerprintAsBitVect(
        mol,
        radius=RADIUS,
        nBits=N_BITS
    )

    arr = np.zeros((N_BITS,), dtype=np.uint8)

    DataStructs.ConvertToNumpyArray(fp, arr)

    fingerprints.append(arr)
    valid_ids.append(smiles_id)
    valid_smiles.append(smiles)


fingerprints = np.asarray(fingerprints, dtype=np.uint8)

print(f"\nFingerprint matrix: {fingerprints.shape}")

# Sanity check: the feature lookup used by the dataset loader is keyed by
# SMILES string, so there must be exactly one fingerprint per unique SMILES.
assert len(valid_smiles) == len(set(valid_smiles)), (
    "Duplicate SMILES keys detected — the dataset loader would silently "
    "overwrite fingerprints for these molecules. Aborting instead of "
    "writing a corrupt feature file."
)

np.savez_compressed(
    OUTPUT_DIR / "morgan_fingerprints.npz",
    smiles_id=np.array(valid_ids),
    smiles=np.array(valid_smiles, dtype=object),
    fingerprints=fingerprints
)

manifest = pd.DataFrame({
    "smiles_id": valid_ids,
    "smiles": valid_smiles
})

manifest.to_csv(
    OUTPUT_DIR / "morgan_manifest.csv",
    index=False
)

invalid_df = pd.DataFrame(invalid_rows)

invalid_df.to_csv(
    OUTPUT_DIR / "invalid_smiles.csv",
    index=False
)


print("\n")
print("SUMMARY")

print(f"Unique molecules      : {len(smiles_df)}")
print(f"Valid molecules       : {len(valid_ids)}")
print(f"Invalid molecules     : {len(invalid_rows)}")
print(f"Fingerprint size      : {N_BITS}")

print("\nSaved:")

print(OUTPUT_DIR / "morgan_fingerprints.npz")
print(OUTPUT_DIR / "morgan_manifest.csv")
print(OUTPUT_DIR / "invalid_smiles.csv")