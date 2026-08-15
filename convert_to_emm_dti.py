"""
Convert mamba-dti dataset format to EMM-DTI format.

Source format (mamba-dti):
- data/raw/{dataset}/train/smiles.csv (index, smiles)
- data/raw/{dataset}/train/sequence.csv (index, sequence)
- data/raw/{dataset}/train/samples.csv (smiles_idx, sequence_idx, label)

Target format (EMM-DTI):
- D:\Projects\EMM_DTI_Replication\data\{dataset}\drugs.csv (drug_id, smiles)
- D:\Projects\EMM_DTI_Replication\data\{dataset}\proteins.csv (protein_id, sequence)
- D:\Projects\EMM_DTI_Replication\data\{dataset}\interactions.csv (drug_id, protein_id, label)
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Use relative paths from script location
SCRIPT_DIR = Path(__file__).parent
MAMBA_DATA_ROOT = SCRIPT_DIR / "data" / "raw"
EMM_DTI_DATA_ROOT = SCRIPT_DIR / "data" / "emm_dti"

# Map mamba-dti dataset names (source) → EMM-DTI names (destination)
DATASET_MAPPING = {
    "human_random": "human",
    "enzyme": "enzyme",
    "biosnap_random": "biosnap",
    "bindingdb_random": "bindingdb",
    "celegans_random": "celegans",
    "drugbank": "drugbank",
}

def convert_dataset(dataset_name: str, mamba_name: str):
    """Convert a single dataset from mamba-dti to EMM-DTI format.

    Handles two formats:
    1. Train/valid/test splits with separate smiles.csv, sequence.csv, samples.csv
    2. Single CSV file with protein_id, drug_id, label, protein_sequence, smiles columns
    """
    print(f"\n{'='*70}")
    print(f"  Converting {mamba_name} → {dataset_name}")
    print(f"{'='*70}")

    mamba_dataset_path = MAMBA_DATA_ROOT / mamba_name
    emm_dti_dataset_path = EMM_DTI_DATA_ROOT / dataset_name
    emm_dti_dataset_path.mkdir(parents=True, exist_ok=True)

    all_smiles = {}
    all_sequences = {}
    all_interactions = []

    # Check if this is the single-file format (enzyme, drugbank)
    csv_files = list(mamba_dataset_path.glob("*.csv"))
    has_splits = (mamba_dataset_path / "train").exists()

    if not has_splits and csv_files:
        # Handle single-file format (enzyme, drugbank)
        print(f"  Detected single-file format (enzyme/drugbank style)")
        for csv_file in csv_files:
            if "Drug_Target_Pair" in csv_file.name:
                print(f"  ✓ Reading {csv_file.name}")
                df = pd.read_csv(csv_file)

                # Extract unique drugs and proteins
                for _, row in df.iterrows():
                    drug_id = row["drug_id"]
                    protein_id = row["protein_id"]
                    smiles = row["smiles"]
                    sequence = row["protein_sequence"]
                    label = int(row["label"])

                    all_smiles[drug_id] = smiles
                    all_sequences[protein_id] = sequence
                    all_interactions.append({
                        "drug_id": drug_id,
                        "protein_id": protein_id,
                        "label": label,
                        "split": "train"  # Single-file datasets will use full data as training
                    })

                print(f"  ✓ Processed {csv_file.name}: {len(df)} interactions")
    else:
        # Handle train/valid/test splits format (standard mamba-dti)
        for split in ["train", "valid", "test"]:
            split_path = mamba_dataset_path / split

            if not split_path.exists():
                print(f"  ⚠️  {split} folder not found, skipping")
                continue

            # Read split data
            smiles_path = split_path / "smiles.csv"
            sequence_path = split_path / "sequence.csv"
            samples_path = split_path / "samples.csv"

            if not all([smiles_path.exists(), sequence_path.exists(), samples_path.exists()]):
                print(f"  ⚠️  Missing data files in {split}, skipping")
                continue

            # Read files
            smiles_df = pd.read_csv(smiles_path)
            sequence_df = pd.read_csv(sequence_path)
            samples_df = pd.read_csv(samples_path)

            # Extract SMILES and sequences
            for _, row in smiles_df.iterrows():
                all_smiles[int(row["index"])] = row["smiles"]

            for _, row in sequence_df.iterrows():
                all_sequences[int(row["index"])] = row["sequence"]

            # Extract interactions
            for _, row in samples_df.iterrows():
                smiles_idx = int(row["smiles"])
                sequence_idx = int(row["sequence"])
                label = int(row["interactions"])

                all_interactions.append({
                    "drug_id": smiles_idx,
                    "protein_id": sequence_idx,
                    "label": label,
                    "split": split  # Track which split (train/valid/test)
                })

            print(f"  ✓ Processed {split}: {len(samples_df)} interactions")

    # Create output CSVs
    # drugs.csv
    drugs_data = []
    for drug_id in sorted(all_smiles.keys()):
        drugs_data.append({"drug_id": drug_id, "smiles": all_smiles[drug_id]})
    drugs_df = pd.DataFrame(drugs_data)
    drugs_df.to_csv(emm_dti_dataset_path / "drugs.csv", index=False)
    print(f"  ✓ Saved {len(drugs_df)} drugs to drugs.csv")

    # proteins.csv
    proteins_data = []
    for protein_id in sorted(all_sequences.keys()):
        proteins_data.append({"protein_id": protein_id, "sequence": all_sequences[protein_id]})
    proteins_df = pd.DataFrame(proteins_data)
    proteins_df.to_csv(emm_dti_dataset_path / "proteins.csv", index=False)
    print(f"  ✓ Saved {len(proteins_df)} proteins to proteins.csv")

    # interactions.csv
    interactions_df = pd.DataFrame(all_interactions)
    interactions_df.to_csv(emm_dti_dataset_path / "interactions.csv", index=False)
    print(f"  ✓ Saved {len(interactions_df)} interactions to interactions.csv")

    print(f"  ✓ {dataset_name} conversion complete!")
    return len(drugs_df), len(proteins_df), len(interactions_df)

def main():
    """Convert all available datasets."""
    print("\n" + "="*70)
    print("  EMM-DTI Dataset Conversion")
    print("="*70)
    print(f"\nSource: {MAMBA_DATA_ROOT}")
    print(f"Target: {EMM_DTI_DATA_ROOT}\n")

    results = {}
    for mamba_name, dataset_name in DATASET_MAPPING.items():
        mamba_path = MAMBA_DATA_ROOT / mamba_name
        if mamba_path.exists():
            try:
                drugs, proteins, interactions = convert_dataset(dataset_name, mamba_name)
                results[dataset_name] = {
                    "drugs": drugs,
                    "proteins": proteins,
                    "interactions": interactions
                }
            except Exception as e:
                print(f"  ✗ Error converting {mamba_name}: {e}")
        else:
            print(f"  ⚠️  {mamba_name} folder not found, skipping")

    # Summary
    print(f"\n{'='*70}")
    print(f"  CONVERSION SUMMARY")
    print(f"{'='*70}")
    for dataset_name, stats in results.items():
        print(f"  {dataset_name:15} | Drugs: {stats['drugs']:5} | Proteins: {stats['proteins']:5} | Interactions: {stats['interactions']:7}")
    print(f"{'='*70}\n")

    print(f"✓ All datasets converted to EMM-DTI format!")
    print(f"  Location: {EMM_DTI_DATA_ROOT}\n")

if __name__ == "__main__":
    main()
