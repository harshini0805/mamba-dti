"""
Dataset config for C. elegans dataset.

This module defines:
  - Data paths and loading logic
  - Dataset-specific constants (drug fingerprint dimension, etc.)

Usage:
    from datasets.celegans import config
    protein_features, drug_embeddings, train_df, val_df, test_df = config.load_data()
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class CElegansDatasetConfig:
    """Configuration for C. elegans DTI dataset."""

    dataset_name: str = "celegans_random"
    data_dir: Path = Path("data")

    # ─── Input Dimensions ───────────────────────────────────────────────
    protein_feature_dim: int = 220  # PsePSSM features
    drug_input_dim: int = 2048  # Morgan fingerprint bits

    # ─── Data File Paths ────────────────────────────────────────────────
    # Protein features CSV
    protein_csv: str = "data/features/celegans_random/psepssm_features.csv"
    # Drug fingerprints NPZ (expected keys: 'drug_ids', 'fingerprints')
    drug_npz: str = "data/features/celegans_random/morgan_fingerprints.npz"
    # Interaction pairs (structured as: protein_id, drug_id, label per split)
    raw_data_dir: str = "data/raw/celegans_random"

    def load_data(self) -> tuple:
        """
        Load protein features, drug embeddings, and interaction pairs (train/valid/test separate).

        Data structure:
          - data/features/{dataset}/psepssm_features.csv: (protein_id → features)
          - data/features/{dataset}/morgan_fingerprints.npz: (drug_id → fingerprints)
          - data/raw/{dataset}/{split}/{samples,smiles,sequence}.csv: interaction triples per split

        Returns:
            (protein_features, drug_embeddings, train_interactions, val_interactions, test_interactions)
              - protein_features: dict[protein_id -> (220,) float32]
              - drug_embeddings: dict[drug_id -> (2048,) float32]
              - train_interactions, val_interactions, test_interactions: DataFrames with ['protein_id', 'drug_id', 'label']
        """
        print(f"Loading {self.dataset_name} dataset...")

        # ─── Load protein features ───────────────────────────────────────
        protein_df = pd.read_csv(self.protein_csv)
        protein_features = {}

        # protein_df has columns: [protein_id, feat_0, feat_1, ..., feat_219]
        for _, row in protein_df.iterrows():
            protein_id = row["protein_id"]
            features = row.iloc[1:].values.astype(np.float32)
            protein_features[protein_id] = features

        # Z-score normalize protein features
        all_feats = np.stack(list(protein_features.values()))
        feat_mean = all_feats.mean(axis=0)
        feat_std = all_feats.std(axis=0) + 1e-8
        for pid in protein_features:
            protein_features[pid] = (protein_features[pid] - feat_mean) / feat_std

        print(f"  ✓ Loaded {len(protein_features):,} protein features")

        # ─── Load drug fingerprints ─────────────────────────────────────
        fp_data = np.load(self.drug_npz, allow_pickle=True)
        drug_ids = fp_data["drug_ids"]
        fingerprints = fp_data["fingerprints"].astype(np.float32)

        # Validate uniqueness
        if len(set(drug_ids)) != len(drug_ids):
            raise ValueError("Duplicate drug_id entries in fingerprint file.")

        drug_embeddings = {drug_id: fingerprints[i] for i, drug_id in enumerate(drug_ids)}
        self.drug_input_dim = fingerprints.shape[1]
        print(f"  ✓ Loaded {len(drug_embeddings):,} drug fingerprints (dim={self.drug_input_dim})")

        # ─── Load interaction pairs per split (train/valid/test separate) ──
        split_interactions = {}
        raw_dir = Path(self.raw_data_dir)

        for split in ["train", "valid", "test"]:
            split_dir = raw_dir / split
            if not split_dir.exists():
                print(f"  ⚠ {split} split not found")
                continue

            samples_file = split_dir / "samples.csv"
            smiles_file = split_dir / "smiles.csv"
            sequence_file = split_dir / "sequence.csv"

            if not all(f.exists() for f in [samples_file, smiles_file, sequence_file]):
                print(f"  ⚠ Skipping {split} (missing files)")
                continue

            # Load mappings
            smiles_df = pd.read_csv(smiles_file)
            sequence_df = pd.read_csv(sequence_file)
            samples_df = pd.read_csv(samples_file)

            # Create ID mappings
            smiles_to_id = dict(zip(smiles_df["index"], smiles_df["smiles"]))
            sequence_to_id = dict(zip(sequence_df["index"], sequence_df["sequence"]))

            # Map samples to protein_id and drug_id
            interactions = pd.DataFrame({
                "protein_id": samples_df["sequence"].map(sequence_to_id),
                "drug_id": samples_df["smiles"].map(smiles_to_id),
                "label": samples_df["interactions"],
            })

            # Filter to proteins and drugs present in feature stores
            before = len(interactions)
            interactions = interactions[
                interactions["protein_id"].isin(protein_features)
                & interactions["drug_id"].isin(drug_embeddings)
            ].reset_index(drop=True)
            after = len(interactions)

            split_interactions[split] = interactions
            print(f"  ✓ {split:6}: {after:,} interactions ({before - after} filtered)")

        # Return separate splits
        train_df = split_interactions.get("train", pd.DataFrame())
        val_df = split_interactions.get("valid", pd.DataFrame())
        test_df = split_interactions.get("test", pd.DataFrame())

        return protein_features, drug_embeddings, train_df, val_df, test_df


# Global config instance
config = CElegansDatasetConfig()
