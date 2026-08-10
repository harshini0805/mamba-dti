"""
Dataset config for DrugBank dataset (5-fold cross-validation).

This module defines data loading logic for 5-fold CV on DrugBank interactions.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold


@dataclass
class DrugBankDatasetConfig:
    """Configuration for DrugBank DTI dataset."""

    dataset_name: str = "drugbank"
    data_dir: Path = Path("data")

    # ─── Input Dimensions ───────────────────────────────────────────────
    protein_feature_dim: int = 220  # PsePSSM features
    drug_input_dim: int = 2048  # Morgan fingerprint bits

    # ─── Data File Paths ────────────────────────────────────────────────
    protein_csv: str = "data/features/drugbank/psepssm_features.csv"
    drug_npz: str = "data/features/drugbank/drugbank_morgan_fingerprints.npz"
    raw_interactions: str = "data/raw/drugbank/Drug_Target_Pair_DrugBank.csv"

    def load_data(self) -> tuple:
        """
        Load protein features, drug embeddings, and all interaction pairs for 5-fold CV.

        Returns:
            (protein_features, drug_embeddings, interactions_df)
              - protein_features: dict[protein_id -> (220,) float32]
              - drug_embeddings: dict[smiles -> (2048,) float32]
              - interactions_df: DataFrame with ['protein_id', 'drug_id', 'label']
        """
        print(f"Loading {self.dataset_name} dataset...")

        # ─── Load protein features ───────────────────────────────────────
        protein_df = pd.read_csv(self.protein_csv)
        protein_features = {}

        # Columns are: protein_id, 0, 1, 2, ..., 219 (220 feature columns)
        feature_cols = [col for col in protein_df.columns if col not in ['protein_id']]
        for _, row in protein_df.iterrows():
            protein_id = row["protein_id"]
            features = row[feature_cols].values.astype(np.float32)
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
        manifest_df = pd.read_csv(Path(self.drug_npz).parent / "drugbank_morgan_manifest.csv")
        fingerprints = fp_data["fingerprints"].astype(np.float32) if "fingerprints" in fp_data else fp_data["fps"].astype(np.float32)

        drug_embeddings = {}
        for idx, row in manifest_df.iterrows():
            smiles = row["smiles"]
            drug_embeddings[smiles] = fingerprints[idx]

        self.drug_input_dim = fingerprints.shape[1]
        print(f"  ✓ Loaded {len(drug_embeddings):,} drug fingerprints (dim={self.drug_input_dim})")

        # ─── Load interaction pairs ──────────────────────────────────────
        raw_df = pd.read_csv(self.raw_interactions)

        interactions = pd.DataFrame({
            "protein_id": raw_df["protein_id"],
            "drug_id": raw_df["smiles"],
            "label": raw_df["label"],
        })

        # Filter to proteins and drugs present in feature stores
        before = len(interactions)
        interactions = interactions[
            interactions["protein_id"].isin(protein_features)
            & interactions["drug_id"].isin(drug_embeddings)
        ].reset_index(drop=True)
        after = len(interactions)

        print(f"  ✓ {after:,} interactions ({before - after} filtered)")

        return protein_features, drug_embeddings, interactions


# Global config instance
config = DrugBankDatasetConfig()
