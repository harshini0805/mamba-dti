"""
Common dataset loader for DTI models.

Shared across all architectures.
"""

import torch
from torch.utils.data import Dataset


class DTIDataset(Dataset):
    """
    Drug-Target Interaction dataset.

    Args:
        interactions: DataFrame with columns ['protein_id', 'drug_id', 'label']
        protein_features: dict[protein_id -> (220,) numpy array]
        drug_embeddings: dict[drug_id -> (fingerprint_dim,) numpy array]
    """

    def __init__(self, interactions, protein_features, drug_embeddings) -> None:
        self.df = interactions
        self.protein_features = protein_features
        self.drug_embeddings = drug_embeddings

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple:
        row = self.df.iloc[idx]

        protein = torch.tensor(
            self.protein_features[row["protein_id"]],
            dtype=torch.float32,
        )
        drug = torch.tensor(
            self.drug_embeddings[row["drug_id"]],
            dtype=torch.float32,
        )
        label = torch.tensor(row["label"], dtype=torch.float32)

        return protein, drug, label


def collate_fn(batch: list) -> tuple:
    """
    Collate function for DataLoader.

    Assumes fixed-length protein features and drug embeddings.

    Args:
        batch: list of (protein, drug, label) tuples from __getitem__

    Returns:
        (proteins, drugs, labels) all stacked
    """
    proteins = torch.stack([x[0] for x in batch])
    drugs = torch.stack([x[1] for x in batch])
    labels = torch.stack([x[2] for x in batch])
    return proteins, drugs, labels
