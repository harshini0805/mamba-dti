import torch
import torch.nn as nn

from .protein_encoder import ProteinEncoder
from .drug_encoder_cnn import CNNDrugEncoder


class MambaCNNDTI(nn.Module):
    """
    Variant of MambaDTI (models/mamba_dti.py) that swaps the drug encoder
    from a plain MLP (models/drug_encoder.py) to a 1D-CNN
    (models/drug_encoder_cnn.py). The protein side (Mamba over PsePSSM,
    models/protein_encoder.py) and the classifier head are IDENTICAL to
    MambaDTI, so any performance difference between the two isolates the
    effect of the drug-encoder architecture specifically, rather than being
    confounded with some other change.

    Standalone — does not modify models/mamba_dti.py or models/drug_encoder.py.
    Trained via train_cnn.py, which does not modify train.py/config.py either
    (same "new file, existing files untouched" pattern as train_swa.py /
    train_ensemble.py).
    """

    def __init__(self, dropout: float = 0.3):

        super().__init__()

        self.protein_encoder = ProteinEncoder()

        self.drug_encoder = CNNDrugEncoder(dropout=dropout)

        self.classifier = nn.Sequential(

            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(128, 1)

        )

    def forward(self, protein, drug):

        protein_embedding = self.protein_encoder(protein)

        drug_embedding = self.drug_encoder(drug)

        x = torch.cat(
            [protein_embedding, drug_embedding],
            dim=1
        )

        logits = self.classifier(x)

        return logits.squeeze(1)
