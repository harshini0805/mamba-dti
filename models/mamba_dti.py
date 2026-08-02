import torch
import torch.nn as nn

from .protein_encoder import ProteinEncoder
from .drug_encoder import DrugEncoder


class MambaDTI(nn.Module):

    def __init__(self, dropout: float = 0.3):
        """
        NOTE: `dropout` previously had no effect no matter what
        config.DROPOUT was set to — this constructor took no arguments,
        called DrugEncoder()/ProteinEncoder() with no args (so they fell
        back to their own hardcoded defaults), and both nn.Dropout calls
        below were hardcoded to 0.3 directly. config.DROPOUT was dead
        config. Now threaded through properly: pass dropout explicitly
        from train.py (`MambaDTI(dropout=DROPOUT)`) to actually control it.
        """

        super().__init__()

        self.protein_encoder = ProteinEncoder()

        self.drug_encoder = DrugEncoder(dropout=dropout)

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
