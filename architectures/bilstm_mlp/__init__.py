"""
BiLSTM+MLP DTI Architecture

Protein encoder uses BiLSTM for sequential modeling of PsePSSM features.
Drug encoder uses MLP for Morgan fingerprints.
Part of ablation study: MeanPool < BiLSTM < Mamba
"""

from .model import BiLSTMDTI, ProteinEncoder, DrugEncoder
from .config import default_config

__all__ = ["BiLSTMDTI", "ProteinEncoder", "DrugEncoder", "default_config"]
