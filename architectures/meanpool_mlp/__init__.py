"""
MeanPool+MLP DTI Architecture (Baseline)

Baseline with NO sequential modeling.
Each protein position treated independently, order discarded by mean pooling.
Floor baseline for ablation studies.
"""

from .model import ProteinBranch, DrugBranch, MeanPoolDTI
from .config import default_config

__all__ = ["ProteinBranch", "DrugBranch", "MeanPoolDTI", "default_config"]
