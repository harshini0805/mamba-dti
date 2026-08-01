"""
Mamba+CrossAttention+MLP DTI Architecture

Protein encoder uses Mamba for sequential modeling (ONLY on protein side).
Bidirectional cross-attention fuses protein and drug representations.
Part of attention mechanism fusion study.
"""

from .model import ProteinEncoder, DrugEncoder, MambaCrossAttentionDTI
from .config import default_config

__all__ = ["ProteinEncoder", "DrugEncoder", "MambaCrossAttentionDTI", "default_config"]
