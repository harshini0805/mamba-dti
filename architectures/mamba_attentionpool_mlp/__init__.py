"""
Mamba+AttentionPool+MLP DTI Architecture

Protein encoder uses Mamba for sequential modeling (ONLY on protein side).
Both protein and drug use learned AttentionPool for pooling.
Part of pooling mechanism ablation study.
"""

from .model import AttentionPool, ProteinEncoder, DrugEncoder, MambaAttnDTI
from .config import default_config

__all__ = ["AttentionPool", "ProteinEncoder", "DrugEncoder", "MambaAttnDTI", "default_config"]
