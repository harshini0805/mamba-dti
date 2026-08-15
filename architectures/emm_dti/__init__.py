"""
EMM-DTI Architecture Module

This module imports the EMM-DTI implementation from the EMM_DTI_Replication folder
to maintain the original faithful reproduction while integrating it as an architecture
variant within the mamba-dti project structure.
"""

import sys
from pathlib import Path

# Add EMM_DTI_Replication to path to import emm_dti package
EMM_DTI_PATH = Path(__file__).parent.parent.parent.parent / "EMM_DTI_Replication"
if str(EMM_DTI_PATH) not in sys.path:
    sys.path.insert(0, str(EMM_DTI_PATH))

__all__ = [
    "emm_dti",
    "train_cv",
    "config",
]
