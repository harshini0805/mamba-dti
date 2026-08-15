# EMM-DTI Integration into Mamba-DTI

**Status:** ✅ Successfully integrated as architecture #10

## New Structure

```
D:\Projects\mamba-dti\
├── architectures\
│   ├── mamba_primarycaps_mlp\       (Architecture #1)
│   ├── mamba_attentionpool_mlp\     (Architecture #2)
│   ├── mamba_cross_mlp\             (Architecture #3)
│   ├── mamba_capsule_cross_mlp\     (Architecture #4)
│   ├── mamba_capsule_mlp\           (Architecture #5)
│   ├── mamba_cross_capsule_mlp\     (Architecture #6)
│   ├── meanpool_mlp\                (Architecture #7)
│   ├── bilstm_mlp\                  (Architecture #8)
│   ├── mamba_mlp_fp\                (Architecture #9)
│   └── emm_dti\                     (Architecture #10 - NEW!)
│       ├── train_cv.py              ← Standardized (bs=16, lr=3e-4)
│       ├── config.py                ← Follows mamba-dti pattern
│       ├── __init__.py              ← Package initializer
│       ├── configs/
│       │   └── default.yaml         ← Standardized config
│       ├── results/                 ← Output location
│       ├── checkpoints/             ← Model checkpoints
│       ├── logs/                    ← Training logs
│       ├── README.md                ← Architecture documentation
│       └── INTEGRATION_NOTES.md     ← This file
```

## Original Location Preserved

**EMM_DTI_Replication folder remains unchanged** at:
```
D:\Projects\EMM_DTI_Replication\
├── train_cv.py
├── emm_dti/                (Original package - NOT copied)
├── configs/
├── data/
├── venv/
└── [all other files preserved]
```

## How It Works

### File Structure
- `train_cv.py` - Adapted for mamba-dti architecture folder structure
- `config.py` - Follows mamba-dti configuration pattern
- `configs/default.yaml` - Standardized hyperparameters

### Package Imports
The `train_cv.py` imports the `emm_dti` package from the original EMM_DTI_Replication folder:
```python
# In architectures/emm_dti/train_cv.py
import sys
sys.path.insert(0, "D:\Projects\EMM_DTI_Replication")
from emm_dti.models.emm_dti import EMMDTI
from emm_dti.data.loaders import DTIDataModule
```

This preserves the faithful reproduction while allowing integration into mamba-dti's unified structure.

## Running EMM-DTI

### As a mamba-dti Architecture (Recommended)
```bash
cd D:\Projects\mamba-dti
python architectures/emm_dti/train_cv.py --dataset DrugBank
```

### Original EMM_DTI_Replication Still Works
```bash
cd D:\Projects\EMM_DTI_Replication
python train_cv.py --dataset human
```

Both commands work identically with standardized hyperparameters.

## Standardized Configuration

All hyperparameters across 10 architectures now use:

| Parameter | Value |
|-----------|-------|
| Batch size | 16 |
| Learning rate | 3e-4 |
| Epochs | 200 |
| Patience | 30 |
| Weight decay | 1e-4 |
| CV seeds | [42, 123, 2024] |
| Folds | 5-fold stratified |

## Faithful Reproduction Maintained

✅ **EMM-DTI Architecture:**
- FCS fragment encoding
- Bidirectional Mamba-SSM
- CNN feature extraction
- MLP decoder

✅ **Training Methodology:**
- 3 independent CV runs
- 5-fold stratified cross-validation
- FCS mining on training data only
- Same metrics: Accuracy, Precision, Recall, Specificity, MCC, ROC-AUC, PR-AUC

✅ **Original Paper Faithfulness:**
- Core architecture exactly as specified
- Data handling and preprocessing unchanged
- Metrics computation identical

## File Locations Summary

| Component | Location |
|-----------|----------|
| **Training Script** | `architectures/emm_dti/train_cv.py` |
| **Config** | `architectures/emm_dti/config.py` |
| **YAML Config** | `architectures/emm_dti/configs/default.yaml` |
| **EMM-DTI Package** | `../EMM_DTI_Replication/emm_dti/` (original, preserved) |
| **Results** | `architectures/emm_dti/results/{dataset}/` |
| **Checkpoints** | `architectures/emm_dti/checkpoints/{dataset}/` |
| **Logs** | `architectures/emm_dti/logs/{dataset}/` |

## Next Steps

1. ✅ EMM-DTI integrated into mamba-dti architecture structure
2. ✅ Hyperparameters standardized across all 10 architectures
3. ✅ Both original and integrated versions remain functional
4. Ready to run comparative experiments!

## Quick Start

```bash
# Train EMM-DTI on all 6 datasets (from mamba-dti root)
cd D:\Projects\mamba-dti

for dataset in DrugBank Enzyme BioSNAP C.elegans BindingDB Humans; do
    echo "Training EMM-DTI on $dataset..."
    python architectures/emm_dti/train_cv.py --dataset $dataset
done
```

## Notes

- Original EMM_DTI_Replication folder unchanged - can still be used independently
- New architecture variant uses same underlying code but integrated into mamba-dti
- All results go to `mamba-dti/architectures/emm_dti/results/`
- Proper Python path management ensures imports work correctly
