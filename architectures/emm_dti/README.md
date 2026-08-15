# EMM-DTI Architecture (10th Variant)

**Architecture Type:** FCS + Bidirectional Mamba-SSM + CNN + MLP

**Faithful Reproduction:** Yes - of EMM-DTI paper with standardized hyperparameters

## Overview

EMM-DTI is the 10th architecture in the standardized comparison, faithfully reproducing the EMM-DTI paper's approach while using hyperparameters consistent with the other 9 mamba-dti variants for fair comparison.

### Architecture Pipeline
```
Drug SMILES → FCS Mining → Fragment Encoding 
→ Bidirectional Mamba-SSM → Interaction Matrix (Dot Product) 
→ CNN Feature Extraction → MLP Predictor → DTI Logit
```

## Configuration

### Default Hyperparameters
- **Epochs:** 200
- **Batch Size:** 16
- **Learning Rate:** 3e-4 (0.0003)
- **Weight Decay:** 1e-4
- **Patience:** 30 (early stopping)
- **Gradient Clip:** 1.0

### Cross-Validation
- **CV Seeds:** [42, 123, 2024]
- **Folds:** 5-fold stratified
- **Total Folds:** 15 (3 runs × 5 folds)

### Model Architecture
- **FCS Embedding Dim:** 128
- **Mamba Hidden Dim:** 256
- **Mamba Layers:** 2
- **CNN Output Channels:** 3
- **Dropout:** 0.1

## Usage

### Basic Training
```bash
cd D:\Projects\mamba-dti
python architectures/emm_dti/train_cv.py --dataset DrugBank
```

### With Custom Hyperparameters
```bash
python architectures/emm_dti/train_cv.py \
  --dataset DrugBank \
  --epochs 200 \
  --batch_size 16 \
  --lr 3e-4
```

## Output Structure

```
architectures/emm_dti/
├── train_cv.py              (Training script)
├── config.py                (Configuration)
├── emm_dti/                 (Package - copied from EMM_DTI_Replication)
├── configs/                 (Configuration files)
├── results/
│   └── {dataset}/
│       ├── results.csv      (15 rows: 3 seeds × 5 folds)
│       ├── results.json     (Summary statistics)
│       └── cv_summary.json
├── checkpoints/
│   └── {dataset}/
│       └── best_model_fold_*.pt
└── logs/
    └── {dataset}/
        └── cv_training.log
```

## Key Features

✅ **Faithfully reproduced** - Core architecture matches EMM-DTI paper  
✅ **Standardized hyperparameters** - Matches 9 other architectures for fair comparison  
✅ **3 CV runs** - [42, 123, 2024] seeds for robust performance estimation  
✅ **5-fold stratified CV** - Maintains class balance per fold  
✅ **FCS data leakage prevention** - Patterns mined from training data only  

## Metrics Tracked

- Accuracy
- Precision
- Recall
- Specificity
- Matthews Correlation Coefficient (MCC)
- ROC-AUC
- PR-AUC (used for early stopping)

## Notes

- EMM-DTI package is imported from EMM_DTI_Replication folder (kept separate to maintain faithfulness)
- All hyperparameters standardized with other 9 mamba-dti architectures
- Training follows 5-fold CV with 3 independent runs per dataset
- Results automatically aggregated with mean ± std metrics

## References

- **Paper:** EMM-DTI (original research)
- **Faithful Replication:** Full architecture reproduction with standard training methodology
- **Comparison Base:** Standardized hyperparameters enable fair architectural comparison

## Files Location

- **Training Script:** `train_cv.py`
- **Config:** `config.py`
- **Data Package:** Located in `D:\Projects\EMM_DTI_Replication\emm_dti\`
- **Results:** `results/{dataset}/`
- **Checkpoints:** `checkpoints/{dataset}/`
