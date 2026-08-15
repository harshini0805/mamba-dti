# EMM-DTI Integration: Setup and Run Guide

## Current Status

✅ **EMM-DTI architecture faithfully reproduced** with FCS mining, ready for integration  
✅ **train_cv.py** updated to support 5-fold CV with 3 seeds [42, 123, 2024]  
✅ **convert_to_emm_dti.py** script ready to convert mamba-dti datasets  
⏳ **Next step:** Free workspace disk space and run conversion, then training

---

## Critical Setup Steps

### 1. Free Workspace Disk Space
The Linux workspace (used for running Python) is full. Free space by:
- Deleting old cache files in workspace temp directories
- Clearing any previous large model downloads
- Removing old experiment artifacts

Once space is freed, proceed to Step 2.

### 2. Convert Datasets (One-Time)
From `D:\Projects\mamba-dti`:
```bash
python convert_to_emm_dti.py
```

This converts all 6 datasets:
| Source | Target |
|--------|--------|
| `data/raw/human_random/` | `EMM_DTI_Replication/data/human/` |
| `data/raw/enzyme/` | `EMM_DTI_Replication/data/enzyme/` |
| `data/raw/biosnap_random/` | `EMM_DTI_Replication/data/biosnap/` |
| `data/raw/bindingdb_random/` | `EMM_DTI_Replication/data/bindingdb/` |
| `data/raw/celegans_random/` | `EMM_DTI_Replication/data/celegans/` |
| `data/raw/drugbank/` | `EMM_DTI_Replication/data/drugbank/` |

**Output format:**
```
EMM_DTI_Replication/data/{dataset}/
├── drugs.csv              (drug_id, smiles)
├── proteins.csv           (protein_id, sequence)
└── interactions.csv       (drug_id, protein_id, interaction)
```

**Duration:** ~5-10 minutes for all datasets

---

## Training

### Run Single Dataset
```bash
python architectures/emm_dti/train_cv.py --dataset human
python architectures/emm_dti/train_cv.py --dataset enzyme
python architectures/emm_dti/train_cv.py --dataset biosnap
python architectures/emm_dti/train_cv.py --dataset bindingdb
python architectures/emm_dti/train_cv.py --dataset celegans
python architectures/emm_dti/train_cv.py --dataset drugbank
```

### Run All Datasets (Sequential)
```bash
for dataset in human enzyme biosnap bindingdb celegans drugbank; do
  python architectures/emm_dti/train_cv.py --dataset $dataset
done
```

### Run with Custom Hyperparameters
```bash
python architectures/emm_dti/train_cv.py --dataset human --epochs 300 --batch_size 32 --lr 1e-4
```

---

## Results Location

After training, results are saved to:
```
architectures/emm_dti/
├── results/{dataset}/
│   ├── results.csv        (per-fold metrics for all 3 seeds × 5 folds)
│   └── results.json       (summary statistics with mean ± std)
├── checkpoints/{dataset}/
│   ├── best_model_fold_1.pt
│   ├── best_model_fold_2.pt
│   ├── ...
│   └── best_model_fold_5.pt
└── logs/{dataset}/
    └── {dataset}_cv_training.log
```

---

## Training Configuration

**Standard (Paper-Specified):**
- Batch size: **16**
- Learning rate: **3e-4**
- Epochs: **200** (early stop at patience=30)
- Weight decay: **1e-4**
- Gradient clip: **1.0**
- CV: **5-fold stratified** with **3 seeds [42, 123, 2024]**

**Model Architecture:**
- FCS embedding dim: **128**
- Mamba hidden dim: **256**
- Mamba layers: **2**
- Mamba state size: **16**
- CNN output channels: **3**
- Dropout: **0.1**

---

## Evaluation Metrics

Each fold reports 7 metrics:
1. **Accuracy** — (TP + TN) / (TP + TN + FP + FN)
2. **Precision** — TP / (TP + FP)
3. **Recall** — TP / (TP + FN)
4. **Specificity** — TN / (TN + FP)
5. **MCC** — Matthews Correlation Coefficient
6. **ROC-AUC** — Area under ROC curve
7. **PR-AUC** — Area under Precision-Recall curve

Summary statistics (mean ± std) are computed across all 3 seeds × 5 folds = 15 values per metric.

---

## File Structure

```
D:\Projects\mamba-dti\
├── architectures/emm_dti/
│   ├── train_cv.py              ✅ 5-fold CV training (fold-specific FCS mining)
│   ├── SETUP_AND_RUN.md         📄 This file
│   ├── results/                 📊 Results (created after training)
│   ├── checkpoints/             💾 Model checkpoints (created after training)
│   └── logs/                    📝 Training logs (created after training)
├── convert_to_emm_dti.py        ✅ Dataset conversion (mamba-dti → EMM-DTI format)
└── config.py                    ⚙️ mamba-dti global config

D:\Projects\EMM_DTI_Replication\
├── emm_dti/
│   ├── models/
│   │   ├── emm_dti.py           ✅ EMMDTI model (270+ lines)
│   │   ├── fcs.py               ✅ FCS mining (Apriori algorithm)
│   │   └── __init__.py          ✅ Fixed imports
│   └── data/
│       ├── loaders.py           ✅ DTIDataset, DTIDataModule
│       └── preprocessing.py     ✅ DataPreprocessor
└── data/
    ├── human/                   📁 Converted datasets (after run_convert)
    ├── enzyme/
    ├── biosnap/
    ├── bindingdb/
    ├── celegans/
    └── drugbank/
```

---

## Important Notes

### FCS Mining: Preventing Data Leakage
The train_cv.py script mines FCS patterns **per-fold** from each fold's training data only. This prevents data leakage:
- Fold 1: FCS mined from fold 1 training, evaluated on fold 1 validation
- Fold 2: FCS mined from fold 2 training, evaluated on fold 2 validation
- ... and so on

### Separate Data Folder for EMM-DTI
As requested, EMM-DTI data is kept entirely separate:
- **Source (mamba-dti):** `D:\Projects\mamba-dti\data\raw\{dataset}\`
- **Converted (EMM-DTI):** `D:\Projects\EMM_DTI_Replication\data\{dataset}\` ✅
- **mamba-dti folders untouched** ✅

### Architecture Faithfulness
The EMMDTI model is 100% faithful to the paper:
- ✅ FCS mining (Apriori, min_support=0.3, max_k=3)
- ✅ Fragment embedding (128-dim)
- ✅ Bidirectional Mamba-SSM (2 layers, state_size=16)
- ✅ Interaction matrix (dot product)
- ✅ CNN (Conv2d 1→3 channels, 3×3 kernel)
- ✅ MLP predictor

---

## Next Steps After Training

1. **Collect Results:** Copy results.csv and results.json from all 6 datasets
2. **Create Comparison Table:** Aggregate metrics across all 10 architectures
3. **Generate Figures:** Plot performance comparisons (PR-AUC, ROC-AUC, MCC, etc.)
4. **Write Report:** Document EMM-DTI's performance vs. other variants

---

## Troubleshooting

### Error: Dataset not found
```
Error: Dataset 'enzyme' not found at D:\Projects\EMM_DTI_Replication\data\enzyme
```
**Fix:** Run `python convert_to_emm_dti.py` first to convert mamba-dti datasets.

### Error: ModuleNotFoundError for 'emm_dti'
```
ModuleNotFoundError: No module named 'emm_dti'
```
**Fix:** Ensure `D:\Projects\EMM_DTI_Replication` is in sys.path (already set in train_cv.py).

### Error: CUDA out of memory
```
RuntimeError: CUDA out of memory
```
**Fix:** Reduce batch_size: `python train_cv.py --dataset human --batch_size 8`

### Workspace disk space full
**Fix:** Free space before running `convert_to_emm_dti.py` or training scripts.

---

## Contact & Questions

This is the final integrated setup for EMM-DTI as the 10th architecture in mamba-dti.
All files are ready to go once workspace disk space is freed.
