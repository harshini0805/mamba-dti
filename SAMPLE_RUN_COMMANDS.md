# Sample Run Commands - Standardized Training

All 10 architectures use identical hyperparameters:
- **Batch size:** 16
- **Learning rate:** 3e-4
- **Epochs:** 200
- **Early stopping patience:** 30
- **CV seeds:** [42, 123, 2024]
- **5-fold stratified cross-validation**

---

## Mamba-DTI Architectures (9 variants)

### Basic Training Template
```bash
cd D:\Projects\mamba-dti
python architectures/<ARCH_NAME>/train_cv.py --dataset <DATASET_NAME>
```

### Example 1: Mamba+PrimaryCaps on DrugBank
```bash
cd D:\Projects\mamba-dti
python architectures/mamba_primarycaps_mlp/train_cv.py --dataset DrugBank
```

### Example 2: Mamba+Cross-Attention on Enzyme
```bash
cd D:\Projects\mamba-dti
python architectures/mamba_cross_mlp/train_cv.py --dataset Enzyme
```

### Example 3: BiLSTM+MLP on BioSNAP
```bash
cd D:\Projects\mamba-dti
python architectures/bilstm_mlp/train_cv.py --dataset BioSNAP
```

### Example 4: Mean Pooling Baseline on C. elegans
```bash
cd D:\Projects\mamba-dti
python architectures/meanpool_mlp/train_cv.py --dataset C.elegans
```

### Example 5: Mamba+MLP with Morgan Fingerprints on Human
```bash
cd D:\Projects\mamba-dti
python architectures/mamba_mlp_fp/train_cv.py --dataset Human
```

### All Architectures Quick Reference
```bash
# 1. Mamba + PrimaryCaps + MLP
python architectures/mamba_primarycaps_mlp/train_cv.py --dataset DrugBank

# 2. Mamba + Attention Pool + MLP
python architectures/mamba_attentionpool_mlp/train_cv.py --dataset Enzyme

# 3. Mamba + Cross-Attention + MLP
python architectures/mamba_cross_mlp/train_cv.py --dataset BioSNAP

# 4. Mamba + Capsule + Cross-Attention + MLP
python architectures/mamba_capsule_cross_mlp/train_cv.py --dataset C.elegans

# 5. Mamba + Capsule + MLP
python architectures/mamba_capsule_mlp/train_cv.py --dataset BindingDB

# 6. Mamba + Cross-Attention + Capsule + MLP
python architectures/mamba_cross_capsule_mlp/train_cv.py --dataset Humans

# 7. Mean Pool + MLP (Baseline)
python architectures/meanpool_mlp/train_cv.py --dataset DrugBank

# 8. BiLSTM + MLP (Baseline)
python architectures/bilstm_mlp/train_cv.py --dataset Enzyme

# 9. Mamba + MLP with Morgan Fingerprints
python architectures/mamba_mlp_fp/train_cv.py --dataset BioSNAP
```

---

## EMM-DTI (New Architecture)

### Basic Training Template
```bash
cd D:\Projects\EMM_DTI_Replication
python train_cv.py --dataset <DATASET_NAME>
```

### Example 1: EMM-DTI on Human (Default)
```bash
cd D:\Projects\EMM_DTI_Replication
python train_cv.py --dataset human
```

### Example 2: EMM-DTI with explicit config
```bash
cd D:\Projects\EMM_DTI_Replication
python train_cv.py --dataset human --config configs/train_human_ssm.yaml
```

### Example 3: EMM-DTI with custom hyperparameters (override defaults)
```bash
cd D:\Projects\EMM_DTI_Replication
python train_cv.py \
  --dataset human \
  --epochs 200 \
  --batch_size 16 \
  --lr 3e-4 \
  --config configs/train_human_ssm.yaml
```

---

## Running All 10 Architectures on All 6 Datasets

### Batch Script (Windows PowerShell)
```powershell
# Run all 9 mamba-dti architectures on all 6 datasets
$architectures = @(
    "mamba_primarycaps_mlp",
    "mamba_attentionpool_mlp",
    "mamba_cross_mlp",
    "mamba_capsule_cross_mlp",
    "mamba_capsule_mlp",
    "mamba_cross_capsule_mlp",
    "meanpool_mlp",
    "bilstm_mlp",
    "mamba_mlp_fp"
)

$datasets = @("DrugBank", "Enzyme", "BioSNAP", "C.elegans", "BindingDB", "Humans")

foreach ($arch in $architectures) {
    foreach ($dataset in $datasets) {
        Write-Host "Training $arch on $dataset..."
        cd D:\Projects\mamba-dti
        python architectures/$arch/train_cv.py --dataset $dataset
    }
}

# Then run EMM-DTI on all 6 datasets
foreach ($dataset in $datasets) {
    Write-Host "Training EMM-DTI on $dataset..."
    cd D:\Projects\EMM_DTI_Replication
    python train_cv.py --dataset $dataset
}
```

### Batch Script (Linux/Mac Bash)
```bash
#!/bin/bash

ARCHITECTURES=(
    "mamba_primarycaps_mlp"
    "mamba_attentionpool_mlp"
    "mamba_cross_mlp"
    "mamba_capsule_cross_mlp"
    "mamba_capsule_mlp"
    "mamba_cross_capsule_mlp"
    "meanpool_mlp"
    "bilstm_mlp"
    "mamba_mlp_fp"
)

DATASETS=("DrugBank" "Enzyme" "BioSNAP" "C.elegans" "BindingDB" "Humans")

# Run all mamba-dti architectures
for arch in "${ARCHITECTURES[@]}"; do
    for dataset in "${DATASETS[@]}"; do
        echo "Training $arch on $dataset..."
        cd ~/Projects/mamba-dti
        python architectures/$arch/train_cv.py --dataset $dataset
    done
done

# Run EMM-DTI
for dataset in "${DATASETS[@]}"; do
    echo "Training EMM-DTI on $dataset..."
    cd ~/Projects/EMM_DTI_Replication
    python train_cv.py --dataset $dataset
done
```

---

## Expected Output

Each run will produce:

```
=================================================================================
  CV Run 1/3 (seed=42)
=================================================================================
  Fold 1/5 | Train: 6,104 | Val: 1,368

    ──────────────────────────────────────────────────
    Fold 1 | Epoch 1
    ──────────────────────────────────────────────────
    ────────────────  ────────────────  ────────────────
    Metric               Train          Val
    ────────────────  ────────────────  ────────────────
    Accuracy             0.7234          0.7156
    Precision            0.7512          0.7402
    Recall               0.6891          0.6824
    Specificity          0.7578          0.7489
    Mcc                  0.4521          0.4312
    Roc Auc              0.8234          0.8156
    Pr Auc               0.8156          0.8042
    Loss                 0.5432          0.5678
    ──────────────────────────────────────────────────

[... continues for 3 CV runs × 5 folds = 15 folds total ...]

=================================================================================
  SUMMARY: 3 CV Runs × 5 Folds
=================================================================================
  val_pr_auc          : 0.8342 ± 0.0124
  val_roc_auc         : 0.8567 ± 0.0156
  val_accuracy        : 0.8234 ± 0.0098
  val_precision       : 0.8156 ± 0.0134
  val_recall          : 0.8267 ± 0.0156
  val_specificity     : 0.8201 ± 0.0145
  val_mcc             : 0.6478 ± 0.0187

  ✓ Saved fold results to results/<dataset>/results.csv
  ✓ Saved summary to results/<dataset>/results.json
  ✓ Saved CV summary to results/<dataset>/cv_summary.json
```

---

## Output Structure

```
mamba-dti/
├── architectures/
│   ├── mamba_primarycaps_mlp/
│   │   ├── results/
│   │   │   └── DrugBank/
│   │   │       ├── results.csv         (15 rows: 3 seeds × 5 folds)
│   │   │       ├── results.json        (summary stats)
│   │   │       └── cv_summary.json
│   │   ├── checkpoints/
│   │   │   └── DrugBank/
│   │   │       ├── best_model_fold_1.pt
│   │   │       ├── best_model_fold_2.pt
│   │   │       └── ... (5 folds × 3 seeds)
│   │   └── logs/
│   │       └── DrugBank/
│   │           └── cv_training.log
│   └── [other architectures...]

EMM_DTI_Replication/
├── results/
│   └── human/
│       ├── results.csv
│       ├── results.json
│       └── cv_summary.json
├── checkpoints/
│   └── human/
│       └── best_model_fold_*.pt
└── logs/
    └── human/
        └── cv_training.log
```

---

## Quick Start (Copy & Paste Ready)

### Single Run - Mamba+PrimaryCaps on DrugBank
```bash
cd D:\Projects\mamba-dti && python architectures/mamba_primarycaps_mlp/train_cv.py --dataset DrugBank
```

### Single Run - EMM-DTI on Human
```bash
cd D:\Projects\EMM_DTI_Replication && python train_cv.py --dataset human
```

### Debug Mode (Single Fold)
Edit config to `num_epochs: 10` temporarily to test the pipeline quickly before full training.

---

## Troubleshooting

**GPU Out of Memory?**
- Reduce batch_size (already at 16, minimum recommended)
- Reduce num_workers in DataLoader

**Slow Training?**
- Ensure CUDA is available: `import torch; print(torch.cuda.is_available())`
- Check GPU usage: `nvidia-smi`

**Reproducibility Issue?**
- All 3 CV seeds (42, 123, 2024) are set consistently
- Results should be identical across runs if hardware/CUDA version unchanged

---

## Notes

- All runs use **3 CV seeds** [42, 123, 2024], so each dataset gets 3 independent runs
- Each run trains 5 folds = **15 total folds per architecture per dataset**
- Total computational cost: 10 architectures × 6 datasets × 15 folds = **900 fold-level trainings**
- Estimate: ~2-4 weeks depending on GPU and dataset sizes
