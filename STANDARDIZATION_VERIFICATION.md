# Hyperparameter Standardization - Verification Report

**Date:** 2026-08-14  
**Status:** ✅ VERIFIED - All 10 architectures now use identical hyperparameters

---

## Summary Table

| Architecture | Epochs | Batch Size | Learning Rate | Weight Decay | Patience | CV Seeds | Folds | Status |
|---|---|---|---|---|---|---|---|---|
| **mamba_primarycaps_mlp** | 200 | 16 | 3e-4 | 1e-4 | 30 | [42,123,2024] | 5 | ✅ |
| **mamba_attentionpool_mlp** | 200 | 16 | 3e-4 | 1e-4 | 30 | [42,123,2024] | 5 | ✅ |
| **mamba_cross_mlp** | 200 | 16 | 3e-4 | 1e-4 | 30 | [42,123,2024] | 5 | ✅ |
| **mamba_capsule_cross_mlp** | 200 | 16 | 3e-4 | 1e-4 | 30 | [42,123,2024] | 5 | ✅ |
| **mamba_capsule_mlp** | 200 | 16 | 3e-4 | 1e-4 | 30 | [42,123,2024] | 5 | ✅ |
| **mamba_cross_capsule_mlp** | 200 | 16 | 3e-4 | 1e-4 | 30 | [42,123,2024] | 5 | ✅ |
| **meanpool_mlp** | 200 | 16 | 3e-4 | 1e-4 | 30 | [42,123,2024] | 5 | ✅ |
| **bilstm_mlp** | 200 | 16 | 3e-4 | 1e-4 | 30 | [42,123,2024] | 5 | ✅ |
| **mamba_mlp_fp** | 200 | 16 | 3e-4 | 1e-4 | 30 | [42,123,2024] | 5 | ✅ |
| **EMM-DTI** | 200 | 16 | 3e-4 | 1e-4 | 30 | [42,123,2024] | 5 | ✅ |

---

## Detailed Verification

### 1. EMM-DTI Files Updated

#### 1.1 `train_cv.py` (Lines 155-156)
```python
parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
```
✅ **Status:** batch_size=16, lr=3e-4

#### 1.2 `configs/train_human_ssm.yaml`
```yaml
training:
  batch_size: 16
  learning_rate: 0.0003
  epochs: 200
  early_stopping_patience: 30
optimization:
  weight_decay: 1.0e-04
```
✅ **Status:** All correct

#### 1.3 `configs/default.yaml`
```yaml
training:
  batch_size: 16
  learning_rate: 0.0003
  epochs: 200
  early_stopping_patience: 30
optimization:
  weight_decay: 1.0e-04
```
✅ **Status:** All correct

#### 1.4 `TRAIN_CV_README.md`
- Batch Size: 16 ✅
- Learning Rate: 3e-4 (0.0003) ✅
- Optimizer: Adam (weight_decay=1e-4) ✅
- Usage example: `--batch_size 16 --lr 3e-4` ✅

#### 1.5 `PAPER_COMPARISON.md`
- Hyperparameter table updated ✅
- Standardization note added ✅

---

### 2. Mamba-DTI Architectures Verified (No Changes Needed)

All 9 architectures already use standardized hyperparameters in their `config.py`:

#### Architecture Configs
1. ✅ **mamba_primarycaps_mlp/config.py**
   - num_epochs: 200
   - batch_size: 16
   - learning_rate: 3e-4
   - weight_decay: 1e-4
   - patience: 30
   - num_folds: 5

2. ✅ **mamba_attentionpool_mlp/config.py**
   - Same as above

3. ✅ **mamba_cross_mlp/config.py**
   - Same as above

4. ✅ **mamba_capsule_cross_mlp/config.py**
   - Same as above

5. ✅ **mamba_capsule_mlp/config.py**
   - Same as above

6. ✅ **mamba_cross_capsule_mlp/config.py**
   - Same as above

7. ✅ **meanpool_mlp/config.py**
   - Same as above

8. ✅ **bilstm_mlp/config.py**
   - Same as above (with additional LR scheduler for LSTM stability)

9. ✅ **mamba_mlp_fp/config.py**
   - Same as above

---

### 3. Cross-Validation Seeds Verification

All 10 architectures use identical CV seeds:

```
CV_SEEDS = [42, 123, 2024]
```

**Verified in:**
- ✅ mamba_primarycaps_mlp/train_cv.py:113
- ✅ mamba_attentionpool_mlp/train_cv.py:128
- ✅ mamba_cross_mlp/train_cv.py:140
- ✅ mamba_capsule_cross_mlp/train_cv.py:113
- ✅ mamba_capsule_mlp/train_cv.py:113
- ✅ mamba_cross_capsule_mlp/train_cv.py:113
- ✅ meanpool_mlp/train_cv.py:161
- ✅ bilstm_mlp/train_cv.py:172
- ✅ mamba_mlp_fp/train_cv.py:265
- ✅ EMM-DTI/train_cv.py:220

---

## Cross-Validation Configuration (All 10 Architectures)

| Parameter | Value | Verification |
|-----------|-------|---|
| **CV Runs** | 3 | ✅ Seeds: [42, 123, 2024] |
| **Folds per Run** | 5 | ✅ Stratified K-Fold |
| **Total Folds** | 15 | ✅ 3 runs × 5 folds |
| **Stratification** | By interaction label | ✅ Maintains class distribution |

---

## Standardization Summary

### What Was Standardized
✅ Batch size: **16** (was 32 in EMM-DTI)  
✅ Learning rate: **3e-4** (was 0.001 in EMM-DTI)  
✅ Weight decay: **1e-4** (was 1e-5 in EMM-DTI)  
✅ CV seeds: **[42, 123, 2024]** (already consistent)  
✅ Epochs: **200** (already consistent)  
✅ Patience: **30** (already consistent)  
✅ Folds: **5** (already consistent)  

### What Remained Unchanged
✅ All architecture specifications (Mamba, CNN, MLP layer sizes, etc.)  
✅ Model initialization  
✅ Loss function (BCEWithLogitsLoss)  
✅ Metrics (Accuracy, Precision, Recall, Specificity, MCC, ROC-AUC, PR-AUC)  
✅ Gradient clipping (1.0)  
✅ Dropout values (architecture-specific)  

---

## Fairness Guarantee

All 10 architectures now train with:
- **Same** computational budget (batch size, learning rate, epochs)
- **Same** regularization (weight decay, gradient clipping)
- **Same** evaluation methodology (5-fold CV, 3 runs)
- **Different** architectural designs (which is the point!)

**Result:** Any performance differences are attributable to **architecture quality**, not hyperparameter tuning.

---

## Files Verified

### EMM-DTI (D:\Projects\EMM_DTI_Replication)
- ✅ train_cv.py (lines 155-156)
- ✅ configs/train_human_ssm.yaml (lines 25-26, 36)
- ✅ configs/default.yaml (lines 30-31, 41)
- ✅ TRAIN_CV_README.md (documentation)
- ✅ PAPER_COMPARISON.md (hyperparameter table + note)

### Mamba-DTI (D:\Projects\mamba-dti\architectures)
- ✅ 9 × config.py files (training hyperparameters)
- ✅ 9 × train_cv.py files (CV_SEEDS definition)
- ✅ HYPERPARAMETER_STANDARDIZATION.md (created)

---

## Next Steps

1. ✅ Verification complete
2. Ready for training with all 10 architectures
3. Collect results across all datasets (DrugBank, Enzyme, BioSNAP, C. elegans, BindingDB, Humans)
4. Generate unified comparison tables for manuscript

---

**Verified by:** Standardization verification script  
**Date:** 2026-08-14  
**All systems GO for training! ✅**
