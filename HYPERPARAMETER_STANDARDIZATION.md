# Hyperparameter Standardization - All 10 Architectures

## Overview
All 10 DTI prediction architectures (9 existing mamba-dti variants + EMM-DTI) have been standardized to use identical hyperparameters for fair architectural comparison.

## Standardized Configuration

### Training Hyperparameters
| Parameter | Value | Rationale |
|-----------|-------|-----------|
| **Epochs** | 200 | Standard across mamba-dti; sufficient for convergence |
| **Batch Size** | 16 | Consistent with existing 9 mamba-dti architectures |
| **Learning Rate** | 3e-4 (0.0003) | Consistent with existing 9 mamba-dti architectures |
| **Weight Decay** | 1e-4 | L2 regularization; consistent across all architectures |
| **Gradient Clipping** | 1.0 | Stability; consistent across all architectures |
| **Early Stopping Patience** | 30 | Consistent across all architectures |

### Cross-Validation Setup
| Parameter | Value |
|-----------|-------|
| **CV Seeds** | [42, 123, 2024] |
| **Number of CV Runs** | 3 independent runs |
| **Folds per Run** | 5 (stratified K-fold) |
| **Total Folds** | 15 (3 runs × 5 folds) |

## Architectures Included

### 9 Existing Mamba-DTI Variants (Already Standardized)
1. **mamba_primarycaps_mlp** - Mamba + Primary Capsule Networks + MLP
2. **mamba_attentionpool_mlp** - Mamba + Attention Pooling + MLP
3. **mamba_cross_mlp** - Mamba + Cross-Attention + MLP
4. **mamba_capsule_cross_mlp** - Mamba + Capsule + Cross-Attention + MLP
5. **mamba_capsule_mlp** - Mamba + Capsule Networks + MLP
6. **mamba_cross_capsule_mlp** - Mamba + Cross-Attention + Capsule + MLP
7. **meanpool_mlp** - Mean Pooling + MLP (baseline)
8. **bilstm_mlp** - BiLSTM + MLP (baseline)
9. **mamba_mlp_fp** - Mamba + MLP with Morgan Fingerprints

### 1 New Architecture (Updated for Standardization)
10. **EMM-DTI** - FCS + Mamba-SSM + CNN + MLP

## Recent Updates

### EMM-DTI (D:\Projects\EMM_DTI_Replication)

**Files Updated:**
1. `train_cv.py` (lines 155-156)
   - Changed: `--batch_size` default from 32 → 16
   - Changed: `--lr` default from 0.001 → 3e-4

2. `configs/train_human_ssm.yaml`
   - batch_size: 32 → 16
   - learning_rate: 0.001 → 0.0003
   - weight_decay: 1.0e-05 → 1.0e-04

3. `configs/default.yaml`
   - batch_size: 32 → 16
   - learning_rate: 0.001 → 0.0003
   - epochs: 100 → 200
   - early_stopping_patience: 10 → 30
   - weight_decay: 1.0e-05 → 1.0e-04

4. `TRAIN_CV_README.md`
   - Updated default hyperparameter documentation
   - Updated usage examples

5. `PAPER_COMPARISON.md`
   - Updated hyperparameter table with new values
   - Added note about standardization for fair comparison

## Consistency Verification

✅ **All 10 architectures now use:**
- 200 epochs
- Batch size: 16
- Learning rate: 3e-4
- Weight decay: 1e-4
- Gradient clipping: 1.0
- Early stopping patience: 30
- CV seeds: [42, 123, 2024]
- 5-fold stratified cross-validation

## Next Steps

1. **Re-run training** for EMM-DTI and optionally re-run other architectures with standardized hyperparameters
2. **Collect results** in unified format for all 10 architectures
3. **Generate comparison tables** showing architecture performance differences (no longer confounded by hyperparameter differences)
4. **Update manuscript** with architectural comparison results

## Notes

- **Fairness**: By standardizing hyperparameters, any performance differences between architectures can now be attributed to architectural design, not hyperparameter tuning.
- **EMM-DTI Paper**: Paper does NOT specify hyperparameters; standardization to mamba-dti values ensures compatibility while maintaining faithful architectural reproduction.
- **Reproducibility**: All 3 CV seeds enable robust performance estimation with mean ± std deviation.

## References

- Mamba-DTI architectures: `D:\Projects\mamba-dti\architectures\*/config.py`
- EMM-DTI: `D:\Projects\EMM_DTI_Replication\configs\*.yaml`
- Previous cross-validation implementations: All `train_cv.py` files use CV_SEEDS = [42, 123, 2024]
