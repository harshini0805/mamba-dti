# Refactoring Summary: Mamba+MLP DTI Architecture

## What Was Done

### 1. **Restructured to architecture-based organization**
```
architectures/
├── mamba_mlp_fp/          ← NEW FIRST ARCHITECTURE
│   ├── config.py          ← Hyperparameters (from original hardcoded values)
│   ├── model.py           ← Architecture definitions only (3 classes)
│   ├── train.py           ← Training pipeline (refactored from original)
│   ├── README.md          ← Documentation
│   ├── results/
│   │   ├── humans/
│   │   ├── bindingdb/
│   │   ├── biosnap/
│   │   └── celegans/
│   ├── logs/
│   │   ├── humans/
│   │   ├── bindingdb/
│   │   ├── biosnap/
│   │   └── celegans/
│   └── checkpoints/
│       ├── humans/
│       ├── bindingdb/
│       ├── biosnap/
│       └── celegans/
```

### 2. **Separated concerns into shared modules**
```
common/
├── dataset_loader.py      ← DTIDataset class, collate_fn (shared)
├── metrics.py             ← compute_metrics function (shared)
└── __init__.py

datasets/
├── humans.py              ← Dataset-specific config (NEW)
├── bindingdb.py           ← Placeholder for next dataset
├── biosnap.py             ← Placeholder for next dataset
├── celegans.py            ← Placeholder for next dataset
└── __init__.py
```

### 3. **Verified Logical Correctness**

✅ **Protein Encoder (Mamba)**
- Input: (B, 220) PsePSSM features
- Unsqueeze → Linear projection to d_model (1 → 128)
- Mamba layer (d_state=16, d_conv=4, expand=2)
- LayerNorm for stability
- Mean pooling over sequence dimension
- Output: (B, 128) fixed-size embedding
- **Sound**: Treats PsePSSM as sequential, learns positional dependencies

✅ **Drug Encoder (MLP)**
- Input: (B, 2048) Morgan fingerprints (fixed-length, no padding needed)
- 2-layer MLP: 2048 → 256 → 128
- ReLU + Dropout for regularization
- Output: (B, 128) dense drug embedding
- **Sound**: Fixed-length inputs don't need sequence models; MLP is efficient

✅ **Decoder (MLP)**
- Input: (B, 256) concatenated embeddings
- 3-layer MLP: 256 → 256 → 128 → 1
- ReLU + Dropout for regularization
- Output: (B,) raw logits (NOT sigmoid-ed)
- Loss: BCEWithLogitsLoss (applies sigmoid internally)
- **Sound**: Numerically stable, correct for binary classification

✅ **Metrics Computation**
- Binarizes at 0.5 threshold for discrete metrics
- Computes all 7 requested metrics correctly
- Handles zero_division edge cases

✅ **Cross-Validation**
- StratifiedKFold preserves label distribution
- Early stopping on ROC-AUC with patience=5
- Checkpoint saving with best ROC-AUC per fold
- Proper train/val split per fold

### 4. **Configuration System**

**Before**: Hardcoded hyperparameters in script (lines 243-255)
```python
NUM_EPOCHS = 50
BATCH_SIZE = 16
LEARNING_RATE = 3e-4
# etc.
```

**After**: Structured config class with override capability
```python
# config.py defines defaults
config = MambaMLPConfig(num_epochs=50, batch_size=16, ...)

# CLI overrides
python train.py --dataset humans --epochs 100 --batch_size 32 --lr 1e-4
```

**Benefits**:
- Hyperparameters easily tunable per dataset
- Config saved with results for reproducibility
- Same architecture used across all datasets

### 5. **Dataset Abstraction**

**Before**: Hardcoded paths embedded in script
```python
protein_df = pd.read_csv(
    "/mnt/c/Users/Harshini J/Engineering/Projects/.../psepssm_features.csv"
)
```

**After**: Dataset config handles loading
```python
# datasets/humans.py
class HumansDatasetConfig:
    def load_data(self):
        return protein_features, drug_embeddings, interactions
```

**Benefits**:
- Easy to add new datasets (just create datasets/{name}.py)
- Same train.py works for all datasets
- Paths configurable, not hardcoded

## Design Decisions

### 1. Mean Pooling (not last-token pooling)
The Mamba protein encoder uses mean pooling over all 220 positions:
```python
x = x.mean(dim=1)  # (B, 220, 128) → (B, 128)
```

**Why**: Averages importance across all pseudo-positions; more stable than relying on final state. Can be easily changed to `x[:, -1, :]` if preferred.

### 2. Separate Drug Embeddings Storage
Drug embeddings are stored as dict[drug_id → ndarray] and reused across folds:
```python
drug_embeddings: dict[str, np.ndarray] = {
    drug_id: _fp_vecs[i] for i, drug_id in enumerate(_fp_ids)
}
```

**Why**: Avoids recomputing Morgan fingerprints; they're fixed by generation.

### 3. Python Config over YAML
Configuration is Python dataclasses (not YAML):
```python
@dataclass
class MambaMLPConfig:
    num_epochs: int = 50
```

**Why**: Single language (Python); can compute derived values; easier for devs to modify.

## How to Run

### Train Mamba+MLP on humans dataset:
```bash
cd mamba-dti
python architectures/mamba_mlp_fp/train.py --dataset humans
```

### With custom hyperparameters:
```bash
python architectures/mamba_mlp_fp/train.py \
  --dataset humans \
  --epochs 100 \
  --batch_size 32 \
  --lr 5e-4
```

### Output structure:
```
architectures/mamba_mlp_fp/
├── results/humans/
│   └── [metrics & logs]
├── logs/humans/
│   └── [training logs]
└── checkpoints/humans/
    └── best_model_fold_*.pt
```

## Next Steps

### For Adding Next 9 Architectures:

1. **Create directory**:
   ```bash
   mkdir -p architectures/{architecture_name}
   ```

2. **Copy template files**:
   ```bash
   cp architectures/mamba_mlp_fp/{model.py,config.py,train.py} \
      architectures/{architecture_name}/
   ```

3. **Modify model.py**: Replace MambaMLPDTI with new architecture
4. **Tune config.py**: Adjust hyperparameters for new architecture
5. **No changes needed to train.py** — it's architecture-agnostic

### For Adding New Datasets:

1. **Create config file** `datasets/{dataset_name}.py`
2. **Implement load_data()** method
3. **Run**:
   ```bash
   python architectures/mamba_mlp_fp/train.py --dataset {dataset_name}
   ```

## File Checklist

- ✅ `architectures/mamba_mlp_fp/model.py` — 3 architecture classes
- ✅ `architectures/mamba_mlp_fp/config.py` — Hyperparameters
- ✅ `architectures/mamba_mlp_fp/train.py` — Training script
- ✅ `architectures/mamba_mlp_fp/README.md` — Documentation
- ✅ `common/dataset_loader.py` — Shared DTI dataset class
- ✅ `common/metrics.py` — Shared metrics computation
- ✅ `datasets/humans.py` — Example dataset config
- ✅ Package __init__.py files for proper imports

## Verification Checklist

- ✅ Uses `from mamba_ssm import Mamba`
- ✅ ProteinEncoder architecture is logically sound
- ✅ DrugEncoder architecture is logically sound
- ✅ Decoder and loss function are correct for binary classification
- ✅ Metrics computation is correct for all 7 metrics
- ✅ Cross-validation properly stratified
- ✅ Config system allows easy customization
- ✅ Dataset abstraction allows multi-dataset training
- ✅ All hard-coded paths removed
- ✅ All hard-coded hyperparameters moved to config
- ✅ Reproducibility: random seeds fixed, configs saved
