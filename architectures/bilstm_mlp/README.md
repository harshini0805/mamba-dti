# BiLSTM+MLP DTI Architecture

## Overview

Binary drug-target interaction (DTI) classifier combining:
- **Protein Encoder**: BiLSTM-based sequence model over PsePSSM features
- **Drug Encoder**: MLP over Morgan fingerprints
- **Decoder**: MLP over concatenated embeddings → binary prediction

## Architecture Details

### Protein Encoder (BiLSTM)
- Input: (B, 220) PsePSSM pseudo-position-specific scoring matrix
- Projection: 220 → d_model (64)
- Pre-norm: LayerNorm
- BiLSTM Block: num_layers=2, hidden_size=d_model//2, bidirectional=True
  - Bidirectional output: forward (d_model//2) + backward (d_model//2) = d_model
  - Dropout between layers for regularization
- Post-norm: LayerNorm
- Pooling: Mean over sequence dimension → (B, 128)
- **Rationale**: PsePSSM is sequence-like; BiLSTM captures positional dependencies via recurrence

### Drug Encoder (MLP)
- Input: (B, 2048) Morgan fingerprints (fixed-length bit-vectors)
- Layer 1: 2048 → 256 (ReLU, Dropout 0.3)
- Layer 2: 256 → 128
- Output: (B, 128) dense embedding
- **Rationale**: Fixed-length vectors don't need sequence models; MLP learns bit co-occurrence patterns

### Decoder (MLP)
- Input: (B, 256) concatenated embeddings
- Layer 1: 256 → 128 (ReLU, Dropout 0.3)
- Layer 2: 128 → 1
- Output: (B,) binary logits (pre-sigmoid)
- Loss: BCEWithLogitsLoss

## Ablation Context

This architecture is part of a sequential modeling ablation:

| Model | Protein Encoder | Properties | Purpose |
|-------|-----------------|-----------|---------|
| MeanPool | None | No sequential modeling | Baseline: no seq structure |
| BiLSTM | LSTM (recurrent) | Sequential + recurrence | Test if seq structure helps |
| Mamba | SSM | Sequential + state space | Test if SSM inductive bias helps |

**Interpretation:**
- If BiLSTM > MeanPool: sequential structure in PsePSSM matters
- If Mamba > BiLSTM: SSM inductive bias is better than pure recurrence
- If BiLSTM ≈ Mamba: recurrence sufficient; Mamba's complexity not justified

## Files

- **`model.py`**: Architecture definitions (ProteinEncoder, DrugEncoder, BiLSTMDTI)
- **`config.py`**: Hyperparameters and model architecture settings
- **`train.py`**: Training loop with 5-fold cross-validation + ReduceLROnPlateau scheduler
- **`results/{dataset}/`**: Evaluation metrics per dataset
- **`logs/{dataset}/`**: Training logs per dataset
- **`checkpoints/{dataset}/`**: Model checkpoints per dataset

## Usage

### Training on a Dataset

```bash
# Train on humans dataset with default hyperparameters
cd mamba-dti
python architectures/bilstm_mlp/train.py --dataset humans

# Override hyperparameters
python architectures/bilstm_mlp/train.py --dataset bindingdb --epochs 100 --batch_size 32 --lr 5e-4
```

### Adding a New Dataset

1. Create `datasets/{dataset_name}.py` with a config class
2. Run training: `python architectures/bilstm_mlp/train.py --dataset {dataset_name}`

See `architectures/bilstm_mlp/../mamba_mlp_fp/README.md` for dataset setup details.

## Key Design Decisions

### BiLSTM Hidden Size Tuning
Hidden size is set to `d_model // 2` so that bidirectional concatenation gives exactly `d_model`:
```
hidden_size = 64 // 2 = 32
bidirectional output = [forward: 32, backward: 32] → 64
```
This matches Mamba output width for fair comparison.

### Weight Initialization
- Linear layers: Xavier Uniform (gain=0.5)
- LSTM hidden-to-hidden: Orthogonal (preserves norm through recurrence)
- LSTM input-to-hidden: Xavier Uniform
- Biases: Zeros

### Learning Rate Scheduler
Uses `ReduceLROnPlateau` (not constant or decay):
- Monitors validation ROC-AUC
- Reduces LR by factor 0.5 if no improvement for 3 epochs
- More stable for LSTM than fixed schedules

### Mean Pooling vs. Attention
Uses mean pooling (not attention) to isolate the encoder difference vs. MeanPool baseline. Attention would add a confound.

## Hyperparameter Recommendations

| Parameter | Default | Notes |
|-----------|---------|-------|
| `bilstm_d_model` | 64 | Output width (hidden_size = 32 for bidirectional) |
| `bilstm_num_layers` | 2 | Stacked LSTM layers; 3+ may overfit |
| `bilstm_dropout` | 0.1 | Dropout between layers |
| `num_epochs` | 50 | BiLSTM may converge faster than Transformers |
| `batch_size` | 16 | LSTM is less parallelizable; avoid very large batches |
| `learning_rate` | 1e-3 | BiLSTM typically uses higher LR than Mamba |
| `plateau_patience` | 3 | Patience for LR scheduler before reducing |
| `patience` | 5 | Early stopping patience (epochs without ROC improvement) |

## Metrics

All seven metrics are computed on validation set per fold:

- **Accuracy**: (TP + TN) / N
- **Precision**: TP / (TP + FP)
- **Recall**: TP / (TP + FN) — sensitivity
- **Specificity**: TN / (TN + FP)
- **MCC**: Matthews Correlation Coefficient — balanced score
- **ROC-AUC**: Area under ROC curve
- **PR-AUC**: Area under precision-recall curve

## Reproducibility

Training is fully reproducible:
- Random seed fixed at 42
- Model checkpoints saved for each fold
- Config files saved alongside results
- Weight initialization deterministic (Xavier/Orthogonal)

## Common Issues

### NaN Loss
- Check for missing values in protein/drug data
- Verify normalization of protein features
- Reduce learning rate (try 5e-4)
- Reduce batch size (LSTMs are sensitive to batch norm)

### Slow Training
- BiLSTM is slower than Mamba; expected
- Use smaller `d_model` (e.g., 32) if speed critical
- Reduce `num_layers` to 1 if convergence is fast

### Poor Validation Performance
- Increase plateau_patience for LR scheduler (allow more exploration)
- Tune `bilstm_dropout` (0.05 for underfitting, 0.15 for overfitting)
- Increase `patience` for early stopping (allow more epochs)

## Comparison to Mamba

**BiLSTM:**
- ✓ Faster training per epoch (no S4 layers)
- ✓ Well-understood recurrent inductive bias
- ✓ Standard LSTM ops (less special-case code)
- ✗ Vanishing gradients on very long sequences
- ✗ Less efficient on length (O(n) recurrence)

**Mamba:**
- ✓ Handles long sequences efficiently (O(n) scan via S4)
- ✓ No vanishing gradients
- ✓ Modern SSM inductive bias
- ✗ Slower per-step (more complex LSTM alternative)
- ✗ Newer, less battle-tested

For DTI (sequence length 220): BiLSTM should be competitive or faster.
"""
