# Mamba+AttentionPool+MLP DTI Architecture

## Overview

Binary drug-target interaction (DTI) classifier combining:
- **Protein Encoder**: Mamba SSM → AttentionPool → (B, 128)
- **Drug Encoder**: MLP → AttentionPool → (B, 128)
- **Decoder**: MLP over concatenated embeddings → binary prediction

**Key constraint**: Mamba is on protein side ONLY ✓

## Architecture Details

### AttentionPool (Learned Pooling)
- Replaces mean pooling: **learned** weighted average instead of uniform
- Mechanism:
  1. Linear layer scores each position → (B, L, 1) scores
  2. Softmax over positions → (B, L, 1) attention weights
  3. Weighted sum across L positions → (B, in_dim)
  4. Linear projection → (B, embed_dim)
  5. LayerNorm → (B, embed_dim)
- NaN safety: Masking uses -1e9 instead of -inf (prevents NaN in softmax)
- Small gain=0.1 on score layer keeps attention uniform initially

### Protein Encoder (Mamba-based)
- Input: (B, 220) PsePSSM features
- Projection: 220 → d_model (64)
- Pre-norm: LayerNorm
- Mamba Block: d_state=16, d_conv=4, expand=2
- Post-norm: LayerNorm (stabilizes output scale)
- Output: (B, 220, d_model)

### Protein Pooling (AttentionPool)
- Input: (B, 220, d_model) Mamba output
- Learned weighted average over 220 positions
- Output: (B, 128)

### Drug Encoder (MLP-based, NO Mamba)
- Input: (B, 2048) Morgan fingerprints
- Layer 1: 2048 → 256 (ReLU, Dropout 0.3)
- Layer 2: 256 → 128
- Output: (B, 128)
- Unsqueezed to: (B, 1, 128) for AttentionPool compatibility

### Drug Pooling (AttentionPool)
- Input: (B, 1, 128) single-position sequence
- Learned attention (trivial for L=1, but consistent interface)
- Output: (B, 128)

### Decoder (MLP)
- Input: (B, 256) concatenated embeddings
- Layer 1: 256 → 128 (ReLU, Dropout 0.3)
- Layer 2: 128 → 1
- Output: (B,) binary logits (pre-sigmoid)
- Loss: BCEWithLogitsLoss

## Files

- **`model.py`**: AttentionPool, ProteinEncoder, DrugEncoder, MambaAttnDTI
- **`config.py`**: Hyperparameters and architecture settings
- **`train.py`**: Training loop with 5-seed evaluation
- **`results/{dataset}/`**: Evaluation metrics per dataset
- **`logs/{dataset}/`**: Training logs per dataset
- **`checkpoints/{dataset}/`**: Model checkpoints per dataset

## Usage

### Training on a Dataset

```bash
cd mamba-dti
python architectures/mamba_attentionpool_mlp/train.py --dataset humans

# Override hyperparameters
python architectures/mamba_attentionpool_mlp/train.py --dataset bindingdb --epochs 100 --lr 1e-4
```

## Key Design Decisions

### Why AttentionPool?
Replaces simple mean pooling with **learned attention weights**:
- Allows model to focus on important positions
- Soft attention: differentiable, no hard selection
- Comparison point: if capsule routing outperforms this, routing's complexity is justified

### Mamba on Protein Only (✓)
- PsePSSM is sequence-like (220 pseudo-positions)
- Morgan fingerprints are fixed-length vectors (no sequence structure)
- Drug uses simple MLP projection + attention (not SSM)

### Drug Unsqueeze to (B, 1, embed_dim)
- Creates a "sequence of length 1"
- Allows AttentionPool to accept it in (B, L, dim) format
- For L=1, attention weights trivially go to that single position
- Keeps interface consistent between protein and drug branches

### NaN Safety in AttentionPool
- Masking uses -1e9 instead of -inf
- Softmax on -1e9 gives ~0 weight (numerically stable)
- If all positions masked: softmax(all -1e9) → uniform small values (not NaN)

### Weight Initialization
- Linear layers: Xavier Uniform (gain=0.5)
- Score layer in AttentionPool: Xavier Uniform (gain=0.1) — keeps attention uniform initially
- Projection in AttentionPool: Xavier Uniform (gain=1.0)
- Biases: Zeros

## Hyperparameter Recommendations

| Parameter | Default | Notes |
|-----------|---------|-------|
| `mamba_d_model` | 64 | Protein sequence encoding dimension |
| `attn_pool_init_gain` | 0.1 | Small gain keeps attention uniform initially |
| `num_epochs` | 50 | Typical convergence point |
| `batch_size` | 16 | Fixed-length inputs allow larger batches |
| `learning_rate` | 3e-4 | Standard for Mamba models |
| `dropout` | 0.3 | Regularization |
| `patience` | 5 | Early stopping patience |

## Metrics

All seven metrics computed on validation set per run:

- **Accuracy**: (TP + TN) / N
- **Precision**: TP / (TP + FP)
- **Recall**: TP / (TP + FN) — sensitivity
- **Specificity**: TN / (TN + FP)
- **MCC**: Matthews Correlation Coefficient
- **ROC-AUC**: Area under ROC curve
- **PR-AUC**: Area under precision-recall curve

## Ablation Context

This architecture sits in a **pooling mechanism ablation**:

| Model | Protein Encoder | Protein Pooling | Protein Attention | Purpose |
|-------|-----------------|-----------------|-------------------|---------|
| MeanPool | None | Mean pooling | None | Baseline: simple averaging |
| MambaPool | Mamba | Mean pooling | None | Seq modeling + simple pooling |
| MambaAttn (this) | Mamba | AttentionPool | Learned weights | Seq modeling + learned pooling |
| MambaCaps | Mamba | CapsuleRoute | Routing | Seq modeling + routing (complex) |

**Interpretation**:
- If MambaAttn > MambaPool: Learned pooling helps
- If MambaCaps > MambaAttn: Routing mechanism justifies complexity

## Reproducibility

Training is fully reproducible:
- Random seeds fixed at 42, 123, 2024, 456, 789
- Model checkpoints saved per seed
- Config saved with results
- All initialization deterministic

## Common Issues

### NaN Loss
- Check protein feature normalization (should be z-scored)
- Verify drug fingerprints are in [0, 1] range
- Reduce learning rate if gradients explode
- Check AttentionPool masking logic (though data shouldn't have masked positions)

### High Variance Across Seeds
- Increase patience for early stopping (allow more training)
- Tune dropout (reduce if underfitting, increase if overfitting)
- Try smaller learning rate for stability

### Memory Issues
- Reduce batch_size
- Reduce mamba_d_model (trades capacity for memory)

## References

AttentionPool is a learned pooling mechanism similar to:
- Self-attention pooling in NLP
- Attention weights over sequence positions
- Soft selection mechanism (vs. hard select or equal weight)

Mamba SSM:
- Efficient sequence modeling with linear complexity
- Alternative to Transformers for long sequences
