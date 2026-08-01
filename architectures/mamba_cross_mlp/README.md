# Mamba+CrossAttention+MLP DTI Architecture

## Overview

Binary drug-target interaction (DTI) classifier using bidirectional cross-attention to fuse protein and drug representations:
- **Protein Encoder**: Mamba SSM → (B, 220, 128)
- **Drug Encoder**: MLP → unsqueeze → (B, 1, 128)
- **Cross-Attention** (bidirectional): Drug queries Protein, Protein queries Drug
- **Pooling**: Mean for protein, squeeze for drug
- **Decoder**: MLP → binary prediction

**Key constraint**: Mamba is on protein side ONLY ✓

## Architecture Details

### ProteinEncoder (Mamba-based)
- Input: (B, 220) PsePSSM features
- Projection: 220 → d_model (128)
- Mamba SSM: d_state=16, d_conv=4, expand=2
- LayerNorm: Stabilize output scale
- Output: (B, 220, 128) full sequence (no pooling yet)

### DrugEncoder (MLP-based, NO Mamba)
- Input: (B, 2048) Morgan fingerprints
- Layer 1: 2048 → 256 (ReLU, Dropout 0.3)
- Layer 2: 256 → 128
- Unsqueeze to: (B, 1, 128) single-position sequence for cross-attention

### Cross-Attention (Bidirectional)
Two MultiheadAttention layers (4 heads each, embed_dim=128):

**cross_dp**: Drug (Q) attends to Protein (K, V)
- Input: Drug (B, 1, 128) queries, Protein (B, 220, 128) keys/values
- Output: (B, 1, 128) attended drug representation
- No padding masks (fixed-length protein sequence)

**cross_pd**: Protein (Q) attends to Drug (K, V)
- Input: Protein (B, 220, 128) queries, Drug (B, 1, 128) keys/values
- Output: (B, 220, 128) attended protein representation
- No padding masks (single-token drug sequence)

### Post-Residual Connections
After cross-attention, apply residual + post-LayerNorm:
```
D = LayerNorm(D + D_attn)  # (B, 1, 128)
P = LayerNorm(P + P_attn)  # (B, 220, 128)
```
Post-residual LN is numerically stable and matches Transformer paper convention.

### Pooling
- Protein: Mean pool over 220 positions → (B, 128)
- Drug: Squeeze dimension 1 → (B, 128)

### Decoder (MLP)
- Input: (B, 256) concatenated embeddings
- Layer 1: 256 → 256 (ReLU, Dropout 0.3)
- Layer 2: 256 → 128 (ReLU, Dropout 0.3)
- Layer 3: 128 → 1
- Output: (B,) binary logits (pre-sigmoid)
- Loss: BCEWithLogitsLoss

## Files

- **`model.py`**: ProteinEncoder, DrugEncoder, MambaCrossAttentionDTI
- **`config.py`**: Hyperparameters and architecture settings
- **`train.py`**: Training loop with 5-seed evaluation
- **`results/{dataset}/`**: Evaluation metrics per dataset
- **`logs/{dataset}/`**: Training logs per dataset
- **`checkpoints/{dataset}/`**: Model checkpoints per dataset

## Usage

### Training on a Dataset

```bash
cd mamba-dti
python architectures/mamba_cross_mlp/train.py --dataset humans

# Override hyperparameters
python architectures/mamba_cross_mlp/train.py --dataset bindingdb --epochs 100 --lr 1e-4
```

## Key Design Decisions

### Bidirectional Cross-Attention
- **Asymmetry**: Drug attends to all 220 protein positions; Protein attends to single drug token
- **Why bidirectional**: Information flow in both directions allows interaction discovery
- **No padding masks**: Fixed-length sequences (220 positions, 1 token) need no masking

### Mamba on Protein Only (✓)
- PsePSSM is sequence-like (220 pseudo-positions)
- Drug fingerprints are fixed-length dense vectors
- Only protein benefits from sequence modeling

### Unsqueeze Drug to (B, 1, 128)
- Creates a degenerate "sequence" of length T=1
- Allows standard (B, T, dim) input to cross-attention layers
- Protein sees drug as a single representative token

### Post-Residual LayerNorm
- Formula: `output = LayerNorm(residual + attention_output)`
- Matches Transformer paper convention
- Numerically stable when attention outputs are well-scaled

### Mamba Output as Full Sequence
- Unlike mean-pool architectures, full sequence (B, 220, 128) is passed to cross-attention
- Allows protein to attend to ALL positions when drug queries it
- Only mean-pooled AFTER cross-attention for final representation

## Hyperparameter Recommendations

| Parameter | Default | Notes |
|-----------|---------|-------|
| `mamba_d_model` | 128 | Protein sequence encoding dimension |
| `cross_attn_num_heads` | 4 | MultiheadAttention heads |
| `cross_attn_embed_dim` | 128 | Attention embedding dimension |
| `num_epochs` | 50 | Typical convergence |
| `batch_size` | 16 | Fixed-length inputs allow larger batches |
| `learning_rate` | 3e-4 | Standard for Mamba models |
| `dropout` | 0.3 | Regularization |
| `patience` | 5 | Early stopping |

## Metrics

All seven metrics computed on validation set per run:

- **Accuracy**: (TP + TN) / N
- **Precision**: TP / (TP + FP)
- **Recall**: TP / (TP + FN)
- **Specificity**: TN / (TN + FP)
- **MCC**: Matthews Correlation Coefficient
- **ROC-AUC**: Area under ROC curve
- **PR-AUC**: Area under precision-recall curve

## Ablation Context

This architecture is part of a **fusion mechanism ablation**:

| Model | Protein Enc | Fusion | Interaction |
|-------|-------------|--------|-------------|
| MeanPool | None | Concat | Direct |
| MambaPool | Mamba | Concat | No interaction |
| MambaCross (this) | Mamba | CrossAttn | Bidirectional |
| MambaCaps | Mamba | Routing | Complex routing |

**Interpretation**:
- If MambaCross > MambaPool: Cross-attention helps
- If MambaCaps > MambaCross: Routing justifies complexity

## Reproducibility

Training is fully reproducible:
- Random seeds: 42, 123, 2024, 456, 789
- Model checkpoints saved per seed
- Config saved with results
- Deterministic initialization

## Common Issues

### NaN Loss
- Check protein feature normalization (z-score)
- Verify drug fingerprints in [0, 1]
- Reduce learning rate if gradients explode
- Cross-attention should be stable with fixed-length sequences

### High Variance Across Seeds
- Increase patience for more epochs
- Adjust dropout (reduce if underfitting, increase if overfitting)
- Try smaller learning rate

### Memory Issues
- Reduce batch_size
- Reduce mamba_d_model
- Cross-attention has quadratic complexity in sequence length

## References

Cross-Attention Mechanisms:
- Standard in Vision Transformers (ViT)
- Used for fusing different modalities
- Allows asymmetric attention patterns

Mamba SSM:
- Efficient sequence modeling
- Linear complexity in sequence length
- Alternative to Transformers
