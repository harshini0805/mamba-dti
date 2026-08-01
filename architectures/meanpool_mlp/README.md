# MeanPool+MLP DTI Architecture (Baseline)

## Overview

**Baseline classifier with NO sequential modeling whatsoever.**

- **Protein**: Linear proj → LayerNorm → mean pool → proj → LayerNorm → (B, 128)
- **Drug**: MLP encoder → (B, 128)
- **Decoder**: MLP → binary prediction

**Purpose**: Floor baseline for ablation series. If sequential/attention/routing versions don't beat this, their added complexity is unjustified.

## Architecture Details

### ProteinBranch (No Seq Modeling)
- Input: (B, 220) PsePSSM features
- Unsqueeze: (B, 220) → (B, 220, 1)
- Project: (B, 220, 1) → (B, 220, d_model) [each position independently]
- LayerNorm: (B, 220, d_model)
- **Mean Pool**: (B, 220, d_model) → (B, d_model) [ORDER DISCARDED]
- Project: (B, d_model) → (B, embed_dim)
- LayerNorm: Final normalization
- Output: (B, 128)

**Key insight**: Each of 220 positions treated as independent feature; spatial order completely ignored.

### DrugBranch (MLP)
- Input: (B, 2048) Morgan fingerprints
- Layer 1: 2048 → 256 (ReLU, Dropout 0.3)
- Layer 2: 256 → 128
- Output: (B, 128)

### Decoder (MLP)
- Input: (B, 256) concatenated embeddings
- Layer 1: 256 → 128 (ReLU, Dropout 0.3)
- Layer 2: 128 → 1
- Output: (B,) binary logits
- Loss: BCEWithLogitsLoss

## Files

- **`model.py`**: ProteinBranch, DrugBranch, MeanPoolDTI
- **`config.py`**: Hyperparameters
- **`train.py`**: Training loop with 5-seed evaluation
- **`results/{dataset}/`**: Metrics
- **`logs/{dataset}/`**: Logs
- **`checkpoints/{dataset}/`**: Checkpoints

## Usage

```bash
cd mamba-dti
python architectures/meanpool_mlp/train.py --dataset humans
```

## Key Design Decisions

### Matched Architecture Across Variants
To isolate sequence modeling as the only variable in ablation:
- d_model = 64 (same as Mamba versions)
- embed_dim = 128 (same as all versions)
- Same drug_encoder hidden_dim
- Same classifier MLP
- Parameter counts comparable

### Why Mean Pool?
- Simplest aggregation
- No learnable parameters
- Completely discards order
- If seq-based models don't beat this, order doesn't matter for this task

### What Order Discarding Means
These two inputs produce identical output:
```
[1, 2, 3, ..., 220]  →  mean pool  →  avg([1, 2, 3, ..., 220])
[220, 219, ..., 1]   →  mean pool  →  avg([220, 219, ..., 1])
```
Position matters NOT AT ALL. Pure bag-of-features.

## Ablation Series

This is the floor baseline:

```
MeanPool (this)      — no seq modeling, just avg
  ↓
BiLSTM+MLP           — recurrent seq modeling
  ↓
Mamba+MLP            — SSM seq modeling
  ↓
Mamba+AttentionPool  — seq + learned pooling
  ↓
Mamba+CrossAttention — seq + bidirectional fusion
  ↓
Mamba+Capsule (future) — seq + routing (?)
```

**Interpretation**:
- If BiLSTM > MeanPool: Sequence structure matters
- If Mamba > BiLSTM: SSM inductive bias better than recurrence
- If AttentionPool > Mamba: Learned pooling helps
- If CrossAttention > AttentionPool: Fusion mechanism helps
- If Capsule > all: Routing mechanism justified

## Hyperparameters

| Parameter | Default | Notes |
|-----------|---------|-------|
| `protein_d_model` | 64 | Intermediate dimension (same as Mamba) |
| `num_epochs` | 50 | Baseline convergence |
| `batch_size` | 16 | Fixed-length inputs |
| `learning_rate` | 3e-4 | Standard |
| `patience` | 5 | Early stopping |

## Metrics

All seven computed per run:
- Accuracy, Precision, Recall, Specificity, MCC, ROC-AUC, PR-AUC

## Reproducibility

- Seeds: [42, 123, 2024, 456, 789]
- 5 independent runs per dataset
- Same train/val split across all 5 runs
- Deterministic initialization

## Expected Performance

**Baseline expectations:**
- Should be fast (no seq modeling overhead)
- May underperform if sequence order matters
- Provides sanity check: if more complex models don't beat this by significant margin, added complexity is unjustified

## When to Use

- As a **reference point** for ablation studies
- To **validate** that sequence order matters in PsePSSM
- To **benchmark** whether added complexity of other models is worthwhile
- As a **lower bound** on expected model performance

## Interpretation

If MeanPool performs well:
- Sequence order in PsePSSM doesn't matter much
- Simpler models may be preferable
- Complex architectures (Mamba, Capsules) add overhead without benefit

If MeanPool performs poorly:
- Sequence structure is important
- More sophisticated models likely needed
- Justifies adding Mamba, attention, routing mechanisms
