# Mamba+MLP DTI Architecture

## Overview

Binary drug-target interaction (DTI) classifier combining:
- **Protein Encoder**: Mamba-based sequence model over PsePSSM features
- **Drug Encoder**: MLP over Morgan fingerprints
- **Decoder**: MLP over concatenated embeddings → binary prediction

## Architecture Details

### Protein Encoder (Mamba)
- Input: (B, 220) PsePSSM pseudo-position-specific scoring matrix
- Projection: 220 → d_model (128)
- Mamba Block: d_state=16, d_conv=4, expand=2
- Pooling: Mean over sequence dimension → (B, 128)
- **Rationale**: PsePSSM is sequence-like; Mamba captures positional dependencies

### Drug Encoder (MLP)
- Input: (B, 2048) Morgan fingerprints (fixed-length bit-vectors)
- Layer 1: 2048 → 256 (ReLU, Dropout 0.3)
- Layer 2: 256 → 128
- Output: (B, 128) dense embedding
- **Rationale**: Fixed-length vectors don't need sequence models; MLP learns bit co-occurrence patterns

### Decoder (MLP)
- Input: (B, 256) concatenated embeddings
- Layer 1: 256 → 256 (ReLU, Dropout 0.3)
- Layer 2: 256 → 128 (ReLU, Dropout 0.3)
- Layer 3: 128 → 1
- Output: (B,) binary logits (pre-sigmoid)
- Loss: BCEWithLogitsLoss

## Files

- **`model.py`**: Architecture definitions (ProteinEncoder, DrugEncoder, MambaMLPDTI)
- **`config.py`**: Hyperparameters and model architecture settings
- **`train.py`**: Training loop with 5-fold cross-validation
- **`results/{dataset}/`**: Evaluation metrics per dataset
- **`logs/{dataset}/`**: Training logs per dataset
- **`checkpoints/{dataset}/`**: Model checkpoints per dataset

## Usage

### Training on a Dataset

```bash
# Train on humans dataset with default hyperparameters
cd mamba-dti
python architectures/mamba_mlp_fp/train.py --dataset humans

# Override hyperparameters
python architectures/mamba_mlp_fp/train.py --dataset bindingdb --epochs 100 --batch_size 32 --lr 1e-4
```

### Adding a New Dataset

1. Create `datasets/{dataset_name}.py` with a config class:

```python
from dataclasses import dataclass
import numpy as np
import pandas as pd

@dataclass
class MyDatasetConfig:
    dataset_name: str = "my_dataset"
    drug_input_dim: int = 2048
    
    def load_data(self):
        # Load and return (protein_features, drug_embeddings, interactions)
        ...

config = MyDatasetConfig()
```

2. Run training:
```bash
python architectures/mamba_mlp_fp/train.py --dataset my_dataset
```

## Key Design Decisions

### Why Mamba for Proteins?
- PsePSSM is a sequential feature (220 pseudo-positions in a scoring matrix)
- Mamba efficiently models long-range dependencies in amino acid properties
- Alternative: simple linear projection would ignore sequential structure

### Why MLP for Drugs?
- Morgan fingerprints are already dense, fixed-length vectors
- No sequential structure to model
- MLP's ability to learn feature interactions is sufficient
- More efficient than sequence models for fixed-length inputs

### Mean Pooling vs. Last Token
- Mean pooling over Mamba outputs provides a more stable aggregate representation
- Accounts for importance across all positions, not just final state

### Binary Cross-Entropy with Logits
- Raw logits from decoder → BCEWithLogitsLoss handles sigmoid + BCE
- Numerically stable and efficient
- Output interpretation: logit > 0 → predicted positive

## Hyperparameter Recommendations

| Parameter | Default | Notes |
|-----------|---------|-------|
| `num_epochs` | 50 | Increase if validation hasn't plateaued |
| `batch_size` | 16 | Increase for larger datasets; decrease if OOM |
| `learning_rate` | 3e-4 | Reduce if loss diverges; increase if training stalls |
| `weight_decay` | 1e-4 | L2 regularization; tune for overfitting |
| `patience` | 5 | Early stopping patience (epochs without improvement) |
| `dropout` | 0.3 | Increase for overfitting; decrease for underfitting |

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
- Random seed fixed at 42 (configurable)
- Model checkpoints saved for each fold
- Config files saved alongside results
- All hyperparameters logged

## Common Issues

### NaN Loss
- Check for missing values in protein/drug data
- Verify normalization of protein features
- Reduce learning rate

### Low Validation Performance
- Increase patience for more epochs
- Tune hyperparameters (LR, dropout, weight_decay)
- Check data quality and label distribution

### OOM Errors
- Reduce batch_size
- Reduce num_epochs per checkpoint save
