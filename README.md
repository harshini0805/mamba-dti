# Quick Start Guide

## Project Structure

```
mamba-dti/
├── architectures/           ← Each architecture has own directory
│   ├── mamba_mlp_fp/        ← First architecture (complete example)
│   ├── attention_mlp/       ← Add next architecture here
│   └── [8 more architectures...]
├── datasets/                ← Dataset configs (shared across architectures)
│   ├── humans.py            ← Load human dataset
│   ├── bindingdb.py         ← Add BindingDB config
│   ├── biosnap.py           ← Add BioSNAP config
│   └── celegans.py          ← Add C. elegans config
├── common/                  ← Shared code (dataset loader, metrics)
│   ├── dataset_loader.py
│   ├── metrics.py
│   └── __init__.py
├── data/                    ← Raw and processed data
│   ├── raw/human_random/
│   └── processed/human_random/
└── scripts/                 ← Data preprocessing scripts
```

## Train Mamba+MLP (Example)

```bash
# Navigate to project root
cd mamba-dti

# Train on humans dataset
python architectures/mamba_mlp_fp/train.py --dataset humans

# Train on other datasets (once configs created)
python architectures/mamba_mlp_fp/train.py --dataset bindingdb
python architectures/mamba_mlp_fp/train.py --dataset biosnap
python architectures/mamba_mlp_fp/train.py --dataset celegans

# With custom hyperparameters
python architectures/mamba_mlp_fp/train.py \
  --dataset humans \
  --epochs 100 \
  --batch_size 32 \
  --lr 1e-4
```

## Add a New Architecture

### Step 1: Create directory
```bash
mkdir -p architectures/your_arch_name
```

### Step 2: Create three files

**`architectures/your_arch_name/model.py`**
```python
import torch.nn as nn
from mamba_ssm import Mamba  # or other imports

class YourProteinEncoder(nn.Module):
    def __init__(self, ...):
        super().__init__()
        # Define layers

    def forward(self, x):
        # Implement forward pass
        return x

class YourDrugEncoder(nn.Module):
    # Similar structure
    pass

class YourDTI(nn.Module):
    """Main model combining protein & drug encoders."""
    def __init__(self, ...):
        super().__init__()
        self.protein_encoder = YourProteinEncoder(...)
        self.drug_encoder = YourDrugEncoder(...)
        self.decoder = nn.Sequential(...)
    
    def forward(self, protein, drug):
        p_vec = self.protein_encoder(protein)  # (B, 128)
        d_vec = self.drug_encoder(drug)         # (B, 128)
        x = torch.cat([p_vec, d_vec], dim=-1)   # (B, 256)
        return self.decoder(x).squeeze(-1)      # (B,)
```

**`architectures/your_arch_name/config.py`**
```python
from dataclasses import dataclass

@dataclass
class YourArchConfig:
    # Model architecture
    protein_embedding_dim: int = 128
    drug_embedding_dim: int = 128
    drug_input_dim: int = 2048
    
    # Training
    num_epochs: int = 50
    batch_size: int = 16
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    patience: int = 5
    
    # Etc.
```

**`architectures/your_arch_name/train.py`**

Copy from `architectures/mamba_mlp_fp/train.py` and change these two imports:
```python
from model import YourDTI  # Change this line
from config import default_config as arch_config
```

The rest stays the same!

### Step 3: Create results/logs/checkpoints directories
```bash
mkdir -p architectures/your_arch_name/{results,logs,checkpoints}
```

### Step 4: Train
```bash
python architectures/your_arch_name/train.py --dataset humans
```

## Add a New Dataset

### Step 1: Create config file

**`datasets/your_dataset.py`**
```python
from dataclasses import dataclass
import pandas as pd
import numpy as np

@dataclass
class YourDatasetConfig:
    dataset_name: str = "your_dataset"
    drug_input_dim: int = 2048  # Update if different
    
    protein_csv: str = "path/to/protein_features.csv"
    drug_npz: str = "path/to/drug_fingerprints.npz"
    interactions_csv: str = "path/to/interactions.csv"
    
    def load_data(self):
        """Must return (protein_features, drug_embeddings, interactions)"""
        # Load protein features
        protein_features = {...}  # dict[protein_id → (220,) array]
        
        # Load drug embeddings
        drug_embeddings = {...}  # dict[drug_id → (2048,) array]
        
        # Load interaction pairs
        interactions = pd.DataFrame({
            'protein_id': [...],
            'drug_id': [...],
            'label': [0, 1, ...]
        })
        
        return protein_features, drug_embeddings, interactions

config = YourDatasetConfig()
```

### Step 2: Train any architecture on this dataset
```bash
python architectures/mamba_mlp_fp/train.py --dataset your_dataset
python architectures/attention_mlp/train.py --dataset your_dataset
python architectures/[any_arch]/train.py --dataset your_dataset
```

## Output Structure

After training on `humans` with `mamba_mlp_fp`:

```
architectures/mamba_mlp_fp/
├── results/humans/
│   └── [results will be saved here]
├── logs/humans/
│   └── [training logs]
└── checkpoints/humans/
    ├── best_model_fold_1.pt
    ├── best_model_fold_2.pt
    ├── best_model_fold_3.pt
    ├── best_model_fold_4.pt
    └── best_model_fold_5.pt
```

## Running Multiple Experiments

Train all 10 architectures on all 4 datasets:

```bash
for arch in mamba_mlp_fp attention_mlp mamba_attention_capsule_mlp ...; do
  for dataset in humans bindingdb biosnap celegans; do
    echo "Training $arch on $dataset..."
    python architectures/$arch/train.py --dataset $dataset
  done
done
```

## Key Files to Understand

1. **`architectures/{arch}/model.py`** — Your architecture implementation
2. **`architectures/{arch}/config.py`** — Hyperparameters
3. **`architectures/{arch}/train.py`** — Training loop (mostly generic, reused)
4. **`datasets/{dataset}.py`** — Data loading logic
5. **`common/dataset_loader.py`** — Shared DTIDataset class
6. **`common/metrics.py`** — Shared metric computation

All other files can be left as-is!

## Troubleshooting

### ImportError: cannot import name 'Mamba'
```bash
pip install mamba-ssm
```

### ModuleNotFoundError: No module named 'datasets'
```bash
# Make sure you're in project root and Python path is correct
cd mamba-dti
python architectures/mamba_mlp_fp/train.py --dataset humans
```

### FileNotFoundError: data/processed/...
Check that data files exist and paths in `datasets/{dataset}.py` are correct.

### NaN loss detected
- Check for missing values in protein/drug data
- Verify protein features are normalized
- Reduce learning rate
