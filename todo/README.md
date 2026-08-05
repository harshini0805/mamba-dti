# Adding and Training on New Datasets (With Precomputed Features)

This guide explains how to properly run the pipeline on the raw datasets in this `todo` folder (`DrugBank_COMPLETE_FIXED.csv`, `IC_COMPLETE.csv`, `protein_smiles_enzyme_filled_94009.csv`) **assuming you already have the precomputed PSSM and Morgan fingerprint files.**

Because you already have the features, you **skip** steps 01, 02, and 03. However, you still must run steps 00 and 04 to properly map and format those features for the training scripts.

---

### Step 1: Prepare the Datasets (Run Once)
First, you need to split the raw CSVs from this folder into train/valid/test directories. 
From the root of the project, run:
```bash
python scripts/prep_todo_datasets.py
```
*(This automatically creates `data/raw/drugbank`, `data/raw/ic_complete`, and `data/raw/protein_smiles` with the correct ID mappings.)*

---

### Step 2: Set Your Dataset Context
You need to tell the configuration which dataset you are actively working on. 

**Windows (PowerShell):**
```powershell
$env:DTI_DATASET="drugbank"
```
**Mac / Linux:**
```bash
export DTI_DATASET="drugbank"
```

---

### Step 3: Build the Protein Lookup
The pipeline needs to assign a unique integer ID (0, 1, 2...) to every unique protein sequence.
```bash
python scripts/00_build_protein_lookup.py
```

---

### Step 4: Drop In Your Precomputed Features
Now that the lookup is built, you can drop your precomputed files into the correct locations:

**1. PSSM Files**
Drop your `.pssm` files into `data/pssm/drugbank/` (or whichever dataset you are on).
> ⚠️ **CRITICAL:** The `.pssm` files must be named using the integer IDs generated in Step 3 (e.g., `0.pssm`, `1.pssm`, `2.pssm`), NOT their Uniprot IDs.

**2. Morgan Fingerprints**
Drop your Morgan files into `data/features/drugbank/`.
> ⚠️ **CRITICAL:** You must include **both** `morgan_fingerprints.npz` AND `morgan_manifest.csv` for the data loader to work.

---

### Step 5: Convert PSSM to PsePSSM
The models do not read raw `.pssm` files. They read a specific `.csv` generated from them. You must run this script to convert your dropped-in PSSM files into the format the model expects:
```bash
python scripts/04_generate_psepssm.py
```

---

### Step 6: Train!
You are now ready to train any of the architectures on your dataset!

```bash
python architectures/mamba_mlp_fp/train.py --dataset drugbank
python architectures/mamba_cross_mlp/train.py --dataset drugbank
python architectures/mamba_attentionpool_mlp/train.py --dataset drugbank
python architectures/bilstm_mlp/train.py --dataset drugbank
python architectures/meanpool_mlp/train.py --dataset drugbank
```

---
*To run the process on the other datasets (`ic_complete` or `protein_smiles`), simply go back to **Step 2**, change the environment variable to the new dataset name, and repeat steps 3 through 6!*
