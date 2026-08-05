import pandas as pd
import numpy as np
from pathlib import Path

def main():
    project_root = Path(__file__).resolve().parent.parent
    todo_dir = project_root / "todo"
    raw_dir = project_root / "data" / "raw"

    datasets = {
        "drugbank": "DrugBank_COMPLETE_FIXED.csv",
        "ic_complete": "IC_COMPLETE.csv",
        "protein_smiles": "protein_smiles_enzyme_filled_94009.csv"
    }

    np.random.seed(42)

    for name, filename in datasets.items():
        print(f"\nProcessing {name} ({filename})...")
        file_path = todo_dir / filename
        if not file_path.exists():
            print(f"Warning: {file_path} not found. Skipping.")
            continue
            
        df = pd.read_csv(file_path)
        
        # Ensure we have the required columns
        required_cols = {"label", "sequence", "smiles"}
        if not required_cols.issubset(df.columns):
            print(f"Warning: {filename} is missing required columns. Has {list(df.columns)}. Skipping.")
            continue
            
        # Shuffle
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        n = len(df)
        n_train = int(0.8 * n)
        n_valid = int(0.1 * n)
        
        splits = {
            "train": df.iloc[:n_train],
            "valid": df.iloc[n_train:n_train+n_valid],
            "test": df.iloc[n_train+n_valid:]
        }
        
        dataset_dir = raw_dir / name
        
        for split_name, split_df in splits.items():
            split_dir = dataset_dir / split_name
            split_dir.mkdir(parents=True, exist_ok=True)
            
            # Create unique mappings for this split
            unique_smiles = split_df["smiles"].drop_duplicates().reset_index(drop=True)
            unique_sequence = split_df["sequence"].drop_duplicates().reset_index(drop=True)
            
            smiles_df = pd.DataFrame({"index": range(len(unique_smiles)), "smiles": unique_smiles})
            sequence_df = pd.DataFrame({"index": range(len(unique_sequence)), "sequence": unique_sequence})
            
            smiles_to_idx = {s: i for i, s in enumerate(unique_smiles)}
            seq_to_idx = {s: i for i, s in enumerate(unique_sequence)}
            
            samples_df = pd.DataFrame({
                "smiles": split_df["smiles"].map(smiles_to_idx),
                "sequence": split_df["sequence"].map(seq_to_idx),
                "interactions": split_df["label"]
            })
            
            smiles_df.to_csv(split_dir / "smiles.csv", index=False)
            sequence_df.to_csv(split_dir / "sequence.csv", index=False)
            samples_df.to_csv(split_dir / "samples.csv", index=False)
            
            print(f"  {split_name}: {len(split_df)} samples saved to {split_dir}")

if __name__ == "__main__":
    main()
