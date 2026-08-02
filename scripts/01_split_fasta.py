import sys
from pathlib import Path

import pandas as pd

# Make the project root importable regardless of cwd, so `config` (which
# derives every path from config.DATASET) resolves the same way it does
# for train.py etc.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import LOOKUP_FILE, FASTA_DIR, DATASET

OUTPUT_DIR = FASTA_DIR

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"[DATASET={DATASET}] reading {LOOKUP_FILE} -> writing FASTA files to {OUTPUT_DIR}")

protein_df = pd.read_csv(LOOKUP_FILE)

for _, row in protein_df.iterrows():

    protein_id = row["protein_id"]
    sequence = row["sequence"]

    fasta_file = OUTPUT_DIR / f"protein_{protein_id}.fasta"

    with open(fasta_file, "w") as f:
        f.write(f">protein_{protein_id}\n")
        f.write(sequence + "\n")

print(f"Generated {len(protein_df)} FASTA files.")