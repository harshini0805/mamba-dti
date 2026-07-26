import pandas as pd
train_smiles = pd.read_csv("data/raw/human_random/train/smiles.csv")

print(train_smiles.iloc[0])