import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_PATH = BASE_DIR / "Processed_data/train_base_price_dataset.csv"

df = pd.read_csv(DATA_PATH)

TARGET = "base_price"


df = df.dropna(subset=[TARGET]).reset_index(drop=True)


train_df, temp_df = train_test_split(
    df,
    test_size=0.20,
    random_state=42,
    shuffle=True
)

val_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    random_state=42,
    shuffle=True
)


out_dir = BASE_DIR / "Processed_data/splits_base_price"
out_dir.mkdir(parents=True, exist_ok=True)

train_df.to_csv(out_dir / "train.csv", index=False)
val_df.to_csv(out_dir / "val.csv", index=False)
test_df.to_csv(out_dir / "test.csv", index=False)

print(f"Saved splits to: {out_dir}")
print(f"train: {train_df.shape}")
print(f"val:   {val_df.shape}")
print(f"test:  {test_df.shape}")
