import pandas as pd
from pathlib import Path


FILES = [
    "calendar.csv",
    "listings.csv",
    "reviews.csv",
    "sessions.csv",
    "users.csv",
]

base_path = Path(".")

for file_name in FILES:
    file_path = base_path / file_name
    print(f"\n================ {file_name} ================")
    df = pd.read_csv(file_path)

    missing_pct = df.isna().mean() * 100

    table = (
        missing_pct
        .reset_index()
        .rename(columns={"index": "column", 0: "missing_percent"})
    )

    table["missing_percent"] = table["missing_percent"].round(2)

    print(table.to_string(index=False))
