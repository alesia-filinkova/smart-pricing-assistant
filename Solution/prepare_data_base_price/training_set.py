import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "IUM25L_Zad_02_01_v2"

calendar = pd.read_csv(DATA_DIR / "calendar.csv")
listings = pd.read_csv(DATA_DIR / "listings.csv")
reviews = pd.read_csv(DATA_DIR / "reviews.csv")


calendar["date"] = pd.to_datetime(calendar["date"])

calendar["price"] = (
    calendar["price"]
    .astype(str)
    .str.replace("$", "", regex=False)
    .str.replace(",", "", regex=False)
)
calendar["price"] = pd.to_numeric(calendar["price"], errors="coerce")


base_price = (
    calendar.dropna(subset=["price"])
    .groupby("listing_id")
    .agg(base_price=("price", "median"))
    .reset_index()
)


listing_location_candidates = [
    "neighbourhood_cleansed",
    "neighbourhood",
    "neighborhood",
    "neighborhood_overview",
]
listing_loc_col = next((c for c in listing_location_candidates if c
                        in listings.columns), None)

if listing_loc_col is None:
    listings["listing_location"] = "unknown"
else:
    listings["listing_location"] = (listings[listing_loc_col].astype(str)
                                    .fillna("unknown"))


if "host_location" in listings.columns:
    listings["host_location"] = (listings["host_location"].astype(str)
                                 .fillna("unknown"))
else:
    listings["host_location"] = "unknown"


review_score_cols = [
    "review_scores_rating",
    "review_scores_accuracy",
    "review_scores_cleanliness",
    "review_scores_checkin",
    "review_scores_communication",
    "review_scores_location",
    "review_scores_value",
]
existing_review_score_cols = [c for c in review_score_cols if c
                              in listings.columns]

listings["amenities_count"] = (
    listings["amenities"].astype(str).str.count(",").fillna(0).astype(int) + 1
)

listings["host_is_superhost"] = (
    listings["host_is_superhost"]
    .astype(str)
    .str.strip()
    .str.lower()
    .map({"t": 1, "f": 0, "true": 1, "false": 0})
)

base_listing_cols = [
    "id",
    "room_type",
    "property_type",
    "accommodates",
    "bedrooms",
    "beds",
    "bathrooms",
    "listing_location",
    "host_location",
    "latitude",
    "longitude",
    "minimum_nights",
    "maximum_nights",
    "host_is_superhost",
    "amenities_count",
    "number_of_reviews",
    "reviews_per_month",
]

base_listing_cols = [c for c in base_listing_cols if c in listings.columns]

listing_features = listings[base_listing_cols +
                            existing_review_score_cols].rename(
    columns={"id": "listing_id"}
)

reviews_agg = (
    reviews.groupby("listing_id")
    .agg(
        avg_rating=("numerical_review", "mean"),
        review_count=("numerical_review", "count"),
    )
    .reset_index()
)

# Joins
df = base_price.merge(listing_features, on="listing_id", how="left")
df = df.merge(reviews_agg, on="listing_id", how="left")

# Final dataset
TARGET = "base_price"

FEATURES = [
    "room_type",
    "property_type",
    "accommodates",
    "bedrooms",
    "beds",
    "bathrooms",
    "listing_location",
    "host_location",
    "latitude",
    "longitude",
    "minimum_nights",
    "maximum_nights",
    "host_is_superhost",
    "amenities_count",
    "number_of_reviews",
    "reviews_per_month",
    "avg_rating",
    "review_count",
] + existing_review_score_cols


FEATURES = [c for c in FEATURES if c in df.columns]

train_df = df[["listing_id", TARGET] + FEATURES]

out_path = BASE_DIR / "Processed_data/train_base_price_dataset.csv"
out_path.parent.mkdir(parents=True, exist_ok=True)
train_df.to_csv(out_path, index=False)

print(f"Saved: {out_path} | rows={len(train_df)} cols={len(train_df.columns)}")
print(f"Listing location column used: {listing_loc_col}")
print(f"Added review score cols: {existing_review_score_cols}")
