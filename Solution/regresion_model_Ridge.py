import pandas as pd
from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error

import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
SPLIT_DIR = BASE_DIR / "Processed_data/splits_base_price"


train_df = pd.read_csv(SPLIT_DIR / "train.csv")
val_df = pd.read_csv(SPLIT_DIR / "val.csv")
test_df = pd.read_csv(SPLIT_DIR / "test.csv")

TARGET = "base_price"


FEATURES = [
    "room_type",
    "property_type",
    "accommodates",
    "bedrooms",
    "beds",
    "bathrooms",
    "city",
    "neighbourhood_cleansed",
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
    "review_scores_rating",
    "review_scores_accuracy",
    "review_scores_cleanliness",
    "review_scores_checkin",
    "review_scores_communication",
    "review_scores_location",
    "review_scores_value",
]

# Keep only columns that exist
FEATURES = [c for c in FEATURES if c in train_df.columns]

X_train = train_df[FEATURES].copy()
y_train = train_df[TARGET].astype(float)

X_val = val_df[FEATURES].copy()
y_val = val_df[TARGET].astype(float)

X_test = test_df[FEATURES].copy()
y_test = test_df[TARGET].astype(float)

# Outlier handling based on TRAIN only
p1 = y_train.quantile(0.01)
p99 = y_train.quantile(0.99)

train_mask = (y_train >= p1) & (y_train <= p99)
X_train = X_train.loc[train_mask]
y_train = y_train.loc[train_mask]


y_val = y_val.clip(lower=p1, upper=p99)
y_test = y_test.clip(lower=p1, upper=p99)

# log(target) to correct price
y_train_log = np.log1p(y_train)


categorical_features = [c for c in ["room_type", "property_type", "city",
                                    "neighbourhood_cleansed"] if c in FEATURES]
numerical_features = [c for c in FEATURES if c not in categorical_features]

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

numerical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", categorical_transformer, categorical_features),
        ("num", numerical_transformer, numerical_features)
    ],
    remainder="drop"
)

model = Ridge(alpha=1.0, random_state=42)

pipeline = Pipeline(steps=[
    ("preprocess", preprocessor),
    ("model", model)
])

pipeline.fit(X_train, y_train_log)


def smape(y_true, y_pred):
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    denom = np.where(denom == 0, np.nan, denom)
    return np.nanmean(np.abs(y_true - y_pred) / denom) * 100


def evaluate(name, X, y_true):
    preds_log = pipeline.predict(X)
    preds = np.expm1(preds_log)

    preds = np.clip(preds, p1, p99)

    mae = mean_absolute_error(y_true, preds)

    # Safe MAPE
    y_safe = (y_true.replace(0, np.nan) if hasattr(y_true, "replace")
              else y_true)
    mape = np.nanmean(np.abs((y_safe - preds) / y_safe)) * 100

    smape_val = smape(y_true.to_numpy() if hasattr(y_true, "to_numpy") else
                      y_true, preds)

    print(f"{name}  MAE:{mae:.2f} | MAPE:{mape:.2f}% | SMAPE:{smape_val:.2f}%")


print("RIDGE BASELINE: BASE_PRICE PER LISTING")
evaluate("TRAIN", X_train, y_train)
evaluate("VAL", X_val, y_val)
evaluate("TEST", X_test, y_test)
