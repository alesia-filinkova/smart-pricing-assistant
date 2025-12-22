import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error

from catboost import CatBoostRegressor
import joblib

BASE_DIR = Path(__file__).resolve().parent.parent.parent
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
    "review_scores_rating",
    "review_scores_accuracy",
    "review_scores_cleanliness",
    "review_scores_checkin",
    "review_scores_communication",
    "review_scores_location",
    "review_scores_value",
]


FEATURES = [c for c in FEATURES if c in train_df.columns]

categorical_features = [
    c for c in ["room_type", "property_type", "listing_location",
                "host_location"]
    if c in FEATURES
]
num_features = [c for c in FEATURES if c not in categorical_features]

X_train = train_df[FEATURES].copy()
y_train = train_df[TARGET].astype(float)

X_val = val_df[FEATURES].copy()
y_val = val_df[TARGET].astype(float)

X_test = test_df[FEATURES].copy()
y_test = test_df[TARGET].astype(float)

# Outlier handling
p1 = y_train.quantile(0.01)
p99 = y_train.quantile(0.99)

mask = (y_train >= p1) & (y_train <= p99)
X_train = X_train.loc[mask]
y_train = y_train.loc[mask]

y_val = y_val.clip(p1, p99)
y_test = y_test.clip(p1, p99)


num_imputer = SimpleImputer(strategy="median")
X_train[num_features] = num_imputer.fit_transform(X_train[num_features])
X_val[num_features] = num_imputer.transform(X_val[num_features])
X_test[num_features] = num_imputer.transform(X_test[num_features])

for c in categorical_features:
    X_train[c] = X_train[c].astype(str).fillna("unknown")
    X_val[c] = X_val[c].astype(str).fillna("unknown")
    X_test[c] = X_test[c].astype(str).fillna("unknown")

cat_feature_indices = [FEATURES.index(c) for c in categorical_features]

USE_LOG_TARGET = False
y_train_fit = np.log1p(y_train) if USE_LOG_TARGET else y_train
y_val_fit = np.log1p(y_val) if USE_LOG_TARGET else y_val

model = CatBoostRegressor(
    loss_function="Quantile:alpha=0.5",
    iterations=2500,
    learning_rate=0.05,
    depth=6,
    l2_leaf_reg=10,
    min_data_in_leaf=20,
    random_seed=42,
    verbose=200,
    od_type="Iter",
    od_wait=150
)

model.fit(
    X_train,
    y_train_fit,
    cat_features=cat_feature_indices,
    eval_set=(X_val, y_val_fit),
    use_best_model=True
)


def evaluate(name, X, y_true):
    preds = model.predict(X)
    if USE_LOG_TARGET:
        preds = np.expm1(preds)

    preds = np.clip(preds, p1, p99)
    mae = mean_absolute_error(y_true, preds)

    y_safe = (y_true.replace(0, np.nan) if hasattr(y_true, "replace")
              else y_true)
    mape = np.nanmean(np.abs((y_safe - preds) / y_safe)) * 100

    print(f"{name} | MAE: {mae:.2f} | MAPE: {mape:.2f}%")


print("CATBOOST BASE_PRICE (clean location fields)")
evaluate("TRAIN", X_train, y_train)
evaluate("VAL", X_val, y_val)
evaluate("TEST", X_test, y_test)

# Save artifacts for serving
out_dir = BASE_DIR / "Processed_data/models"
out_dir.mkdir(parents=True, exist_ok=True)

model.save_model(str(out_dir / "catboost_base_price.cbm"))
joblib.dump(num_imputer, out_dir / "num_imputer.joblib")

meta = {
    "features": FEATURES,
    "categorical_features": categorical_features,
    "cat_feature_indices": cat_feature_indices,
    "p1": float(p1),
    "p99": float(p99),
    "use_log_target": USE_LOG_TARGET
}
joblib.dump(meta, out_dir / "meta.joblib")

print("Saved model + preprocessors to:", out_dir)
