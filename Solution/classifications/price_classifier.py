from pathlib import Path
import pandas as pd
import joblib

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, f1_score

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

BASE_DIR = Path(__file__).resolve().parent.parent.parent
SPLIT_DIR = BASE_DIR / "Processed_data/splits_base_price"
MODEL_DIR = BASE_DIR / "Processed_data/models"
TRAIN_PATH = SPLIT_DIR / "train.csv"
VAL_PATH = SPLIT_DIR / "val.csv"
TEST_PATH = SPLIT_DIR / "test.csv"

train_df = pd.read_csv(TRAIN_PATH)
val_df = pd.read_csv(VAL_PATH)
test_df = pd.read_csv(TEST_PATH)

q_low = train_df["base_price"].quantile(0.25)
q_high = train_df["base_price"].quantile(0.75)

def price_to_class(price):
    if price < q_low:
        return "low"
    elif price > q_high:
        return "high"
    else:
        return "normal"

for df in [train_df, val_df, test_df]:
    df["price_class"] = df["base_price"].apply(price_to_class)

TARGET = "price_class"
DROP_COLS = ["listing_id", "base_price"]

numeric_features = [
    "accommodates", "bedrooms", "beds", "bathrooms",
    "latitude", "longitude",
    "minimum_nights", "maximum_nights",
    "amenities_count", "number_of_reviews",
    "reviews_per_month", "avg_rating",
    "review_scores_rating", "review_scores_accuracy",
    "review_scores_cleanliness", "review_scores_checkin",
    "review_scores_communication", "review_scores_location",
    "review_scores_value"
]

categorical_features = [
    "room_type", "property_type",
    "listing_location", "host_location",
    "host_is_superhost"
]


X_train = train_df.drop(columns=DROP_COLS + [TARGET])
y_train = train_df[TARGET]

X_val = val_df.drop(columns=DROP_COLS + [TARGET])
y_val = val_df[TARGET]

X_test = test_df.drop(columns=DROP_COLS + [TARGET])
y_test = test_df[TARGET]

numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features)
])


classifiers = {
    "RandomForest": RandomForestClassifier(
        random_state=42, class_weight="balanced"
    ),
    "GradientBoosting": GradientBoostingClassifier(
        random_state=42
    ),
    "SVM": SVC(
        random_state=42, class_weight="balanced", probability=True
    ),
    "KNN": KNeighborsClassifier(),
    "DecisionTree": DecisionTreeClassifier(
        random_state=42, class_weight="balanced"
    )
}

results = {}
trained_pipelines = {}

print("\n=== VALIDATION RESULTS ===\n")

for name, clf in classifiers.items():
    pipeline = Pipeline([
        ("preprocess", preprocessor),
        ("model", clf)
    ])

    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_val)

    f1 = f1_score(y_val, preds, average="macro")
    results[name] = f1
    trained_pipelines[name] = pipeline

    print(f"Model: {name}")
    print(f"Macro F1: {f1:.4f}")
    print(classification_report(y_val, preds))
    print("-" * 60)


best_model_name = max(results, key=results.get)
best_pipeline = trained_pipelines[best_model_name]

print(f"\nBEST MODEL: {best_model_name}")
print(f"BEST MACRO F1: {results[best_model_name]:.4f}")

X_final = pd.concat([X_train, X_val])
y_final = pd.concat([y_train, y_val])

best_pipeline.fit(X_final, y_final)

test_preds = best_pipeline.predict(X_test)

print("\n=== TEST RESULTS ===\n")
print(classification_report(y_test, test_preds))

model_artifact = {
    "model": best_pipeline,
    "price_quantiles": {
        "low": q_low,
        "high": q_high
    },
    "classes": ["low", "normal", "high"],
    "numeric_features": numeric_features,
    "categorical_features": categorical_features,
    "best_model_name": best_model_name
}

MODEL_PATH = MODEL_DIR / "price_classifier.pkl"
joblib.dump(model_artifact, MODEL_PATH)

print("\nModel saved as price_classifier.pkl")
