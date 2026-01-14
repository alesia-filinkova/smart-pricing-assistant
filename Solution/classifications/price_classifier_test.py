import joblib
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
MODEL_PATH = BASE_DIR / "Processed_data/models"

artifact = joblib.load(MODEL_PATH / "price_classifier.pkl")

model = artifact["model"]
classes = model.classes_

print(f"Loaded model: {artifact['best_model_name']}")
print("Class order used by model:", classes)

tests = [
    {
        "description": "Very cheap small apartment with no ratings",
        "expected": "low",
        "data": {
            "accommodates": 1, "bedrooms": 0, "beds": 1, "bathrooms": 1,
            "latitude": 37.98, "longitude": 23.72,
            "minimum_nights": 3, "maximum_nights": 365,
            "amenities_count": 5, "number_of_reviews": 0,
            "reviews_per_month": 0, "avg_rating": 0,
            "review_scores_rating": 0, "review_scores_accuracy": 0,
            "review_scores_cleanliness": 0, "review_scores_checkin": 0,
            "review_scores_communication": 0, "review_scores_location": 0,
            "review_scores_value": 0,
            "room_type": "Private room",
            "property_type": "Apartment",
            "listing_location": "ΚΥΨΕΛΗ",
            "host_location": "Greece",
            "host_is_superhost": 0
        }
    },
    {
        "description": "Average apartment with normal reviews",
        "expected": "normal",
        "data": {
            "accommodates": 3, "bedrooms": 1, "beds": 2, "bathrooms": 1,
            "latitude": 37.97, "longitude": 23.73,
            "minimum_nights": 2, "maximum_nights": 365,
            "amenities_count": 25, "number_of_reviews": 20,
            "reviews_per_month": 1.5, "avg_rating": 4.5,
            "review_scores_rating": 4.6, "review_scores_accuracy": 4.5,
            "review_scores_cleanliness": 4.4, "review_scores_checkin": 4.7,
            "review_scores_communication": 4.6, "review_scores_location": 4.5,
            "review_scores_value": 4.4,
            "room_type": "Entire home/apt",
            "property_type": "Apartment",
            "listing_location": "ΝΕΟΣ ΚΟΣΜΟΣ",
            "host_location": "Greece",
            "host_is_superhost": 0
        }
    },
    {
        "description": "Large family apartment with good rating",
        "expected": "normal",
        "data": {
            "accommodates": 6, "bedrooms": 3, "beds": 4, "bathrooms": 2,
            "latitude": 37.99, "longitude": 23.73,
            "minimum_nights": 3, "maximum_nights": 365,
            "amenities_count": 40, "number_of_reviews": 60,
            "reviews_per_month": 3.2, "avg_rating": 4.8,
            "review_scores_rating": 4.9, "review_scores_accuracy": 4.8,
            "review_scores_cleanliness": 4.7, "review_scores_checkin": 4.9,
            "review_scores_communication": 4.9, "review_scores_location": 4.8,
            "review_scores_value": 4.7,
            "room_type": "Entire home/apt",
            "property_type": "Apartment",
            "listing_location": "ΚΥΨΕΛΗ",
            "host_location": "Greece",
            "host_is_superhost": 1
        }
    },
    {
        "description": "Premium loft in the center, superhost",
        "expected": "high",
        "data": {
            "accommodates": 4, "bedrooms": 1, "beds": 2, "bathrooms": 1,
            "latitude": 37.97, "longitude": 23.72,
            "minimum_nights": 2, "maximum_nights": 30,
            "amenities_count": 60, "number_of_reviews": 150,
            "reviews_per_month": 4.5, "avg_rating": 4.95,
            "review_scores_rating": 5.0, "review_scores_accuracy": 5.0,
            "review_scores_cleanliness": 5.0, "review_scores_checkin": 5.0,
            "review_scores_communication": 5.0, "review_scores_location": 5.0,
            "review_scores_value": 4.9,
            "room_type": "Entire home/apt",
            "property_type": "Loft",
            "listing_location": "ΕΜΠΟΡΙΚΟ ΤΡΙΓΩΝΟ-ΠΛΑΚΑ",
            "host_location": "Greece",
            "host_is_superhost": 1
        }
    }
]

for i, test in enumerate(tests, 1):
    df = pd.DataFrame([test["data"]])

    pred = model.predict(df)[0]
    probs = model.predict_proba(df)[0]

    print("=" * 80)
    print(f"TEST #{i}: {test['description']}")
    print(f"Expected class: {test['expected']}")
    print(f"Predicted class: {pred}\n")

    print("Class probabilities:")
    for cls, prob in zip(classes, probs):
        print(f"  {cls}: {prob:.3f}")

    print("\nResult:")
    print("Correct prediction" if pred == test["expected"] else "Wrong prediction")

