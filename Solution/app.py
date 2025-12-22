from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional

import numpy as np
import pandas as pd
import joblib
from catboost import CatBoostRegressor

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "Processed_data/models"

app = FastAPI(title="Base Price Prediction Service", version="1.0.0")


model = CatBoostRegressor()
model.load_model(str(MODEL_DIR / "catboost_base_price.cbm"))

num_imputer = joblib.load(MODEL_DIR / "num_imputer.joblib")
meta = joblib.load(MODEL_DIR / "meta.joblib")

FEATURES = meta["features"]
CATEGORICAL = meta["categorical_features"]
P1 = meta["p1"]
P99 = meta["p99"]
USE_LOG = meta["use_log_target"]


class PredictRequest(BaseModel):
    room_type: str
    property_type: str
    accommodates: float
    bedrooms: Optional[float] = None
    beds: Optional[float] = None
    bathrooms: Optional[float] = None
    city: str
    neighbourhood_cleansed: str
    latitude: float
    longitude: float
    minimum_nights: float
    maximum_nights: float
    host_is_superhost: Optional[float] = None
    amenities_count: float
    number_of_reviews: float
    reviews_per_month: Optional[float] = None
    avg_rating: Optional[float] = None
    review_count: float
    review_scores_rating: Optional[float] = None
    review_scores_accuracy: Optional[float] = None
    review_scores_cleanliness: Optional[float] = None
    review_scores_checkin: Optional[float] = None
    review_scores_communication: Optional[float] = None
    review_scores_location: Optional[float] = None
    review_scores_value: Optional[float] = None


class PredictResponse(BaseModel):
    base_price_usd: float


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    row = pd.DataFrame([{f: getattr(req, f, None) for f in FEATURES}])

    for c in CATEGORICAL:
        row[c] = row[c].astype(str).fillna("unknown")

    num_features = [f for f in FEATURES if f not in CATEGORICAL]
    row[num_features] = num_imputer.transform(row[num_features])

    pred = model.predict(row)[0]
    if USE_LOG:
        pred = float(np.expm1(pred))

    pred = float(np.clip(pred, P1, P99))
    return PredictResponse(base_price_usd=pred)
