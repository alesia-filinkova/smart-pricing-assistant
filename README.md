# Smart Pricing Assistant

**Smart Pricing Assistant** is a machine learning service designed to help Airbnb (and similar) hosts price their listings intelligently.  
It combines **price prediction**, **price category classification**, and **pricing optimization strategies** to guide hosts in setting competitive and revenue-driving prices based on listing features and local market patterns.

---

## Project Overview

Hosts often struggle to determine the right price for their listings.  
This repository provides a complete ML-backed service that:

- Predicts the optimal **base price** for a listing
- Classifies current pricing as **low, normal, or high**
- Recommends price adjustments with expected revenue insights
- Compares multiple optimization strategies

The service is built with **FastAPI** and scalable machine learning pipelines.

---

## Features

### Price Prediction
Predicts the optimal USD price for a listing using a CatBoost regression model.

### Price Category Classification
Classifies price into one of three classes:
- `normal` — competitive price
- `high` — overpriced
- `low` — underpriced

Model uses group-based percentiles (by *location + room type*) for robust classification.

### Price Optimization
Suggests a pricing adjustment based on strategy:
- `balanced` (default)
- `conservative`
- `aggressive`
- `revenue_max`

Each recommendation includes expected impact and model confidence.

### Strategy Comparison
Evaluate multiple strategies side-by-side for decision support.

---

## Dependencies

All required Python packages are listed in requirements.txt.

Install them using:

pip install -r requirements.txt


The project relies on the following key libraries:

fastapi – REST API framework used to build the pricing service

uvicorn – ASGI server for running the FastAPI application

scikit-learn – preprocessing pipelines, classical ML models, and evaluation

catboost – gradient boosting model used for price regression

pandas, numpy – data manipulation and numerical computations

joblib – serialization of trained models and preprocessing pipelines

matplotlib, seaborn – optional visualization libraries used in notebooks and analysis

---

## Running the Service

To start the FastAPI service locally, run:

cd Solution/app
uvicorn app:app --reload --host 0.0.0.0 --port 8000


Once the server is running, the following endpoints will be available:

- Interactive API documentation (Swagger UI)
    - http://localhost:8000/docs
- Root status endpoint
    - http://localhost:8000/

---
## Model Insights and Evaluation

The repository includes notebooks and scripts that demonstrate:
- training, validation, and test performance
- confusion matrices for classification models
- feature importance analysis
- sample predictions with explanations
- visualization and comparison of pricing strategies

---

## Credits

Smart Pricing Assistant
Developed by Alesia Filinkova, Diana Pelin
Warsaw University of Technologies project — IUM, Semester 5
