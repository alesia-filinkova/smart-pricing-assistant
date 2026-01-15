from pathlib import Path
import pandas as pd
import joblib
import json
from datetime import datetime
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import cross_val_score

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

def create_output_dirs(model_name="price_classifier"):
    """Create directories for model outputs"""
    model_dir = MODEL_DIR / model_name
    info_dir = model_dir / "info"
    
    model_dir.mkdir(parents=True, exist_ok=True)
    info_dir.mkdir(parents=True, exist_ok=True)
    
    return model_dir, info_dir

def save_model_info(info_dir, info_dict, filename="model_info.json"):
    """Save model information as JSON"""
    info_path = info_dir / filename
    with open(info_path, 'w') as f:
        json.dump(info_dict, f, indent=4, default=str)
    return info_path


train_df = pd.read_csv(TRAIN_PATH)
val_df = pd.read_csv(VAL_PATH)
test_df = pd.read_csv(TEST_PATH)

numeric_features = [
    "accommodates", "bedrooms", "beds", "bathrooms",
    "latitude", "longitude",
    "minimum_nights", "maximum_nights",
    "amenities_count", "number_of_reviews",
    "reviews_per_month", "avg_rating",
    "review_scores_rating", "review_scores_accuracy",
    "review_scores_cleanliness", "review_scores_checkin",
    "review_scores_communication", "review_scores_location",
    "review_scores_value",
    "base_price" 
]

categorical_features = [
    "room_type", "property_type",
    "listing_location", "host_location",
    "host_is_superhost"
]

def create_price_target_with_group_percentiles(df, is_training=True, train_percentiles=None):
    """
    Create price class target using percentiles within each (location, room_type) group
    """
    if is_training:
        train_percentiles = {}
        results = []
        
        for (loc, room_type), group in df.groupby(['listing_location', 'room_type']):
            if len(group) >= 4:  
                q25 = group['base_price'].quantile(0.25)
                q75 = group['base_price'].quantile(0.75)
                train_percentiles[(loc, room_type)] = (q25, q75)
                
                conditions = [
                    group['base_price'] < q25,
                    group['base_price'] > q75
                ]
                choices = ['low', 'high']
                classes = np.select(conditions, choices, default='normal')
                results.append(pd.Series(classes, index=group.index))
            else:
                results.append(pd.Series(['normal'] * len(group), index=group.index))
                train_percentiles[(loc, room_type)] = (None, None)
        
        y_target = pd.concat(results).sort_index()
        return y_target, train_percentiles
    else:
        results = []
        
        for idx, row in df.iterrows():
            key = (row['listing_location'], row['room_type'])
            if key in train_percentiles and train_percentiles[key][0] is not None:
                q25, q75 = train_percentiles[key]
                if row['base_price'] < q25:
                    price_class = 'low'
                elif row['base_price'] > q75:
                    price_class = 'high'
                else:
                    price_class = 'normal'
            else:
                price_class = 'normal'
            
            results.append(price_class)
        
        return pd.Series(results, index=df.index), train_percentiles


y_train, train_percentiles = create_price_target_with_group_percentiles(train_df, is_training=True)
y_val, _ = create_price_target_with_group_percentiles(val_df, is_training=False, train_percentiles=train_percentiles)
y_test, _ = create_price_target_with_group_percentiles(test_df, is_training=False, train_percentiles=train_percentiles)

X_train = train_df[numeric_features + categorical_features].copy()
X_val = val_df[numeric_features + categorical_features].copy()
X_test = test_df[numeric_features + categorical_features].copy()

numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])

preprocessor = ColumnTransformer([
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features)
])


classifiers = {
    "RandomForest": RandomForestClassifier(
        random_state=42, class_weight="balanced", n_estimators=200
    ),
    "GradientBoosting": GradientBoostingClassifier(
        random_state=42
    ),
    "SVM": SVC(
        random_state=42, class_weight="balanced", probability=True
    ),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "DecisionTree": DecisionTreeClassifier(
        random_state=42, class_weight="balanced"
    )
}

results = {}
trained_pipelines = {}
training_log = []

for name, clf in classifiers.items():
    pipeline = Pipeline([
        ("preprocess", preprocessor),
        ("model", clf)
    ])
    
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=3, scoring='f1_macro')
    train_cv_f1 = cv_scores.mean()
    
    pipeline.fit(X_train, y_train)
    
    val_preds = pipeline.predict(X_val)
    val_f1 = f1_score(y_val, val_preds, average="macro")
    val_acc = accuracy_score(y_val, val_preds)
    
    results[name] = {
        'train_cv_f1': train_cv_f1,
        'val_f1': val_f1,
        'val_acc': val_acc
    }
    trained_pipelines[name] = pipeline

best_model_name = max(results, key=lambda x: results[x]['val_f1'])
best_model = trained_pipelines[best_model_name]
best_val_f1 = results[best_model_name]['val_f1']

def evaluate_model(model, X):
    y_pred = model.predict(X)
    return y_pred

y_train_pred = evaluate_model(best_model, X_train)
y_val_pred = evaluate_model(best_model, X_val)
y_test_pred = best_model.predict(X_test)


model_dir, info_dir = create_output_dirs("price_classifier")
model_path = model_dir / "model.pkl"
joblib.dump(best_model, model_path)
print(f"Model saved to: {model_path}")

model_info = {
    'model_name': 'price_classifier',
    'best_model_type': best_model_name,
    'features': {
        'numeric': numeric_features,
        'categorical': categorical_features
    },
    'target_classes': ['low', 'normal', 'high'],
    'target_description': '3-class classification based on group percentiles (25th, 75th)',
    'training_date': datetime.now().isoformat(),
    'best_val_f1': float(best_val_f1),
    'train_percentiles_sample': {str(k): v for k, v in list(train_percentiles.items())[:5]},
    'model_performance': {
        'train': {
            'accuracy': float(accuracy_score(y_train, y_train_pred)),
            'f1_macro': float(f1_score(y_train, y_train_pred, average='macro'))
        },
        'val': {
            'accuracy': float(accuracy_score(y_val, y_val_pred)),
            'f1_macro': float(f1_score(y_val, y_val_pred, average='macro'))
        },
        'test': {
            'accuracy': float(accuracy_score(y_test, y_test_pred)),
            'f1_macro': float(f1_score(y_test, y_test_pred, average='macro'))
        }
    },
    'important_note': 'Model includes base_price as a feature. When making predictions, ensure base_price is included in the input features.'
}

save_model_info(info_dir, model_info)

if hasattr(best_model.named_steps.get('model', None), 'feature_importances_'):
    try:
        importances = best_model.named_steps['model'].feature_importances_
        try:
            feature_names = []
            if 'cat' in best_model.named_steps['preprocess'].named_transformers_:
                cat_transformer = best_model.named_steps['preprocess'].named_transformers_['cat']
                cat_encoder = cat_transformer.named_steps['onehot']
                cat_feature_names = cat_encoder.get_feature_names_out(categorical_features)
                feature_names.extend(cat_feature_names)
            
            if 'num' in best_model.named_steps['preprocess'].named_transformers_:
                feature_names.extend(numeric_features)
            
            if len(feature_names) == len(importances):
                feature_importance_df = pd.DataFrame({
                    'feature': feature_names,
                    'importance': importances
                }).sort_values('importance', ascending=False)
                
                feature_importance_df.to_csv(info_dir / "feature_importance.tsv", sep='\t', index=False)
                print(f"Feature importance saved to: {info_dir / 'feature_importance.tsv'}")
            else:
                print("Warning: Feature count mismatch. Showing top indices instead.")
                idx_importance_df = pd.DataFrame({
                    'feature_index': range(len(importances)),
                    'importance': importances
                }).sort_values('importance', ascending=False)
                
                print("\nMapping top indices to original features:")
                top_indices = idx_importance_df.head(10)['feature_index'].values
                for idx in top_indices:
                    if idx < len(numeric_features):
                        print(f"  Index {idx}: {numeric_features[idx]}")
                    else:
                        cat_idx = idx - len(numeric_features)
                        if cat_idx < len(feature_names) - len(numeric_features):
                            print(f"  Index {idx}: {feature_names[cat_idx + len(numeric_features)]}")
                
        except Exception as e:
            print(f"Could not get feature names: {e}")
            print(f"\nFeature importance range: {importances.min():.6f} to {importances.max():.6f}")
            print(f"Most important feature index: {np.argmax(importances)}")
            print(f"Least important feature index: {np.argmin(importances)}")
            
    except Exception as e:
        print(f"Error in feature importance analysis: {e}")
