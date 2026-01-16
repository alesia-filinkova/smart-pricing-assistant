from pathlib import Path
import joblib
from pyparsing import Enum

BASE_DIR = Path(__file__).resolve().parent.parent.parent
MODEL_DIR = BASE_DIR / "Processed_data/models"

class PriceCategory(str, Enum):
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"

class OptimizationAction(str, Enum):
    INCREASE = "increase"
    DECREASE = "decrease"
    MAINTAIN = "maintain"
    ADJUST = "adjust"

class ModelConfig:
    def __init__(self):
        self.regression_model_path = MODEL_DIR / "catboost_base_price.cbm"
        self.num_imputer_path = MODEL_DIR / "num_imputer.joblib"
        self.meta_path = MODEL_DIR / "meta.joblib"
        self.classifier_dir = MODEL_DIR / "price_classifier"
        
    def load_meta(self):
        meta = joblib.load(self.meta_path)
        return meta