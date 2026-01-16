from typing import Any, Dict
import numpy as np
import pandas as pd
import joblib
import json
from catboost import CatBoostRegressor
from config import ModelConfig
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

class ModelService:
    def __init__(self):
        self.config = ModelConfig()
        self.regression_model = None
        self.num_imputer = None
        self.meta = None
        self.classifier_model = None
        self.classifier_info = None
        self.classifier_loaded = False
        
        self._load_models()
    
    def _load_models(self):
        """Load regression and classification models along with necessary artifacts"""
        try:
            self.regression_model = CatBoostRegressor()
            self.regression_model.load_model(str(self.config.regression_model_path))
            
            self.num_imputer = joblib.load(self.config.num_imputer_path)
            self.meta = joblib.load(self.config.meta_path)
            
            self.classifier_model = joblib.load(self.config.classifier_dir / "model.pkl")
            
            with open(self.config.classifier_dir / "info" / "model_info.json", 'r') as f:
                self.classifier_info = json.load(f)
            
            self.classifier_loaded = True
            logger.info("All models loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            self.classifier_loaded = False
    
    @property
    def regression_features(self):
        return self.meta["features"]
    
    @property
    def categorical_features(self):
        return self.meta["categorical_features"]
    
    @property
    def price_bounds(self):
        return {"p1": self.meta["p1"], "p99": self.meta["p99"]}
    
    @property
    def use_log_target(self):
        return self.meta["use_log_target"]
    
    @property
    def classifier_features(self):
        if self.classifier_loaded:
            return self.classifier_info['features']['numeric'] + self.classifier_info['features']['categorical']
        return []

class PricePredictionService:
    def __init__(self, model_service: ModelService):
        self.model_service = model_service
    
    def prepare_regression_features(self, features_dict: Dict[str, Any]) -> pd.DataFrame:
        """Prepare features for regression model"""
        features = self.model_service.regression_features
        categorical = self.model_service.categorical_features
        
        row = pd.DataFrame([{f: features_dict.get(f, None) for f in features}])
        
        for c in categorical:
            row[c] = row[c].astype(str).fillna("unknown")
        
        num_features = [f for f in features if f not in categorical]
        if num_features:
            row[num_features] = self.model_service.num_imputer.transform(row[num_features])
        
        return row
    
    def prepare_classification_features(self, features_dict: Dict[str, Any], price: float) -> pd.DataFrame:
        """Prepare features for classifier"""
        if not self.model_service.classifier_loaded:
            raise ValueError("Price classifier not available")
        
        features_with_price = features_dict.copy()
        features_with_price['base_price'] = price
        
        classifier_features = self.model_service.classifier_features
        
        for feature in classifier_features:
            if feature not in features_with_price:
                if feature in self.model_service.classifier_info['features']['numeric']:
                    features_with_price[feature] = 0.0
                elif feature in self.model_service.classifier_info['features']['categorical']:
                    features_with_price[feature] = 'unknown'
                else:
                    features_with_price[feature] = 0.0
        
        row = pd.DataFrame([features_with_price])[classifier_features]
        
        return row
    
    def predict_price(self, features_dict: Dict[str, Any]) -> float:
        """Predict base price"""
        row = self.prepare_regression_features(features_dict)
        
        pred = self.model_service.regression_model.predict(row)[0]
        if self.model_service.use_log_target:
            pred = float(np.expm1(pred))
        
        bounds = self.model_service.price_bounds
        pred = float(np.clip(pred, bounds["p1"], bounds["p99"]))
        return pred

class PriceAnalysisService:
    def __init__(self, model_service: ModelService):
        self.model_service = model_service
        self.prediction_service = PricePredictionService(model_service)
    
    def classify_price(self, features_dict: Dict[str, Any], price: float) -> Dict[str, Any]:
        """Classify price"""
        if not self.model_service.classifier_loaded:
            raise ValueError("Price classifier not available")
        
        row = self.prediction_service.prepare_classification_features(features_dict, price)
        
        category = self.model_service.classifier_model.predict(row)[0]
        
        if hasattr(self.model_service.classifier_model.named_steps['model'], 'predict_proba'):
            probabilities = self.model_service.classifier_model.predict_proba(row)[0]
            prob_dict = {
                'high': float(probabilities[0]),
                'low': float(probabilities[1]),
                'normal': float(probabilities[2])
            }
            confidence = float(max(probabilities))
        else:
            prob_dict = {'high': 0.33, 'low': 0.33, 'normal': 0.33}
            confidence = 0.5
        
        return {
            'category': category,
            'probabilities': prob_dict,
            'confidence': confidence,
            'is_competitive': category == 'normal'
        }
    
    def generate_recommendations(self, category: str, price: float, predicted_price: float) -> list:
        """Generate recommendations"""
        recommendations = []
        
        if category == 'low':
            recommendations.append(f"Price is likely too low. Consider increasing to ${predicted_price:.2f}")
            recommendations.append("You're leaving money on the table!")
        elif category == 'high':
            recommendations.append(f"Price is likely too high. Consider decreasing to ${predicted_price:.2f}")
            recommendations.append("You might be losing potential bookings")
        else:
            recommendations.append("Price appears competitive in the market")
            recommendations.append("Monitor performance and adjust if needed")
        
        price_diff = predicted_price - price
        if abs(price_diff) > 10:
            action = "increase" if price_diff > 0 else "decrease"
            recommendations.append(f"Suggested {action}: ${abs(price_diff):.2f}")
        
        return recommendations


@dataclass
class OptimizationStrategy:
    name: str
    price_adjustment_factor: float  
    competitiveness_weight: float 
    revenue_weight: float        
    risk_tolerance: float 

class PriceOptimizationService:
    def __init__(self, model_service: ModelService):
        self.model_service = model_service
        self.prediction_service = PricePredictionService(model_service)
        self.analysis_service = PriceAnalysisService(model_service)
        
        self.strategies = {
            "balanced": OptimizationStrategy(
                name="balanced",
                price_adjustment_factor=1.0,
                competitiveness_weight=0.5,
                revenue_weight=0.5,
                risk_tolerance=0.5
            ),
            "aggressive": OptimizationStrategy(
                name="aggressive",
                price_adjustment_factor=0.8,
                competitiveness_weight=0.8,
                revenue_weight=0.2,
                risk_tolerance=0.8
            ),
            "conservative": OptimizationStrategy(
                name="conservative",
                price_adjustment_factor=0.9,
                competitiveness_weight=0.9,
                revenue_weight=0.1,
                risk_tolerance=0.3
            ),
            "revenue_max": OptimizationStrategy(
                name="revenue_max",
                price_adjustment_factor=1.2, 
                competitiveness_weight=0.2,
                revenue_weight=0.8,
                risk_tolerance=0.7
            )
        }
    
    def _calculate_competitiveness_score(self, price: float, category: str, probabilities: Dict[str, float]) -> float:
        """Calculate competitiveness score"""
        if category == 'low':
            return 0.8 + probabilities.get('low', 0) * 0.2
        elif category == 'normal':
            return 0.5 + probabilities.get('normal', 0) * 0.5
        else: 
            return 0.2 + probabilities.get('high', 0) * 0.2
    
    def _calculate_revenue_potential(self, current_price: float, new_price: float, 
                                   current_category: str, new_category: str) -> float:
        """Calculate revenue potential"""
        price_change_pct = (new_price - current_price) / current_price * 100
        
        category_factors = {
            ('low', 'normal'): 1.5, 
            ('low', 'high'): 1.2,     
            ('normal', 'high'): 0.8,   
            ('high', 'normal'): 1.2,   
            ('high', 'low'): 1.5,      
            ('normal', 'low'): 0.6,    
        }
        
        factor = category_factors.get((current_category, new_category), 1.0)
        return price_change_pct * factor
    
    def _estimate_demand_curve(self, features_dict: Dict[str, Any], price: float) -> float:
        """Calculate demand curve"""
        base_price = self.prediction_service.predict_price(features_dict)
        price_diff_ratio = price / base_price
        
        if price_diff_ratio <= 0.8:  
            demand = 2.0  
        elif price_diff_ratio <= 0.95:  
            demand = 1.5 
        elif price_diff_ratio <= 1.05:  
            demand = 1.0  
        elif price_diff_ratio <= 1.2:  
            demand = 0.7  
        else:
            demand = 0.3  
        
        return demand
    
    def optimize_price(self, features_dict: Dict[str, Any], current_price: float, 
                      strategy_name: str = "balanced") -> Dict[str, Any]:
        """Provide optimized price based on selected strategy"""
        if strategy_name not in self.strategies:
            strategy_name = "balanced"
        
        strategy = self.strategies[strategy_name]
        optimal_price = self.prediction_service.predict_price(features_dict)
        current_analysis = self.analysis_service.classify_price(features_dict, current_price)
        adjusted_price = optimal_price * strategy.price_adjustment_factor
        
        bounds = self.model_service.price_bounds
        min_price = bounds["p1"] * 0.8 
        max_price = bounds["p99"] * 1.2 
        
        if strategy_name == "aggressive":
            adjusted_price = optimal_price * np.random.uniform(0.8, 0.9)
            explanation = "Aggressive pricing to maximize bookings and market share"
        
        elif strategy_name == "conservative":
            adjusted_price = optimal_price * np.random.uniform(0.95, 0.98)
            explanation = "Conservative pricing to ensure high occupancy rate"
        
        elif strategy_name == "revenue_max":
            demand_estimate = self._estimate_demand_curve(features_dict, optimal_price * 1.1)
            if demand_estimate > 0.5: 
                adjusted_price = optimal_price * np.random.uniform(1.05, 1.15)
                explanation = "Revenue maximization strategy with premium pricing"
            else:
                adjusted_price = optimal_price
                explanation = "Revenue optimization balanced with demand"
        
        else:  # balanced
            adjusted_price = optimal_price
            explanation = "Balanced strategy for optimal price-performance ratio"
        
        adjusted_price = max(min_price, min(max_price, adjusted_price))
        
        adjusted_analysis = self.analysis_service.classify_price(features_dict, adjusted_price)
        
        price_change_pct = (adjusted_price - current_price) / current_price * 100
        
        current_competitiveness = self._calculate_competitiveness_score(
            current_price, current_analysis['category'], current_analysis['probabilities']
        )
        
        adjusted_competitiveness = self._calculate_competitiveness_score(
            adjusted_price, adjusted_analysis['category'], adjusted_analysis['probabilities']
        )
        
        revenue_potential = self._calculate_revenue_potential(
            current_price, adjusted_price, 
            current_analysis['category'], adjusted_analysis['category']
        )
        
        if abs(price_change_pct) < 2:
            action = "maintain"
            explanation = "Current price is already optimal"
        elif price_change_pct > 5:
            action = "increase"
            if strategy_name == "revenue_max":
                explanation = f"Increase price to ${adjusted_price:.2f} for revenue maximization"
            else:
                explanation = f"Increase price to ${adjusted_price:.2f} for better positioning"
        elif price_change_pct < -5:
            action = "decrease"
            if strategy_name == "aggressive":
                explanation = f"Decrease price to ${adjusted_price:.2f} to capture market share"
            else:
                explanation = f"Decrease price to ${adjusted_price:.2f} to be more competitive"
        else:
            action = "adjust"
            explanation = f"Minor adjustment to ${adjusted_price:.2f} recommended"
        
        demand_current = self._estimate_demand_curve(features_dict, current_price)
        demand_adjusted = self._estimate_demand_curve(features_dict, adjusted_price)
        
        revenue_current = current_price * demand_current
        revenue_adjusted = adjusted_price * demand_adjusted
        expected_revenue_change_pct = (revenue_adjusted - revenue_current) / revenue_current * 100

        confidence = min(current_analysis['confidence'], adjusted_analysis['confidence'])        
        if strategy_name == "balanced":
            confidence *= 1.0
        elif strategy_name == "aggressive":
            confidence *= 0.8 
        elif strategy_name == "conservative":
            confidence *= 0.9
        elif strategy_name == "revenue_max":
            confidence *= 0.7 
        
        return {
            'recommended_price': round(adjusted_price, 2),
            'price_change_pct': round(price_change_pct, 2),
            'current_category': current_analysis['category'],
            'recommended_category': adjusted_analysis['category'],
            'expected_revenue_change_pct': round(expected_revenue_change_pct, 2),
            'action': action,
            'explanation': explanation,
            'confidence': round(min(confidence, 0.95), 2),
            'strategy_used': strategy_name,
            'competitiveness_score': round(adjusted_competitiveness, 3),
            'revenue_potential': round(revenue_potential, 2)
        }
    
    def get_strategy_comparison(self, features_dict: Dict[str, Any], current_price: float) -> Dict[str, Any]:
        """Compare all strategies"""
        results = {}
        
        for strategy_name in self.strategies.keys():
            try:
                result = self.optimize_price(features_dict, current_price, strategy_name)
                results[strategy_name] = {
                    'recommended_price': result['recommended_price'],
                    'price_change_pct': result['price_change_pct'],
                    'expected_revenue_change_pct': result['expected_revenue_change_pct'],
                    'confidence': result['confidence'],
                    'action': result['action'],
                    'competitiveness_score': result.get('competitiveness_score', 0.5)
                }
            except Exception as e:
                logger.error(f"Error in strategy {strategy_name}: {e}")
                results[strategy_name] = {
                    'error': str(e),
                    'recommended_price': current_price,
                    'price_change_pct': 0,
                    'expected_revenue_change_pct': 0,
                    'confidence': 0
                }
        
        return results