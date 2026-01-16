from fastapi import FastAPI, HTTPException, Depends
from datetime import datetime
import logging

from models import *
from services import ModelService
from dependencies import *

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Price Prediction & Optimization Service",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

model_service = ModelService()

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "service": "Price Prediction & Optimization",
        "version": "2.0.0",
        "endpoints": {
            "/docs": "Interactive API documentation",
            "/predict": "Basic price prediction",
            "/analyze-price": "Price category analysis",
            "/optimize-price": "Price optimization",
            "/full-analysis": "Complete analysis"
        },
        "models_loaded": {
            "regression": True,
            "classification": model_service.classifier_loaded
        }
    }

@app.post("/predict", response_model=PredictResponse)
def predict(
    req: PredictRequest,
    prediction_service: PricePredictionService = Depends(get_prediction_service)
):
    """
    Predict optimal base price for a listing
    
    Returns the predicted optimal price in USD
    """
    try:
        features_dict = req.model_dump()
        predicted_price = prediction_service.predict_price(features_dict)
        
        return PredictResponse(
            base_price_usd=predicted_price,
            message=f"Predicted optimal price: ${predicted_price:.2f}"
        )
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/analyze-price", response_model=PriceAnalysisResponse)
def analyze_price(
    req: PriceAnalysisRequest,
    analysis_service: PriceAnalysisService = Depends(get_analysis_service)
):
    """
    Analyze current price and classify it as low/normal/high
    
    Returns price category, probabilities, and recommendations
    """
    try:
        if not model_service.classifier_loaded:
            raise HTTPException(status_code=503, detail="Price classifier not available")
        
        analysis = analysis_service.classify_price(req.listing_features, req.current_price)
        predicted_price = analysis_service.prediction_service.predict_price(req.listing_features)
        recommendations = analysis_service.generate_recommendations(
            analysis['category'], 
            req.current_price, 
            predicted_price
        )
        
        return PriceAnalysisResponse(
            current_price=req.current_price,
            predicted_category=analysis['category'],
            probabilities=analysis['probabilities'],
            is_competitive=analysis['is_competitive'],
            confidence=analysis['confidence'],
            recommendations=recommendations
        )
    except Exception as e:
        logger.error(f"Price analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Price analysis failed: {str(e)}")

@app.post("/optimize-price", response_model=PriceOptimizationResponse)
def optimize_price_endpoint(
    req: PriceOptimizationRequest,
    optimization_service: PriceOptimizationService = Depends(get_optimization_service)
):
    """
    Optimize listing price based on strategy
    
    Strategies:
    - balanced: Balance between revenue and competitiveness (default)
    - aggressive: Maximize bookings (lower prices)
    - conservative: Maximize price competitiveness (slightly lower prices)
    - revenue_max: Maximize expected revenue (higher prices)
    """
    try:
        optimization = optimization_service.optimize_price(
            req.listing_features,
            req.current_price,
            req.strategy
        )
        
        return PriceOptimizationResponse(
            current_price=req.current_price,
            recommended_price=optimization['recommended_price'],
            price_change_pct=optimization['price_change_pct'],
            current_category=optimization['current_category'],
            recommended_category=optimization['recommended_category'],
            expected_revenue_change_pct=optimization.get('expected_revenue_change_pct'),
            action=optimization['action'],
            explanation=optimization['explanation'],
            confidence=optimization['confidence']
        )
    except Exception as e:
        logger.error(f"Price optimization failed: {e}")
        raise HTTPException(status_code=500, detail=f"Price optimization failed: {str(e)}")

@app.post("/compare-strategies")
async def compare_strategies(
    req: PriceOptimizationRequest,
    optimization_service: PriceOptimizationService = Depends(get_optimization_service)
):
    """
    Compare results of different pricing strategies
    """
    try:
        comparison = optimization_service.get_strategy_comparison(
            req.listing_features,
            req.current_price
        )
        
        return {
            "current_price": req.current_price,
            "strategies_comparison": comparison,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Strategy comparison failed: {e}")
        raise HTTPException(status_code=500, detail=f"Strategy comparison failed: {str(e)}")

@app.post("/full-analysis", response_model=FullAnalysisResponse)
def full_analysis(
    req: PriceAnalysisRequest,
    prediction_service: PricePredictionService = Depends(get_prediction_service),
    analysis_service: PriceAnalysisService = Depends(get_analysis_service),
    optimization_service: PriceOptimizationService = Depends(get_optimization_service)
):
    """
    Complete analysis: regression prediction + classification + optimization
    
    Returns all analysis in one response
    """
    try:
        predicted_price = prediction_service.predict_price(req.listing_features)
        regression_response = PredictResponse(
            base_price_usd=predicted_price,
            message=f"Predicted optimal price: ${predicted_price:.2f}"
        )
        
        if model_service.classifier_loaded:
            analysis = analysis_service.classify_price(req.listing_features, req.current_price)
            recommendations = analysis_service.generate_recommendations(
                analysis['category'], 
                req.current_price, 
                predicted_price
            )
            
            price_analysis_response = PriceAnalysisResponse(
                current_price=req.current_price,
                predicted_category=analysis['category'],
                probabilities=analysis['probabilities'],
                is_competitive=analysis['is_competitive'],
                confidence=analysis['confidence'],
                recommendations=recommendations
            )
            
            optimization = optimization_service.optimize_price(req.listing_features, req.current_price)
            optimization_response = PriceOptimizationResponse(
                current_price=req.current_price,
                recommended_price=optimization['recommended_price'],
                price_change_pct=optimization['price_change_pct'],
                current_category=optimization['current_category'],
                recommended_category=optimization['recommended_category'],
                expected_revenue_change_pct=optimization.get('expected_revenue_change_pct'),
                action=optimization['action'],
                explanation=optimization['explanation'],
                confidence=optimization['confidence']
            )
        else:
            price_analysis_response = PriceAnalysisResponse(
                current_price=req.current_price,
                predicted_category="normal",  # Default
                probabilities={"low": 0.0, "normal": 1.0, "high": 0.0},
                is_competitive=True,
                confidence=0.5,
                recommendations=["Price classifier not available"]
            )
            optimization_response = None
        
        return FullAnalysisResponse(
            regression_prediction=regression_response,
            price_analysis=price_analysis_response,
            optimization=optimization_response,
            timestamp=datetime.now().isoformat()
        )
    except Exception as e:
        logger.error(f"Full analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Full analysis failed: {str(e)}")

@app.get("/example-request")
async def example_request():
    """Get example request structure"""
    example_features = {
        "room_type": "Entire home/apt",
        "property_type": "Entire rental unit",
        "accommodates": 4.0,
        "bedrooms": 2.0,
        "beds": 2.0,
        "bathrooms": 1.0,
        "city": "Athens",
        "neighbourhood_cleansed": "ΕΜΠΟΡΙΚΟ ΤΡΙΓΩΝΟ-ΠΛΑΚΑ",
        "latitude": 37.97506,
        "longitude": 23.73068,
        "minimum_nights": 3.0,
        "maximum_nights": 365.0,
        "host_is_superhost": 1.0,
        "amenities_count": 45.0,
        "number_of_reviews": 33.0,
        "reviews_per_month": 0.22,
        "avg_rating": 4.5,
        "review_count": 33.0,
        "review_scores_rating": 4.84,
        "review_scores_accuracy": 4.91,
        "review_scores_cleanliness": 4.91,
        "review_scores_checkin": 4.88,
        "review_scores_communication": 4.88,
        "review_scores_location": 5.0,
        "review_scores_value": 4.76
    }
    
    return {
        "example_for_predict_endpoint": PredictRequest(**example_features).model_dump(),
        "example_for_analyze_endpoint": {
            "listing_features": example_features,
            "current_price": 476.86
        },
        "note": "Use these examples to test the API endpoints"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)