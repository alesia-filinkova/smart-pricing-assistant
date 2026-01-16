from pydantic import BaseModel
from typing import Optional, Dict, Any

from config import OptimizationAction, PriceCategory


class PredictRequest(BaseModel):
    """Request model for price prediction"""
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

class PriceAnalysisRequest(BaseModel):
    """Request model for price analysis with specific price"""
    listing_features: Dict[str, Any] 
    current_price: float

class PriceOptimizationRequest(BaseModel):
    """Request model for price optimization"""
    listing_features: Dict[str, Any]
    current_price: float
    strategy: Optional[str] = "balanced" 

class PredictResponse(BaseModel):
    """Response for basic price prediction"""
    base_price_usd: float
    message: Optional[str] = None

class PriceAnalysisResponse(BaseModel):
    """Response for price category analysis"""
    current_price: float
    predicted_category: PriceCategory
    probabilities: Dict[str, float]  
    is_competitive: bool  # True if category is 'normal'
    confidence: float
    recommendations: Optional[list] = None

class PriceOptimizationResponse(BaseModel):
    """Response for price optimization"""
    current_price: float
    recommended_price: float
    price_change_pct: float  # Percentage change
    current_category: PriceCategory
    recommended_category: PriceCategory
    expected_revenue_change_pct: Optional[float] = None
    action: OptimizationAction
    explanation: str
    confidence: float 

class FullAnalysisResponse(BaseModel):
    """Complete analysis response"""
    regression_prediction: PredictResponse
    price_analysis: PriceAnalysisResponse
    optimization: Optional[PriceOptimizationResponse] = None
    timestamp: str

class StrategyComparisonResponse(BaseModel):
    """Response for strategy comparison"""
    current_price: float
    strategies_comparison: Dict[str, Dict[str, Any]]
    timestamp: str
    best_strategy: Optional[str] = None
    best_revenue_strategy: Optional[str] = None
    best_competitiveness_strategy: Optional[str] = None