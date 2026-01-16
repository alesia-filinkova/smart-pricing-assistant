from fastapi import Depends
from services import ModelService, PricePredictionService, PriceAnalysisService, PriceOptimizationService

def get_model_service():
    return ModelService()

def get_prediction_service(model_service: ModelService = Depends(get_model_service)):
    return PricePredictionService(model_service)

def get_analysis_service(model_service: ModelService = Depends(get_model_service)):
    return PriceAnalysisService(model_service)

def get_optimization_service(model_service: ModelService = Depends(get_model_service)):
    return PriceOptimizationService(model_service)