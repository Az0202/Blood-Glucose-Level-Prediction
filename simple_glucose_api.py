#!/usr/bin/env python
import os
import json
import numpy as np
import pandas as pd
from typing import List, Dict, Optional
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
import uvicorn
from datetime import datetime

# Create FastAPI app
app = FastAPI(
    title="Simple Glucose Prediction API",
    description="API for demonstrating glucose prediction concepts",
    version="1.0.0"
)

class PredictionInput(BaseModel):
    """Input schema for glucose prediction"""
    patient_id: int
    glucose_value: float
    glucose_diff: float
    glucose_diff_rate: Optional[float] = None
    glucose_rolling_mean_1h: Optional[float] = None
    glucose_rolling_std_1h: Optional[float] = None
    hour: int
    day_of_week: Optional[int] = None
    glucose_lags: Optional[List[float]] = None
    insulin_dose: Optional[float] = None
    insulin_dose_1h: Optional[float] = None
    insulin_dose_2h: Optional[float] = None
    insulin_dose_4h: Optional[float] = None
    carbs_1h: Optional[float] = None
    carbs_2h: Optional[float] = None
    carbs_4h: Optional[float] = None

class PredictionOutput(BaseModel):
    """Output schema for glucose prediction"""
    patient_id: int
    current_glucose: float
    predictions: Dict[str, float]
    prediction_time: str
    features_used: Dict[str, List[str]]
    model_types: Dict[str, str]

def make_simple_prediction(data: PredictionInput, horizon: int) -> float:
    """
    Make a simple rule-based prediction based on recent glucose values and trends.
    
    Args:
        data: Input data
        horizon: Prediction horizon in minutes
        
    Returns:
        Predicted glucose value
    """
    # Current glucose is the starting point
    current_glucose = data.glucose_value
    
    # Recent trend (mg/dL per 5 minutes)
    trend = data.glucose_diff if data.glucose_diff is not None else 0
    
    # Calculate how many 5-minute periods in the horizon
    periods = horizon / 5
    
    # Simple linear projection based on recent trend
    # But with dampening factor for longer horizons
    dampening = 1.0 / (1.0 + 0.05 * periods)  # Dampens the trend for longer horizons
    projected_change = trend * periods * dampening
    
    # Adjust prediction based on time of day (example: dawn phenomenon)
    time_factor = 1.0
    if 4 <= data.hour <= 8:  # Dawn phenomenon: glucose tends to rise
        time_factor = 1.2
    elif 1 <= data.hour <= 3:  # Overnight: glucose can drop
        time_factor = 0.9
        
    # Adjust for any recent insulin or carbs
    insulin_effect = 0
    carb_effect = 0
    
    if data.insulin_dose_1h and data.insulin_dose_1h > 0:
        insulin_effect = -10 * data.insulin_dose_1h * (horizon / 60)  # More effect with longer horizon
    
    if data.carbs_1h and data.carbs_1h > 0:
        carb_effect = 0.5 * data.carbs_1h * (1 - (horizon / 120))  # Less effect with longer horizon
    
    # Calculate predicted glucose
    predicted_glucose = current_glucose + (projected_change * time_factor) + insulin_effect + carb_effect
    
    # Ensure prediction is within physiological limits
    predicted_glucose = max(40, min(400, predicted_glucose))
    
    return predicted_glucose

@app.get("/")
def read_root():
    """Root endpoint with API info"""
    return {
        "name": "Simple Glucose Prediction API",
        "version": "1.0.0",
        "description": "API for demonstrating glucose prediction concepts",
        "endpoints": {
            "/predict": "Make glucose predictions",
            "/patients": "Get available patient IDs (demo)",
            "/horizons": "Get available prediction horizons"
        }
    }

@app.get("/patients")
def get_patients():
    """Get list of available patient IDs (demo)"""
    return {"patients": [570, 575, 588, 591]}

@app.get("/horizons")
def get_horizons(patient_id: int):
    """Get available prediction horizons"""
    return {"horizons": [15, 30, 45, 60]}

@app.post("/predict", response_model=PredictionOutput)
def predict(
    input_data: PredictionInput,
    horizons: str = Query("15,30", description="Comma-separated prediction horizons in minutes")
):
    """
    Make glucose predictions using a rule-based approach
    
    Args:
        input_data: Input data for prediction
        horizons: Comma-separated prediction horizons in minutes
        
    Returns:
        Predictions for each horizon
    """
    try:
        # Parse horizons
        horizon_list = [int(h) for h in horizons.split(",")]
        print(f"Making predictions for horizons: {horizon_list}")
        
        # Get key features used for prediction
        key_features = [
            "glucose_value", 
            "glucose_diff", 
            "hour"
        ]
        
        # Add optional features if provided
        if input_data.glucose_diff_rate is not None:
            key_features.append("glucose_diff_rate")
        if input_data.glucose_rolling_mean_1h is not None:
            key_features.append("glucose_rolling_mean_1h")
        if input_data.insulin_dose_1h is not None and input_data.insulin_dose_1h > 0:
            key_features.append("insulin_dose_1h")
        if input_data.carbs_1h is not None and input_data.carbs_1h > 0:
            key_features.append("carbs_1h")
        
        # Make predictions for each horizon
        predictions = {}
        features_used = {}
        model_types = {}
        
        for horizon in horizon_list:
            # Make prediction
            predicted_value = make_simple_prediction(input_data, horizon)
            
            # Store results
            predictions[f"{horizon}min"] = float(predicted_value)
            features_used[f"{horizon}min"] = key_features
            model_types[f"{horizon}min"] = "rule_based"
            
            print(f"Prediction for {horizon}min: {predicted_value:.1f} mg/dL")
        
        # Return result
        return {
            "patient_id": input_data.patient_id,
            "current_glucose": input_data.glucose_value,
            "predictions": predictions,
            "prediction_time": datetime.now().isoformat(),
            "features_used": features_used,
            "model_types": model_types
        }
    
    except Exception as e:
        print(f"Error making prediction: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error making prediction: {str(e)}")

# Run the app when executed directly
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001) 