#!/usr/bin/env python
import os
import json
import joblib
import numpy as np
import pandas as pd
from typing import List, Dict, Optional
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
import uvicorn

# Create FastAPI app
app = FastAPI(
    title="Glucose Prediction API",
    description="API for real-time glucose prediction using patient-specific models",
    version="1.0.0"
)

# Model cache to avoid reloading models for every request
model_cache = {}

class PredictionInput(BaseModel):
    """Input schema for glucose prediction"""
    patient_id: int
    glucose_value: float
    glucose_diff: float
    glucose_diff_rate: float
    glucose_rolling_mean_1h: Optional[float] = None
    glucose_rolling_std_1h: Optional[float] = None
    hour: int
    day_of_week: Optional[int] = None
    # Add optional fields for all potential features
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

def load_patient_model(patient_id: int, horizon: int, models_dir: str = "ensemble_models"):
    """
    Load the model for a specific patient and prediction horizon
    
    Args:
        patient_id: Patient ID
        horizon: Prediction horizon in minutes
        models_dir: Directory containing models
        
    Returns:
        model: Loaded model
        features: List of features used by the model
    """
    # Create cache key
    cache_key = f"{patient_id}_{horizon}"
    
    # Check if model is in cache
    if cache_key in model_cache:
        return model_cache[cache_key]
    
    # Patient directory
    patient_dir = os.path.join(models_dir, f"patient_{patient_id}")
    
    if not os.path.exists(patient_dir):
        raise HTTPException(status_code=404, detail=f"No models found for patient {patient_id}")
    
    # Try to load models in preference order (best to worst)
    model = None
    features = None
    model_type = None
    
    for model_prefix in ['stacking', 'voting', 'gb', 'rf']:
        model_path = os.path.join(patient_dir, f"{model_prefix}_model_{horizon}min.joblib")
        features_path = os.path.join(patient_dir, f"{model_prefix}_features_{horizon}min.json")
        
        if os.path.exists(model_path) and os.path.exists(features_path):
            model = joblib.load(model_path)
            with open(features_path, 'r') as f:
                features = json.load(f)
            model_type = model_prefix
            break
    
    if model is None or features is None:
        raise HTTPException(status_code=404, 
                           detail=f"No model or features found for patient {patient_id}, horizon {horizon}min")
    
    # Cache the model, features and type
    model_cache[cache_key] = (model, features, model_type)
    
    return model, features, model_type

def prepare_input_data(input_data: PredictionInput, features: List[str]) -> pd.DataFrame:
    """
    Prepare the input data for the model
    
    Args:
        input_data: Input data from request
        features: List of features required by the model
        
    Returns:
        DataFrame with prepared features
    """
    # Create a dictionary with available features
    data_dict = {
        "glucose_value": input_data.glucose_value,
        "glucose_diff": input_data.glucose_diff,
        "glucose_diff_rate": input_data.glucose_diff_rate,
        "glucose_rolling_mean_1h": input_data.glucose_rolling_mean_1h,
        "glucose_rolling_std_1h": input_data.glucose_rolling_std_1h,
        "hour": input_data.hour,
        "day_of_week": input_data.day_of_week,
    }
    
    # Add glucose lags if provided
    if input_data.glucose_lags:
        for i, lag in enumerate(input_data.glucose_lags, 1):
            if i <= 12:  # We support up to 12 lags
                data_dict[f"glucose_lag_{i}"] = lag
    
    # Add insulin and carbs features if provided
    if input_data.insulin_dose is not None:
        data_dict["insulin_dose"] = input_data.insulin_dose
    if input_data.insulin_dose_1h is not None:
        data_dict["insulin_dose_1h"] = input_data.insulin_dose_1h
    if input_data.insulin_dose_2h is not None:
        data_dict["insulin_dose_2h"] = input_data.insulin_dose_2h
    if input_data.insulin_dose_4h is not None:
        data_dict["insulin_dose_4h"] = input_data.insulin_dose_4h
    if input_data.carbs_1h is not None:
        data_dict["carbs_1h"] = input_data.carbs_1h
    if input_data.carbs_2h is not None:
        data_dict["carbs_2h"] = input_data.carbs_2h
    if input_data.carbs_4h is not None:
        data_dict["carbs_4h"] = input_data.carbs_4h
    
    # Create DataFrame
    df = pd.DataFrame([data_dict])
    
    # Check for missing required features
    missing_features = [f for f in features if f not in df.columns]
    if missing_features:
        # Fill missing features with zeros
        for feature in missing_features:
            df[feature] = 0
    
    # Select and order features according to the model's requirements
    return df[features]

@app.get("/")
def read_root():
    """Root endpoint with API info"""
    return {
        "name": "Glucose Prediction API",
        "version": "1.0.0",
        "description": "API for real-time glucose prediction",
        "endpoints": {
            "/predict": "Make glucose predictions",
            "/patients": "Get list of available patient models",
            "/horizons": "Get available prediction horizons"
        }
    }

@app.get("/patients")
def get_patients(models_dir: str = "ensemble_models"):
    """Get list of patients with available models"""
    try:
        # Get all directories in models_dir that start with "patient_"
        patients = []
        for item in os.listdir(models_dir):
            if os.path.isdir(os.path.join(models_dir, item)) and item.startswith("patient_"):
                patient_id = item.replace("patient_", "")
                patients.append(int(patient_id))
        
        return {"patients": sorted(patients)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing patients: {str(e)}")

@app.get("/horizons")
def get_horizons(patient_id: int, models_dir: str = "ensemble_models"):
    """Get available prediction horizons for a patient"""
    try:
        patient_dir = os.path.join(models_dir, f"patient_{patient_id}")
        if not os.path.exists(patient_dir):
            raise HTTPException(status_code=404, detail=f"No models found for patient {patient_id}")
        
        # Get available horizons by looking for model files
        horizons = set()
        for filename in os.listdir(patient_dir):
            if filename.endswith(".joblib") and "model" in filename:
                # Extract horizon from filename (format: prefix_model_XXmin.joblib)
                parts = filename.split("_")
                for part in parts:
                    if part.endswith("min.joblib"):
                        horizon = int(part.replace("min.joblib", ""))
                        horizons.add(horizon)
        
        return {"horizons": sorted(list(horizons))}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing horizons: {str(e)}")

@app.post("/predict", response_model=PredictionOutput)
def predict(
    input_data: PredictionInput,
    horizons: str = Query("15,30", description="Comma-separated prediction horizons in minutes"),
    models_dir: str = Query("ensemble_models", description="Directory containing models")
):
    """
    Make glucose predictions for the specified patient and horizons
    
    Args:
        input_data: Input data for prediction
        horizons: Comma-separated prediction horizons in minutes
        models_dir: Directory containing models
        
    Returns:
        Predictions for each horizon
    """
    try:
        # Parse horizons
        horizon_list = [int(h) for h in horizons.split(",")]
        print(f"Making predictions for horizons: {horizon_list}")
        
        # Initialize output
        predictions = {}
        features_used = {}
        model_types = {}
        
        # Make predictions for each horizon
        for horizon in horizon_list:
            try:
                print(f"\nProcessing horizon {horizon}min:")
                # Load model
                print(f"Loading model for patient {input_data.patient_id}, horizon {horizon}min")
                model, features, model_type = load_patient_model(
                    input_data.patient_id, horizon, models_dir
                )
                print(f"Loaded {model_type} model with {len(features)} features")
                
                # Prepare input data
                print(f"Preparing input data with features: {features[:5]}...")
                X = prepare_input_data(input_data, features)
                print(f"Input data shape: {X.shape}")
                
                # Make prediction
                print(f"Making prediction with {model_type} model")
                pred = model.predict(X)[0]
                print(f"Raw prediction result: {pred}")
                
                # Store results
                predictions[f"{horizon}min"] = float(pred)
                features_used[f"{horizon}min"] = features
                model_types[f"{horizon}min"] = model_type
                print(f"Prediction for {horizon}min: {float(pred):.1f} mg/dL")
            
            except Exception as e:
                # If prediction fails for a horizon, skip it
                print(f"Error predicting for horizon {horizon}min: {str(e)}")
                import traceback
                traceback.print_exc()
        
        if not predictions:
            print("No predictions were successful")
            raise HTTPException(
                status_code=500, 
                detail="Failed to make predictions for all requested horizons"
            )
        
        print(f"Successful predictions: {list(predictions.keys())}")
        
        # Return result
        return {
            "patient_id": input_data.patient_id,
            "current_glucose": input_data.glucose_value,
            "predictions": predictions,
            "prediction_time": pd.Timestamp.now().isoformat(),
            "features_used": features_used,
            "model_types": model_types
        }
    
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        # Convert other exceptions to HTTP exceptions
        print(f"Unexpected error in prediction endpoint: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error making prediction: {str(e)}")

# Run the app when executed directly
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000) 