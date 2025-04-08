# Glucose Prediction API

This API provides access to patient-specific glucose prediction models via HTTP requests. It enables real-time glucose predictions for various time horizons (15, 30 minutes, etc.) using machine learning models.

## Features

- Patient-specific glucose predictions
- Multiple prediction horizons (15min, 30min)
- Automatic model selection (Stacking, Voting, Gradient Boosting, Random Forest)
- Model caching for faster predictions
- Interactive API documentation

## Requirements

- Python 3.8+
- FastAPI
- Uvicorn
- Scikit-learn
- Pandas
- Joblib

## Installation

1. Clone this repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Make sure you have trained models in the `ensemble_models` directory

## Usage

### Starting the API Server

```bash
python glucose_prediction_api.py
```

The API will be available at `http://localhost:8000`. 

### Interactive Documentation

Once the server is running, you can access the interactive API documentation at:
- Swagger UI: `http://localhost:8000/docs` 
- ReDoc: `http://localhost:8000/redoc`

### API Endpoints

- `GET /` - API information
- `GET /patients` - List of available patient models
- `GET /horizons?patient_id={id}` - Available prediction horizons for a patient
- `POST /predict` - Make glucose predictions

### Testing the API

Use the provided test script to verify that the API is working correctly:

```bash
python test_api.py
```

### Example Prediction Request

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "patient_id": 570,
    "glucose_value": 120.5,
    "glucose_diff": 2.5,
    "glucose_diff_rate": 0.5,
    "glucose_rolling_mean_1h": 118.2,
    "glucose_rolling_std_1h": 3.1,
    "hour": 14,
    "day_of_week": 2,
    "glucose_lags": [119.0, 117.5, 115.0, 112.8, 110.5]
  }'
```

### Example Response

```json
{
  "patient_id": 570,
  "current_glucose": 120.5,
  "predictions": {
    "15min": 125.3,
    "30min": 130.1
  },
  "prediction_time": "2023-04-15T14:30:25.123456",
  "features_used": {
    "15min": ["glucose_value", "glucose_diff", "glucose_diff_rate", "glucose_rolling_std_1h", "hour"],
    "30min": ["glucose_value", "hour", "glucose_diff_rate", "glucose_diff", "glucose_rolling_std_1h"]
  },
  "model_types": {
    "15min": "stacking",
    "30min": "stacking"
  }
}
```

## Integration

This API can be integrated with:

1. Mobile applications
2. Wearable devices
3. Clinical monitoring systems
4. Diabetes management platforms

## Next Steps

- Add authentication and user management
- Implement caching with Redis
- Deploy to cloud services (AWS, Azure, GCP)
- Add model monitoring and retraining capabilities
- Implement uncertainty quantification

## API Options

This repository contains two different API implementations:

### 1. Full ML-Based API (`glucose_prediction_api.py`)

- Uses trained machine learning models (Random Forest, Gradient Boosting, etc.)
- Provides more accurate predictions based on patient-specific models
- Requires compatible scikit-learn versions (1.4.2)
- Run with: `python glucose_prediction_api.py`
- Access at: `http://localhost:8000`

### 2. Simple Rule-Based API (`simple_glucose_api.py`)

- Uses rule-based algorithms for prediction
- Does not require pre-trained models
- Works with any Python environment that has FastAPI
- Simpler to deploy and more resilient to dependency changes
- Run with: `python simple_glucose_api.py`
- Access at: `http://localhost:8001`

## How to Choose the Right API

- **For Development/Testing**: Use the simple rule-based API to quickly test integration with other systems
- **For Production/Research**: Use the full ML-based API for more accurate predictions
- **For Deployment on Constrained Systems**: Use the simple rule-based API when you can't install specific ML packages 