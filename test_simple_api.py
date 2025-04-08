#!/usr/bin/env python
import requests
import json
import pandas as pd
import argparse
import sys

def test_api_root(base_url="http://localhost:8001"):
    """Test the root endpoint"""
    try:
        response = requests.get(f"{base_url}/")
        print("API Info:")
        print(json.dumps(response.json(), indent=2))
        print("-" * 50)
        return response.status_code == 200
    except Exception as e:
        print(f"Error testing root endpoint: {str(e)}")
        return False

def test_patients_endpoint(base_url="http://localhost:8001"):
    """Test the patients endpoint"""
    try:
        response = requests.get(f"{base_url}/patients")
        data = response.json()
        print("Available patients:")
        print(json.dumps(data, indent=2))
        print("-" * 50)
        return len(data["patients"]) > 0
    except Exception as e:
        print(f"Error testing patients endpoint: {str(e)}")
        return False

def test_horizons_endpoint(patient_id, base_url="http://localhost:8001"):
    """Test the horizons endpoint"""
    try:
        response = requests.get(f"{base_url}/horizons", params={"patient_id": patient_id})
        data = response.json()
        print(f"Available horizons for patient {patient_id}:")
        print(json.dumps(data, indent=2))
        print("-" * 50)
        return len(data["horizons"]) > 0
    except Exception as e:
        print(f"Error testing horizons endpoint: {str(e)}")
        return False

def test_predict_endpoint(patient_id, test_file=None, base_url="http://localhost:8001"):
    """Test the predict endpoint with sample data"""
    try:
        # Sample request data
        request_data = {
            "patient_id": patient_id,
            "glucose_value": 180.0,
            "glucose_diff": 2.5,
            "glucose_diff_rate": 0.5,
            "glucose_rolling_mean_1h": 175.0,
            "glucose_rolling_std_1h": 7.5,
            "hour": 14,
            "day_of_week": 2,
            "glucose_lags": [178.0, 175.0, 170.0, 168.0, 165.0],
            "insulin_dose": 0.0,
            "insulin_dose_1h": 2.0,
            "insulin_dose_2h": 0.0,
            "insulin_dose_4h": 0.0,
            "carbs_1h": 15.0,
            "carbs_2h": 0.0,
            "carbs_4h": 0.0
        }
        
        # If test file is provided, use real data
        if test_file:
            try:
                df = pd.read_csv(test_file)
                patient_data = df[df["patient_id"] == patient_id]
                
                if len(patient_data) > 0:
                    sample = patient_data.iloc[0]
                    
                    glucose_lags = [
                        sample[f"glucose_lag_{i}"] 
                        for i in range(1, 13)
                        if f"glucose_lag_{i}" in sample.index and not pd.isna(sample[f"glucose_lag_{i}"])
                    ]
                    
                    request_data.update({
                        "patient_id": int(sample["patient_id"]),
                        "glucose_value": float(sample["glucose_value"]),
                        "glucose_diff": float(sample["glucose_diff"]) if "glucose_diff" in sample.index and not pd.isna(sample["glucose_diff"]) else 0.0,
                        "glucose_diff_rate": float(sample["glucose_diff_rate"]) if "glucose_diff_rate" in sample.index and not pd.isna(sample["glucose_diff_rate"]) else None,
                        "glucose_rolling_mean_1h": float(sample["glucose_rolling_mean_1h"]) if "glucose_rolling_mean_1h" in sample.index and not pd.isna(sample["glucose_rolling_mean_1h"]) else None,
                        "glucose_rolling_std_1h": float(sample["glucose_rolling_std_1h"]) if "glucose_rolling_std_1h" in sample.index and not pd.isna(sample["glucose_rolling_std_1h"]) else None,
                        "hour": int(sample["hour"]) if "hour" in sample.index and not pd.isna(sample["hour"]) else 0,
                        "day_of_week": int(sample["day_of_week"]) if "day_of_week" in sample.index and not pd.isna(sample["day_of_week"]) else None,
                        "glucose_lags": glucose_lags
                    })
                    
                    # Add additional features if they exist
                    for feature in ["insulin_dose", "insulin_dose_1h", "insulin_dose_2h", "insulin_dose_4h", 
                                   "carbs_1h", "carbs_2h", "carbs_4h"]:
                        if feature in sample.index and not pd.isna(sample[feature]):
                            request_data[feature] = float(sample[feature])
            except Exception as e:
                print(f"Error loading test data: {str(e)}")
                print("Using default test data")
        
        # Print request data
        print("Request data:")
        print(json.dumps(request_data, indent=2))
        
        # Make request
        response = requests.post(f"{base_url}/predict", json=request_data)
        data = response.json()
        
        # Print response
        print("Prediction response:")
        print(json.dumps(data, indent=2))
        print("-" * 50)
        
        return "predictions" in data and len(data["predictions"]) > 0
    except Exception as e:
        print(f"Error testing predict endpoint: {str(e)}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Test Simple Glucose Prediction API")
    parser.add_argument("--url", default="http://localhost:8001", help="Base URL of the API")
    parser.add_argument("--patient_id", type=int, default=570, help="Patient ID to test")
    parser.add_argument("--test_file", default=None, help="Test data file (optional)")
    
    args = parser.parse_args()
    
    # Run tests
    print(f"Testing API at {args.url}...")
    
    # First make sure the API is running
    if not test_api_root(args.url):
        print("Failed to connect to API. Is it running?")
        sys.exit(1)
    
    # Test patients endpoint
    if not test_patients_endpoint(args.url):
        print("Failed to get patient list")
        sys.exit(1)
    
    # Test horizons endpoint
    if not test_horizons_endpoint(args.patient_id, args.url):
        print(f"Failed to get horizons for patient {args.patient_id}")
        sys.exit(1)
    
    # Test predict endpoint
    if not test_predict_endpoint(args.patient_id, args.test_file, args.url):
        print(f"Failed to make prediction for patient {args.patient_id}")
        sys.exit(1)
    
    print("All tests passed successfully!")

if __name__ == "__main__":
    main() 