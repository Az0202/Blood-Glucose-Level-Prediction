#!/usr/bin/env python
import requests
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import argparse
import os
import json
from tqdm import tqdm

def make_api_prediction(base_url, input_data):
    """Make a prediction using the API"""
    try:
        response = requests.post(f"{base_url}/predict", json=input_data, timeout=5)
        return response.json()
    except Exception as e:
        print(f"Error making prediction: {e}")
        return None

def compare_apis(test_file, patient_id, horizons, api_url_1, api_url_2, api_1_name, api_2_name, output_dir):
    """Compare predictions from two different APIs"""
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load test data
    try:
        test_df = pd.read_csv(test_file)
    except Exception as e:
        print(f"Error loading test file: {e}")
        return None
    
    # Filter for the patient
    patient_data = test_df[test_df['patient_id'] == patient_id].copy()
    
    if len(patient_data) == 0:
        print(f"No data found for patient {patient_id}")
        return None
    
    # Sample data points (to limit API calls)
    sample_size = min(100, len(patient_data))
    stride = max(1, len(patient_data) // sample_size)
    sample_indices = list(range(0, len(patient_data), stride))[:sample_size]
    
    # Storage for results
    comparison_results = {
        'patient_id': patient_id,
        'horizons': horizons,
        'api_1': api_1_name,
        'api_2': api_2_name,
        'predictions': {}
    }
    
    for horizon in horizons:
        comparison_results['predictions'][horizon] = {
            'api_1_predictions': [],
            'api_2_predictions': [],
            'ground_truth': []
        }
    
    # Process each sample
    print(f"Comparing API predictions for patient {patient_id} ({len(sample_indices)} samples)")
    
    for idx in tqdm(sample_indices, desc=f"Processing patient {patient_id}"):
        row = patient_data.iloc[idx]
        
        # Skip rows with missing target values
        if any(pd.isna(row.get(f'target_{h}min', None)) for h in horizons):
            continue
        
        # Prepare input data for API
        input_data = {
            "patient_id": int(row['patient_id']),
            "glucose_value": float(row['glucose_value']),
            "glucose_diff": float(row.get('glucose_diff', 0)),
            "glucose_diff_rate": float(row.get('glucose_diff_rate', 0)),
            "glucose_rolling_mean_1h": float(row.get('glucose_rolling_mean_1h', row['glucose_value'])),
            "glucose_rolling_std_1h": float(row.get('glucose_rolling_std_1h', 0)),
            "hour": int(row.get('hour', 0)),
            "day_of_week": int(row.get('day_of_week', 0)),
            "insulin_dose": float(row.get('insulin_dose', 0)),
            "insulin_dose_1h": float(row.get('insulin_dose_1h', 0)),
            "insulin_dose_2h": float(row.get('insulin_dose_2h', 0)),
            "insulin_dose_4h": float(row.get('insulin_dose_4h', 0)),
            "carbs_1h": float(row.get('carbs_1h', 0)),
            "carbs_2h": float(row.get('carbs_2h', 0)),
            "carbs_4h": float(row.get('carbs_4h', 0)),
        }
        
        # Add glucose lags if available
        glucose_lags = []
        for i in range(1, 13):
            lag_col = f'glucose_lag_{i}'
            if lag_col in row and not pd.isna(row[lag_col]):
                glucose_lags.append(float(row[lag_col]))
        
        if glucose_lags:
            input_data["glucose_lags"] = glucose_lags
        
        # Get predictions from both APIs
        prediction_1 = make_api_prediction(api_url_1, input_data)
        prediction_2 = make_api_prediction(api_url_2, input_data)
        
        # Process predictions
        for horizon in horizons:
            horizon_key = f"{horizon}min"
            target_key = f"target_{horizon}min"
            
            # Extract values if available
            api_1_value = None
            if prediction_1 and 'predictions' in prediction_1 and horizon_key in prediction_1['predictions']:
                api_1_value = prediction_1['predictions'][horizon_key]
            
            api_2_value = None
            if prediction_2 and 'predictions' in prediction_2 and horizon_key in prediction_2['predictions']:
                api_2_value = prediction_2['predictions'][horizon_key]
            
            ground_truth = row.get(target_key)
            
            # Only store if we have both predictions and ground truth
            if api_1_value is not None and api_2_value is not None and not pd.isna(ground_truth):
                comparison_results['predictions'][horizon]['api_1_predictions'].append(api_1_value)
                comparison_results['predictions'][horizon]['api_2_predictions'].append(api_2_value)
                comparison_results['predictions'][horizon]['ground_truth'].append(ground_truth)
    
    # Create visualizations and compute metrics
    for horizon in horizons:
        horizon_data = comparison_results['predictions'][horizon]
        
        # Skip if no predictions
        if not horizon_data['ground_truth']:
            print(f"No valid predictions for horizon {horizon}min")
            continue
        
        api_1_pred = np.array(horizon_data['api_1_predictions'])
        api_2_pred = np.array(horizon_data['api_2_predictions'])
        ground_truth = np.array(horizon_data['ground_truth'])
        
        # Calculate differences between APIs
        api_diff = api_1_pred - api_2_pred
        api_1_error = api_1_pred - ground_truth
        api_2_error = api_2_pred - ground_truth
        
        # Calculate metrics
        mean_diff = np.mean(np.abs(api_diff))
        max_diff = np.max(np.abs(api_diff))
        
        api_1_mae = np.mean(np.abs(api_1_error))
        api_2_mae = np.mean(np.abs(api_2_error))
        
        api_1_rmse = np.sqrt(np.mean(np.square(api_1_error)))
        api_2_rmse = np.sqrt(np.mean(np.square(api_2_error)))
        
        # Store metrics
        comparison_results['predictions'][horizon]['metrics'] = {
            'mean_absolute_difference': float(mean_diff),
            'max_absolute_difference': float(max_diff),
            f'{api_1_name}_mae': float(api_1_mae),
            f'{api_2_name}_mae': float(api_2_mae),
            f'{api_1_name}_rmse': float(api_1_rmse),
            f'{api_2_name}_rmse': float(api_2_rmse)
        }
        
        # Print metric comparison
        print(f"\nHorizon: {horizon}min")
        print(f"Mean absolute difference between APIs: {mean_diff:.2f} mg/dL")
        print(f"Maximum absolute difference between APIs: {max_diff:.2f} mg/dL")
        print(f"{api_1_name} MAE: {api_1_mae:.2f} mg/dL, RMSE: {api_1_rmse:.2f} mg/dL")
        print(f"{api_2_name} MAE: {api_2_mae:.2f} mg/dL, RMSE: {api_2_rmse:.2f} mg/dL")
        
        # Create comparison visualizations
        plt.figure(figsize=(18, 10))
        
        # Scatter plot of predictions vs ground truth
        plt.subplot(2, 2, 1)
        plt.scatter(ground_truth, api_1_pred, alpha=0.7, label=api_1_name)
        plt.scatter(ground_truth, api_2_pred, alpha=0.7, label=api_2_name)
        plt.plot([min(ground_truth), max(ground_truth)], 
                 [min(ground_truth), max(ground_truth)], 'k--')
        plt.xlabel('Ground Truth (mg/dL)')
        plt.ylabel('Predicted (mg/dL)')
        plt.title(f'Predictions vs Ground Truth - {horizon}min Horizon')
        plt.legend()
        
        # Histogram of differences
        plt.subplot(2, 2, 2)
        plt.hist(api_diff, bins=20, alpha=0.7)
        plt.axvline(x=0, color='k', linestyle='--')
        plt.xlabel(f'Difference: {api_1_name} - {api_2_name} (mg/dL)')
        plt.ylabel('Frequency')
        plt.title(f'Histogram of API Differences - Mean Abs Diff: {mean_diff:.2f} mg/dL')
        
        # Error comparison
        plt.subplot(2, 2, 3)
        plt.scatter(api_1_error, api_2_error, alpha=0.7)
        plt.axhline(y=0, color='k', linestyle='--')
        plt.axvline(x=0, color='k', linestyle='--')
        plt.xlabel(f'{api_1_name} Error (mg/dL)')
        plt.ylabel(f'{api_2_name} Error (mg/dL)')
        plt.title('Error Comparison')
        
        # Time series view of a few predictions
        plt.subplot(2, 2, 4)
        sample_size = min(30, len(ground_truth))
        plt.plot(range(sample_size), ground_truth[:sample_size], label='Ground Truth')
        plt.plot(range(sample_size), api_1_pred[:sample_size], label=api_1_name)
        plt.plot(range(sample_size), api_2_pred[:sample_size], label=api_2_name)
        plt.xlabel('Sample Index')
        plt.ylabel('Glucose (mg/dL)')
        plt.title(f'Time Series Comparison - {horizon}min Horizon')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'api_comparison_{patient_id}_{horizon}min.png'))
        plt.close()
    
    # Save comparison results
    with open(os.path.join(output_dir, f'api_comparison_results_{patient_id}.json'), 'w') as f:
        json.dump(comparison_results, f, indent=2)
    
    return comparison_results

def main():
    parser = argparse.ArgumentParser(description='Compare Glucose Prediction APIs')
    parser.add_argument('--test_file', required=True, help='Path to test data CSV file')
    parser.add_argument('--patient_id', type=int, required=True, help='Patient ID to compare')
    parser.add_argument('--horizons', default='15,30', help='Comma-separated list of prediction horizons (minutes)')
    parser.add_argument('--api_url_1', required=True, help='Base URL of the first API')
    parser.add_argument('--api_url_2', required=True, help='Base URL of the second API')
    parser.add_argument('--api_1_name', default='API_1', help='Name of the first API')
    parser.add_argument('--api_2_name', default='API_2', help='Name of the second API')
    parser.add_argument('--output_dir', default='api_comparison_results', help='Directory to save results')
    
    args = parser.parse_args()
    
    # Parse horizons
    horizons = [int(h.strip()) for h in args.horizons.split(',')]
    
    # Run comparison
    compare_apis(
        test_file=args.test_file,
        patient_id=args.patient_id,
        horizons=horizons,
        api_url_1=args.api_url_1,
        api_url_2=args.api_url_2,
        api_1_name=args.api_1_name,
        api_2_name=args.api_2_name,
        output_dir=args.output_dir
    )
    
    print(f"Comparison results saved to {args.output_dir}")

if __name__ == "__main__":
    main() 