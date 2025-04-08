#!/usr/bin/env python
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import argparse
import os
import json
from datetime import datetime
import time
from tqdm import tqdm
import sys

# Set up styling for plots
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)

def format_elapsed_time(seconds):
    """Format elapsed time in a human-readable format"""
    if seconds < 60:
        return f"{seconds:.2f} seconds"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.2f} minutes"
    else:
        hours = seconds / 3600
        return f"{hours:.2f} hours"

def clarke_error_grid(reference, prediction):
    """
    Implement the Clarke Error Grid Analysis
    Returns the percentage of points in each zone (A through E)
    """
    # Create zones
    zones = {
        'A': 0,
        'B': 0,
        'C': 0,
        'D': 0,
        'E': 0
    }
    
    total_points = len(reference)
    
    for i in range(total_points):
        bg_ref = reference[i]
        bg_pred = prediction[i]
        
        # Zone A: Clinically accurate
        if (bg_ref <= 70 and bg_pred <= 70) or \
           (bg_pred <= 1.2 * bg_ref and bg_pred >= 0.8 * bg_ref):
            zones['A'] += 1
        
        # Zone B: Benign errors
        elif (bg_ref >= 180 and bg_pred <= 70) or \
             (bg_ref <= 70 and bg_pred >= 180):
            zones['E'] += 1
        elif ((bg_pred > bg_ref + 110) and (bg_ref < 70)) or \
             ((bg_pred < bg_ref - 110) and (bg_ref > 180)):
            zones['D'] += 1
        elif (((bg_ref >= 70) and (bg_ref <= 290)) and \
             ((bg_pred >= 0) and (bg_pred <= 70))) or \
             (((bg_ref >= 130) and (bg_ref <= 180)) and \
             ((bg_pred > 180) and (bg_pred <= 300))):
            zones['C'] += 1
        else:
            zones['B'] += 1
    
    # Convert to percentages
    for zone in zones:
        zones[zone] = (zones[zone] / total_points) * 100
    
    return zones

def plot_clarke_error_grid(reference, prediction, patient_id, horizon, output_dir):
    """Create Clarke Error Grid visualization"""
    plt.figure(figsize=(10, 10))
    
    # Plot the data
    plt.scatter(reference, prediction, c='blue', alpha=0.5)
    
    # Draw grid lines
    plt.plot([0, 400], [0, 400], 'k--')
    plt.plot([0, 175], [70, 70], 'k-')
    plt.plot([70, 70], [0, 175], 'k-')
    plt.plot([0, 70], [180, 180], 'k-')
    plt.plot([180, 180], [0, 70], 'k-')
    plt.plot([70, 400], [70, 400 * 0.8], 'k-')
    plt.plot([70, 400], [70, 400 * 1.2], 'k-')
    
    # Zone labels
    plt.text(50, 100, 'A', fontsize=15)
    plt.text(100, 50, 'B', fontsize=15)
    plt.text(50, 200, 'C', fontsize=15)
    plt.text(200, 50, 'D', fontsize=15)
    plt.text(200, 10, 'E', fontsize=15)
    
    # Calculate zone percentages
    zones = clarke_error_grid(reference, prediction)
    
    # Add zone percentages to the plot
    plt.text(300, 380, f"Zone A: {zones['A']:.1f}%", fontsize=12)
    plt.text(300, 360, f"Zone B: {zones['B']:.1f}%", fontsize=12)
    plt.text(300, 340, f"Zone C: {zones['C']:.1f}%", fontsize=12)
    plt.text(300, 320, f"Zone D: {zones['D']:.1f}%", fontsize=12)
    plt.text(300, 300, f"Zone E: {zones['E']:.1f}%", fontsize=12)
    
    plt.title(f'Clarke Error Grid - Patient {patient_id} - {horizon}min Predictions')
    plt.xlabel('Reference Glucose (mg/dL)')
    plt.ylabel('Predicted Glucose (mg/dL)')
    plt.xlim(0, 400)
    plt.ylim(0, 400)
    plt.grid(True)
    
    # Save the plot
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, f'clarke_grid_patient_{patient_id}_{horizon}min.png'))
    plt.close()
    
    return zones

def calculate_in_range_percentage(actual, predicted, range_min=70, range_max=180):
    """Calculate the percentage of predictions that correctly identify in/out of range"""
    actual_in_range = (actual >= range_min) & (actual <= range_max)
    predicted_in_range = (predicted >= range_min) & (predicted <= range_max)
    
    # Count correct classifications
    correct = (actual_in_range & predicted_in_range) | (~actual_in_range & ~predicted_in_range)
    
    return (np.sum(correct) / len(actual)) * 100

def make_api_prediction(base_url, input_data):
    """Make a prediction using the API"""
    try:
        response = requests.post(f"{base_url}/predict", json=input_data, timeout=5)
        return response.json()
    except Exception as e:
        print(f"Error making prediction: {e}")
        return None

def validate_patient(patient_id, test_file, horizons, base_url, output_dir):
    """Validate predictions for a specific patient"""
    # Load test data
    try:
        test_df = pd.read_csv(test_file)
    except Exception as e:
        print(f"Error loading test file: {e}")
        return None
    
    # Filter to patient data
    patient_data = test_df[test_df['patient_id'] == patient_id].copy()
    
    if len(patient_data) == 0:
        print(f"No data found for patient {patient_id}")
        return None
    
    print(f"Validating predictions for patient {patient_id} - {len(patient_data)} data points")
    
    # Results storage
    validation_results = {
        'patient_id': patient_id,
        'metrics': {},
        'timing': {},
        'validation_count': 0
    }
    
    # Create patient output directory
    patient_dir = os.path.join(output_dir, f'patient_{patient_id}')
    os.makedirs(patient_dir, exist_ok=True)
    
    # Process a sample of the data (to avoid too many API calls)
    # Use stride to sample evenly throughout the dataset
    sample_size = min(100, len(patient_data))
    stride = max(1, len(patient_data) // sample_size)
    sample_indices = list(range(0, len(patient_data), stride))[:sample_size]
    
    # Initialize prediction storage
    all_predictions = {}
    all_actual = {}
    for horizon in horizons:
        all_predictions[horizon] = []
        all_actual[horizon] = []
    
    # Track API call timing
    api_times = []
    
    # Process each sample point
    progress_bar = tqdm(sample_indices, desc=f"Processing patient {patient_id}")
    for idx in progress_bar:
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
        
        # Time API call
        start_time = time.time()
        prediction_result = make_api_prediction(base_url, input_data)
        end_time = time.time()
        api_times.append(end_time - start_time)
        
        # Process prediction result
        if prediction_result and 'predictions' in prediction_result:
            for horizon in horizons:
                horizon_key = f"{horizon}min"
                if horizon_key in prediction_result['predictions']:
                    # Store predictions
                    prediction = prediction_result['predictions'][horizon_key]
                    actual = row.get(f'target_{horizon}min')
                    if not pd.isna(actual) and prediction is not None:
                        all_predictions[horizon].append(prediction)
                        all_actual[horizon].append(actual)
    
    # Calculate metrics for each horizon
    for horizon in horizons:
        if not all_predictions[horizon]:
            print(f"No valid predictions for horizon {horizon}min")
            continue
        
        # Convert to numpy arrays
        predictions = np.array(all_predictions[horizon])
        actual = np.array(all_actual[horizon])
        
        # Calculate metrics
        mae = mean_absolute_error(actual, predictions)
        rmse = np.sqrt(mean_squared_error(actual, predictions))
        r2 = r2_score(actual, predictions)
        in_range_pct = calculate_in_range_percentage(actual, predictions)
        
        # Calculate Clarke Error Grid zones
        zones = plot_clarke_error_grid(actual, predictions, patient_id, horizon, patient_dir)
        
        # Store metrics
        validation_results['metrics'][horizon] = {
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'in_range_pct': in_range_pct,
            'clarke_grid': zones
        }
        
        # Print metrics
        print(f"\nPatient {patient_id} - {horizon}min Prediction Metrics:")
        print(f"  MAE: {mae:.2f} mg/dL")
        print(f"  RMSE: {rmse:.2f} mg/dL")
        print(f"  R²: {r2:.4f}")
        print(f"  In-Range Accuracy: {in_range_pct:.2f}%")
        print(f"  Clarke Grid: A={zones['A']:.1f}%, B={zones['B']:.1f}%, C={zones['C']:.1f}%, D={zones['D']:.1f}%, E={zones['E']:.1f}%")
        
        # Create visualization
        plt.figure(figsize=(15, 10))
        
        # Scatter plot
        plt.subplot(2, 2, 1)
        plt.scatter(actual, predictions, alpha=0.5)
        plt.plot([actual.min(), actual.max()], [actual.min(), actual.max()], 'r--')
        plt.xlabel('Actual Glucose (mg/dL)')
        plt.ylabel('Predicted Glucose (mg/dL)')
        plt.title(f'{horizon}min Prediction - Actual vs Predicted')
        
        # Histogram of errors
        plt.subplot(2, 2, 2)
        errors = predictions - actual
        plt.hist(errors, bins=20, alpha=0.7)
        plt.xlabel('Prediction Error (mg/dL)')
        plt.ylabel('Frequency')
        plt.title(f'Error Distribution - RMSE: {rmse:.2f}')
        
        # Time series sample
        plt.subplot(2, 2, 3)
        sample_size = min(30, len(actual))
        plt.plot(range(sample_size), actual[:sample_size], label='Actual')
        plt.plot(range(sample_size), predictions[:sample_size], label='Predicted')
        plt.xlabel('Sample Index')
        plt.ylabel('Glucose (mg/dL)')
        plt.title(f'Time Series Sample - {horizon}min Horizon')
        plt.legend()
        
        # Boxplot of errors by glucose range
        plt.subplot(2, 2, 4)
        
        # Create glucose categories
        glucose_categories = []
        for a, error in zip(actual, errors):
            if a < 70:
                glucose_categories.append('Hypo (<70)')
            elif a <= 180:
                glucose_categories.append('Normal (70-180)')
            else:
                glucose_categories.append('Hyper (>180)')
        
        error_df = pd.DataFrame({
            'Category': glucose_categories,
            'Error': errors
        })
        
        sns.boxplot(x='Category', y='Error', data=error_df)
        plt.axhline(y=0, color='r', linestyle='-')
        plt.title('Prediction Errors by Glucose Category')
        plt.ylabel('Error (mg/dL)')
        
        plt.tight_layout()
        plt.savefig(os.path.join(patient_dir, f'prediction_analysis_{horizon}min.png'))
        plt.close()
    
    # Store API timing statistics
    validation_results['timing'] = {
        'mean_api_time': np.mean(api_times),
        'median_api_time': np.median(api_times),
        'min_api_time': np.min(api_times),
        'max_api_time': np.max(api_times),
        'total_api_time': np.sum(api_times)
    }
    
    validation_results['validation_count'] = len(api_times)
    
    # Save validation results
    with open(os.path.join(patient_dir, 'validation_results.json'), 'w') as f:
        json.dump(validation_results, f, indent=2)
    
    return validation_results

def summarize_results(all_results, output_dir):
    """Create summary of all validation results"""
    # Create summary DataFrame
    summary_rows = []
    
    for patient_id, result in all_results.items():
        if not result or 'metrics' not in result:
            continue
        
        for horizon, metrics in result['metrics'].items():
            row = {
                'patient_id': patient_id,
                'horizon': horizon,
                'mae': metrics['mae'],
                'rmse': metrics['rmse'],
                'r2': metrics['r2'],
                'in_range_pct': metrics['in_range_pct'],
                'clarke_a': metrics['clarke_grid']['A'],
                'clarke_b': metrics['clarke_grid']['B'],
                'validation_count': result['validation_count'],
                'mean_api_time': result['timing']['mean_api_time']
            }
            summary_rows.append(row)
    
    if not summary_rows:
        print("No valid results to summarize")
        return
    
    summary_df = pd.DataFrame(summary_rows)
    
    # Save summary to CSV
    summary_df.to_csv(os.path.join(output_dir, 'validation_summary.csv'), index=False)
    
    # Create summary visualizations
    
    # RMSE by horizon
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='horizon', y='rmse', data=summary_df)
    plt.title('RMSE by Prediction Horizon')
    plt.xlabel('Prediction Horizon (minutes)')
    plt.ylabel('RMSE (mg/dL)')
    plt.savefig(os.path.join(output_dir, 'rmse_by_horizon.png'))
    plt.close()
    
    # Clarke Zone A by horizon
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='horizon', y='clarke_a', data=summary_df)
    plt.title('Clarke Zone A Percentage by Prediction Horizon')
    plt.xlabel('Prediction Horizon (minutes)')
    plt.ylabel('Clarke Zone A (%)')
    plt.savefig(os.path.join(output_dir, 'clarke_a_by_horizon.png'))
    plt.close()
    
    # In-range percentage by horizon
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='horizon', y='in_range_pct', data=summary_df)
    plt.title('In-Range Accuracy by Prediction Horizon')
    plt.xlabel('Prediction Horizon (minutes)')
    plt.ylabel('In-Range Accuracy (%)')
    plt.savefig(os.path.join(output_dir, 'in_range_by_horizon.png'))
    plt.close()
    
    # Print overall summary
    print("\n===== VALIDATION SUMMARY =====")
    for horizon in summary_df['horizon'].unique():
        horizon_df = summary_df[summary_df['horizon'] == horizon]
        print(f"\n{horizon}min Horizon Summary:")
        print(f"  Mean RMSE: {horizon_df['rmse'].mean():.2f} mg/dL (± {horizon_df['rmse'].std():.2f})")
        print(f"  Mean MAE: {horizon_df['mae'].mean():.2f} mg/dL (± {horizon_df['mae'].std():.2f})")
        print(f"  Mean R²: {horizon_df['r2'].mean():.4f} (± {horizon_df['r2'].std():.4f})")
        print(f"  Mean In-Range Accuracy: {horizon_df['in_range_pct'].mean():.2f}% (± {horizon_df['in_range_pct'].std():.2f})")
        print(f"  Mean Clarke Zone A: {horizon_df['clarke_a'].mean():.2f}% (± {horizon_df['clarke_a'].std():.2f})")
    
    print(f"\nAPI Performance:")
    print(f"  Mean Response Time: {summary_df['mean_api_time'].mean() * 1000:.2f} ms")
    print(f"  Total Validations: {summary_df['validation_count'].sum()}")
    
    return summary_df

def main():
    parser = argparse.ArgumentParser(description='Validate Glucose Prediction API')
    parser.add_argument('--test_file', required=True, help='Path to test data CSV file')
    parser.add_argument('--patient_ids', default='570', help='Comma-separated list of patient IDs to validate')
    parser.add_argument('--horizons', default='15,30', help='Comma-separated list of prediction horizons (minutes)')
    parser.add_argument('--api_url', default='http://localhost:8001', help='Base URL of the API')
    parser.add_argument('--output_dir', default='validation_results', help='Directory to save validation results')
    
    args = parser.parse_args()
    
    # Parse arguments
    patient_ids = [int(pid.strip()) for pid in args.patient_ids.split(',')]
    horizons = [int(h.strip()) for h in args.horizons.split(',')]
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Validating API {args.api_url} for patients {patient_ids} with horizons {horizons}")
    print(f"Using test file: {args.test_file}")
    print(f"Results will be saved to: {args.output_dir}")
    
    # Validate API is running
    try:
        response = requests.get(f"{args.api_url}/")
        api_info = response.json()
        print(f"Connected to API: {api_info.get('name', 'Unknown')} v{api_info.get('version', 'Unknown')}")
    except Exception as e:
        print(f"Error connecting to API: {e}")
        print("Make sure the API is running before starting validation.")
        sys.exit(1)
    
    # Start validation
    all_results = {}
    start_time = time.time()
    
    for patient_id in patient_ids:
        patient_results = validate_patient(patient_id, args.test_file, horizons, args.api_url, args.output_dir)
        if patient_results:
            all_results[patient_id] = patient_results
    
    # Summarize results
    summary_df = summarize_results(all_results, args.output_dir)
    
    # Print timing information
    total_time = time.time() - start_time
    print(f"\nValidation completed in {format_elapsed_time(total_time)}")
    print(f"Results saved to {args.output_dir}")

if __name__ == "__main__":
    main() 