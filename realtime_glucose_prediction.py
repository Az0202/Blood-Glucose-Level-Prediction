#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse
import numpy as np
import pandas as pd
import joblib
import json
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import time
from sklearn.preprocessing import StandardScaler
import warnings
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import math
warnings.filterwarnings('ignore')

class RealTimeGlucosePredictor:
    """Class for making real-time glucose predictions using trained models"""
    
    def __init__(self, models_dir, patient_id, horizons=None):
        """
        Initialize the predictor with models for different prediction horizons.
        
        Args:
            models_dir: Directory containing saved models
            patient_id: ID of the patient for prediction
            horizons: List of prediction horizons in minutes (default: [15, 30, 45, 60])
        """
        self.models_dir = models_dir
        self.patient_id = patient_id
        self.horizons = horizons if horizons is not None else [15, 30, 45, 60]
        self.models = {}
        self.features = {}
        self.scalers_X = {}
        self.scalers_y = {}
        
        # Load models, feature lists, and scalers
        self.load_models()
        
        # History for plotting
        self.history = {
            'timestamps': [],
            'actual_values': [],
            'predictions': {horizon: [] for horizon in self.horizons}
        }
    
    def load_models(self):
        """Load the trained models and their associated metadata."""
        patient_dir = os.path.join(self.models_dir, f"patient_{self.patient_id}")
        
        if not os.path.exists(patient_dir):
            print(f"Warning: Patient directory {patient_dir} does not exist")
            return
        
        for horizon in self.horizons:
            # Check for available model types (RF, GB, Stacking, Voting)
            model_found = False
            
            # Try to load models in preference order (best to worst)
            for model_prefix in ['stacking', 'voting', 'gb', 'rf']:
                model_path = os.path.join(patient_dir, f"{model_prefix}_model_{horizon}min.joblib")
                if os.path.exists(model_path):
                    self.models[horizon] = joblib.load(model_path)
                    print(f"Loaded {model_prefix} model for {horizon}min prediction")
                    
                    # Load feature list for this model
                    features_path = os.path.join(patient_dir, f"{model_prefix}_features_{horizon}min.json")
                    if os.path.exists(features_path):
                        with open(features_path, 'r') as f:
                            self.features[horizon] = json.load(f)
                        print(f"Loaded {len(self.features[horizon])} features for {horizon}min prediction")
                    else:
                        print(f"Warning: Features file {features_path} not found")
                    
                    # Model found for this horizon, no need to check others
                    model_found = True
                    break
            
            if not model_found:
                print(f"Warning: No model found for {horizon} minute horizon")
    
    def add_enhanced_features(self, df):
        """
        Add enhanced features to the dataframe for better prediction.
        
        Args:
            df: DataFrame with glucose data
            
        Returns:
            DataFrame with added features
        """
        # Make a copy to avoid modifying the original
        df_enhanced = df.copy()
        
        # Convert timestamp to datetime if it's not already
        if 'timestamp' in df_enhanced.columns and not pd.api.types.is_datetime64_dtype(df_enhanced['timestamp']):
            try:
                df_enhanced['timestamp'] = pd.to_datetime(df_enhanced['timestamp'])
            except:
                print("Warning: Could not convert timestamp to datetime")
        
        # Add time-based features if not already present
        if 'minute' not in df_enhanced.columns and 'timestamp' in df_enhanced.columns:
            df_enhanced['minute'] = df_enhanced['timestamp'].dt.minute
        
        if 'hour' not in df_enhanced.columns and 'timestamp' in df_enhanced.columns:
            df_enhanced['hour'] = df_enhanced['timestamp'].dt.hour
        
        if 'day_of_week' not in df_enhanced.columns and 'timestamp' in df_enhanced.columns:
            df_enhanced['day_of_week'] = df_enhanced['timestamp'].dt.dayofweek
        
        # Add cyclical time features
        if 'time_sin' not in df_enhanced.columns and 'hour' in df_enhanced.columns and 'minute' in df_enhanced.columns:
            # Time of day in minutes
            minutes_in_day = df_enhanced['hour'] * 60 + df_enhanced['minute']
            df_enhanced['time_sin'] = np.sin(2 * np.pi * minutes_in_day / 1440)
            df_enhanced['time_cos'] = np.cos(2 * np.pi * minutes_in_day / 1440)
        
        if 'day_sin' not in df_enhanced.columns and 'day_of_week' in df_enhanced.columns:
            # Day of week (0-6)
            df_enhanced['day_sin'] = np.sin(2 * np.pi * df_enhanced['day_of_week'] / 7)
            df_enhanced['day_cos'] = np.cos(2 * np.pi * df_enhanced['day_of_week'] / 7)
        
        # Calculate glucose velocity and acceleration if they don't exist
        if 'glucose_velocity' not in df_enhanced.columns and 'glucose_value' in df_enhanced.columns:
            # Calculate velocity (rate of change) - mg/dL per 5 minutes
            df_enhanced['glucose_diff'] = df_enhanced['glucose_value'].diff()
            df_enhanced['glucose_velocity'] = df_enhanced['glucose_diff']
            
            # Calculate acceleration (change in velocity)
            df_enhanced['glucose_acceleration'] = df_enhanced['glucose_velocity'].diff()
            
            # Fill NaN values
            df_enhanced['glucose_velocity'] = df_enhanced['glucose_velocity'].fillna(0)
            df_enhanced['glucose_acceleration'] = df_enhanced['glucose_acceleration'].fillna(0)
        
        # Calculate glucose statistics for different windows if not already present
        if 'glucose_mean_3' not in df_enhanced.columns and 'glucose_value' in df_enhanced.columns:
            for window in [3, 6, 12, 24]:  # Hours converted to number of readings (assuming 5-min intervals)
                window_size = min(int(window * 60 / 5), len(df_enhanced))  # Convert hours to number of 5-minute readings, but cap at dataframe length
                if window_size < 1:
                    window_size = 1
                
                # Rolling statistics
                df_enhanced[f'glucose_mean_{window}'] = df_enhanced['glucose_value'].rolling(window=window_size, min_periods=1).mean()
                df_enhanced[f'glucose_std_{window}'] = df_enhanced['glucose_value'].rolling(window=window_size, min_periods=1).std()
                df_enhanced[f'glucose_min_{window}'] = df_enhanced['glucose_value'].rolling(window=window_size, min_periods=1).min()
                df_enhanced[f'glucose_max_{window}'] = df_enhanced['glucose_value'].rolling(window=window_size, min_periods=1).max()
                
                # Fill NaN values
                df_enhanced[f'glucose_mean_{window}'] = df_enhanced[f'glucose_mean_{window}'].fillna(df_enhanced['glucose_value'])
                df_enhanced[f'glucose_std_{window}'] = df_enhanced[f'glucose_std_{window}'].fillna(0)
                df_enhanced[f'glucose_min_{window}'] = df_enhanced[f'glucose_min_{window}'].fillna(df_enhanced['glucose_value'])
                df_enhanced[f'glucose_max_{window}'] = df_enhanced[f'glucose_max_{window}'].fillna(df_enhanced['glucose_value'])
        
        # Create lagged features
        if 'glucose_lag_1' not in df_enhanced.columns and 'glucose_value' in df_enhanced.columns:
            # Limit lags to available data
            max_lags = min(12, len(df_enhanced) - 1)
            for i in range(1, max_lags + 1):  # Create up to 12 lag features
                df_enhanced[f'glucose_lag_{i}'] = df_enhanced['glucose_value'].shift(i).fillna(df_enhanced['glucose_value'].iloc[0] if len(df_enhanced) > 0 else 0)
        
        # Set default values for insulin and carb features if not present
        for feature in ['insulin_dose', 'insulin_dose_1h', 'insulin_dose_2h', 'insulin_dose_4h', 
                       'carbs_1h', 'carbs_2h', 'carbs_4h']:
            if feature not in df_enhanced.columns:
                df_enhanced[feature] = 0
        
        # Fill NaN values
        df_enhanced = df_enhanced.fillna(0)
        
        return df_enhanced
    
    def prepare_input_data(self, data, horizon):
        """
        Prepare input data for prediction.
        
        Args:
            data: DataFrame with recent glucose data
            horizon: Prediction horizon in minutes
            
        Returns:
            Processed input data ready for model prediction
        """
        if horizon not in self.features:
            print(f"No feature list found for {horizon}min prediction")
            return None
        
        # Get required features for this horizon
        required_features = self.features[horizon]
        print(f"Required features for {horizon}min prediction: {required_features[:5]}...")
        
        # Check if all required features are available
        missing_features = [f for f in required_features if f not in data.columns]
        if missing_features:
            print(f"Warning: Missing {len(missing_features)} features for {horizon}min prediction")
            print(f"First few missing: {missing_features[:5]}")
            
            # Add missing features with zeros - this is a fallback
            for feature in missing_features:
                data[feature] = 0
        
        # Select only the needed features in the right order
        X = data[required_features].copy()
        
        # Check for NaN values and replace them
        if X.isna().any().any():
            print(f"Warning: Input data contains NaN values. Filling with zeros.")
            X = X.fillna(0)
        
        # Return the last row as input for the model
        if len(X) > 0:
            return X.tail(1)
        else:
            print("Error: No valid input data available after preprocessing")
            return None
    
    def make_prediction(self, current_data, horizon):
        """
        Make glucose prediction for a specific horizon.
        
        Args:
            current_data: DataFrame with recent glucose data
            horizon: Prediction horizon in minutes
            
        Returns:
            Predicted glucose value
        """
        if horizon not in self.models:
            print(f"No model available for {horizon}min prediction")
            return None
        
        # Prepare input data
        X = self.prepare_input_data(current_data, horizon)
        if X is None:
            return None
        
        try:
            # Make prediction
            prediction = self.models[horizon].predict(X)
            
            # Extract single value from prediction if it's an array
            if isinstance(prediction, np.ndarray) and len(prediction) > 0:
                prediction = prediction[0]
            
            # Print prediction for logging
            print(f"Predicted glucose at {horizon}min: {prediction:.1f} mg/dL")
            
            return prediction
        
        except Exception as e:
            print(f"Error making prediction for {horizon}min: {e}")
            return None
    
    def visualize_predictions(self, timestamps, actual_values, predictions_dict, output_dir=None):
        """
        Visualize actual glucose values and predictions.
        
        Args:
            timestamps: List of datetime objects
            actual_values: List of actual glucose values
            predictions_dict: Dictionary of predictions for each horizon
            output_dir: Directory to save visualization (optional)
        """
        plt.figure(figsize=(12, 8))
        
        # Convert timestamps to datetime if they're not already
        datetime_timestamps = []
        for ts in timestamps:
            if isinstance(ts, str):
                try:
                    datetime_timestamps.append(pd.to_datetime(ts))
                except:
                    datetime_timestamps.append(None)
            else:
                datetime_timestamps.append(ts)
        
        # Plot actual values
        plt.plot(datetime_timestamps, actual_values, 'b-', label='Actual Glucose')
        
        # Plot predictions for each horizon
        colors = ['g-', 'r-', 'c-', 'm-']
        for i, (horizon, predictions) in enumerate(predictions_dict.items()):
            # Filter out None values
            valid_indices = [j for j, p in enumerate(predictions) if p is not None]
            if valid_indices:
                valid_timestamps = [datetime_timestamps[j] for j in valid_indices]
                valid_predictions = [predictions[j] for j in valid_indices]
                plt.plot(valid_timestamps, valid_predictions, colors[i % len(colors)], 
                         label=f'{horizon}min Prediction')
        
        # Add clinical ranges
        plt.axhspan(70, 180, alpha=0.2, color='green', label='Target Range (70-180 mg/dL)')
        plt.axhspan(0, 70, alpha=0.2, color='red', label='Hypoglycemia (<70 mg/dL)')
        plt.axhspan(180, 400, alpha=0.2, color='orange', label='Hyperglycemia (>180 mg/dL)')
        
        plt.title(f'Glucose Predictions for Patient {self.patient_id}')
        plt.xlabel('Time')
        plt.ylabel('Glucose (mg/dL)')
        plt.legend()
        plt.grid(True)
        
        # Format x-axis with dates
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
        plt.gcf().autofmt_xdate()
        
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            plt.savefig(os.path.join(output_dir, f'patient_{self.patient_id}_predictions.png'))
        
        plt.close()
    
    def evaluate_predictions(self, timestamps, actual_values, predictions_dict, output_dir=None):
        """
        Evaluate prediction accuracy for each horizon.
        
        Args:
            timestamps: List of datetime objects
            actual_values: List of actual glucose values
            predictions_dict: Dictionary of predictions for each horizon
            output_dir: Directory to save evaluation results (optional)
            
        Returns:
            Dictionary with evaluation metrics for each horizon
        """
        results = {}
        
        for horizon, predictions in predictions_dict.items():
            # Account for the prediction offset
            prediction_offset = horizon // 5  # Convert minutes to 5-min intervals
            
            # Get predictions and corresponding actual values
            if prediction_offset > 0 and len(actual_values) > prediction_offset:
                aligned_actual = actual_values[prediction_offset:]
                aligned_predictions = predictions[:-prediction_offset] if prediction_offset > 0 else predictions
            else:
                aligned_actual = actual_values
                aligned_predictions = predictions
            
            # Truncate to the same length
            min_len = min(len(aligned_actual), len(aligned_predictions))
            aligned_actual = aligned_actual[:min_len]
            aligned_predictions = aligned_predictions[:min_len]
            
            # Filter out None values
            valid_indices = [i for i, p in enumerate(aligned_predictions) if p is not None]
            if valid_indices:
                valid_actual = [aligned_actual[i] for i in valid_indices]
                valid_predictions = [aligned_predictions[i] for i in valid_indices]
                
                # Calculate metrics
                metrics = {}
                metrics['rmse'] = np.sqrt(mean_squared_error(valid_actual, valid_predictions))
                metrics['mae'] = mean_absolute_error(valid_actual, valid_predictions)
                metrics['r2'] = r2_score(valid_actual, valid_predictions)
                
                results[horizon] = metrics
                
                print(f"{horizon}min Prediction - RMSE: {metrics['rmse']:.2f}, MAE: {metrics['mae']:.2f}, R²: {metrics['r2']:.4f}")
        
        # Save results if output directory is provided
        if output_dir and results:
            os.makedirs(output_dir, exist_ok=True)
            with open(os.path.join(output_dir, f'patient_{self.patient_id}_metrics.json'), 'w') as f:
                json.dump(results, f, indent=4)
        
        return results


def simulate_realtime_prediction(test_data, models_dir, patient_id, horizons=None, 
                               output_dir=None, update_interval=5, duration=60):
    """
    Simulate real-time prediction using historical test data.
    
    Args:
        test_data: Path to test data CSV file
        models_dir: Directory containing saved models
        patient_id: ID of the patient for prediction
        horizons: List of prediction horizons in minutes (default: [15, 30, 45, 60])
        output_dir: Directory to save results (optional)
        update_interval: Interval between updates in seconds (default: 5)
        duration: Duration of simulation in minutes (default: 60)
    """
    # Load test data
    print(f"Loading test data from {test_data}")
    df = pd.read_csv(test_data)
    
    # Print columns to debug
    print(f"Available columns in test data: {df.columns.tolist()}")
    
    # Filter by patient if needed
    if 'patient_id' in df.columns:
        df = df[df['patient_id'] == int(patient_id)].reset_index(drop=True)
        print(f"Filtered data for patient {patient_id}: {len(df)} records")
    
    # Handle timestamp conversion properly
    if 'timestamp' in df.columns:
        print(f"Timestamp data type before conversion: {df['timestamp'].dtype}")
        print(f"First few timestamps: {df['timestamp'].head().tolist()}")
        
        try:
            # First try if it's already a datetime string
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        except:
            try:
                # Try to convert from Unix timestamp (seconds since epoch)
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
                print("Converted Unix timestamp (seconds) to datetime")
            except:
                try:
                    # Try to convert from Unix timestamp (milliseconds since epoch)
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                    print("Converted Unix timestamp (milliseconds) to datetime")
                except Exception as e:
                    print(f"Failed to convert timestamps: {e}")
                    # Create a new timestamp column
                    start_date = datetime.now() - timedelta(days=len(df) // 288)
                    df['timestamp'] = [start_date + timedelta(minutes=i*5) for i in range(len(df))]
                    print("Created artificial timestamps")
        
        print(f"Timestamp data type after conversion: {df['timestamp'].dtype}")
        print(f"First few converted timestamps: {df['timestamp'].head().tolist()}")
    else:
        # If no timestamp column exists, create one
        start_date = datetime.now() - timedelta(days=len(df) // 288)
        df['timestamp'] = [start_date + timedelta(minutes=i*5) for i in range(len(df))]
        print("Created artificial timestamps")
    
    # Initialize predictor
    predictor = RealTimeGlucosePredictor(models_dir, patient_id, horizons)
    
    # Add enhanced features
    print("Preparing features...")
    df = predictor.add_enhanced_features(df)
    print(f"Features prepared. Data shape: {df.shape}")
    
    # Print sample of prepared data for debugging
    print("\nSample of prepared data:")
    print(df[['timestamp', 'glucose_value'] + [col for col in df.columns if col.startswith('glucose_lag_')][:3]].head())
    
    # Prepare data structures for storing results
    num_samples = min(int(duration * 60 / update_interval), len(df))
    timestamps = []
    actual_values = []
    predictions = {h: [] for h in horizons}
    
    print(f"\nStarting real-time prediction simulation for patient {patient_id}")
    print(f"Prediction horizons: {horizons} minutes")
    print(f"Simulating {num_samples} time points\n")
    
    # Simulate real-time prediction
    for i in range(num_samples):
        # Get current data window
        current_data = df.iloc[:i+1]
        
        # Get current glucose and timestamp
        if 'glucose_value' in current_data.columns:
            current_glucose = current_data['glucose_value'].iloc[-1]
        else:
            current_glucose = None
        
        if 'timestamp' in current_data.columns:
            current_time = current_data['timestamp'].iloc[-1]
        else:
            # Use current time if timestamp not available
            current_time = datetime.now() + timedelta(minutes=i*5)
        
        timestamps.append(current_time)
        actual_values.append(current_glucose)
        
        # Make predictions for each horizon
        for horizon in horizons:
            prediction = predictor.make_prediction(current_data, horizon)
            predictions[horizon].append(prediction)
        
        # Print current status
        pred_str = " | ".join([f"{h}min: {predictions[h][-1]:.1f} mg/dL" if predictions[h][-1] is not None else f"{h}min: NA" for h in horizons])
        print(f"Time: {current_time.strftime('%Y-%m-%d %H:%M:%S')} | Current Glucose: {current_glucose:.1f} mg/dL | {pred_str}")
        
        # Sleep to simulate real-time updates
        time.sleep(0.01)  # Use a small value for faster simulation
    
    print("\nSimulation complete. Evaluating predictions...")
    
    # Visualize predictions
    if output_dir:
        predictor.visualize_predictions(timestamps, actual_values, predictions, output_dir)
        print(f"Prediction visualization saved to {output_dir}")
    
    # Evaluate predictions
    evaluation = predictor.evaluate_predictions(timestamps, actual_values, predictions, output_dir)
    
    if output_dir:
        print(f"Evaluation metrics saved to {output_dir}")
    
    return timestamps, actual_values, predictions, evaluation


def main():
    parser = argparse.ArgumentParser(description='Real-time glucose prediction')
    parser.add_argument('--test_file', required=True, help='Path to test data file')
    parser.add_argument('--models_dir', required=True, help='Directory containing trained models')
    parser.add_argument('--patient_id', required=True, help='Patient ID for prediction')
    parser.add_argument('--horizons', default='15,30', help='Comma-separated prediction horizons in minutes')
    parser.add_argument('--output_dir', default='realtime_predictions', help='Directory to save results')
    parser.add_argument('--duration', type=int, default=60, help='Duration of simulation in minutes')
    parser.add_argument('--update_interval', type=int, default=5, help='Interval between updates in seconds')
    
    args = parser.parse_args()
    
    # Parse horizons
    horizons = [int(h) for h in args.horizons.split(',')]
    
    # Run simulation
    simulate_realtime_prediction(
        args.test_file,
        args.models_dir,
        args.patient_id,
        horizons=horizons,
        output_dir=args.output_dir,
        update_interval=args.update_interval,
        duration=args.duration
    )

if __name__ == "__main__":
    main() 