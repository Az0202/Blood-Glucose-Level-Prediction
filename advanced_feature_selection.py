#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse
import numpy as np
import pandas as pd
import joblib
import json
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.feature_selection import SelectFromModel, RFE, RFECV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.inspection import permutation_importance
import warnings
import logging
from sklearn.impute import SimpleImputer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def load_features(file_path):
    """Load features from a CSV file"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Features file not found: {file_path}")
    return pd.read_csv(file_path)

def add_physiological_features(df):
    """Add enhanced physiologically relevant features for glucose prediction."""
    logging.info("Adding enhanced physiological features...")
    
    df = df.copy()
    
    # Ensure timestamp is in datetime format
    if 'timestamp' in df.columns:
        if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
            df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Calculate glucose rate of change if historic glucose is available
    if 'historic_glucose' in df.columns:
        # First derivative (velocity)
        df['glucose_velocity'] = df.groupby('patient_id')['historic_glucose'].diff()
        
        # Second derivative (acceleration)
        df['glucose_acceleration'] = df.groupby('patient_id')['glucose_velocity'].diff()
        
        # Volatility (standard deviation in rolling windows)
        for window in [3, 6, 12]:
            df[f'glucose_std_{window}'] = df.groupby('patient_id')['historic_glucose'].rolling(
                window, min_periods=1).std().reset_index(0, drop=True)
            
        # Momentum indicators
        df['glucose_momentum'] = df.groupby('patient_id')['historic_glucose'].diff(periods=3)
        
        # Time-in-range indicators
        df['is_hypo'] = (df['historic_glucose'] < 70).astype(int)
        df['is_hyper'] = (df['historic_glucose'] > 180).astype(int)
        df['is_normal'] = ((df['historic_glucose'] >= 70) & (df['historic_glucose'] <= 180)).astype(int)
    
    # Time of day features
    if 'timestamp' in df.columns:
        df['hour_of_day'] = df['timestamp'].dt.hour
        df['is_overnight'] = ((df['hour_of_day'] >= 0) & (df['hour_of_day'] < 6)).astype(int)
        df['is_morning'] = ((df['hour_of_day'] >= 6) & (df['hour_of_day'] < 12)).astype(int)
        df['is_afternoon'] = ((df['hour_of_day'] >= 12) & (df['hour_of_day'] < 18)).astype(int)
        df['is_evening'] = ((df['hour_of_day'] >= 18) & (df['hour_of_day'] < 24)).astype(int)
    
    logging.info(f"Added physiological features. New shape: {df.shape}")
    return df

def select_features_for_patient(data, patient_id, horizon, n_features):
    """
    Select the most important features for a specific patient and prediction horizon.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        The input dataset containing features
    patient_id : int
        The patient ID to filter data for
    horizon : int
        Prediction horizon in minutes
    n_features : int
        Number of features to select
        
    Returns:
    --------
    selected_features : list
        List of selected feature names
    """
    logging.info(f"Starting feature selection for patient {patient_id}, horizon {horizon}min")
    
    # Filter data for this patient
    patient_data = data[data['patient_id'] == patient_id].copy()
    logging.info(f"Found {len(patient_data)} records for patient {patient_id}")
    
    if len(patient_data) == 0:
        logging.error(f"No data found for patient {patient_id}")
        return []
    
    # Define target column - use the correct naming convention
    target_col = f'target_{horizon}min'
    
    # Check if target column exists
    if target_col not in patient_data.columns:
        logging.error(f"Target column '{target_col}' not found in data")
        return []
    
    # Add physiological features
    patient_data = add_physiological_features(patient_data)
    
    # Drop rows with missing target values
    patient_data = patient_data.dropna(subset=[target_col])
    logging.info(f"After removing rows with missing targets: {len(patient_data)} records")
    
    if len(patient_data) == 0:
        logging.error("No data left after removing missing targets")
        return []
    
    # Exclude these columns from feature selection
    exclude_cols = ['patient_id', 'timestamp', 'datetime'] + [f'target_{h}min' for h in [5, 10, 15, 20, 25, 30, 45, 60, 90]]
    
    # Identify categorical and numerical columns
    categorical_cols = []
    numerical_cols = []
    
    for col in patient_data.columns:
        if col in exclude_cols or col == target_col:
            continue
            
        # Skip columns with too many missing values
        if patient_data[col].isna().mean() > 0.5:
            logging.warning(f"Skipping column '{col}' with >50% missing values")
            continue
            
        # Check if column is categorical
        if patient_data[col].dtype == 'object' or patient_data[col].nunique() < 10:
            # Try to convert to numeric
            try:
                pd.to_numeric(patient_data[col])
                numerical_cols.append(col)
            except:
                categorical_cols.append(col)
        else:
            numerical_cols.append(col)
    
    logging.info(f"Identified {len(numerical_cols)} numerical and {len(categorical_cols)} categorical features")
    
    # Create feature dataframe
    X = patient_data[numerical_cols].copy()
    
    # Handle categorical features
    if categorical_cols:
        # One-hot encode categorical features
        encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        cat_encoded = encoder.fit_transform(patient_data[categorical_cols])
        
        # Get encoded feature names
        cat_feature_names = []
        for i, col in enumerate(categorical_cols):
            for j, category in enumerate(encoder.categories_[i]):
                cat_feature_names.append(f"{col}_{category}")
        
        # Add encoded features to X
        X_cat = pd.DataFrame(cat_encoded, index=X.index, columns=cat_feature_names)
        X = pd.concat([X, X_cat], axis=1)
    
    logging.info(f"Final feature matrix shape: {X.shape}")
    
    # Handle missing values in features
    imputer = SimpleImputer(strategy='median')
    X_imputed = imputer.fit_transform(X)
    X = pd.DataFrame(X_imputed, columns=X.columns, index=X.index)
    
    # Get target values
    y = patient_data[target_col]
    
    # Apply feature selection methods
    selected_features = {}
    
    # 1. Random Forest Importance
    logging.info("Running Random Forest feature importance...")
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X, y)
    rf_importance = pd.Series(rf.feature_importances_, index=X.columns)
    rf_selected = rf_importance.nlargest(n_features).index.tolist()
    selected_features['rf'] = rf_selected
    
    # 2. Gradient Boosting Importance
    logging.info("Running Gradient Boosting feature importance...")
    gb = GradientBoostingRegressor(n_estimators=100, random_state=42)
    gb.fit(X, y)
    gb_importance = pd.Series(gb.feature_importances_, index=X.columns)
    gb_selected = gb_importance.nlargest(n_features).index.tolist()
    selected_features['gb'] = gb_selected
    
    # 3. Recursive Feature Elimination
    logging.info("Running Recursive Feature Elimination...")
    rfe = RFE(estimator=RandomForestRegressor(n_estimators=50, random_state=42), 
              n_features_to_select=n_features)
    rfe.fit(X, y)
    rfe_selected = [X.columns[i] for i in range(len(X.columns)) if rfe.support_[i]]
    selected_features['rfe'] = rfe_selected
    
    # Get consensus features
    feature_votes = {}
    for feature in X.columns:
        votes = sum([1 for method, features in selected_features.items() if feature in features])
        feature_votes[feature] = votes
    
    # Sort by votes and then by RF importance
    sorted_features = sorted(
        feature_votes.items(), 
        key=lambda x: (x[1], rf_importance.get(x[0], 0)), 
        reverse=True
    )
    
    # Take top n_features
    final_selected = [feature for feature, votes in sorted_features[:n_features]]
    
    logging.info(f"Selected {len(final_selected)} features. Top 5: {final_selected[:5]}")
    
    return final_selected

def main():
    parser = argparse.ArgumentParser(description='Patient-specific feature selection')
    parser.add_argument('--train_file', required=True, help='Path to training data CSV')
    parser.add_argument('--patient_id', required=True, type=int, help='Patient ID')
    parser.add_argument('--horizon', type=int, default=30, help='Prediction horizon in minutes')
    parser.add_argument('--n_features', type=int, default=20, help='Number of features to select')
    parser.add_argument('--output_dir', default='selected_features', help='Output directory')
    
    args = parser.parse_args()
    
    # Create output directory
    output_path = os.path.join(args.output_dir, f'patient_{args.patient_id}')
    os.makedirs(output_path, exist_ok=True)
    
    logging.info(f"Loading training data from {args.train_file}")
    try:
        data = pd.read_csv(args.train_file)
        logging.info(f"Loaded data with shape: {data.shape}")
        
        # Convert timestamp to datetime if present
        if 'timestamp' in data.columns:
            data['timestamp'] = pd.to_datetime(data['timestamp'])
    except Exception as e:
        logging.error(f"Error loading data: {str(e)}")
        return
    
    # Select features
    selected_features = select_features_for_patient(
        data, 
        args.patient_id, 
        args.horizon, 
        args.n_features
    )
    
    if selected_features:
        # Save selected features
        output_file = os.path.join(output_path, f'selected_features_{args.horizon}min.json')
        with open(output_file, 'w') as f:
            json.dump(selected_features, f, indent=2)
        
        logging.info(f"Selected features saved to {output_file}")
    else:
        logging.error("Feature selection failed - no features were selected")

if __name__ == '__main__':
    main() 