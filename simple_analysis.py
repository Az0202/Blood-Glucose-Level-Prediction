import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import argparse

def load_features(file_path):
    """Load CSV file into DataFrame"""
    if os.path.exists(file_path):
        return pd.read_csv(file_path)
    else:
        raise FileNotFoundError(f"Feature file not found: {file_path}")

def train_xgboost_model(X_train, y_train, num_boost_round=100):
    """Train a simple XGBoost model"""
    dtrain = xgb.DMatrix(X_train, label=y_train)
    params = {'objective': 'reg:squarederror', 'eval_metric': 'rmse'}
    model = xgb.train(params, dtrain, num_boost_round=num_boost_round)
    return model

def prepare_xgboost_data(df, target_column):
    """Prepare data for XGBoost model"""
    features_copy = df.copy()
    
    # Drop unnecessary columns
    drop_cols = ['timestamp', 'datetime', 'glucose_value']
    target_cols = [col for col in df.columns if col.startswith('target_') and col != target_column]
    all_drop_cols = drop_cols + target_cols
    drop_cols_exist = [col for col in all_drop_cols if col in features_copy.columns]
    
    # Get feature columns
    feature_cols = [col for col in features_copy.columns if col not in drop_cols_exist 
                   and col != target_column and col != 'patient_id']
    
    # Extract features and target
    X = features_copy[feature_cols].values
    y = features_copy[target_column].values
    
    return X, y, feature_cols

def analyze_horizon(train_df, test_df, horizon, output_dir):
    """Analyze a specific prediction horizon"""
    print(f"\n{'='*10} Analyzing {horizon}-minute prediction horizon {'='*10}")
    
    # Create output directory for this horizon
    horizon_dir = os.path.join(output_dir, f"{horizon}min")
    os.makedirs(horizon_dir, exist_ok=True)
    
    # Set target column
    target_column = f"target_{horizon}min"
    
    # Check if target column exists
    if target_column not in train_df.columns or target_column not in test_df.columns:
        print(f"Warning: Target column {target_column} not found in data!")
        return
    
    # Prepare data
    X_train, y_train, feature_cols = prepare_xgboost_data(train_df, target_column)
    X_test, y_test, _ = prepare_xgboost_data(test_df, target_column)
    
    # Train model
    print(f"Training XGBoost model for {horizon}-minute horizon...")
    model = train_xgboost_model(X_train, y_train)
    
    # Generate predictions
    dtest = xgb.DMatrix(X_test)
    y_pred = model.predict(dtest)
    
    # Calculate metrics
    metrics = {}
    metrics['mae'] = mean_absolute_error(y_test, y_pred)
    metrics['rmse'] = np.sqrt(mean_squared_error(y_test, y_pred))
    metrics['r2'] = r2_score(y_test, y_pred)
    
    print(f"MAE: {metrics['mae']:.2f}, RMSE: {metrics['rmse']:.2f}, R²: {metrics['r2']:.4f}")
    
    # Save metrics
    metrics_df = pd.DataFrame({
        'Metric': ['MAE', 'RMSE', 'R²'],
        'Value': [metrics['mae'], metrics['rmse'], metrics['r2']]
    })
    metrics_df.to_csv(os.path.join(horizon_dir, f'metrics_{horizon}min.csv'), index=False)
    
    # Create feature importance plot
    importance = model.get_score(importance_type='weight')
    importance_df = pd.DataFrame(
        {'Feature': list(importance.keys()), 'Importance': list(importance.values())}
    )
    importance_df = importance_df.sort_values('Importance', ascending=False)
    
    plt.figure(figsize=(12, 8))
    plt.barh(importance_df['Feature'][:20], importance_df['Importance'][:20])
    plt.title(f'Top 20 Feature Importances for {horizon}-minute Horizon', fontsize=16)
    plt.xlabel('Importance', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(horizon_dir, f'feature_importance_{horizon}min.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot predictions vs actual
    idx = np.random.choice(len(y_test), min(1000, len(y_test)), replace=False)
    
    plt.figure(figsize=(12, 8))
    plt.scatter(y_test[idx], y_pred[idx], alpha=0.5)
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--')
    plt.title(f'Predictions vs Actual for {horizon}-minute Horizon', fontsize=16)
    plt.xlabel('Actual Blood Glucose', fontsize=14)
    plt.ylabel('Predicted Blood Glucose', fontsize=14)
    plt.grid(True)
    plt.savefig(os.path.join(horizon_dir, f'pred_vs_actual_{horizon}min.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot error histogram
    errors = y_pred - y_test
    
    plt.figure(figsize=(12, 8))
    plt.hist(errors, bins=50, alpha=0.75)
    plt.axvline(x=0, color='r', linestyle='--')
    plt.title(f'Error Distribution for {horizon}-minute Horizon', fontsize=16)
    plt.xlabel('Prediction Error', fontsize=14)
    plt.ylabel('Frequency', fontsize=14)
    plt.grid(True)
    plt.savefig(os.path.join(horizon_dir, f'error_hist_{horizon}min.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    return metrics

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Simple analysis of blood glucose prediction models')
    parser.add_argument('--horizons', type=str, default='15,30,45,60,90',
                        help='Comma-separated list of prediction horizons in minutes')
    parser.add_argument('--output_dir', type=str, default='simple_analysis',
                        help='Directory to save analysis results')
    
    args = parser.parse_args()
    
    # Parse horizons
    horizons = [int(h) for h in args.horizons.split(',')]
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load data
    train_file = os.path.join("features", "2018_train_features.csv")
    test_file = os.path.join("features", "2018_test_features.csv")
    
    train_df = load_features(train_file)
    test_df = load_features(test_file)
    
    # Initialize summary
    summary = []
    
    # Analyze each horizon
    for horizon in horizons:
        metrics = analyze_horizon(train_df, test_df, horizon, args.output_dir)
        if metrics:
            summary.append({
                'Horizon': horizon,
                'MAE': metrics['mae'],
                'RMSE': metrics['rmse'],
                'R²': metrics['r2']
            })
    
    # Create summary DataFrame
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(os.path.join(args.output_dir, 'summary.csv'), index=False)
    
    # Plot metrics by horizon
    plt.figure(figsize=(14, 8))
    
    plt.subplot(1, 3, 1)
    plt.plot(summary_df['Horizon'], summary_df['MAE'], 'o-', linewidth=2)
    plt.title('MAE by Horizon', fontsize=14)
    plt.xlabel('Prediction Horizon (minutes)', fontsize=12)
    plt.ylabel('Mean Absolute Error', fontsize=12)
    plt.grid(True)
    
    plt.subplot(1, 3, 2)
    plt.plot(summary_df['Horizon'], summary_df['RMSE'], 'o-', linewidth=2)
    plt.title('RMSE by Horizon', fontsize=14)
    plt.xlabel('Prediction Horizon (minutes)', fontsize=12)
    plt.ylabel('Root Mean Squared Error', fontsize=12)
    plt.grid(True)
    
    plt.subplot(1, 3, 3)
    plt.plot(summary_df['Horizon'], summary_df['R²'], 'o-', linewidth=2)
    plt.title('R² by Horizon', fontsize=14)
    plt.xlabel('Prediction Horizon (minutes)', fontsize=12)
    plt.ylabel('R² Score', fontsize=12)
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'metrics_by_horizon.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nAnalysis complete! Results saved to {args.output_dir}")

if __name__ == "__main__":
    main() 