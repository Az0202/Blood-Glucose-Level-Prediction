import os
import pandas as pd
import numpy as np
from datetime import timedelta

def generate_additional_horizons(features_dir, dataset="2018", horizons=[45, 60, 90]):
    """
    Generate additional target horizon columns for glucose prediction
    
    Parameters:
    -----------
    features_dir : str
        Directory containing feature files
    dataset : str
        '2018' or '2020'
    horizons : list
        List of new prediction horizons in minutes to generate
    """
    print(f"Generating additional target horizons {horizons} for {dataset} dataset")
    
    # Process both train and test datasets
    for dataset_type in ["train", "test"]:
        file_path = os.path.join(features_dir, f"{dataset}_{dataset_type}_features.csv")
        
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            continue
        
        # Load the features file
        print(f"Loading {file_path}")
        df = pd.read_csv(file_path)
        
        # Ensure datetime column is in datetime format
        if 'datetime' in df.columns:
            df['datetime'] = pd.to_datetime(df['datetime'])
        
        # Create a copy for working with
        df_with_new_targets = df.copy()
        
        # Iterate through each patient
        patient_ids = df['patient_id'].unique()
        
        for patient_id in patient_ids:
            print(f"Processing patient {patient_id}")
            patient_df = df[df['patient_id'] == patient_id].copy()
            
            # Sort by datetime to ensure correct time sequence
            if 'datetime' in patient_df.columns:
                patient_df = patient_df.sort_values('datetime')
            
            # Process each new horizon
            for horizon in horizons:
                target_col = f'target_{horizon}min'
                
                if target_col in df.columns:
                    print(f"Column {target_col} already exists, skipping")
                    continue
                
                # Generate target values by shifting the glucose_value
                # The shift is calculated based on the sampling frequency
                # Assuming 5-minute intervals between readings
                shift_periods = horizon // 5
                
                # Shift the glucose values to create future targets
                patient_df[target_col] = patient_df['glucose_value'].shift(-shift_periods)
                
                # Handle missing values and outliers
                # Replace missing values with the mean of non-NaN values
                patient_mean = patient_df[target_col].mean(skipna=True)
                
                # Replace NaN values with patient mean
                patient_df[target_col] = patient_df[target_col].fillna(patient_mean)
                
                # Handle extreme outliers (values > 600 or < 20 mg/dL are likely errors)
                # These thresholds are based on physiological limits
                mask_high = patient_df[target_col] > 600
                mask_low = patient_df[target_col] < 20
                
                if mask_high.any() or mask_low.any():
                    print(f"Fixing outliers for patient {patient_id}, horizon {horizon}")
                    # Replace outliers with the mean
                    patient_df.loc[mask_high | mask_low, target_col] = patient_mean
                
                # Ensure no infinity values
                patient_df[target_col] = patient_df[target_col].replace([np.inf, -np.inf], patient_mean)
                
                # Perform a final check for any remaining NaN or inf values
                if patient_df[target_col].isna().any() or np.isinf(patient_df[target_col]).any():
                    print(f"WARNING: Still have NaN or inf values for patient {patient_id}, horizon {horizon}")
                    # Use a global mean as last resort
                    global_mean = df['glucose_value'].mean()
                    patient_df[target_col] = patient_df[target_col].fillna(global_mean)
                    patient_df[target_col] = patient_df[target_col].replace([np.inf, -np.inf], global_mean)
                
                # Update the main dataframe with the new target column for this patient
                df_with_new_targets.loc[df_with_new_targets['patient_id'] == patient_id, target_col] = patient_df[target_col].values
        
        # Final verification for the entire dataset
        for horizon in horizons:
            target_col = f'target_{horizon}min'
            if target_col in df_with_new_targets.columns:
                # Check for any remaining NaN or inf values
                if df_with_new_targets[target_col].isna().any() or np.isinf(df_with_new_targets[target_col]).any():
                    print(f"WARNING: Final check found NaN or inf values in {target_col}")
                    # Use global mean as last resort
                    global_mean = df['glucose_value'].mean()
                    df_with_new_targets[target_col] = df_with_new_targets[target_col].fillna(global_mean)
                    df_with_new_targets[target_col] = df_with_new_targets[target_col].replace([np.inf, -np.inf], global_mean)
                
                # Report statistics
                print(f"Target column {target_col} statistics:")
                print(f"  Min: {df_with_new_targets[target_col].min()}")
                print(f"  Max: {df_with_new_targets[target_col].max()}")
                print(f"  Mean: {df_with_new_targets[target_col].mean()}")
                print(f"  NaN count: {df_with_new_targets[target_col].isna().sum()}")
        
        # Save the updated dataframe
        output_path = os.path.join(features_dir, f"{dataset}_{dataset_type}_features_extended.csv")
        print(f"Saving to {output_path}")
        df_with_new_targets.to_csv(output_path, index=False)
        
        # Create backup of original file
        backup_path = os.path.join(features_dir, f"{dataset}_{dataset_type}_features_original.csv")
        print(f"Creating backup at {backup_path}")
        df.to_csv(backup_path, index=False)
        
        # Replace original file with extended version
        print(f"Replacing {file_path} with extended version")
        df_with_new_targets.to_csv(file_path, index=False)
        
    print("Target horizon generation completed!")

def main():
    """
    Main function to generate additional target horizons
    """
    # Create required directories
    features_dir = 'features'
    
    # Define new horizons to generate
    new_horizons = [45, 60, 90]
    
    # Generate for 2018 dataset
    generate_additional_horizons(features_dir, dataset="2018", horizons=new_horizons)
    
    # Generate for 2020 dataset
    generate_additional_horizons(features_dir, dataset="2020", horizons=new_horizons)
    
    print("All target horizons generated successfully!")

if __name__ == "__main__":
    main() 