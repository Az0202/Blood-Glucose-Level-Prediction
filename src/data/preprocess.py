import os
import pandas as pd
import xml.etree.ElementTree as ET
from datetime import datetime
import numpy as np

def parse_xml_file(file_path):
    """
    Parse an OhioT1DM XML file and extract glucose readings
    
    Parameters:
    -----------
    file_path : str
        Path to XML file
        
    Returns:
    --------
    tuple: (patient_info, glucose_df, insulin_df, meal_df)
    """
    print(f"Processing file: {file_path}")
    
    # Parse XML file
    tree = ET.parse(file_path)
    root = tree.getroot()
    
    # Extract patient information
    patient_id = root.get('id')
    weight = float(root.get('weight'))
    insulin_type = root.get('insulin_type')
    
    patient_info = {
        'id': patient_id,
        'weight': weight,
        'insulin_type': insulin_type
    }
    
    # Initialize empty lists for each data type
    glucose_data = []
    insulin_data = []
    meal_data = []
    
    # Extract glucose levels - they are in event tags within glucose_level
    for event in root.findall('.//glucose_level/event'):
        try:
            ts_str = event.get('ts')
            if ts_str is None:
                continue  # Skip entries with missing timestamp
            
            # Convert timestamp string to unix timestamp
            # Parse date in DD-MM-YYYY format
            dt = datetime.strptime(ts_str, '%d-%m-%Y %H:%M:%S')
            ts = int(dt.timestamp())
            
            value = float(event.get('value'))
            glucose_data.append({
                'timestamp': ts,
                'datetime': dt,
                'glucose_value': value
            })
        except (TypeError, ValueError) as e:
            print(f"Error processing glucose entry: {e}")
            continue
    
    # Extract insulin data - check if they're in event tags too
    for insulin_parent in root.findall('.//insulin'):
        for event in insulin_parent.findall('./event'):
            try:
                ts_str = event.get('ts')
                if ts_str is None:
                    continue  # Skip entries with missing timestamp
                
                # Convert timestamp string to unix timestamp
                # Parse date in DD-MM-YYYY format
                dt = datetime.strptime(ts_str, '%d-%m-%Y %H:%M:%S')
                ts = int(dt.timestamp())
                
                dose = float(event.get('dose', 0))
                insulin_type = event.get('type', 'unknown')
                insulin_data.append({
                    'timestamp': ts,
                    'datetime': dt,
                    'insulin_dose': dose,
                    'insulin_type': insulin_type
                })
            except (TypeError, ValueError) as e:
                print(f"Error processing insulin entry: {e}")
                continue
    
    # Extract meal data - check if they're in event tags too
    for meal_parent in root.findall('.//meal'):
        for event in meal_parent.findall('./event'):
            try:
                ts_str = event.get('ts')
                if ts_str is None:
                    continue  # Skip entries with missing timestamp
                
                # Convert timestamp string to unix timestamp
                # Parse date in DD-MM-YYYY format
                dt = datetime.strptime(ts_str, '%d-%m-%Y %H:%M:%S')
                ts = int(dt.timestamp())
                
                carbs = float(event.get('carbs', 0))
                meal_data.append({
                    'timestamp': ts,
                    'datetime': dt,
                    'carbs': carbs
                })
            except (TypeError, ValueError) as e:
                print(f"Error processing meal entry: {e}")
                continue
    
    # Convert to DataFrames
    glucose_df = pd.DataFrame(glucose_data)
    insulin_df = pd.DataFrame(insulin_data) if insulin_data else pd.DataFrame(columns=['timestamp', 'datetime', 'insulin_dose', 'insulin_type'])
    meal_df = pd.DataFrame(meal_data) if meal_data else pd.DataFrame(columns=['timestamp', 'datetime', 'carbs'])
    
    # Sort by timestamp
    if not glucose_df.empty:
        glucose_df = glucose_df.sort_values('timestamp')
    if not insulin_df.empty:
        insulin_df = insulin_df.sort_values('timestamp')
    if not meal_df.empty:
        meal_df = meal_df.sort_values('timestamp')
    
    print(f"Found {len(glucose_df)} glucose readings, {len(insulin_df)} insulin records, {len(meal_df)} meal records")
    
    return patient_info, glucose_df, insulin_df, meal_df

def process_dataset(dataset_dir, output_dir, dataset_type='train'):
    """
    Process all XML files in the given directory
    
    Parameters:
    -----------
    dataset_dir : str
        Directory containing XML files
    output_dir : str
        Directory to save processed data
    dataset_type : str
        'train' or 'test'
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all XML files in the directory
    xml_files = [f for f in os.listdir(dataset_dir) if f.endswith('-ws-training.xml') and dataset_type == 'train' or 
                 f.endswith('-ws-testing.xml') and dataset_type == 'test']
    
    patient_info_list = []
    
    for xml_file in xml_files:
        file_path = os.path.join(dataset_dir, xml_file)
        patient_id = xml_file.split('-')[0]
        
        patient_info, glucose_df, insulin_df, meal_df = parse_xml_file(file_path)
        patient_info_list.append(patient_info)
        
        # Save to CSV
        glucose_df.to_csv(os.path.join(output_dir, f"{patient_id}_glucose.csv"), index=False)
        insulin_df.to_csv(os.path.join(output_dir, f"{patient_id}_insulin.csv"), index=False)
        meal_df.to_csv(os.path.join(output_dir, f"{patient_id}_meal.csv"), index=False)
        
        print(f"Processed patient {patient_id}: {len(glucose_df)} glucose readings, {len(insulin_df)} insulin records, {len(meal_df)} meal records")
    
    # Save patient information
    pd.DataFrame(patient_info_list).to_csv(os.path.join(output_dir, "patient_info.csv"), index=False)

def generate_features(patient_id, glucose_df, insulin_df, meal_df, 
                      history_window=12, prediction_horizon=6):
    """
    Generate features for model training, including:
    - Historical glucose values
    - Time-based features
    - Insulin and meal information
    
    Parameters:
    -----------
    patient_id : str
        Patient identifier
    glucose_df : DataFrame
        Glucose readings
    insulin_df : DataFrame
        Insulin records
    meal_df : DataFrame
        Meal records
    history_window : int
        Number of historical readings to use (each reading is typically 5 minutes)
    prediction_horizon : int
        Number of steps to predict into the future
        
    Returns:
    --------
    DataFrame
        Features ready for model training
    """
    # Ensure data is sorted
    glucose_df = glucose_df.sort_values('timestamp')
    
    # Create time features
    glucose_df['hour'] = glucose_df['datetime'].dt.hour
    glucose_df['day_of_week'] = glucose_df['datetime'].dt.dayofweek
    
    # Calculate glucose rate of change
    glucose_df['glucose_diff'] = glucose_df['glucose_value'].diff()
    glucose_df['glucose_diff_rate'] = glucose_df['glucose_diff'] / (glucose_df['timestamp'].diff() / 60)  # per minute
    
    # Add rolling statistics
    glucose_df['glucose_rolling_mean_1h'] = glucose_df['glucose_value'].rolling(window=12).mean()
    glucose_df['glucose_rolling_std_1h'] = glucose_df['glucose_value'].rolling(window=12).std()
    
    # Create lag features
    for i in range(1, history_window + 1):
        glucose_df[f'glucose_lag_{i}'] = glucose_df['glucose_value'].shift(i)
    
    # Add insulin information
    # Merge the closest previous insulin dose
    feature_df = glucose_df.copy()
    feature_df['insulin_dose'] = 0
    feature_df['insulin_dose_1h'] = 0
    feature_df['insulin_dose_2h'] = 0
    feature_df['insulin_dose_4h'] = 0
    
    # For each glucose reading, find recent insulin doses
    if not insulin_df.empty:
        for idx, row in feature_df.iterrows():
            glucose_time = row['timestamp']
            
            # Find insulin doses in the last 1, 2, and 4 hours
            last_1h = insulin_df[(insulin_df['timestamp'] <= glucose_time) & 
                                (insulin_df['timestamp'] > glucose_time - 3600)]
            last_2h = insulin_df[(insulin_df['timestamp'] <= glucose_time) & 
                                (insulin_df['timestamp'] > glucose_time - 7200)]
            last_4h = insulin_df[(insulin_df['timestamp'] <= glucose_time) & 
                                (insulin_df['timestamp'] > glucose_time - 14400)]
            
            # Convert to float to avoid dtype incompatibility warnings
            feature_df.at[idx, 'insulin_dose_1h'] = float(last_1h['insulin_dose'].sum() if not last_1h.empty else 0)
            feature_df.at[idx, 'insulin_dose_2h'] = float(last_2h['insulin_dose'].sum() if not last_2h.empty else 0)
            feature_df.at[idx, 'insulin_dose_4h'] = float(last_4h['insulin_dose'].sum() if not last_4h.empty else 0)
    
    # Add meal information
    feature_df['carbs_1h'] = 0
    feature_df['carbs_2h'] = 0
    feature_df['carbs_4h'] = 0
    
    # For each glucose reading, find recent meal carbs
    if not meal_df.empty:
        for idx, row in feature_df.iterrows():
            glucose_time = row['timestamp']
            
            # Find meals in the last 1, 2, and 4 hours
            last_1h = meal_df[(meal_df['timestamp'] <= glucose_time) & 
                             (meal_df['timestamp'] > glucose_time - 3600)]
            last_2h = meal_df[(meal_df['timestamp'] <= glucose_time) & 
                             (meal_df['timestamp'] > glucose_time - 7200)]
            last_4h = meal_df[(meal_df['timestamp'] <= glucose_time) & 
                             (meal_df['timestamp'] > glucose_time - 14400)]
            
            # Convert to float to avoid dtype incompatibility warnings
            feature_df.at[idx, 'carbs_1h'] = float(last_1h['carbs'].sum() if not last_1h.empty else 0)
            feature_df.at[idx, 'carbs_2h'] = float(last_2h['carbs'].sum() if not last_2h.empty else 0)
            feature_df.at[idx, 'carbs_4h'] = float(last_4h['carbs'].sum() if not last_4h.empty else 0)
    
    # Create target variables for different prediction horizons
    for horizon in range(1, prediction_horizon + 1):
        feature_df[f'target_{horizon*5}min'] = feature_df['glucose_value'].shift(-horizon)
    
    # Drop rows with NaN values (beginning and end of the series)
    feature_df = feature_df.dropna()
    
    # Add patient ID
    feature_df['patient_id'] = patient_id
    
    return feature_df

def main():
    # Process 2018 dataset
    data_dir = "DATA/2018"
    processed_dir = "processed_data/2018"
    
    # Create processed data directory
    os.makedirs(processed_dir, exist_ok=True)
    
    # Process training data
    train_dir = os.path.join(data_dir, "Train")
    train_output_dir = os.path.join(processed_dir, "train")
    process_dataset(train_dir, train_output_dir, dataset_type='train')
    
    # Process testing data
    test_dir = os.path.join(data_dir, "Test")
    test_output_dir = os.path.join(processed_dir, "test")
    process_dataset(test_dir, test_output_dir, dataset_type='test')
    
    # Process 2020 dataset
    data_dir = "DATA/2020"
    processed_dir = "processed_data/2020"
    
    # Create processed data directory
    os.makedirs(processed_dir, exist_ok=True)
    
    # Process training data
    train_dir = os.path.join(data_dir, "Train")
    train_output_dir = os.path.join(processed_dir, "train")
    process_dataset(train_dir, train_output_dir, dataset_type='train')
    
    # Process testing data
    test_dir = os.path.join(data_dir, "Test")
    test_output_dir = os.path.join(processed_dir, "test")
    process_dataset(test_dir, test_output_dir, dataset_type='test')
    
    print("Data preprocessing complete!")
    
    # Generate features for machine learning models
    print("\nGenerating features for machine learning models...")
    
    # Create feature directory
    features_dir = "features"
    os.makedirs(features_dir, exist_ok=True)
    
    # Generate features for 2018 dataset
    for dataset_type in ['train', 'test']:
        all_features = []
        input_dir = f"processed_data/2018/{dataset_type}"
        
        # Load patient info
        patient_info = pd.read_csv(os.path.join(input_dir, "patient_info.csv"))
        
        for _, patient in patient_info.iterrows():
            patient_id = patient['id']
            print(f"Generating features for patient {patient_id} ({dataset_type})...")
            
            # Load data
            glucose_df = pd.read_csv(os.path.join(input_dir, f"{patient_id}_glucose.csv"))
            glucose_df['datetime'] = pd.to_datetime(glucose_df['datetime'])
            
            meal_df = pd.read_csv(os.path.join(input_dir, f"{patient_id}_meal.csv"))
            meal_df['datetime'] = pd.to_datetime(meal_df['datetime'])
            
            insulin_df = pd.read_csv(os.path.join(input_dir, f"{patient_id}_insulin.csv"))
            if not insulin_df.empty:
                insulin_df['datetime'] = pd.to_datetime(insulin_df['datetime'])
            
            # Generate features
            features = generate_features(patient_id, glucose_df, insulin_df, meal_df)
            all_features.append(features)
            
            # Save individual patient features
            patient_feature_file = os.path.join(features_dir, f"{patient_id}_{dataset_type}_features.csv")
            features.to_csv(patient_feature_file, index=False)
            print(f"  - Saved {len(features)} feature rows to {patient_feature_file}")
        
        # Combine all patient features
        if all_features:
            combined_features = pd.concat(all_features, ignore_index=True)
            combined_feature_file = os.path.join(features_dir, f"2018_{dataset_type}_features.csv")
            combined_features.to_csv(combined_feature_file, index=False)
            print(f"Saved combined features for 2018 {dataset_type} set: {len(combined_features)} rows")
    
    # Generate features for 2020 dataset
    for dataset_type in ['train', 'test']:
        all_features = []
        input_dir = f"processed_data/2020/{dataset_type}"
        
        # Load patient info
        patient_info = pd.read_csv(os.path.join(input_dir, "patient_info.csv"))
        
        for _, patient in patient_info.iterrows():
            patient_id = patient['id']
            print(f"Generating features for patient {patient_id} ({dataset_type})...")
            
            # Load data
            glucose_df = pd.read_csv(os.path.join(input_dir, f"{patient_id}_glucose.csv"))
            glucose_df['datetime'] = pd.to_datetime(glucose_df['datetime'])
            
            meal_df = pd.read_csv(os.path.join(input_dir, f"{patient_id}_meal.csv"))
            meal_df['datetime'] = pd.to_datetime(meal_df['datetime'])
            
            insulin_df = pd.read_csv(os.path.join(input_dir, f"{patient_id}_insulin.csv"))
            if not insulin_df.empty:
                insulin_df['datetime'] = pd.to_datetime(insulin_df['datetime'])
            
            # Generate features
            features = generate_features(patient_id, glucose_df, insulin_df, meal_df)
            all_features.append(features)
            
            # Save individual patient features
            patient_feature_file = os.path.join(features_dir, f"{patient_id}_{dataset_type}_features.csv")
            features.to_csv(patient_feature_file, index=False)
            print(f"  - Saved {len(features)} feature rows to {patient_feature_file}")
        
        # Combine all patient features
        if all_features:
            combined_features = pd.concat(all_features, ignore_index=True)
            combined_feature_file = os.path.join(features_dir, f"2020_{dataset_type}_features.csv")
            combined_features.to_csv(combined_feature_file, index=False)
            print(f"Saved combined features for 2020 {dataset_type} set: {len(combined_features)} rows")
    
    print("Feature generation complete!")

if __name__ == "__main__":
    main() 