#!/usr/bin/env python3
"""
create_sample_data.py

This script creates a small sample dataset in the same format as the OhioT1DM dataset
for demonstration and testing purposes. This allows the project to work without
requiring access to the actual OhioT1DM dataset which requires a Data Use Agreement.
"""

import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import xml.etree.ElementTree as ET
import xml.dom.minidom

def ensure_directory(directory):
    """Create directory if it doesn't exist"""
    if not os.path.exists(directory):
        os.makedirs(directory)

def generate_sample_glucose_data(start_date, days=2, interval_minutes=5):
    """
    Generate realistic but synthetic glucose data
    
    Parameters:
    -----------
    start_date : datetime
        Start date for the data
    days : int
        Number of days to generate
    interval_minutes : int
        Interval between readings in minutes
        
    Returns:
    --------
    DataFrame: Synthetic glucose data
    """
    # Calculate number of readings
    total_readings = int((days * 24 * 60) / interval_minutes)
    
    # Generate timestamps
    timestamps = [start_date + timedelta(minutes=i*interval_minutes) 
                 for i in range(total_readings)]
    
    # Generate baseline glucose values with daily patterns
    # Morning rise, post-meal peaks, night-time drop
    hours = np.array([(t.hour + t.minute/60) for t in timestamps])
    
    # Base level around 100 mg/dL
    base_level = 100 * np.ones(total_readings)
    
    # Morning rise (dawn phenomenon)
    morning_effect = 30 * np.exp(-((hours - 7) ** 2) / 5)
    
    # Meal peaks (breakfast, lunch, dinner)
    meal_times = [7, 12, 18]
    meal_effects = np.zeros(total_readings)
    for meal_time in meal_times:
        # Random variation in meal timing
        meal_variation = np.random.normal(0, 0.5, total_readings)
        meal_effect = 50 * np.exp(-((hours - meal_time - meal_variation) ** 2) / 1)
        meal_effects += meal_effect
    
    # Night drop
    night_effect = -20 * np.exp(-((hours - 2) ** 2) / 10)
    
    # Combine effects
    glucose_values = base_level + morning_effect + meal_effects + night_effect
    
    # Add some noise
    noise = np.random.normal(0, 5, total_readings)
    glucose_values += noise
    
    # Ensure values are realistic
    glucose_values = np.clip(glucose_values, 40, 400)
    
    # Create dataframe
    df = pd.DataFrame({
        'datetime': timestamps,
        'glucose': glucose_values.astype(int)
    })
    
    return df

def generate_sample_insulin_data(glucose_df, basal_rate=1.0):
    """
    Generate synthetic insulin data based on glucose values
    
    Parameters:
    -----------
    glucose_df : DataFrame
        Glucose data
    basal_rate : float
        Basal insulin rate in U/hr
        
    Returns:
    --------
    DataFrame: Synthetic insulin data
    """
    # Extract timestamps
    timestamps = glucose_df['datetime'].values
    
    # Generate basal insulin entries (hourly)
    basal_times = []
    basal_values = []
    
    # Convert numpy.datetime64 to Python datetime if needed
    first_timestamp = pd.Timestamp(timestamps[0]).to_pydatetime()
    last_timestamp = pd.Timestamp(timestamps[-1]).to_pydatetime()
    
    start_time = first_timestamp.replace(minute=0, second=0, microsecond=0)
    end_time = last_timestamp
    
    current_time = start_time
    while current_time <= end_time:
        basal_times.append(current_time)
        # Add some variation to basal rate
        basal_values.append(basal_rate * np.random.uniform(0.95, 1.05))
        current_time += timedelta(hours=1)
    
    # Generate bolus insulin for meals
    bolus_times = []
    bolus_values = []
    
    # Typical meal times with some variation
    meal_hours = [7, 12, 18]  # breakfast, lunch, dinner
    
    # Group by day
    days = glucose_df['datetime'].dt.date.unique()
    
    for day in days:
        for meal_hour in meal_hours:
            # Add variation to meal time
            meal_time = datetime.combine(day, datetime.min.time()) + timedelta(hours=meal_hour)
            meal_time += timedelta(minutes=np.random.randint(-30, 30))
            
            # Only include if within our data range
            if meal_time >= first_timestamp and meal_time <= last_timestamp:
                bolus_times.append(meal_time)
                # Randomize bolus amount
                bolus_values.append(np.random.uniform(4, 8))
    
    # Create separate dataframes
    basal_df = pd.DataFrame({
        'datetime': basal_times,
        'insulin': basal_values,
        'type': 'basal'
    })
    
    bolus_df = pd.DataFrame({
        'datetime': bolus_times,
        'insulin': bolus_values,
        'type': 'bolus'
    })
    
    # Combine
    insulin_df = pd.concat([basal_df, bolus_df], ignore_index=True)
    insulin_df = insulin_df.sort_values('datetime').reset_index(drop=True)
    
    return insulin_df

def generate_sample_meal_data(insulin_df):
    """
    Generate synthetic meal data based on bolus insulin events
    
    Parameters:
    -----------
    insulin_df : DataFrame
        Insulin data with bolus entries
        
    Returns:
    --------
    DataFrame: Synthetic meal data
    """
    # Extract bolus events as meal times
    bolus_entries = insulin_df[insulin_df['type'] == 'bolus']
    
    # Create meal dataframe
    meal_df = pd.DataFrame({
        'datetime': bolus_entries['datetime'].values,
        'carbs': bolus_entries['insulin'].apply(lambda x: int(x * 10 + np.random.uniform(-10, 10)))
    })
    
    # Ensure carbs are positive
    meal_df['carbs'] = meal_df['carbs'].apply(lambda x: max(15, x))
    
    return meal_df

def create_xml_file(patient_id, glucose_df, insulin_df, meal_df, output_file):
    """
    Create an XML file in the OhioT1DM format
    
    Parameters:
    -----------
    patient_id : str
        Patient ID
    glucose_df : DataFrame
        Glucose data
    insulin_df : DataFrame
        Insulin data
    meal_df : DataFrame
        Meal data
    output_file : str
        Output file path
    """
    # Create the root element
    root = ET.Element("patient")
    root.set("id", patient_id)
    
    # Add glucose entries
    for _, row in glucose_df.iterrows():
        entry = ET.SubElement(root, "glucose_level")
        timestamp = int(row['datetime'].timestamp())
        entry.set("ts", str(timestamp))
        entry.set("value", str(int(row['glucose'])))
    
    # Add insulin entries
    for _, row in insulin_df.iterrows():
        entry = ET.SubElement(root, "insulin")
        timestamp = int(row['datetime'].timestamp())
        entry.set("ts", str(timestamp))
        entry.set("dose", str(round(row['insulin'], 2)))
        entry.set("type", row['type'])
    
    # Add meal entries
    for _, row in meal_df.iterrows():
        entry = ET.SubElement(root, "meal")
        timestamp = int(row['datetime'].timestamp())
        entry.set("ts", str(timestamp))
        entry.set("carbs", str(int(row['carbs'])))
    
    # Create a formatted XML string
    rough_string = ET.tostring(root, 'utf-8')
    reparsed = xml.dom.minidom.parseString(rough_string)
    pretty_xml = reparsed.toprettyxml(indent="  ")
    
    # Write to file
    with open(output_file, 'w') as f:
        f.write(pretty_xml)

def generate_sample_patient_data(patient_id, start_date, output_file):
    """
    Generate a full sample dataset for a patient
    
    Parameters:
    -----------
    patient_id : str
        Patient ID
    start_date : datetime
        Start date for the data
    output_file : str
        Output file path
    """
    # Generate glucose data
    glucose_df = generate_sample_glucose_data(start_date, days=2)
    
    # Generate insulin data
    insulin_df = generate_sample_insulin_data(glucose_df)
    
    # Generate meal data
    meal_df = generate_sample_meal_data(insulin_df)
    
    # Create XML file
    create_xml_file(patient_id, glucose_df, insulin_df, meal_df, output_file)
    
    print(f"Generated sample data for patient {patient_id} in {output_file}")
    print(f"  - {len(glucose_df)} glucose readings")
    print(f"  - {len(insulin_df)} insulin entries")
    print(f"  - {len(meal_df)} meal records")

def main():
    """Generate sample data for the 2018 and 2020 datasets"""
    print("Generating sample data for the OhioT1DM dataset format")
    
    # Ensure directories exist
    for year in ["2018", "2020"]:
        for split in ["Train", "Test"]:
            directory = os.path.join("DATA", year, split)
            ensure_directory(directory)
    
    # Generate 2018 dataset (3 patients)
    patient_ids_2018 = ["559", "563", "570"]
    start_date = datetime(2018, 1, 1)
    
    for patient_id in patient_ids_2018:
        # Training set
        output_file = os.path.join("DATA", "2018", "Train", f"{patient_id}-ws-training.xml")
        generate_sample_patient_data(patient_id, start_date, output_file)
        
        # Test set (use a different date)
        test_start_date = start_date + timedelta(days=60)
        output_file = os.path.join("DATA", "2018", "Test", f"{patient_id}-ws-testing.xml")
        generate_sample_patient_data(patient_id, test_start_date, output_file)
    
    # Generate 2020 dataset (3 patients)
    patient_ids_2020 = ["540", "544", "552"]
    start_date = datetime(2020, 1, 1)
    
    for patient_id in patient_ids_2020:
        # Training set
        output_file = os.path.join("DATA", "2020", "Train", f"{patient_id}-ws-training.xml")
        generate_sample_patient_data(patient_id, start_date, output_file)
        
        # Test set (use a different date)
        test_start_date = start_date + timedelta(days=60)
        output_file = os.path.join("DATA", "2020", "Test", f"{patient_id}-ws-testing.xml")
        generate_sample_patient_data(patient_id, test_start_date, output_file)
    
    print("\nSample data generation complete!")
    print("These are synthetic samples for testing and demonstration only.")
    print("For research, please obtain the actual OhioT1DM dataset by signing the Data Use Agreement.")

if __name__ == "__main__":
    main() 