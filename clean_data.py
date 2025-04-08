#!/usr/bin/env python3
"""
clean_data.py

This script helps to remove actual patient data from the OhioT1DM dataset
while preserving the sample data for demo and testing purposes.
"""

import os
import sys
import shutil
import argparse
import glob

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Clean OhioT1DM dataset files while preserving sample data"
    )
    parser.add_argument(
        "--keep_sample", 
        action="store_true", 
        help="Keep the sample data files (default is to remove all)"
    )
    parser.add_argument(
        "--backup", 
        action="store_true", 
        help="Backup data before cleaning"
    )
    return parser.parse_args()

def backup_data(data_dir="DATA"):
    """Create a backup of the data directory"""
    if os.path.exists(data_dir):
        backup_dir = f"{data_dir}_backup"
        print(f"Creating backup of {data_dir} to {backup_dir}...")
        
        # Remove existing backup if it exists
        if os.path.exists(backup_dir):
            print(f"Removing existing backup at {backup_dir}")
            shutil.rmtree(backup_dir)
        
        # Create backup
        shutil.copytree(data_dir, backup_dir)
        print(f"Backup created at {backup_dir}")
    else:
        print(f"No data directory found at {data_dir}, skipping backup")

def is_sample_data(filename):
    """
    Check if a file is part of the sample data
    Sample data files were generated with specific patient IDs
    """
    sample_ids_2018 = ["559", "563", "570"]
    sample_ids_2020 = ["540", "544", "552"]
    
    for patient_id in sample_ids_2018 + sample_ids_2020:
        if f"{patient_id}-ws-" in filename:
            return True
    
    return False

def clean_data(keep_sample=False, data_dir="DATA"):
    """
    Clean data files from the OhioT1DM dataset
    
    Parameters:
    -----------
    keep_sample : bool
        Whether to keep the sample data files
    data_dir : str
        Directory containing the data files
    """
    if not os.path.exists(data_dir):
        print(f"No data directory found at {data_dir}, nothing to clean")
        return
    
    # Find all XML files
    xml_files = glob.glob(f"{data_dir}/**/*.xml", recursive=True)
    
    if not xml_files:
        print(f"No XML files found in {data_dir}")
        return
    
    print(f"Found {len(xml_files)} XML files in {data_dir}")
    
    # Count files to remove
    removed_count = 0
    kept_count = 0
    
    # Process each file
    for file_path in xml_files:
        filename = os.path.basename(file_path)
        
        # Check if this is sample data
        is_sample = is_sample_data(filename)
        
        # Determine whether to keep this file
        should_keep = keep_sample and is_sample
        
        if should_keep:
            print(f"Keeping sample file: {file_path}")
            kept_count += 1
        else:
            # Remove the file
            print(f"Removing: {file_path}")
            os.remove(file_path)
            removed_count += 1
    
    print(f"\nCleaning complete:")
    print(f"  - Removed {removed_count} files")
    print(f"  - Kept {kept_count} files")
    
    if keep_sample:
        print("\nSample data has been preserved for demonstration purposes.")
        print("For research, please obtain the actual OhioT1DM dataset by signing the Data Use Agreement.")
    else:
        print("\nAll data files have been removed.")
        print("Run 'python create_sample_data.py' to generate sample data for demonstration.")

def main():
    """Main function"""
    args = parse_arguments()
    
    # Create backup if requested
    if args.backup:
        backup_data()
    
    # Clean data
    clean_data(keep_sample=args.keep_sample)
    
    print("\nDone.")

if __name__ == "__main__":
    main() 