#!/bin/bash
# run_demo.sh - Script to run through the entire blood glucose prediction demo

echo "===== Blood Glucose Level Prediction Demo ====="
echo "This script will run through the entire workflow."

# Step 0: Check dependencies
echo -e "\n[Step 0] Checking dependencies..."
if ! command -v pip &> /dev/null; then
    echo "Error: pip is not installed"
    exit 1
fi

echo "Installing dependencies..."
pip install -r requirements.txt

# Step 1: Generate sample data
echo -e "\n[Step 1] Generating sample data..."
python create_sample_data.py

# Step 2: Preprocess data
echo -e "\n[Step 2] Preprocessing data..."
python src/data/preprocess.py

# Step 3: Start the API (in background)
echo -e "\n[Step 3] Starting the API..."
python simple_glucose_api.py &
API_PID=$!

# Wait for API to start
echo "Waiting for API to start..."
sleep 5

# Step 4: Test the API
echo -e "\n[Step 4] Testing the API..."
python test_simple_api.py --patient_id 559 --test_file features/559_test_features.csv

# Step 5: Stop the API
echo -e "\n[Step 5] Stopping the API..."
kill $API_PID

echo -e "\n===== Demo completed successfully! ====="
echo "See END_TO_END_DEMO.md for more details on each step." 