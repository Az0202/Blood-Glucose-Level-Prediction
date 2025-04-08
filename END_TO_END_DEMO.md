# Blood Glucose Level Prediction: End-to-End Demo

This document demonstrates the complete workflow for blood glucose level prediction using the sample data.

## 0. Installation and Setup

Before starting, make sure to install the required dependencies:

```bash
# Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

All required packages are listed in the requirements.txt file, including:
- fastapi and uvicorn for the API
- pandas and numpy for data processing
- scikit-learn for model training and evaluation
- matplotlib and seaborn for visualization
- requests for API testing

## 1. Generating Sample Data

The first step is to generate synthetic sample data in the same format as the OhioT1DM dataset:

```bash
python create_sample_data.py
```

This will create synthetic XML files in the DATA directory:
- DATA/2018/Train/ - Training data for 2018 dataset
- DATA/2018/Test/ - Testing data for 2018 dataset
- DATA/2020/Train/ - Training data for 2020 dataset
- DATA/2020/Test/ - Testing data for 2020 dataset

Each file contains glucose readings, insulin doses, and meal information.

## 2. Preprocessing the Data

Next, we preprocess the raw XML data to extract structured information and save it as CSV files:

```bash
python src/data/preprocess.py
```

This creates:
- processed_data/2018/train/ - Processed training data
- processed_data/2018/test/ - Processed testing data
- processed_data/2020/train/ - Processed training data
- processed_data/2020/test/ - Processed testing data

## 3. Feature Engineering

The preprocessing script also generates features for machine learning models:

- features/559_train_features.csv
- features/559_test_features.csv
- features/563_train_features.csv
- features/563_test_features.csv
- features/570_train_features.csv
- features/570_test_features.csv
- features/540_train_features.csv
- features/540_test_features.csv
- features/544_train_features.csv
- features/544_test_features.csv
- features/552_train_features.csv
- features/552_test_features.csv

As well as combined features:
- features/2018_train_features.csv
- features/2018_test_features.csv
- features/2020_train_features.csv
- features/2020_test_features.csv

These features include:
- Current glucose value
- Glucose difference (rate of change)
- Time-based features (hour, day of week)
- Recent insulin doses
- Recent carbohydrate intake
- Target values for different prediction horizons

## 4. Running the API

The Simple Glucose API provides predictions based on the current glucose value and other features:

```bash
python simple_glucose_api.py
```

This starts a FastAPI-based web service on http://localhost:8001/

## 5. Making Predictions

You can test the API using the test script:

```bash
python test_simple_api.py --patient_id 559 --test_file features/559_test_features.csv
```

Or you can make requests directly:

```python
import requests
import json
import pandas as pd

# Load features data
features_df = pd.read_csv('features/559_train_features.csv')
sample = features_df.iloc[100]

# Prepare request data
request_data = {
    "patient_id": int(sample["patient_id"]),
    "glucose_value": float(sample["glucose_value"]),
    "glucose_diff": float(sample["glucose_diff"]),
    "hour": int(sample["hour"]),
    "day_of_week": int(sample["day_of_week"]),
    "insulin_dose_1h": float(sample["insulin_dose_1h"]),
    "insulin_dose_2h": float(sample["insulin_dose_2h"]),
    "insulin_dose_4h": float(sample["insulin_dose_4h"]),
    "carbs_1h": float(sample["carbs_1h"]),
    "carbs_2h": float(sample["carbs_2h"]),
    "carbs_4h": float(sample["carbs_4h"])
}

# Make the prediction
response = requests.post("http://localhost:8001/predict", json=request_data)
prediction = response.json()
print(json.dumps(prediction, indent=2))
```

## 6. Advanced Models

For more accurate predictions, you could train advanced models using the sample data:

```bash
python train_improved.py --horizons 15,30,45,60 --batch_size 128 --epochs 10
```

This would train neural network models (LSTM, GRU) for different prediction horizons.

## 7. Evaluating Predictions

To evaluate prediction accuracy, you can use the validation script:

```bash
python validate_api.py --test_file features/559_test_features.csv --patient_ids 559 --horizons 15,30 --api_url http://localhost:8001
```

This will calculate metrics like MAE, RMSE, and Clarke Error Grid analysis.

## 8. Web Interface

For a user-friendly interface, you can run the web interface:

```bash
python web_interface.py
```

This provides a browser-based UI for making predictions and visualizing results.

## Conclusion

This workflow demonstrates how to:
1. Generate synthetic data that mimics the structure of the OhioT1DM dataset
2. Preprocess the data and extract features
3. Make predictions using the API
4. Evaluate the accuracy of predictions

For research purposes, you should obtain the actual OhioT1DM dataset by signing the Data Use Agreement as described in SAMPLE_DATA_README.md. 