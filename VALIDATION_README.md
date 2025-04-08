# Glucose Prediction API Validation

This document explains how to use the validation script to assess the accuracy and performance of the Glucose Prediction API against historical data.

## Overview

The `validate_api.py` script provides a comprehensive validation framework for testing the Glucose Prediction API. It:

1. Validates predictions against historical ground truth data
2. Analyzes prediction accuracy using multiple metrics
3. Generates visualizations to understand model performance
4. Tests API response times and reliability
5. Creates detailed reports per patient and summary statistics

## Requirements

- Python 3.8+
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- requests
- tqdm

Install required packages:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn requests tqdm
```

## Running the Validation

### Basic Usage

```bash
python validate_api.py --test_file /path/to/test_features.csv --patient_ids 570,575,588,591 --horizons 15,30 --api_url http://localhost:8001
```

### Command-line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--test_file` | Path to CSV file with test data (required) | - |
| `--patient_ids` | Comma-separated list of patient IDs to validate | 570 |
| `--horizons` | Comma-separated list of prediction horizons (minutes) | 15,30 |
| `--api_url` | Base URL of the API | http://localhost:8001 |
| `--output_dir` | Directory to save validation results | validation_results |

### Important Notes About Parameters

- **Multiple Patient IDs**: Always test with multiple patients to get a comprehensive assessment of performance across different glucose patterns. For example: `--patient_ids 570,575,588,591`
- **API URL**: Make sure to use the correct port where your API is running:
  - The prediction API typically runs on port 8001 (`http://localhost:8001`)
  - The web interface typically runs on port 5001 (`http://localhost:5001`)
  - For validation, you need to target the API directly, not the web interface

### Test Data Format

The test file should be a CSV with the following columns:
- `patient_id`: The patient identifier
- `glucose_value`: Current glucose value
- `target_Xmin`: Ground truth future glucose value (where X is the horizon in minutes)

Additional optional columns:
- `glucose_diff`: Change in glucose from previous reading
- `glucose_diff_rate`: Rate of change in glucose
- `glucose_rolling_mean_1h`: Average glucose over past hour
- `glucose_rolling_std_1h`: Standard deviation of glucose over past hour
- `hour`: Hour of day (0-23)
- `day_of_week`: Day of week (0-6)
- `insulin_dose`, `insulin_dose_1h`, etc.: Insulin dosage information
- `carbs_1h`, `carbs_2h`, etc.: Carbohydrate intake information
- `glucose_lag_X`: Historical glucose readings

## Output

The validation script generates the following outputs in the specified output directory:

### Per-Patient Results

For each patient, a subdirectory `patient_X` is created containing:

1. **Clarke Error Grid Analysis**: `clarke_grid_patient_X_Ymin.png`
   - Plots predictions against actual glucose values in the clinical context
   - Classifies prediction errors into zones A-E based on clinical significance

2. **Prediction Analysis**: `prediction_analysis_Ymin.png`
   - Scatter plot of actual vs. predicted values
   - Histogram of prediction errors
   - Time series comparison of sample predictions
   - Error distribution by glucose category (hypo, normal, hyper)

3. **Validation Results**: `validation_results.json`
   - Complete metrics including MAE, RMSE, R², in-range accuracy
   - Clarke Error Grid zone percentages
   - API performance statistics

### Summary Results

1. **Summary CSV**: `validation_summary.csv`
   - Tabular data with metrics for all patients and horizons

2. **Summary Visualizations**:
   - `rmse_by_horizon.png`: RMSE comparison across horizons
   - `clarke_a_by_horizon.png`: Clinical accuracy comparison
   - `in_range_accuracy_by_horizon.png`: In-range detection accuracy

## Understanding the Metrics

### Accuracy Metrics

- **MAE**: Mean Absolute Error - average magnitude of errors in mg/dL
- **RMSE**: Root Mean Squared Error - square root of average squared errors in mg/dL (penalizes large errors)
- **R²**: Coefficient of determination - proportion of variance explained by model (higher is better)

### Clinical Metrics

- **In-Range Accuracy**: Percentage of predictions that correctly identify whether glucose is in target range (70-180 mg/dL)
- **Clarke Error Grid**:
  - **Zone A**: Clinically accurate predictions (clinical decisions would be correct)
  - **Zone B**: Benign errors (would not lead to dangerous treatment decisions)
  - **Zone C**: Overcorrection errors (could lead to unnecessary treatment)
  - **Zone D**: Dangerous failure to detect (could lead to failure to treat)
  - **Zone E**: Erroneous treatment (could lead to opposite of correct treatment)

### Performance Metrics

- **API Response Time**: Average time for the API to respond to prediction requests
- **Validation Count**: Number of test cases successfully validated

## Example Usage Scenarios

### Validating Model Updates

After updating prediction models, run validation to ensure accuracy hasn't regressed:

```bash
python validate_api.py --test_file features/2018_test_features.csv --patient_ids 570,575,588,591 --horizons 15,30,45,60
```

### Comparing Performance Across Patients

To analyze how prediction accuracy varies across patients:

```bash
python validate_api.py --test_file features/2018_test_features.csv --patient_ids 570,575,588,591 --output_dir patient_comparison
```

Example output comparison:
```
15min Horizon Summary by Patient:
- Patient 570: RMSE 24.55 mg/dL, Clarke Zone A 89.0%
- Patient 575: RMSE 20.31 mg/dL, Clarke Zone A 91.0%
- Patient 588: RMSE 14.90 mg/dL, Clarke Zone A 97.0% (best performer)
- Patient 591: RMSE 15.72 mg/dL, Clarke Zone A 88.0%
```

This patient-specific analysis helps identify which glucose patterns your model handles best and which need improvement.

### Testing Different Horizons

To evaluate how prediction accuracy degrades with longer horizons:

```bash
python validate_api.py --test_file features/2018_test_features.csv --patient_ids 570 --horizons 15,30,45,60 --output_dir horizon_analysis
```

## Troubleshooting

1. **API Connection Issues**: Ensure the API is running at the specified URL before starting validation
   - For the simple API: `python simple_glucose_api.py` (runs on port 8001)
   - Check if it's running: `curl http://localhost:8001/`

2. **Missing Data**: Verify test file includes required columns and target values for selected horizons

3. **Memory Issues**: For large test files, consider validating fewer patients or horizons at once

## Interpreting Results

Good performance generally means:
- RMSE < 20 mg/dL for 15-minute predictions
- Clarke Error Grid Zone A > 95% for 15-minute predictions
- In-range accuracy > 90%
- Performance degrades gracefully with longer horizons

## Advanced Usage

### Using in Continuous Integration

This validation script can be integrated into CI/CD pipelines to ensure model quality before deployment:

```bash
python validate_api.py --test_file test_data.csv --output_dir ci_results --patient_ids 570,575,588,591
# Add thresholds for CI pass/fail based on metrics
if grep -q "Mean RMSE: [0-9]\{1,2\}\.[0-9]\{2\}" ci_results/validation_summary.txt; then
  echo "Validation passed"
  exit 0
else
  echo "Validation failed - RMSE too high"
  exit 1
fi
``` 