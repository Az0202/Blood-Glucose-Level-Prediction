# Glucose Prediction API Validation Suite

## Overview

This validation suite contains comprehensive tools for testing and validating the accuracy and performance of glucose prediction APIs. It allows for thorough evaluation of prediction quality against historical data, as well as comparisons between different API implementations.

## Components

### 1. API Validation Script (`validate_api.py`)

This script performs comprehensive validation of a glucose prediction API against historical test data:

- Tests API predictions against ground truth values
- Calculates key metrics (MAE, RMSE, R²)
- Performs Clarke Error Grid analysis for clinical relevance
- Analyzes prediction accuracy across different glucose ranges
- Generates visualizations and detailed reports
- Supports multiple patients and prediction horizons

**Usage Example:**
```bash
./validate_api.py --test_file features/570_test_features.csv --patient_ids 570,575 --horizons 15,30 --api_url http://localhost:8001
```

### 2. API Comparison Tool (`compare_api_predictions.py`)

This tool compares predictions between two different API implementations:

- Makes identical requests to two different APIs
- Calculates differences between predictions
- Analyzes how predictions compare to ground truth
- Visualizes agreement and disagreement between implementations
- Supports side-by-side metric comparison

**Usage Example:**
```bash
./compare_api_predictions.py --test_file features/570_test_features.csv --patient_id 570 --api_url_1 http://localhost:8000 --api_url_2 http://localhost:8001 --api_1_name "Original" --api_2_name "Simple"
```

### 3. Documentation

- **VALIDATION_README.md**: Comprehensive documentation for using the validation script
- **VALIDATION_RESULTS_REPORT.md**: Analysis of validation results from the simple glucose API

## Validation Results Summary

The validation of the Simple Glucose Prediction API showed:

- Strong performance for 15-minute predictions (RMSE: 18.87 mg/dL)
- High clinical accuracy (91.25% in Clarke Error Grid Zone A for 15-minute predictions)
- Degradation for longer horizons (30-minute RMSE: 29.84 mg/dL)
- Excellent API performance (average response time: 1.86ms)
- Variability across patients, with Patient 588 showing the best overall performance

## Next Steps

1. **Fix Original API**: Once the version mismatch in the original API is resolved, use the comparison tool to analyze differences between the original model-based predictions and the simple rule-based predictions.

2. **Extend Validation**: Add support for additional metrics such as prediction lag analysis and glucose rate-of-change accuracy.

3. **Continuous Integration**: Integrate validation into a CI/CD pipeline to automatically test API changes.

4. **Optimize Models**: Use validation insights to improve prediction models, especially for longer prediction horizons.

5. **Real-time Validation**: Implement a real-time validation component that continuously monitors prediction accuracy during live operation.

## Conclusion

This validation suite provides the tools needed to ensure the glucose prediction system meets accuracy and performance requirements. It enables data-driven decision making about model improvements and helps ensure clinical safety of the predictions for diabetes management. 