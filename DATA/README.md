# OhioT1DM Dataset

## Dataset Information

The OhioT1DM dataset contains 8 weeks worth of data for each of 12 people with type 1 diabetes. The dataset includes:
- CGM blood glucose level every 5 minutes
- Blood glucose levels from finger sticks
- Insulin doses (bolus and basal)
- Self-reported meal times with carbohydrate estimates
- Self-reported times of exercise, sleep, work, stress, and illness
- Physiological data from fitness bands

## How to Obtain the Dataset

This dataset is not included in this repository as it requires a Data Use Agreement (DUA) to ensure it is used only for research purposes.

To obtain the dataset:
1. Visit the official dataset page: [Ohio T1DM Dataset](https://www.taliaandeducation.org/datasets/ohio-t1dm-dataset)
2. Request the Data Use Agreement by following the instructions on the website
3. Have the DUA signed by a legal signatory for your research institution and the principal investigator
4. Once your DUA is approved, you will receive access to download the dataset

## Directory Structure

After obtaining the dataset, place the files in the following structure:

```
DATA/
├── 2018/           # 2018 dataset
│   ├── Train/      # Training data files
│   └── Test/       # Test data files
└── 2020/           # 2020 dataset
    ├── Train/      # Training data files
    └── Test/       # Test data files
```

## Reference

The OhioT1DM dataset is described in the paper:
"The OhioT1DM Dataset for Blood Glucose Level Prediction: Update 2020"
Available at: https://pmc.ncbi.nlm.nih.gov/articles/PMC7881904/ 