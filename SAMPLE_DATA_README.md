# Sample Data for Blood Glucose Prediction

## Overview

This project uses a sample dataset that mimics the structure and format of the OhioT1DM dataset but contains only synthetic data. This is because:

1. The actual OhioT1DM dataset requires a Data Use Agreement (DUA) to be signed by a legal signatory for your research institution and the principal investigator
2. The dataset is meant to be used only for research purposes under specific terms

## Data Management Scripts

### Quick Management Script

For convenience, we provide a shell script that combines all data management operations:

```bash
# Make the script executable (if needed)
chmod +x manage_data.sh

# Generate synthetic sample data
./manage_data.sh create-sample

# Remove real data but keep sample data
./manage_data.sh clean-keep

# Remove all data files
./manage_data.sh clean

# Create a backup of the data directory
./manage_data.sh backup

# Restore from backup
./manage_data.sh restore

# Display help
./manage_data.sh help
```

### Creating Sample Data

We provide a script `create_sample_data.py` that generates synthetic data in the same format as the OhioT1DM dataset for demonstration and testing purposes:

```bash
python create_sample_data.py
```

This will create synthetic data files with the same structure as the original dataset but containing only artificial values.

### Cleaning Data

If you have the real OhioT1DM dataset in your repository and want to clean it up before sharing your code, you can use the `clean_data.py` script:

```bash
# Create a backup and remove all real data files, keeping only sample data
python clean_data.py --backup --keep_sample

# Remove all data files (including sample data)
python clean_data.py

# Just create a backup without removing any files
python clean_data.py --backup
```

This is useful for:
- Cleaning up your repository before sharing it
- Removing real patient data to comply with the Data Use Agreement
- Creating a backup of your data before making changes

## Using the Sample Data

The sample data can be used with all the scripts and tools in this repository to demonstrate functionality. However, it is important to note that:

1. The sample data is synthetic and does not represent real patient data
2. Results obtained using this sample data should not be used for research purposes
3. For actual research, you should obtain the OhioT1DM dataset through proper channels

## Obtaining the Real Dataset

To obtain the full OhioT1DM dataset for research:

1. Visit the official dataset page: [Ohio T1DM Dataset](https://www.taliaandeducation.org/datasets/ohio-t1dm-dataset)
2. Request the Data Use Agreement by following the instructions on the website
3. Have the DUA signed by a legal signatory for your research institution and the principal investigator
4. Once your DUA is approved, you will receive access to download the dataset

## Directory Structure

After running the sample data generation script or obtaining the actual dataset, place the files in the following structure:

```
DATA/
├── 2018/           # 2018 dataset
│   ├── Train/      # Training data files
│   └── Test/       # Test data files
└── 2020/           # 2020 dataset
    ├── Train/      # Training data files
    └── Test/       # Test data files
``` 