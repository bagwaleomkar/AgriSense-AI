"""
AgriSense AI - Data Exploration Script
This script loads and explores the weather and crop datasets
"""

import pandas as pd
import numpy as np

# Load datasets
print("=" * 80)
print("LOADING DATASETS")
print("=" * 80)

# Load Maharashtra Weather Dataset
weather_df = pd.read_csv('Maharashtra_Weather - Maharashtra_Weather (1).csv')
print("\n✓ Maharashtra Weather dataset loaded successfully")

# Load Crop Dataset
crop_df = pd.read_csv('Crop Dataset.csv')
print("✓ Crop dataset loaded successfully")

# ============================================================================
# MAHARASHTRA WEATHER DATASET ANALYSIS
# ============================================================================
print("\n" + "=" * 80)
print("MAHARASHTRA WEATHER DATASET")
print("=" * 80)

# Display shape
print(f"\n1. SHAPE:")
print(f"   Rows: {weather_df.shape[0]}")
print(f"   Columns: {weather_df.shape[1]}")

# Display column names
print(f"\n2. COLUMN NAMES:")
for idx, col in enumerate(weather_df.columns, 1):
    print(f"   {idx}. {col}")

# Display data types
print(f"\n3. DATA TYPES:")
print(weather_df.dtypes.to_string())

# Display first 5 rows
print(f"\n4. FIRST 5 ROWS:")
print("-" * 80)
print(weather_df.head())

# ============================================================================
# CROP DATASET ANALYSIS
# ============================================================================
print("\n\n" + "=" * 80)
print("CROP DATASET")
print("=" * 80)

# Display shape
print(f"\n1. SHAPE:")
print(f"   Rows: {crop_df.shape[0]}")
print(f"   Columns: {crop_df.shape[1]}")

# Display column names
print(f"\n2. COLUMN NAMES:")
for idx, col in enumerate(crop_df.columns, 1):
    print(f"   {idx}. {col}")

# Display data types
print(f"\n3. DATA TYPES:")
print(crop_df.dtypes.to_string())

# Display first 5 rows
print(f"\n4. FIRST 5 ROWS:")
print("-" * 80)
print(crop_df.head())

# ============================================================================
# SUMMARY STATISTICS
# ============================================================================
print("\n\n" + "=" * 80)
print("SUMMARY STATISTICS")
print("=" * 80)

print("\n1. WEATHER DATASET - Missing Values:")
print(weather_df.isnull().sum())

print("\n2. CROP DATASET - Missing Values:")
print(crop_df.isnull().sum())

print("\n" + "=" * 80)
print("DATA EXPLORATION COMPLETE")
print("=" * 80)
