"""
AgriSense AI - Data Cleaning and Preprocessing Script
This script performs comprehensive data cleaning on weather and crop datasets
"""

import pandas as pd
import numpy as np
from datetime import datetime

# ============================================================================
# STEP 1: LOAD DATASETS
# ============================================================================
print("=" * 80)
print("STEP 1: LOADING DATASETS")
print("=" * 80)

weather_df = pd.read_csv('Maharashtra_Weather - Maharashtra_Weather (1).csv')
crop_df = pd.read_csv('Crop Dataset.csv')

print(f"✓ Weather dataset loaded: {weather_df.shape}")
print(f"✓ Crop dataset loaded: {crop_df.shape}")

# ============================================================================
# STEP 2: HANDLE MISSING VALUES
# ============================================================================
print("\n" + "=" * 80)
print("STEP 2: HANDLING MISSING VALUES")
print("=" * 80)

print("\nBefore cleaning:")
print(f"Weather dataset - Missing values:\n{weather_df.isnull().sum()}")
print(f"\nCrop dataset - Missing values:\n{crop_df.isnull().sum()}")

# Weather Dataset: Handle missing values
# Strategy:
# - Numerical columns: Fill with median (robust to outliers)
# - Categorical columns: Fill with mode or 'Unknown'
for col in weather_df.columns:
    if weather_df[col].dtype in ['float64', 'int64']:
        # Fill numerical columns with median
        weather_df[col].fillna(weather_df[col].median(), inplace=True)
    else:
        # Fill categorical columns with mode or 'Unknown'
        if not weather_df[col].mode().empty:
            weather_df[col].fillna(weather_df[col].mode()[0], inplace=True)
        else:
            weather_df[col].fillna('Unknown', inplace=True)

print("\n✓ Weather dataset: Missing values handled using median for numerical, mode for categorical")

# Crop Dataset: Handle missing values
for col in crop_df.columns:
    if crop_df[col].dtype in ['float64', 'int64']:
        crop_df[col].fillna(crop_df[col].median(), inplace=True)
    else:
        if not crop_df[col].mode().empty:
            crop_df[col].fillna(crop_df[col].mode()[0], inplace=True)
        else:
            crop_df[col].fillna('Unknown', inplace=True)

print("✓ Crop dataset: Missing values handled using median for numerical, mode for categorical")

print("\nAfter cleaning:")
print(f"Weather dataset - Missing values: {weather_df.isnull().sum().sum()}")
print(f"Crop dataset - Missing values: {crop_df.isnull().sum().sum()}")

# ============================================================================
# STEP 3: CONVERT DATE COLUMNS TO DATETIME FORMAT
# ============================================================================
print("\n" + "=" * 80)
print("STEP 3: CONVERTING DATE COLUMNS TO DATETIME FORMAT")
print("=" * 80)

# Identify date columns (columns with 'date', 'year', 'month' in name)
weather_date_cols = [col for col in weather_df.columns 
                     if any(keyword in col.lower() for keyword in ['date', 'time'])]
crop_date_cols = [col for col in crop_df.columns 
                  if any(keyword in col.lower() for keyword in ['date', 'time', 'year'])]

# Convert weather date columns
for col in weather_date_cols:
    try:
        weather_df[col] = pd.to_datetime(weather_df[col], errors='coerce')
        print(f"✓ Weather dataset: Converted '{col}' to datetime")
    except Exception as e:
        print(f"✗ Weather dataset: Could not convert '{col}' - {str(e)}")

# Convert crop date columns
for col in crop_date_cols:
    try:
        # Try different datetime formats
        crop_df[col] = pd.to_datetime(crop_df[col], errors='coerce')
        print(f"✓ Crop dataset: Converted '{col}' to datetime")
    except Exception as e:
        print(f"✗ Crop dataset: Could not convert '{col}' - {str(e)}")

print(f"\n✓ Date conversion complete")

# ============================================================================
# STEP 4: REMOVE DUPLICATE ROWS
# ============================================================================
print("\n" + "=" * 80)
print("STEP 4: REMOVING DUPLICATE ROWS")
print("=" * 80)

weather_before = len(weather_df)
crop_before = len(crop_df)

# Remove exact duplicates
weather_df.drop_duplicates(inplace=True)
crop_df.drop_duplicates(inplace=True)

weather_removed = weather_before - len(weather_df)
crop_removed = crop_before - len(crop_df)

print(f"✓ Weather dataset: Removed {weather_removed} duplicate rows")
print(f"✓ Crop dataset: Removed {crop_removed} duplicate rows")
print(f"\nNew shapes:")
print(f"  Weather: {weather_df.shape}")
print(f"  Crop: {crop_df.shape}")

# ============================================================================
# STEP 5: DETECT AND REMOVE OUTLIERS USING IQR METHOD
# ============================================================================
print("\n" + "=" * 80)
print("STEP 5: DETECTING AND REMOVING OUTLIERS (IQR METHOD)")
print("=" * 80)

def remove_outliers_iqr(df, columns=None):
    """
    Remove outliers using Interquartile Range (IQR) method
    
    IQR Method:
    - Q1 = 25th percentile, Q3 = 75th percentile
    - IQR = Q3 - Q1
    - Lower bound = Q1 - 1.5 * IQR
    - Upper bound = Q3 + 1.5 * IQR
    - Values outside these bounds are considered outliers
    """
    df_clean = df.copy()
    
    if columns is None:
        # Select only numerical columns
        columns = df_clean.select_dtypes(include=['float64', 'int64']).columns
    
    outliers_removed = 0
    
    for col in columns:
        if df_clean[col].dtype in ['float64', 'int64']:
            Q1 = df_clean[col].quantile(0.25)
            Q3 = df_clean[col].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # Count outliers
            outliers = ((df_clean[col] < lower_bound) | (df_clean[col] > upper_bound)).sum()
            
            if outliers > 0:
                print(f"  {col}: Found {outliers} outliers [Range: {lower_bound:.2f} to {upper_bound:.2f}]")
                outliers_removed += outliers
            
            # Remove outliers
            df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]
    
    return df_clean, outliers_removed

print("\nWeather Dataset - Outlier Detection:")
weather_before_outliers = len(weather_df)
weather_df, weather_outliers = remove_outliers_iqr(weather_df)
print(f"✓ Removed {weather_outliers} outlier records from weather dataset")

print("\nCrop Dataset - Outlier Detection:")
crop_before_outliers = len(crop_df)
crop_df, crop_outliers = remove_outliers_iqr(crop_df)
print(f"✓ Removed {crop_outliers} outlier records from crop dataset")

print(f"\nFinal shapes after outlier removal:")
print(f"  Weather: {weather_df.shape}")
print(f"  Crop: {crop_df.shape}")

# ============================================================================
# STEP 6: STANDARDIZE DISTRICT NAMES
# ============================================================================
print("\n" + "=" * 80)
print("STEP 6: STANDARDIZING DISTRICT NAMES")
print("=" * 80)

# Identify district columns
district_cols_weather = [col for col in weather_df.columns 
                         if any(keyword in col.lower() for keyword in ['district', 'location', 'area', 'region'])]
district_cols_crop = [col for col in crop_df.columns 
                      if any(keyword in col.lower() for keyword in ['district', 'location', 'area', 'region'])]

# Standardize weather district columns
for col in district_cols_weather:
    if weather_df[col].dtype == 'object':
        # Convert to lowercase, strip whitespace, remove extra spaces
        weather_df[col] = weather_df[col].str.lower().str.strip().str.replace(r'\s+', ' ', regex=True)
        print(f"✓ Weather dataset: Standardized '{col}'")

# Standardize crop district columns
for col in district_cols_crop:
    if crop_df[col].dtype == 'object':
        crop_df[col] = crop_df[col].str.lower().str.strip().str.replace(r'\s+', ' ', regex=True)
        print(f"✓ Crop dataset: Standardized '{col}'")

print(f"\n✓ District name standardization complete")

# ============================================================================
# STEP 7: SAVE CLEANED DATASETS
# ============================================================================
print("\n" + "=" * 80)
print("STEP 7: SAVING CLEANED DATASETS")
print("=" * 80)

weather_df.to_csv('Maharashtra_Weather_Cleaned.csv', index=False)
crop_df.to_csv('Crop_Dataset_Cleaned.csv', index=False)

print("✓ Cleaned weather dataset saved as: Maharashtra_Weather_Cleaned.csv")
print("✓ Cleaned crop dataset saved as: Crop_Dataset_Cleaned.csv")

# ============================================================================
# SUMMARY REPORT
# ============================================================================
print("\n" + "=" * 80)
print("DATA CLEANING SUMMARY REPORT")
print("=" * 80)

print("\n📊 WEATHER DATASET:")
print(f"   Original rows: {weather_before}")
print(f"   After removing duplicates: {weather_before - weather_removed}")
print(f"   After removing outliers: {len(weather_df)}")
print(f"   Total rows removed: {weather_before - len(weather_df)} ({((weather_before - len(weather_df))/weather_before*100):.2f}%)")
print(f"   Final shape: {weather_df.shape}")

print("\n📊 CROP DATASET:")
print(f"   Original rows: {crop_before}")
print(f"   After removing duplicates: {crop_before - crop_removed}")
print(f"   After removing outliers: {len(crop_df)}")
print(f"   Total rows removed: {crop_before - len(crop_df)} ({((crop_before - len(crop_df))/crop_before*100):.2f}%)")
print(f"   Final shape: {crop_df.shape}")

print("\n✅ DATA CLEANING PROCESS COMPLETED SUCCESSFULLY")
print("=" * 80)

# ============================================================================
# PREPROCESSING STEPS EXPLANATION
# ============================================================================
print("\n" + "=" * 80)
print("PREPROCESSING STEPS EXPLANATION")
print("=" * 80)

explanation = """
1. MISSING VALUE HANDLING:
   - Numerical columns: Filled with MEDIAN (robust to outliers, better than mean)
   - Categorical columns: Filled with MODE (most frequent value)
   - Reason: Preserves data distribution without introducing bias

2. DATE CONVERSION:
   - Converts date strings to datetime objects
   - Enables time-series analysis and date-based operations
   - Reason: Required for temporal analysis and forecasting

3. DUPLICATE REMOVAL:
   - Removes exact duplicate rows
   - Prevents data redundancy and bias in ML models
   - Reason: Duplicates can skew statistical analysis and model training

4. OUTLIER REMOVAL (IQR METHOD):
   - Q1 (25th percentile), Q3 (75th percentile)
   - IQR = Q3 - Q1
   - Outlier bounds: [Q1 - 1.5*IQR, Q3 + 1.5*IQR]
   - Reason: Removes extreme values that can distort model predictions

5. DISTRICT NAME STANDARDIZATION:
   - Lowercase conversion for consistency
   - Whitespace trimming (leading/trailing)
   - Multiple space reduction to single space
   - Reason: Ensures consistent matching and joining across datasets
"""

print(explanation)
print("=" * 80)
