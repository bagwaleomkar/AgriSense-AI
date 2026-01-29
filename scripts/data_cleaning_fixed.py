"""
AgriSense AI - Fixed Data Cleaning Script
This script performs data cleaning without removing all crop data
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
for col in weather_df.columns:
    if weather_df[col].dtype in ['float64', 'int64']:
        weather_df[col].fillna(weather_df[col].median(), inplace=True)
    else:
        if not weather_df[col].mode().empty:
            weather_df[col].fillna(weather_df[col].mode()[0], inplace=True)
        else:
            weather_df[col].fillna('Unknown', inplace=True)

print("\n✓ Weather dataset: Missing values handled")

# Crop Dataset: Handle missing values
for col in crop_df.columns:
    if crop_df[col].dtype in ['float64', 'int64']:
        crop_df[col].fillna(crop_df[col].median(), inplace=True)
    else:
        if not crop_df[col].mode().empty:
            crop_df[col].fillna(crop_df[col].mode()[0], inplace=True)
        else:
            crop_df[col].fillna('Unknown', inplace=True)

print("✓ Crop dataset: Missing values handled")

# Drop columns with all missing values or unnamed columns
if 'Unnamed: 6' in crop_df.columns:
    crop_df.drop('Unnamed: 6', axis=1, inplace=True)
    print("✓ Dropped 'Unnamed: 6' column from crop dataset")

print("\nAfter cleaning:")
print(f"Weather dataset - Missing values: {weather_df.isnull().sum().sum()}")
print(f"Crop dataset - Missing values: {crop_df.isnull().sum().sum()}")

# ============================================================================
# STEP 3: CONVERT DATE COLUMNS TO DATETIME FORMAT
# ============================================================================
print("\n" + "=" * 80)
print("STEP 3: CONVERTING DATE COLUMNS TO DATETIME FORMAT")
print("=" * 80)

# Convert weather date column
if 'Date' in weather_df.columns:
    weather_df['Date'] = pd.to_datetime(weather_df['Date'], errors='coerce')
    print(f"✓ Weather dataset: Converted 'Date' to datetime")

# Convert crop date column (DD-MM-YYYY format)
if 'Date' in crop_df.columns:
    crop_df['Date'] = pd.to_datetime(crop_df['Date'], format='%d-%m-%Y', errors='coerce')
    print(f"✓ Crop dataset: Converted 'Date' to datetime")

# Extract year from date for crop dataset if not present
if 'Date' in crop_df.columns and 'Year' not in crop_df.columns:
    crop_df['Year'] = crop_df['Date'].dt.year
    print(f"✓ Crop dataset: Extracted 'Year' from Date")

print(f"\n✓ Date conversion complete")

# ============================================================================
# STEP 4: REMOVE DUPLICATE ROWS
# ============================================================================
print("\n" + "=" * 80)
print("STEP 4: REMOVING DUPLICATE ROWS")
print("=" * 80)

weather_before = len(weather_df)
crop_before = len(crop_df)

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
# STEP 5: DETECT AND REMOVE OUTLIERS (SELECTIVE APPROACH)
# ============================================================================
print("\n" + "=" * 80)
print("STEP 5: DETECTING AND REMOVING OUTLIERS (SELECTIVE METHOD)")
print("=" * 80)

def remove_outliers_selective(df, columns_to_check):
    """
    Remove outliers only from specified columns using IQR method
    This prevents removing all data when some columns have unusual distributions
    """
    df_clean = df.copy()
    total_outliers = 0
    
    for col in columns_to_check:
        if col in df_clean.columns and df_clean[col].dtype in ['float64', 'int64']:
            Q1 = df_clean[col].quantile(0.25)
            Q3 = df_clean[col].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # Count outliers
            outliers = ((df_clean[col] < lower_bound) | (df_clean[col] > upper_bound)).sum()
            
            if outliers > 0:
                print(f"  {col}: Found {outliers} outliers [Range: {lower_bound:.2f} to {upper_bound:.2f}]")
                total_outliers += outliers
            
            # Remove outliers
            df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]
    
    return df_clean, total_outliers

# For weather dataset - check temperature, rainfall, humidity, wind speed
print("\nWeather Dataset - Outlier Detection:")
weather_cols_to_check = ['Daily_Rainfall_mm', 'Max_Temp_C', 'Min_Temp_C', 
                         'Avg_Humidity_%', 'Wind_Speed_km_h']
weather_before_outliers = len(weather_df)
weather_df, weather_outliers = remove_outliers_selective(weather_df, weather_cols_to_check)
print(f"✓ Removed {weather_before_outliers - len(weather_df)} outlier records from weather dataset")

# For crop dataset - check prices and arrivals (be lenient)
print("\nCrop Dataset - Outlier Detection:")
print("  Note: Using lenient outlier detection for price data")
crop_cols_to_check = ['Arrivals', 'Min_Price', 'Max_Price', 'Modal_Price']

# Use a more lenient threshold (3*IQR instead of 1.5*IQR) for crop prices
df_clean = crop_df.copy()
crop_outliers = 0

for col in crop_cols_to_check:
    if col in df_clean.columns and df_clean[col].dtype in ['float64', 'int64']:
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        
        # More lenient bounds (3*IQR instead of 1.5*IQR)
        lower_bound = Q1 - 3 * IQR
        upper_bound = Q3 + 3 * IQR
        
        outliers = ((df_clean[col] < lower_bound) | (df_clean[col] > upper_bound)).sum()
        
        if outliers > 0:
            print(f"  {col}: Found {outliers} extreme outliers [Range: {lower_bound:.2f} to {upper_bound:.2f}]")
            crop_outliers += outliers
        
        df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]

crop_df = df_clean
print(f"✓ Removed {crop_outliers} extreme outlier records from crop dataset")

print(f"\nFinal shapes after outlier removal:")
print(f"  Weather: {weather_df.shape}")
print(f"  Crop: {crop_df.shape}")

# ============================================================================
# STEP 6: STANDARDIZE DISTRICT NAMES
# ============================================================================
print("\n" + "=" * 80)
print("STEP 6: STANDARDIZING DISTRICT NAMES")
print("=" * 80)

# Standardize weather district column
if 'District' in weather_df.columns:
    weather_df['District'] = weather_df['District'].str.lower().str.strip().str.replace(r'\s+', ' ', regex=True)
    print(f"✓ Weather dataset: Standardized 'District'")

# Standardize crop district column
if 'District' in crop_df.columns:
    crop_df['District'] = crop_df['District'].str.lower().str.strip().str.replace(r'\s+', ' ', regex=True)
    print(f"✓ Crop dataset: Standardized 'District'")

print(f"\n✓ District name standardization complete")

# Display sample districts for verification
print(f"\nSample districts from weather dataset:")
print(f"  {weather_df['District'].unique()[:5].tolist()}")
print(f"\nSample districts from crop dataset:")
print(f"  {crop_df['District'].unique()[:5].tolist()}")

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
print(f"   Final shape: {weather_df.shape}")
print(f"   Unique districts: {weather_df['District'].nunique()}")

print("\n📊 CROP DATASET:")
print(f"   Original rows: {crop_before}")
print(f"   After removing duplicates: {crop_before - crop_removed}")
print(f"   After removing outliers: {len(crop_df)}")
print(f"   Final shape: {crop_df.shape}")
print(f"   Unique districts: {crop_df['District'].nunique()}")
print(f"   Unique commodities: {crop_df['Commodity'].nunique()}")
print(f"   Date range: {crop_df['Date'].min()} to {crop_df['Date'].max()}")

print("\n✅ DATA CLEANING PROCESS COMPLETED SUCCESSFULLY")
print("=" * 80)
