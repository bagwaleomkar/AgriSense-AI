"""
AgriSense AI - Data Merging Script
This script merges cleaned crop and weather datasets on District and Date
"""

import pandas as pd
import numpy as np
from datetime import datetime

# ============================================================================
# STEP 1: LOAD CLEANED DATASETS
# ============================================================================
print("=" * 80)
print("STEP 1: LOADING CLEANED DATASETS")
print("=" * 80)

weather_df = pd.read_csv('Maharashtra_Weather_Cleaned.csv')
crop_df = pd.read_csv('Crop_Dataset_Cleaned.csv')

print(f"✓ Weather dataset loaded: {weather_df.shape}")
print(f"✓ Crop dataset loaded: {crop_df.shape}")

print("\nWeather dataset columns:")
print(weather_df.columns.tolist())

print("\nCrop dataset columns:")
print(crop_df.columns.tolist())

# ============================================================================
# STEP 2: IDENTIFY MERGE KEYS (DISTRICT AND DATE COLUMNS)
# ============================================================================
print("\n" + "=" * 80)
print("STEP 2: IDENTIFYING MERGE KEYS")
print("=" * 80)

# Identify district columns
weather_district_col = None
crop_district_col = None

for col in weather_df.columns:
    if any(keyword in col.lower() for keyword in ['district', 'location', 'area', 'region']):
        weather_district_col = col
        break

for col in crop_df.columns:
    if any(keyword in col.lower() for keyword in ['district', 'location', 'area', 'region']):
        crop_district_col = col
        break

print(f"Weather district column: {weather_district_col}")
print(f"Crop district column: {crop_district_col}")

# Identify date columns
weather_date_col = None
crop_date_col = None

for col in weather_df.columns:
    if any(keyword in col.lower() for keyword in ['date', 'time']) and weather_df[col].dtype != 'object':
        weather_date_col = col
        break

for col in crop_df.columns:
    if any(keyword in col.lower() for keyword in ['date', 'time']) and crop_df[col].dtype != 'object':
        crop_date_col = col
        break

# If date columns not found in datetime format, check for year/month columns
if weather_date_col is None:
    print("\n⚠ No datetime column found in weather dataset. Checking for year/month columns...")
    year_cols = [col for col in weather_df.columns if 'year' in col.lower()]
    month_cols = [col for col in weather_df.columns if 'month' in col.lower()]
    if year_cols and month_cols:
        print(f"  Found: {year_cols[0]} and {month_cols[0]}")
        # Create date column from year and month
        weather_df['date'] = pd.to_datetime(
            weather_df[year_cols[0]].astype(str) + '-' + 
            weather_df[month_cols[0]].astype(str) + '-01',
            errors='coerce'
        )
        weather_date_col = 'date'
        print(f"  ✓ Created date column from year and month")

if crop_date_col is None:
    print("\n⚠ No datetime column found in crop dataset. Checking for year/month columns...")
    year_cols = [col for col in crop_df.columns if 'year' in col.lower()]
    month_cols = [col for col in crop_df.columns if 'month' in col.lower()]
    if year_cols and month_cols:
        print(f"  Found: {year_cols[0]} and {month_cols[0]}")
        # Create date column from year and month
        crop_df['date'] = pd.to_datetime(
            crop_df[year_cols[0]].astype(str) + '-' + 
            crop_df[month_cols[0]].astype(str) + '-01',
            errors='coerce'
        )
        crop_date_col = 'date'
        print(f"  ✓ Created date column from year and month")

print(f"\nWeather date column: {weather_date_col}")
print(f"Crop date column: {crop_date_col}")

# ============================================================================
# STEP 3: PREPARE DATA FOR MERGING
# ============================================================================
print("\n" + "=" * 80)
print("STEP 3: PREPARING DATA FOR MERGING")
print("=" * 80)

# Ensure district columns exist
if weather_district_col is None or crop_district_col is None:
    print("❌ ERROR: Could not identify district columns in both datasets")
    print("   Please check your data structure")
else:
    # Standardize district names (ensure consistency)
    weather_df[weather_district_col] = weather_df[weather_district_col].str.lower().str.strip()
    crop_df[crop_district_col] = crop_df[crop_district_col].str.lower().str.strip()
    print(f"✓ Standardized district names")

# Ensure date columns exist and are in datetime format
if weather_date_col and crop_date_col:
    # Convert to datetime if not already
    if weather_df[weather_date_col].dtype != 'datetime64[ns]':
        weather_df[weather_date_col] = pd.to_datetime(weather_df[weather_date_col], errors='coerce')
    
    if crop_df[crop_date_col].dtype != 'datetime64[ns]':
        crop_df[crop_date_col] = pd.to_datetime(crop_df[crop_date_col], errors='coerce')
    
    print(f"✓ Date columns converted to datetime format")
    
    # Handle date mismatches by normalizing to month-level
    # This helps match data even if exact dates don't align
    print("\n📅 Handling date mismatches:")
    print("   Strategy: Normalizing dates to year-month level for better matching")
    
    weather_df['year_month'] = weather_df[weather_date_col].dt.to_period('M')
    crop_df['year_month'] = crop_df[crop_date_col].dt.to_period('M')
    
    print(f"   ✓ Created year_month period columns")
    print(f"   Weather date range: {weather_df[weather_date_col].min()} to {weather_df[weather_date_col].max()}")
    print(f"   Crop date range: {crop_df[crop_date_col].min()} to {crop_df[crop_date_col].max()}")

# Display unique districts to check for overlap
if weather_district_col and crop_district_col:
    print(f"\n📍 Unique districts:")
    weather_districts = set(weather_df[weather_district_col].unique())
    crop_districts = set(crop_df[crop_district_col].unique())
    common_districts = weather_districts.intersection(crop_districts)
    
    print(f"   Weather dataset: {len(weather_districts)} districts")
    print(f"   Crop dataset: {len(crop_districts)} districts")
    print(f"   Common districts: {len(common_districts)} districts")
    
    if len(common_districts) > 0:
        print(f"\n   Sample common districts: {list(common_districts)[:5]}")
    else:
        print("\n   ⚠ WARNING: No common districts found! Check district name standardization")

# ============================================================================
# STEP 4: PERFORM INNER JOIN MERGE
# ============================================================================
print("\n" + "=" * 80)
print("STEP 4: MERGING DATASETS (INNER JOIN)")
print("=" * 80)

if weather_district_col and crop_district_col and weather_date_col and crop_date_col:
    # Rename columns to standardize before merge
    weather_df_merge = weather_df.rename(columns={
        weather_district_col: 'district',
        weather_date_col: 'date'
    })
    
    crop_df_merge = crop_df.rename(columns={
        crop_district_col: 'district',
        crop_date_col: 'date'
    })
    
    # Perform inner join on district and year_month
    merged_df = pd.merge(
        crop_df_merge,
        weather_df_merge,
        on=['district', 'year_month'],
        how='inner',
        suffixes=('_crop', '_weather')
    )
    
    print(f"✓ Merge completed successfully!")
    print(f"  Merge keys: district + year_month")
    print(f"  Join type: INNER JOIN")
    
else:
    print("❌ ERROR: Could not perform merge - missing required columns")
    merged_df = None

# ============================================================================
# STEP 5: DISPLAY MERGED DATASET INFORMATION
# ============================================================================
if merged_df is not None:
    print("\n" + "=" * 80)
    print("STEP 5: MERGED DATASET INFORMATION")
    print("=" * 80)
    
    print(f"\n📊 DATASET SHAPE:")
    print(f"   Rows: {merged_df.shape[0]:,}")
    print(f"   Columns: {merged_df.shape[1]}")
    
    print(f"\n📋 COLUMN NAMES ({merged_df.shape[1]} columns):")
    for idx, col in enumerate(merged_df.columns, 1):
        print(f"   {idx:2d}. {col}")
    
    print(f"\n🔢 DATA TYPES:")
    dtype_summary = merged_df.dtypes.value_counts()
    for dtype, count in dtype_summary.items():
        print(f"   {dtype}: {count} columns")
    
    print(f"\n📈 SUMMARY STATISTICS:")
    print(f"   Numerical columns: {merged_df.select_dtypes(include=['float64', 'int64']).shape[1]}")
    print(f"   Categorical columns: {merged_df.select_dtypes(include=['object']).shape[1]}")
    print(f"   Datetime columns: {merged_df.select_dtypes(include=['datetime64']).shape[1]}")
    
    print(f"\n❓ MISSING VALUES:")
    missing_values = merged_df.isnull().sum()
    total_missing = missing_values.sum()
    if total_missing > 0:
        print(f"   Total missing values: {total_missing:,}")
        print(f"   Columns with missing values:")
        for col, count in missing_values[missing_values > 0].items():
            print(f"      {col}: {count} ({count/len(merged_df)*100:.2f}%)")
    else:
        print(f"   ✓ No missing values in merged dataset!")
    
    print(f"\n🎯 FIRST 5 ROWS OF MERGED DATASET:")
    print("-" * 80)
    print(merged_df.head())
    
    print(f"\n📊 STATISTICAL SUMMARY OF NUMERICAL COLUMNS:")
    print("-" * 80)
    print(merged_df.describe())
    
    # ========================================================================
    # STEP 6: SAVE MERGED DATASET
    # ========================================================================
    print("\n" + "=" * 80)
    print("STEP 6: SAVING MERGED DATASET")
    print("=" * 80)
    
    merged_df.to_csv('Merged_Crop_Weather_Data.csv', index=False)
    print("✓ Merged dataset saved as: Merged_Crop_Weather_Data.csv")
    
    # ========================================================================
    # MERGE SUMMARY REPORT
    # ========================================================================
    print("\n" + "=" * 80)
    print("MERGE SUMMARY REPORT")
    print("=" * 80)
    
    print(f"""
📊 INPUT DATASETS:
   Weather dataset: {weather_df.shape[0]:,} rows × {weather_df.shape[1]} columns
   Crop dataset: {crop_df.shape[0]:,} rows × {crop_df.shape[1]} columns

🔗 MERGE CONFIGURATION:
   Join type: INNER JOIN
   Merge keys: District + Year-Month
   Date handling: Normalized to monthly periods

📈 OUTPUT DATASET:
   Merged dataset: {merged_df.shape[0]:,} rows × {merged_df.shape[1]} columns
   
💡 DATA RETENTION:
   Rows retained: {merged_df.shape[0]:,} / {min(weather_df.shape[0], crop_df.shape[0]):,} 
   Retention rate: {(merged_df.shape[0] / min(weather_df.shape[0], crop_df.shape[0]) * 100):.2f}%
   
✅ MERGE COMPLETED SUCCESSFULLY!
    """)
    
    print("=" * 80)
    
    # Additional insights
    print("\n💡 MERGE INSIGHTS:")
    print(f"   • Districts in merged data: {merged_df['district'].nunique()}")
    print(f"   • Date range: {merged_df['date_crop'].min()} to {merged_df['date_crop'].max()}")
    print(f"   • Average records per district: {merged_df.shape[0] / merged_df['district'].nunique():.1f}")
    
else:
    print("\n❌ Merge failed - please check the error messages above")

print("\n" + "=" * 80)
print("DATA MERGING PROCESS COMPLETED")
print("=" * 80)
