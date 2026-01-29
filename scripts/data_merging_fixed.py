"""
AgriSense AI - Improved Data Merging Script
This script intelligently merges crop and weather data handling date mismatches
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

# ============================================================================
# STEP 2: PREPARE DATE COLUMNS
# ============================================================================
print("\n" + "=" * 80)
print("STEP 2: PREPARING DATE COLUMNS")
print("=" * 80)

# Convert date columns to datetime
weather_df['Date'] = pd.to_datetime(weather_df['Date'])
crop_df['Date'] = pd.to_datetime(crop_df['Date'])

print(f"✓ Date columns converted to datetime")

# Extract year and month for merging
weather_df['Year'] = weather_df['Date'].dt.year
weather_df['Month'] = weather_df['Date'].dt.month
crop_df['Year'] = crop_df['Date'].dt.year
crop_df['Month'] = crop_df['Date'].dt.month

print(f"✓ Extracted Year and Month for merging")

print(f"\n📅 Date Ranges:")
print(f"   Weather: {weather_df['Date'].min()} to {weather_df['Date'].max()}")
print(f"   Crop: {crop_df['Date'].min()} to {crop_df['Date'].max()}")

# Check for date overlap
weather_years = set(weather_df['Year'].unique())
crop_years = set(crop_df['Year'].unique())
common_years = weather_years.intersection(crop_years)

print(f"\n📆 Year Analysis:")
print(f"   Weather years: {sorted(weather_years)}")
print(f"   Crop years: {sorted(crop_years)}")
print(f"   Common years: {sorted(common_years) if common_years else 'No overlap'}")

# ============================================================================
# STEP 3: HANDLE DATE MISMATCH
# ============================================================================
print("\n" + "=" * 80)
print("STEP 3: HANDLING DATE MISMATCH")
print("=" * 80)

if not common_years:
    print("⚠ WARNING: No common years between datasets!")
    print("\n🔧 SOLUTION: Using month-based aggregation instead of exact dates")
    print("   Strategy: Match crop data with weather patterns by district + month")
    print("   This allows us to correlate prices with typical weather patterns\n")
    
    # Aggregate weather data by district and month (average values)
    weather_agg = weather_df.groupby(['District', 'Month']).agg({
        'Daily_Rainfall_mm': 'mean',
        'Max_Temp_C': 'mean',
        'Min_Temp_C': 'mean',
        'Avg_Humidity_%': 'mean',
        'Wind_Speed_km_h': 'mean'
    }).reset_index()
    
    # Rename columns for clarity
    weather_agg.columns = ['District', 'Month', 'Avg_Rainfall_mm', 
                           'Avg_Max_Temp_C', 'Avg_Min_Temp_C', 
                           'Avg_Humidity_%', 'Avg_Wind_Speed_km_h']
    
    print(f"✓ Aggregated weather data by district and month: {weather_agg.shape}")
    
    # Prepare crop data for merging
    crop_merge = crop_df.copy()
    
else:
    print("✓ Found common years - using date-based matching")
    # Filter for common years only
    weather_merge = weather_df[weather_df['Year'].isin(common_years)].copy()
    crop_merge = crop_df[crop_df['Year'].isin(common_years)].copy()
    
    # Ensure Month column is numeric in both
    weather_merge['Month'] = pd.to_numeric(weather_merge['Month'], errors='coerce')
    crop_merge['Month'] = pd.to_numeric(crop_merge['Month'], errors='coerce')
    
    print(f"✓ Filtered to common years: {sorted(common_years)}")
    print(f"   Weather: {weather_merge.shape}")
    print(f"   Crop: {crop_merge.shape}")

# ============================================================================
# STEP 4: CHECK DISTRICT OVERLAP
# ============================================================================
print("\n" + "=" * 80)
print("STEP 4: CHECKING DISTRICT OVERLAP")
print("=" * 80)

weather_districts = set(weather_df['District'].unique())
crop_districts = set(crop_df['District'].unique())
common_districts = weather_districts.intersection(crop_districts)

print(f"📍 District Analysis:")
print(f"   Weather districts: {len(weather_districts)}")
print(f"   Crop districts: {len(crop_districts)}")
print(f"   Common districts: {len(common_districts)}")

if len(common_districts) > 0:
    print(f"\n   ✓ Common districts found: {sorted(list(common_districts))[:10]}")
else:
    print(f"\n   ⚠ No exact matches - showing samples:")
    print(f"   Weather sample: {sorted(list(weather_districts))[:5]}")
    print(f"   Crop sample: {sorted(list(crop_districts))[:5]}")

# ============================================================================
# STEP 5: PERFORM MERGE
# ============================================================================
print("\n" + "=" * 80)
print("STEP 5: PERFORMING INNER JOIN MERGE")
print("=" * 80)

if not common_years:
    # Merge on District and Month only (weather patterns)
    merged_df = pd.merge(
        crop_df,
        weather_agg,
        on=['District', 'Month'],
        how='inner',
        suffixes=('', '_weather')
    )
    merge_keys = ['District', 'Month']
    
else:
    # Merge on District, Year, and Month
    merged_df = pd.merge(
        crop_merge,
        weather_merge,
        on=['District', 'Year', 'Month'],
        how='inner',
        suffixes=('_crop', '_weather')
    )
    merge_keys = ['District', 'Year', 'Month']

print(f"✓ Merge completed!")
print(f"  Merge keys: {', '.join(merge_keys)}")
print(f"  Join type: INNER JOIN")
print(f"  Result: {merged_df.shape}")

# ============================================================================
# STEP 6: MERGED DATASET INFORMATION
# ============================================================================
if len(merged_df) > 0:
    print("\n" + "=" * 80)
    print("STEP 6: MERGED DATASET INFORMATION")
    print("=" * 80)
    
    print(f"\n📊 DATASET SHAPE:")
    print(f"   Rows: {merged_df.shape[0]:,}")
    print(f"   Columns: {merged_df.shape[1]}")
    
    print(f"\n📋 COLUMN NAMES ({merged_df.shape[1]} columns):")
    for idx, col in enumerate(merged_df.columns, 1):
        print(f"   {idx:2d}. {col}")
    
    print(f"\n📈 DATA SUMMARY:")
    print(f"   Numerical columns: {merged_df.select_dtypes(include=['float64', 'int64']).shape[1]}")
    print(f"   Categorical columns: {merged_df.select_dtypes(include=['object']).shape[1]}")
    print(f"   Districts: {merged_df['District'].nunique()}")
    print(f"   Commodities: {merged_df['Commodity'].nunique() if 'Commodity' in merged_df.columns else 'N/A'}")
    
    print(f"\n❓ MISSING VALUES:")
    total_missing = merged_df.isnull().sum().sum()
    if total_missing > 0:
        print(f"   Total: {total_missing:,}")
        missing_cols = merged_df.isnull().sum()[merged_df.isnull().sum() > 0]
        for col, count in missing_cols.items():
            print(f"      {col}: {count} ({count/len(merged_df)*100:.2f}%)")
    else:
        print(f"   ✓ No missing values!")
    
    print(f"\n🎯 FIRST 5 ROWS:")
    print("-" * 80)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    print(merged_df.head())
    
    print(f"\n📊 NUMERICAL STATISTICS:")
    print("-" * 80)
    print(merged_df.describe())
    
    # ========================================================================
    # STEP 7: SAVE MERGED DATASET
    # ========================================================================
    print("\n" + "=" * 80)
    print("STEP 7: SAVING MERGED DATASET")
    print("=" * 80)
    
    merged_df.to_csv('Merged_Crop_Weather_Data.csv', index=False)
    print("✓ Merged dataset saved as: Merged_Crop_Weather_Data.csv")
    
    # ========================================================================
    # MERGE SUMMARY
    # ========================================================================
    print("\n" + "=" * 80)
    print("MERGE SUMMARY REPORT")
    print("=" * 80)
    
    retention_rate = (merged_df.shape[0] / min(len(weather_df), len(crop_df)) * 100)
    
    print(f"""
📊 INPUT DATASETS:
   Weather dataset: {weather_df.shape[0]:,} rows × {weather_df.shape[1]} columns
   Crop dataset: {crop_df.shape[0]:,} rows × {crop_df.shape[1]} columns

🔗 MERGE CONFIGURATION:
   Join type: INNER JOIN
   Merge keys: {', '.join(merge_keys)}
   Matching strategy: {'Month-based (no year overlap)' if not common_years else 'Year-Month based'}

📈 OUTPUT DATASET:
   Merged dataset: {merged_df.shape[0]:,} rows × {merged_df.shape[1]} columns
   Data retention: {retention_rate:.2f}%
   
💡 INSIGHTS:
   • Unique districts: {merged_df['District'].nunique()}
   • Date range: {merged_df['Date_crop'].min() if 'Date_crop' in merged_df.columns else merged_df['Date'].min()} to {merged_df['Date_crop'].max() if 'Date_crop' in merged_df.columns else merged_df['Date'].max()}
   • Avg records/district: {merged_df.shape[0] / merged_df['District'].nunique():.1f}
   
✅ MERGE COMPLETED SUCCESSFULLY!
    """)
    
else:
    print("\n❌ MERGE RESULTED IN EMPTY DATASET")
    print("\n🔍 Debugging Information:")
    print(f"   Common districts: {len(common_districts)}")
    print(f"   Common years: {len(common_years) if common_years else 0}")
    print("\n💡 Suggestions:")
    print("   1. Check district name standardization")
    print("   2. Verify date formats and ranges")
    print("   3. Review data cleaning steps")

print("\n" + "=" * 80)
print("DATA MERGING PROCESS COMPLETED")
print("=" * 80)
