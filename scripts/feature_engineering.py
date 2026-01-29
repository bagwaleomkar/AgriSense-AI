"""
AgriSense AI - Feature Engineering for Crop Price Prediction
This script creates advanced features for ML model training
"""

import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("FEATURE ENGINEERING FOR CROP PRICE PREDICTION")
print("=" * 80)

# ============================================================================
# STEP 1: LOAD MERGED DATASET
# ============================================================================
print("\n" + "=" * 80)
print("STEP 1: LOADING MERGED DATASET")
print("=" * 80)

df = pd.read_csv('Merged_Crop_Weather_Data.csv')
print(f"✓ Dataset loaded: {df.shape}")

# Convert date columns
df['Date_crop'] = pd.to_datetime(df['Date_crop'])
df['Date_weather'] = pd.to_datetime(df['Date_weather'])

# Sort by district, commodity, and date for proper time series features
df = df.sort_values(['District', 'Commodity', 'Date_crop']).reset_index(drop=True)
print(f"✓ Data sorted by District, Commodity, and Date")

print(f"\nOriginal features: {df.shape[1]}")

# ============================================================================
# STEP 2: ROLLING AVERAGE FEATURES (7-DAY AND 30-DAY)
# ============================================================================
print("\n" + "=" * 80)
print("STEP 2: CREATING ROLLING AVERAGE FEATURES")
print("=" * 80)

print("""
📊 WHY ROLLING AVERAGES ARE IMPORTANT:

1. SMOOTHING SHORT-TERM FLUCTUATIONS:
   • Daily prices can be volatile due to temporary market conditions
   • Rolling averages smooth out noise and reveal underlying trends
   • Help models focus on genuine patterns rather than random spikes

2. CAPTURING MOMENTUM:
   • 7-day average captures recent price momentum (short-term trend)
   • 30-day average captures medium-term trend direction
   • Relationship between short and long averages indicates trend strength

3. TREND IDENTIFICATION:
   • When 7-day avg > 30-day avg → Upward trend (bullish signal)
   • When 7-day avg < 30-day avg → Downward trend (bearish signal)
   • Crossovers indicate potential trend reversals

4. REDUCING OVERFITTING:
   • Smoothed values are more stable and generalizable
   • Help model learn from patterns rather than noise
""")

# Calculate rolling averages for each district-commodity combination
df['Price_MA_7'] = df.groupby(['District', 'Commodity'])['Modal_Price'].transform(
    lambda x: x.rolling(window=7, min_periods=1).mean()
)

df['Price_MA_30'] = df.groupby(['District', 'Commodity'])['Modal_Price'].transform(
    lambda x: x.rolling(window=30, min_periods=1).mean()
)

# Calculate moving average convergence/divergence indicator
df['Price_MA_Ratio'] = df['Price_MA_7'] / df['Price_MA_30']

print(f"✓ Created 7-day rolling average: Price_MA_7")
print(f"✓ Created 30-day rolling average: Price_MA_30")
print(f"✓ Created MA ratio indicator: Price_MA_Ratio")

print(f"\nSample values:")
print(f"  Original Price: {df['Modal_Price'].iloc[100]:.2f}")
print(f"  7-day MA: {df['Price_MA_7'].iloc[100]:.2f}")
print(f"  30-day MA: {df['Price_MA_30'].iloc[100]:.2f}")
print(f"  MA Ratio: {df['Price_MA_Ratio'].iloc[100]:.3f}")

# ============================================================================
# STEP 3: PRICE CHANGE PERCENTAGE FEATURES
# ============================================================================
print("\n" + "=" * 80)
print("STEP 3: CREATING PRICE CHANGE PERCENTAGE FEATURES")
print("=" * 80)

print("""
📊 WHY PRICE CHANGE PERCENTAGE IS IMPORTANT:

1. RELATIVE MAGNITUDE OF CHANGE:
   • Absolute price changes can be misleading (₹100 change means different things
     for ₹1000 vs ₹5000 crops)
   • Percentage change normalizes across different price levels
   • Allows comparison across different commodities

2. VOLATILITY MEASUREMENT:
   • Large percentage changes indicate high volatility/risk
   • Helps identify unstable market conditions
   • Important for risk assessment and forecasting uncertainty

3. MOMENTUM INDICATOR:
   • Consistent positive changes indicate strong upward momentum
   • Negative changes signal downward pressure
   • Rate of change helps predict acceleration/deceleration

4. FEATURE SCALING:
   • Percentages naturally bounded and comparable
   • Easier for models to learn from normalized values
   • Reduces bias from absolute price levels
""")

# Calculate daily price change percentage
df['Price_Change_Pct'] = df.groupby(['District', 'Commodity'])['Modal_Price'].pct_change() * 100

# Calculate 7-day price change percentage
df['Price_Change_7d_Pct'] = df.groupby(['District', 'Commodity'])['Modal_Price'].transform(
    lambda x: ((x - x.shift(7)) / x.shift(7)) * 100
)

# Calculate 30-day price change percentage
df['Price_Change_30d_Pct'] = df.groupby(['District', 'Commodity'])['Modal_Price'].transform(
    lambda x: ((x - x.shift(30)) / x.shift(30)) * 100
)

# Calculate volatility (rolling standard deviation of price changes)
df['Price_Volatility'] = df.groupby(['District', 'Commodity'])['Price_Change_Pct'].transform(
    lambda x: x.rolling(window=7, min_periods=1).std()
)

print(f"✓ Created daily price change: Price_Change_Pct")
print(f"✓ Created 7-day price change: Price_Change_7d_Pct")
print(f"✓ Created 30-day price change: Price_Change_30d_Pct")
print(f"✓ Created price volatility: Price_Volatility")

# ============================================================================
# STEP 4: MONTHLY AND SEASONAL FEATURES
# ============================================================================
print("\n" + "=" * 80)
print("STEP 4: CREATING MONTHLY AND SEASONAL FEATURES")
print("=" * 80)

print("""
📊 WHY TEMPORAL FEATURES ARE IMPORTANT:

1. SEASONALITY CAPTURE:
   • Agriculture is highly seasonal (planting, growing, harvest cycles)
   • Different crops have different harvest seasons
   • Weather patterns follow seasonal cycles

2. CYCLICAL PATTERNS:
   • Sin/Cos encoding preserves cyclical nature (Dec and Jan are close)
   • Linear encoding would treat Jan(1) and Dec(12) as far apart
   • Helps model understand periodic patterns

3. MARKET DYNAMICS:
   • Festival seasons affect demand (Diwali, harvest festivals)
   • End of season often sees price drops (oversupply)
   • Quarter-level trends capture longer economic cycles

4. FEATURE RICHNESS:
   • Multiple representations give model flexibility
   • Can learn different patterns at different time scales
   • Improves model's ability to generalize
""")

# Extract temporal features
df['Day_of_Week'] = df['Date_crop'].dt.dayofweek
df['Day_of_Month'] = df['Date_crop'].dt.day
df['Week_of_Year'] = df['Date_crop'].dt.isocalendar().week
df['Quarter'] = df['Date_crop'].dt.quarter

# Cyclical encoding for month (sin and cos to preserve cyclical nature)
df['Month_Sin'] = np.sin(2 * np.pi * df['Month'] / 12)
df['Month_Cos'] = np.cos(2 * np.pi * df['Month'] / 12)

# Day of year (1-365)
df['Day_of_Year'] = df['Date_crop'].dt.dayofyear

# Season encoding (Maharashtra seasons)
def get_season(month):
    if month in [12, 1, 2]:
        return 'Winter'
    elif month in [3, 4, 5]:
        return 'Summer'
    elif month in [6, 7, 8, 9]:
        return 'Monsoon'
    else:  # 10, 11
        return 'Post_Monsoon'

df['Season'] = df['Month'].apply(get_season)

# One-hot encode season
season_dummies = pd.get_dummies(df['Season'], prefix='Season')
df = pd.concat([df, season_dummies], axis=1)

# Harvest season indicator (varies by crop, using general pattern)
# Rabi crops: Oct-March (harvest Mar-Apr)
# Kharif crops: Jun-Sep (harvest Oct-Nov)
def is_harvest_season(month, commodity):
    # Simplified logic - can be refined per commodity
    if commodity.lower() in ['wheat', 'gram', 'mustard']:  # Rabi crops
        return 1 if month in [3, 4] else 0
    elif commodity.lower() in ['rice', 'bajra', 'jowar', 'maize']:  # Kharif crops
        return 1 if month in [10, 11] else 0
    else:
        return 0

df['Is_Harvest_Season'] = df.apply(lambda x: is_harvest_season(x['Month'], x['Commodity']), axis=1)

print(f"✓ Created day of week: Day_of_Week (0=Monday, 6=Sunday)")
print(f"✓ Created day of month: Day_of_Month")
print(f"✓ Created week of year: Week_of_Year")
print(f"✓ Created quarter: Quarter")
print(f"✓ Created cyclical month encoding: Month_Sin, Month_Cos")
print(f"✓ Created day of year: Day_of_Year")
print(f"✓ Created season categories with one-hot encoding")
print(f"✓ Created harvest season indicator: Is_Harvest_Season")

# ============================================================================
# STEP 5: RAINFALL DEVIATION FROM MONTHLY AVERAGE
# ============================================================================
print("\n" + "=" * 80)
print("STEP 5: CREATING RAINFALL DEVIATION FEATURES")
print("=" * 80)

print("""
📊 WHY RAINFALL DEVIATION IS IMPORTANT:

1. ANOMALY DETECTION:
   • Absolute rainfall doesn't tell the full story
   • What matters is deviation from normal/expected rainfall
   • Excess or deficit rainfall both impact crop yields and prices

2. CONTEXT-AWARE WEATHER IMPACT:
   • 50mm rain in monsoon = normal (expected)
   • 50mm rain in winter = exceptional (unexpected impact)
   • Deviation captures the surprise factor

3. REGIONAL NORMALIZATION:
   • Different districts have different rainfall patterns
   • Deviation normalizes across regions
   • 10mm in drought-prone area ≠ 10mm in high-rainfall area

4. CROP STRESS INDICATOR:
   • Large positive deviation → flooding risk, disease
   • Large negative deviation → drought stress, yield reduction
   • Both extremes typically increase prices (reduced supply or panic)

5. PREDICTIVE POWER:
   • Historical deviations help model learn threshold effects
   • Non-linear relationships with prices easier to capture
   • Better than absolute rainfall for price prediction
""")

# Calculate monthly average rainfall by district
monthly_avg_rainfall = df.groupby(['District', 'Month'])['Daily_Rainfall_mm'].transform('mean')
df['Rainfall_Monthly_Avg'] = monthly_avg_rainfall

# Calculate deviation from monthly average
df['Rainfall_Deviation'] = df['Daily_Rainfall_mm'] - df['Rainfall_Monthly_Avg']

# Calculate percentage deviation
df['Rainfall_Deviation_Pct'] = (df['Rainfall_Deviation'] / (df['Rainfall_Monthly_Avg'] + 0.001)) * 100

# Calculate cumulative rainfall for the month
df['Year_Month'] = df['Date_crop'].dt.to_period('M')
df['Cumulative_Rainfall_Month'] = df.groupby(['District', 'Year_Month'])['Daily_Rainfall_mm'].cumsum()

# Rainfall categories based on deviation
def rainfall_category(deviation_pct):
    if deviation_pct < -50:
        return 'Severe_Deficit'
    elif deviation_pct < -20:
        return 'Deficit'
    elif deviation_pct < 20:
        return 'Normal'
    elif deviation_pct < 50:
        return 'Excess'
    else:
        return 'Severe_Excess'

df['Rainfall_Category'] = df['Rainfall_Deviation_Pct'].apply(rainfall_category)

# One-hot encode rainfall category
rainfall_dummies = pd.get_dummies(df['Rainfall_Category'], prefix='Rainfall')
df = pd.concat([df, rainfall_dummies], axis=1)

print(f"✓ Created monthly average rainfall: Rainfall_Monthly_Avg")
print(f"✓ Created rainfall deviation: Rainfall_Deviation")
print(f"✓ Created percentage deviation: Rainfall_Deviation_Pct")
print(f"✓ Created cumulative monthly rainfall: Cumulative_Rainfall_Month")
print(f"✓ Created rainfall categories with one-hot encoding")

print(f"\nSample values:")
sample_idx = 500
print(f"  Actual Rainfall: {df['Daily_Rainfall_mm'].iloc[sample_idx]:.2f} mm")
print(f"  Monthly Average: {df['Rainfall_Monthly_Avg'].iloc[sample_idx]:.2f} mm")
print(f"  Deviation: {df['Rainfall_Deviation'].iloc[sample_idx]:.2f} mm")
print(f"  Deviation %: {df['Rainfall_Deviation_Pct'].iloc[sample_idx]:.2f}%")

# ============================================================================
# STEP 6: LAG FEATURES FOR PRICES
# ============================================================================
print("\n" + "=" * 80)
print("STEP 6: CREATING LAG FEATURES FOR PRICES")
print("=" * 80)

print("""
📊 WHY LAG FEATURES ARE IMPORTANT:

1. TIME SERIES AUTOCORRELATION:
   • Prices exhibit autocorrelation (today's price related to yesterday's)
   • Past prices contain information about future prices
   • Lag features explicitly provide historical context

2. MOMENTUM AND INERTIA:
   • Markets don't change instantly - there's inertia
   • Recent price history indicates current momentum
   • Multiple lags capture different time horizons

3. LEADING INDICATORS:
   • 1-day lag captures immediate past (strong correlation)
   • 7-day lag captures weekly patterns
   • 30-day lag captures monthly trends
   • Multiple lags help model learn temporal dependencies

4. FEATURE INTERACTIONS:
   • Combination of lags creates complex patterns
   • Model can learn from differences between lags
   • Captures acceleration/deceleration of price changes

5. AVOIDING DATA LEAKAGE:
   • Lags ensure we only use past information
   • Critical for time series to avoid future data contamination
   • Maintains temporal integrity of predictions

6. STATISTICAL FOUNDATION:
   • Autoregressive models (AR, ARIMA) rely on lags
   • Proven effective in time series forecasting
   • Provides baseline features for any time series model
""")

# Create lag features for modal price (1, 3, 7, 14, 30 days)
lag_periods = [1, 3, 7, 14, 30]

for lag in lag_periods:
    df[f'Price_Lag_{lag}d'] = df.groupby(['District', 'Commodity'])['Modal_Price'].shift(lag)
    print(f"✓ Created lag feature: Price_Lag_{lag}d")

# Create lag features for arrivals
for lag in [1, 7, 14]:
    df[f'Arrivals_Lag_{lag}d'] = df.groupby(['District', 'Commodity'])['Arrivals'].shift(lag)
    print(f"✓ Created arrivals lag feature: Arrivals_Lag_{lag}d")

# Create lag features for weather variables (7-day lag)
weather_vars = ['Daily_Rainfall_mm', 'Max_Temp_C', 'Min_Temp_C', 'Avg_Humidity_%']
for var in weather_vars:
    df[f'{var}_Lag_7d'] = df.groupby(['District'])[var].shift(7)

# Calculate differences between lags (price change indicators)
df['Price_Diff_1_7'] = df['Price_Lag_1d'] - df['Price_Lag_7d']
df['Price_Diff_7_30'] = df['Price_Lag_7d'] - df['Price_Lag_30d']

print(f"✓ Created lag differences: Price_Diff_1_7, Price_Diff_7_30")

# ============================================================================
# STEP 7: ADDITIONAL ADVANCED FEATURES
# ============================================================================
print("\n" + "=" * 80)
print("STEP 7: CREATING ADDITIONAL ADVANCED FEATURES")
print("=" * 80)

# Price spread (difference between max and min price)
df['Price_Spread'] = df['Max_Price'] - df['Min_Price']
df['Price_Spread_Pct'] = (df['Price_Spread'] / df['Modal_Price']) * 100

# Temperature range
df['Temp_Range'] = df['Max_Temp_C'] - df['Min_Temp_C']

# Weather comfort index (simplified)
df['Weather_Comfort_Index'] = (df['Max_Temp_C'] + df['Min_Temp_C']) / 2 - df['Avg_Humidity_%'] / 10

# Price position (where modal price sits between min and max)
df['Price_Position'] = (df['Modal_Price'] - df['Min_Price']) / (df['Price_Spread'] + 0.001)

# Log transformations for skewed variables
df['Log_Modal_Price'] = np.log1p(df['Modal_Price'])
df['Log_Arrivals'] = np.log1p(df['Arrivals'])

# Interaction features
df['Rainfall_Temp_Interaction'] = df['Daily_Rainfall_mm'] * df['Max_Temp_C']
df['Humidity_Temp_Interaction'] = df['Avg_Humidity_%'] * df['Max_Temp_C']

print(f"✓ Created price spread features")
print(f"✓ Created temperature range")
print(f"✓ Created weather comfort index")
print(f"✓ Created price position indicator")
print(f"✓ Created log-transformed features")
print(f"✓ Created interaction features")

# ============================================================================
# STEP 8: ENCODE CATEGORICAL VARIABLES
# ============================================================================
print("\n" + "=" * 80)
print("STEP 8: ENCODING CATEGORICAL VARIABLES")
print("=" * 80)

# District encoding (label encoding)
from sklearn.preprocessing import LabelEncoder

le_district = LabelEncoder()
df['District_Encoded'] = le_district.fit_transform(df['District'])

# Commodity encoding
le_commodity = LabelEncoder()
df['Commodity_Encoded'] = le_commodity.fit_transform(df['Commodity'])

# Market encoding
le_market = LabelEncoder()
df['Market_Encoded'] = le_market.fit_transform(df['Market'])

print(f"✓ Encoded District: {df['District'].nunique()} unique values")
print(f"✓ Encoded Commodity: {df['Commodity'].nunique()} unique values")
print(f"✓ Encoded Market: {df['Market'].nunique()} unique values")

# ============================================================================
# STEP 9: HANDLE MISSING VALUES AND SAVE
# ============================================================================
print("\n" + "=" * 80)
print("STEP 9: HANDLING MISSING VALUES AND SAVING")
print("=" * 80)

print(f"\nMissing values before handling:")
missing_summary = df.isnull().sum()
missing_cols = missing_summary[missing_summary > 0]
if len(missing_cols) > 0:
    for col, count in missing_cols.items():
        print(f"  {col}: {count} ({count/len(df)*100:.2f}%)")
else:
    print("  No missing values!")

# Fill missing values for lag features with forward fill (appropriate for time series)
lag_cols = [col for col in df.columns if 'Lag' in col or 'MA' in col]
for col in lag_cols:
    df[col] = df[col].ffill()
    df[col] = df[col].bfill()

# Fill remaining missing values
df.fillna(0, inplace=True)

print(f"\n✓ Missing values handled using forward/backward fill for time series")

# Remove temporary columns
if 'Year_Month' in df.columns:
    df = df.drop('Year_Month', axis=1)

# Save engineered dataset
df.to_csv('Crop_Price_Features_Engineered.csv', index=False)
print(f"\n✓ Feature-engineered dataset saved: Crop_Price_Features_Engineered.csv")

# ============================================================================
# STEP 10: FEATURE SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("FEATURE ENGINEERING SUMMARY")
print("=" * 80)

print(f"""
📊 FEATURE SUMMARY:

Original Features: 20
New Features Created: {df.shape[1] - 20}
Total Features: {df.shape[1]}
Total Records: {len(df):,}

FEATURE CATEGORIES:

1. Rolling Averages (3 features):
   • Price_MA_7, Price_MA_30, Price_MA_Ratio
   • Smooth noise and capture trends

2. Price Change Features (4 features):
   • Price_Change_Pct, Price_Change_7d_Pct, Price_Change_30d_Pct, Price_Volatility
   • Measure momentum and volatility

3. Temporal Features (13+ features):
   • Day/Week/Month/Quarter/Season indicators
   • Cyclical encodings
   • Capture seasonality and cycles

4. Rainfall Deviation Features (8+ features):
   • Rainfall_Deviation, Rainfall_Deviation_Pct, Cumulative_Rainfall_Month
   • Rainfall categories
   • Context-aware weather impact

5. Lag Features (8+ features):
   • Price lags: 1, 3, 7, 14, 30 days
   • Arrivals lags: 1, 7, 14 days
   • Provide historical context

6. Advanced Features (10+ features):
   • Price spread, temperature range, comfort index
   • Log transformations
   • Interaction features

7. Encoded Variables (3 features):
   • District, Commodity, Market encoding
   • Machine-readable categorical variables
""")

# Display sample of engineered features
print("\n" + "=" * 80)
print("SAMPLE OF ENGINEERED FEATURES")
print("=" * 80)

feature_cols = ['Modal_Price', 'Price_MA_7', 'Price_MA_30', 'Price_Change_Pct', 
                'Rainfall_Deviation', 'Price_Lag_1d', 'Price_Lag_7d', 'Season']
print(df[feature_cols].head(10).to_string())

print("\n" + "=" * 80)
print("✅ FEATURE ENGINEERING COMPLETED SUCCESSFULLY!")
print("=" * 80)

print("""
🎯 NEXT STEPS FOR ML MODEL TRAINING:

1. FEATURE SELECTION:
   • Use correlation analysis to remove redundant features
   • Apply feature importance from tree-based models
   • Consider domain knowledge for feature subset

2. FEATURE SCALING:
   • Normalize/standardize continuous features
   • Keep encoded categorical features as-is
   • Consider separate scaling for different feature groups

3. TRAIN-TEST SPLIT:
   • Use time-based split (e.g., first 80% train, last 20% test)
   • Maintain temporal order - no random shuffling
   • Consider multiple validation periods

4. MODEL SELECTION:
   • Start with tree-based models (XGBoost, Random Forest)
   • Try time series models (ARIMA, Prophet)
   • Experiment with deep learning (LSTM, Transformer)

5. EVALUATION METRICS:
   • MAE, RMSE for price prediction accuracy
   • MAPE for percentage error
   • Directional accuracy (up/down prediction)
""")

print("=" * 80)
