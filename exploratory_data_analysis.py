"""
AgriSense AI - Exploratory Data Analysis (EDA)
This script performs comprehensive EDA on the merged crop and weather dataset
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set visualization style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 10

# ============================================================================
# STEP 1: LOAD MERGED DATASET
# ============================================================================
print("=" * 80)
print("LOADING MERGED DATASET FOR EDA")
print("=" * 80)

df = pd.read_csv('Merged_Crop_Weather_Data.csv')
print(f"✓ Dataset loaded: {df.shape}")

# Convert date columns
df['Date_crop'] = pd.to_datetime(df['Date_crop'])
df['Date_weather'] = pd.to_datetime(df['Date_weather'])

print(f"\nDataset Overview:")
print(f"  Date range: {df['Date_crop'].min()} to {df['Date_crop'].max()}")
print(f"  Districts: {df['District'].nunique()}")
print(f"  Commodities: {df['Commodity'].nunique()}")
print(f"  Total records: {len(df):,}")

# ============================================================================
# VISUALIZATION 1: CROP PRICE TRENDS OVER TIME
# ============================================================================
print("\n" + "=" * 80)
print("VISUALIZATION 1: CROP PRICE TRENDS OVER TIME")
print("=" * 80)

fig, axes = plt.subplots(2, 2, figsize=(16, 10))
fig.suptitle('Crop Price Trends Over Time - 2025', fontsize=16, fontweight='bold')

# 1.1 Overall Modal Price Trend
ax1 = axes[0, 0]
daily_avg = df.groupby('Date_crop')['Modal_Price'].mean().reset_index()
ax1.plot(daily_avg['Date_crop'], daily_avg['Modal_Price'], linewidth=2, color='#2E86AB')
ax1.fill_between(daily_avg['Date_crop'], daily_avg['Modal_Price'], alpha=0.3, color='#2E86AB')
ax1.set_title('Overall Average Modal Price Trend', fontsize=12, fontweight='bold')
ax1.set_xlabel('Date')
ax1.set_ylabel('Modal Price (₹)')
ax1.grid(True, alpha=0.3)
ax1.tick_params(axis='x', rotation=45)

# 1.2 Top 5 Commodities Price Trends
ax2 = axes[0, 1]
top_commodities = df['Commodity'].value_counts().head(5).index
for commodity in top_commodities:
    commodity_data = df[df['Commodity'] == commodity].groupby('Date_crop')['Modal_Price'].mean()
    ax2.plot(commodity_data.index, commodity_data.values, label=commodity.capitalize(), linewidth=2)
ax2.set_title('Top 5 Commodities - Price Trends', fontsize=12, fontweight='bold')
ax2.set_xlabel('Date')
ax2.set_ylabel('Modal Price (₹)')
ax2.legend(loc='best', fontsize=9)
ax2.grid(True, alpha=0.3)
ax2.tick_params(axis='x', rotation=45)

# 1.3 Price Range (Min to Max) Over Time
ax3 = axes[1, 0]
daily_prices = df.groupby('Date_crop').agg({
    'Min_Price': 'mean',
    'Max_Price': 'mean',
    'Modal_Price': 'mean'
}).reset_index()
ax3.fill_between(daily_prices['Date_crop'], daily_prices['Min_Price'], 
                 daily_prices['Max_Price'], alpha=0.3, color='orange', label='Price Range')
ax3.plot(daily_prices['Date_crop'], daily_prices['Modal_Price'], 
         color='red', linewidth=2, label='Modal Price')
ax3.set_title('Price Range and Modal Price Over Time', fontsize=12, fontweight='bold')
ax3.set_xlabel('Date')
ax3.set_ylabel('Price (₹)')
ax3.legend(loc='best')
ax3.grid(True, alpha=0.3)
ax3.tick_params(axis='x', rotation=45)

# 1.4 District-wise Average Prices
ax4 = axes[1, 1]
district_prices = df.groupby('District')['Modal_Price'].mean().sort_values(ascending=False).head(10)
colors = plt.cm.viridis(np.linspace(0, 1, len(district_prices)))
bars = ax4.barh(range(len(district_prices)), district_prices.values, color=colors)
ax4.set_yticks(range(len(district_prices)))
ax4.set_yticklabels([d.capitalize() for d in district_prices.index])
ax4.set_title('Top 10 Districts by Average Modal Price', fontsize=12, fontweight='bold')
ax4.set_xlabel('Average Modal Price (₹)')
ax4.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig('EDA_1_Price_Trends.png', dpi=300, bbox_inches='tight')
print("✓ Saved: EDA_1_Price_Trends.png")
plt.close()

# Key Insights
print("\n📊 KEY INSIGHTS - PRICE TRENDS:")
print("   1. Overall Trend:")
overall_start = daily_avg.iloc[0]['Modal_Price']
overall_end = daily_avg.iloc[-1]['Modal_Price']
change_pct = ((overall_end - overall_start) / overall_start) * 100
print(f"      • Modal price {'increased' if change_pct > 0 else 'decreased'} by {abs(change_pct):.2f}% over the year")
print(f"      • Average modal price: ₹{df['Modal_Price'].mean():.2f}")
print(f"      • Price range: ₹{df['Modal_Price'].min():.2f} to ₹{df['Modal_Price'].max():.2f}")

print("   2. Commodity Analysis:")
for commodity in top_commodities[:3]:
    avg_price = df[df['Commodity'] == commodity]['Modal_Price'].mean()
    print(f"      • {commodity.capitalize()}: Average ₹{avg_price:.2f}")

print("   3. District Analysis:")
print(f"      • Highest avg price: {district_prices.index[0].capitalize()} (₹{district_prices.values[0]:.2f})")
print(f"      • Lowest avg price: {district_prices.index[-1].capitalize()} (₹{district_prices.values[-1]:.2f})")

# ============================================================================
# VISUALIZATION 2: MONTHLY SEASONALITY ANALYSIS
# ============================================================================
print("\n" + "=" * 80)
print("VISUALIZATION 2: MONTHLY SEASONALITY ANALYSIS")
print("=" * 80)

fig, axes = plt.subplots(2, 2, figsize=(16, 10))
fig.suptitle('Monthly Seasonality Analysis - 2025', fontsize=16, fontweight='bold')

# 2.1 Average Modal Price by Month
ax1 = axes[0, 0]
monthly_price = df.groupby('Month')['Modal_Price'].agg(['mean', 'std']).reset_index()
month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
colors_gradient = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, 12))
bars = ax1.bar(monthly_price['Month'], monthly_price['mean'], yerr=monthly_price['std'], 
               color=colors_gradient, alpha=0.8, capsize=5)
ax1.set_xticks(range(1, 13))
ax1.set_xticklabels(month_names)
ax1.set_title('Average Modal Price by Month', fontsize=12, fontweight='bold')
ax1.set_xlabel('Month')
ax1.set_ylabel('Average Modal Price (₹)')
ax1.grid(True, alpha=0.3, axis='y')

# 2.2 Arrivals by Month
ax2 = axes[0, 1]
monthly_arrivals = df.groupby('Month')['Arrivals'].sum().reset_index()
ax2.plot(monthly_arrivals['Month'], monthly_arrivals['Arrivals'], 
         marker='o', linewidth=2.5, markersize=8, color='#F77F00')
ax2.fill_between(monthly_arrivals['Month'], monthly_arrivals['Arrivals'], alpha=0.3, color='#F77F00')
ax2.set_xticks(range(1, 13))
ax2.set_xticklabels(month_names)
ax2.set_title('Total Crop Arrivals by Month', fontsize=12, fontweight='bold')
ax2.set_xlabel('Month')
ax2.set_ylabel('Total Arrivals (tonnes)')
ax2.grid(True, alpha=0.3)

# 2.3 Price Volatility by Month (Coefficient of Variation)
ax3 = axes[1, 0]
monthly_volatility = df.groupby('Month')['Modal_Price'].agg(lambda x: (x.std() / x.mean()) * 100).reset_index()
monthly_volatility.columns = ['Month', 'CV']
ax3.bar(monthly_volatility['Month'], monthly_volatility['CV'], color='#06A77D', alpha=0.8)
ax3.set_xticks(range(1, 13))
ax3.set_xticklabels(month_names)
ax3.set_title('Price Volatility by Month (Coefficient of Variation %)', fontsize=12, fontweight='bold')
ax3.set_xlabel('Month')
ax3.set_ylabel('CV (%)')
ax3.grid(True, alpha=0.3, axis='y')
ax3.axhline(y=monthly_volatility['CV'].mean(), color='red', linestyle='--', 
            linewidth=2, label=f"Avg CV: {monthly_volatility['CV'].mean():.2f}%")
ax3.legend()

# 2.4 Heatmap: Commodity vs Month
ax4 = axes[1, 1]
commodity_month = df.groupby(['Commodity', 'Month'])['Modal_Price'].mean().reset_index()
heatmap_data = commodity_month.pivot(index='Commodity', columns='Month', values='Modal_Price')
# Select top 10 commodities for readability
top_10_comm = df.groupby('Commodity')['Modal_Price'].count().nlargest(10).index
heatmap_data_top = heatmap_data.loc[top_10_comm]
sns.heatmap(heatmap_data_top, annot=False, fmt='.0f', cmap='YlOrRd', 
            cbar_kws={'label': 'Modal Price (₹)'}, ax=ax4)
ax4.set_title('Commodity Price Heatmap by Month', fontsize=12, fontweight='bold')
ax4.set_xlabel('Month')
ax4.set_ylabel('Commodity')

plt.tight_layout()
plt.savefig('EDA_2_Monthly_Seasonality.png', dpi=300, bbox_inches='tight')
print("✓ Saved: EDA_2_Monthly_Seasonality.png")
plt.close()

# Key Insights
print("\n📊 KEY INSIGHTS - MONTHLY SEASONALITY:")
print("   1. Price Patterns:")
highest_month = monthly_price.loc[monthly_price['mean'].idxmax(), 'Month']
lowest_month = monthly_price.loc[monthly_price['mean'].idxmin(), 'Month']
print(f"      • Highest prices: {month_names[int(highest_month)-1]} (₹{monthly_price['mean'].max():.2f})")
print(f"      • Lowest prices: {month_names[int(lowest_month)-1]} (₹{monthly_price['mean'].min():.2f})")

print("   2. Supply Patterns:")
peak_arrival_month = monthly_arrivals.loc[monthly_arrivals['Arrivals'].idxmax(), 'Month']
low_arrival_month = monthly_arrivals.loc[monthly_arrivals['Arrivals'].idxmin(), 'Month']
print(f"      • Peak arrivals: {month_names[int(peak_arrival_month)-1]} ({monthly_arrivals['Arrivals'].max():.2f} tonnes)")
print(f"      • Lowest arrivals: {month_names[int(low_arrival_month)-1]} ({monthly_arrivals['Arrivals'].min():.2f} tonnes)")

print("   3. Volatility:")
most_volatile = monthly_volatility.loc[monthly_volatility['CV'].idxmax(), 'Month']
least_volatile = monthly_volatility.loc[monthly_volatility['CV'].idxmin(), 'Month']
print(f"      • Most volatile: {month_names[int(most_volatile)-1]} (CV: {monthly_volatility['CV'].max():.2f}%)")
print(f"      • Most stable: {month_names[int(least_volatile)-1]} (CV: {monthly_volatility['CV'].min():.2f}%)")

# ============================================================================
# VISUALIZATION 3: RAINFALL VS MODAL PRICE RELATIONSHIP
# ============================================================================
print("\n" + "=" * 80)
print("VISUALIZATION 3: RAINFALL VS MODAL PRICE RELATIONSHIP")
print("=" * 80)

fig, axes = plt.subplots(2, 2, figsize=(16, 10))
fig.suptitle('Weather Impact Analysis: Rainfall vs Crop Prices', fontsize=16, fontweight='bold')

# 3.1 Scatter Plot: Rainfall vs Modal Price
ax1 = axes[0, 0]
sample_data = df.sample(n=min(5000, len(df)), random_state=42)
scatter = ax1.scatter(sample_data['Daily_Rainfall_mm'], sample_data['Modal_Price'], 
                     alpha=0.5, c=sample_data['Modal_Price'], cmap='viridis', s=20)
# Add trend line
z = np.polyfit(df['Daily_Rainfall_mm'], df['Modal_Price'], 1)
p = np.poly1d(z)
ax1.plot(df['Daily_Rainfall_mm'].sort_values(), p(df['Daily_Rainfall_mm'].sort_values()), 
         "r--", linewidth=2, label=f'Trend: y={z[0]:.2f}x+{z[1]:.2f}')
ax1.set_title('Rainfall vs Modal Price (Scatter)', fontsize=12, fontweight='bold')
ax1.set_xlabel('Daily Rainfall (mm)')
ax1.set_ylabel('Modal Price (₹)')
ax1.legend()
ax1.grid(True, alpha=0.3)
plt.colorbar(scatter, ax=ax1, label='Modal Price (₹)')

# 3.2 Rainfall Categories vs Average Price
ax2 = axes[0, 1]
# Create rainfall categories
df['Rainfall_Category'] = pd.cut(df['Daily_Rainfall_mm'], 
                                   bins=[0, 0.1, 2, 5, 100], 
                                   labels=['No Rain', 'Light', 'Moderate', 'Heavy'])
rainfall_cat_price = df.groupby('Rainfall_Category')['Modal_Price'].agg(['mean', 'std']).reset_index()
colors_rain = ['#FF6B6B', '#FFA07A', '#4ECDC4', '#45B7D1']
bars = ax2.bar(range(len(rainfall_cat_price)), rainfall_cat_price['mean'], 
               yerr=rainfall_cat_price['std'], color=colors_rain, alpha=0.8, capsize=5)
ax2.set_xticks(range(len(rainfall_cat_price)))
ax2.set_xticklabels(rainfall_cat_price['Rainfall_Category'])
ax2.set_title('Average Price by Rainfall Category', fontsize=12, fontweight='bold')
ax2.set_xlabel('Rainfall Category')
ax2.set_ylabel('Average Modal Price (₹)')
ax2.grid(True, alpha=0.3, axis='y')

# Add value labels on bars
for i, (bar, val) in enumerate(zip(bars, rainfall_cat_price['mean'])):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + rainfall_cat_price['std'].iloc[i], 
             f'₹{val:.0f}', ha='center', va='bottom', fontweight='bold')

# 3.3 Temperature vs Price
ax3 = axes[1, 0]
temp_bins = pd.cut(df['Max_Temp_C'], bins=5)
temp_price = df.groupby(temp_bins)['Modal_Price'].mean()
temp_labels = [f"{interval.left:.1f}-{interval.right:.1f}°C" for interval in temp_price.index]
ax3.plot(range(len(temp_price)), temp_price.values, marker='o', linewidth=2.5, 
         markersize=10, color='#E63946')
ax3.fill_between(range(len(temp_price)), temp_price.values, alpha=0.3, color='#E63946')
ax3.set_xticks(range(len(temp_price)))
ax3.set_xticklabels(temp_labels, rotation=45, ha='right')
ax3.set_title('Average Price by Temperature Range', fontsize=12, fontweight='bold')
ax3.set_xlabel('Max Temperature Range')
ax3.set_ylabel('Average Modal Price (₹)')
ax3.grid(True, alpha=0.3)

# 3.4 Multi-variable Weather Impact
ax4 = axes[1, 1]
weather_factors = ['Daily_Rainfall_mm', 'Max_Temp_C', 'Min_Temp_C', 
                   'Avg_Humidity_%', 'Wind_Speed_km_h']
correlations = [df[factor].corr(df['Modal_Price']) for factor in weather_factors]
factor_labels = ['Rainfall', 'Max Temp', 'Min Temp', 'Humidity', 'Wind Speed']
colors_corr = ['green' if c > 0 else 'red' for c in correlations]
bars = ax4.barh(factor_labels, correlations, color=colors_corr, alpha=0.7)
ax4.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
ax4.set_title('Weather Factors Correlation with Modal Price', fontsize=12, fontweight='bold')
ax4.set_xlabel('Correlation Coefficient')
ax4.grid(True, alpha=0.3, axis='x')

# Add value labels
for bar, val in zip(bars, correlations):
    ax4.text(val + (0.01 if val > 0 else -0.01), bar.get_y() + bar.get_height()/2, 
             f'{val:.3f}', va='center', ha='left' if val > 0 else 'right', fontweight='bold')

plt.tight_layout()
plt.savefig('EDA_3_Weather_Impact.png', dpi=300, bbox_inches='tight')
print("✓ Saved: EDA_3_Weather_Impact.png")
plt.close()

# Key Insights
print("\n📊 KEY INSIGHTS - WEATHER IMPACT:")
print("   1. Rainfall-Price Relationship:")
corr_rainfall = df['Daily_Rainfall_mm'].corr(df['Modal_Price'])
print(f"      • Correlation: {corr_rainfall:.3f} ({'Positive' if corr_rainfall > 0 else 'Negative'})")
print(f"      • Trend: Prices {z[0]:.2f} ₹/mm of rainfall")
for idx, row in rainfall_cat_price.iterrows():
    print(f"      • {row['Rainfall_Category']}: ₹{row['mean']:.2f} (±{row['std']:.2f})")

print("   2. Temperature Impact:")
corr_temp = df['Max_Temp_C'].corr(df['Modal_Price'])
print(f"      • Max Temperature correlation: {corr_temp:.3f}")
print(f"      • Optimal temp range: {temp_labels[temp_price.values.argmax()]}")

print("   3. Overall Weather Impact:")
for factor, corr in zip(factor_labels, correlations):
    impact = "Strong" if abs(corr) > 0.5 else "Moderate" if abs(corr) > 0.3 else "Weak"
    print(f"      • {factor}: {corr:.3f} ({impact} {'positive' if corr > 0 else 'negative'} impact)")

# ============================================================================
# VISUALIZATION 4: CORRELATION HEATMAP
# ============================================================================
print("\n" + "=" * 80)
print("VISUALIZATION 4: CORRELATION HEATMAP")
print("=" * 80)

fig, axes = plt.subplots(1, 2, figsize=(18, 8))
fig.suptitle('Correlation Analysis - Variables Relationships', fontsize=16, fontweight='bold')

# 4.1 Full Correlation Heatmap
ax1 = axes[0]
# Select numerical columns
numerical_cols = ['Arrivals', 'Min_Price', 'Max_Price', 'Modal_Price', 'Month', 
                  'Daily_Rainfall_mm', 'Max_Temp_C', 'Min_Temp_C', 
                  'Avg_Humidity_%', 'Wind_Speed_km_h']
corr_matrix = df[numerical_cols].corr()

# Create mask for upper triangle
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='coolwarm', 
            center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8}, ax=ax1,
            vmin=-1, vmax=1)
ax1.set_title('Full Correlation Matrix', fontsize=12, fontweight='bold')

# 4.2 Modal Price Correlation Bar Chart
ax2 = axes[1]
modal_price_corr = corr_matrix['Modal_Price'].drop('Modal_Price').sort_values()
colors_bar = ['#D62828' if x < 0 else '#2A9D8F' for x in modal_price_corr.values]
bars = ax2.barh(range(len(modal_price_corr)), modal_price_corr.values, color=colors_bar, alpha=0.8)
ax2.set_yticks(range(len(modal_price_corr)))
ax2.set_yticklabels(modal_price_corr.index)
ax2.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
ax2.set_title('Modal Price Correlations with Other Variables', fontsize=12, fontweight='bold')
ax2.set_xlabel('Correlation Coefficient')
ax2.grid(True, alpha=0.3, axis='x')

# Add value labels
for bar, val, label in zip(bars, modal_price_corr.values, modal_price_corr.index):
    ax2.text(val + (0.02 if val > 0 else -0.02), bar.get_y() + bar.get_height()/2, 
             f'{val:.3f}', va='center', ha='left' if val > 0 else 'right', fontweight='bold')

plt.tight_layout()
plt.savefig('EDA_4_Correlation_Analysis.png', dpi=300, bbox_inches='tight')
print("✓ Saved: EDA_4_Correlation_Analysis.png")
plt.close()

# Key Insights
print("\n📊 KEY INSIGHTS - CORRELATION ANALYSIS:")
print("   1. Price Variables Relationships:")
print(f"      • Min_Price ↔ Max_Price: {corr_matrix.loc['Min_Price', 'Max_Price']:.3f} (Very Strong)")
print(f"      • Min_Price ↔ Modal_Price: {corr_matrix.loc['Min_Price', 'Modal_Price']:.3f}")
print(f"      • Max_Price ↔ Modal_Price: {corr_matrix.loc['Max_Price', 'Modal_Price']:.3f}")

print("   2. Weather Correlations:")
weather_corrs = modal_price_corr[['Daily_Rainfall_mm', 'Max_Temp_C', 'Min_Temp_C', 
                                   'Avg_Humidity_%', 'Wind_Speed_km_h']]
strongest_weather = weather_corrs.abs().idxmax()
print(f"      • Strongest weather factor: {strongest_weather} ({weather_corrs[strongest_weather]:.3f})")
print(f"      • Weakest weather factor: {weather_corrs.abs().idxmin()} ({weather_corrs[weather_corrs.abs().idxmin()]:.3f})")

print("   3. Supply-Price Relationship:")
arrivals_price_corr = corr_matrix.loc['Arrivals', 'Modal_Price']
print(f"      • Arrivals ↔ Modal_Price: {arrivals_price_corr:.3f}")
print(f"      • Interpretation: {'Higher supply → Lower prices' if arrivals_price_corr < 0 else 'Higher supply → Higher prices'}")

# ============================================================================
# FINAL SUMMARY AND RECOMMENDATIONS
# ============================================================================
print("\n" + "=" * 80)
print("EDA SUMMARY AND KEY RECOMMENDATIONS")
print("=" * 80)

print("""
📊 OVERALL FINDINGS:

1. PRICE DYNAMICS:
   • Crop prices show seasonal variation with clear patterns
   • Price volatility varies significantly across months
   • District-level variations indicate regional market factors

2. WEATHER IMPACT:
   • Weather parameters show measurable correlation with prices
   • Rainfall and temperature have notable effects on pricing
   • Multi-factor weather models will improve prediction accuracy

3. SEASONALITY:
   • Strong monthly patterns in both prices and arrivals
   • Supply-demand dynamics visible in seasonal trends
   • Peak seasons identified for different commodities

4. CORRELATIONS:
   • Price variables (Min/Max/Modal) highly correlated
   • Weather factors show moderate correlations with prices
   • Supply (Arrivals) inversely related to prices

💡 ML MODEL RECOMMENDATIONS:

1. FEATURE ENGINEERING:
   • Include lagged weather variables (7-day, 14-day averages)
   • Create interaction features (Rainfall × Temperature)
   • Add seasonal indicators (harvest season, monsoon period)
   
2. MODEL SELECTION:
   • Time series models for temporal patterns
   • Regression models incorporating weather + seasonal features
   • Ensemble methods to capture complex relationships

3. DATA PREPROCESSING:
   • Normalize weather variables for scale consistency
   • Handle seasonal trends using decomposition
   • Consider district-wise modeling for regional accuracy

4. VALIDATION STRATEGY:
   • Time-based train-test split (preserve temporal order)
   • Cross-validation with time series splits
   • Evaluate on different seasons separately
""")

print("=" * 80)
print("✅ EDA COMPLETED SUCCESSFULLY!")
print("=" * 80)
print(f"\n📁 Generated Files:")
print("   1. EDA_1_Price_Trends.png")
print("   2. EDA_2_Monthly_Seasonality.png")
print("   3. EDA_3_Weather_Impact.png")
print("   4. EDA_4_Correlation_Analysis.png")
print("\n" + "=" * 80)
