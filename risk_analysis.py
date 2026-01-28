"""
AgriSense AI - Risk Analysis for Crop Price Prediction
This script uses K-Means clustering and anomaly detection to classify crop risks
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("AI-BASED CROP PRICE RISK ANALYSIS")
print("=" * 80)

# ============================================================================
# STEP 1: LOAD AND PREPARE DATA
# ============================================================================
print("\n" + "=" * 80)
print("STEP 1: LOADING AND PREPARING DATA")
print("=" * 80)

df = pd.read_csv('Crop_Price_Features_Engineered.csv')
df['Date_crop'] = pd.to_datetime(df['Date_crop'])

print(f"✓ Dataset loaded: {df.shape}")
print(f"  Date range: {df['Date_crop'].min()} to {df['Date_crop'].max()}")
print(f"  Commodities: {df['Commodity'].nunique()}")
print(f"  Districts: {df['District'].nunique()}")

# ============================================================================
# STEP 2: CALCULATE RISK INDICATORS
# ============================================================================
print("\n" + "=" * 80)
print("STEP 2: CALCULATING RISK INDICATORS")
print("=" * 80)

print("""
📊 RISK INDICATORS BEING CALCULATED:

1. PRICE VOLATILITY (σ)
   • How much prices jump up and down
   • High volatility = Unpredictable prices = Higher risk
   
2. PRICE CRASH FREQUENCY
   • How often prices suddenly drop >10% in a day
   • More crashes = Higher risk for farmers
   
3. PRICE STABILITY SCORE
   • Measures consistency of prices over time
   • Lower score = More stable = Lower risk
   
4. SUPPLY VARIATION
   • How much arrival quantities fluctuate
   • High variation = Market uncertainty
""")

# Calculate risk metrics for each Commodity-District combination
risk_metrics = []

for (commodity, district), group in df.groupby(['Commodity', 'District']):
    if len(group) < 30:  # Skip if insufficient data
        continue
    
    # Sort by date
    group = group.sort_values('Date_crop')
    
    # 1. Price Volatility (Standard Deviation)
    price_volatility = group['Modal_Price'].std()
    price_cv = (price_volatility / group['Modal_Price'].mean()) * 100  # Coefficient of Variation
    
    # 2. Price Crash Detection (>10% drop in single day)
    price_changes = group['Modal_Price'].pct_change()
    crash_count = (price_changes < -0.10).sum()  # 10% drop threshold
    crash_frequency = (crash_count / len(group)) * 100
    
    # 3. Price Stability Score (Inverse of CV)
    stability_score = 100 - min(price_cv, 100)  # Scale 0-100
    
    # 4. Supply Variation
    supply_volatility = group['Arrivals'].std()
    supply_cv = (supply_volatility / group['Arrivals'].mean()) * 100 if group['Arrivals'].mean() > 0 else 0
    
    # 5. Average Price and Trends
    avg_price = group['Modal_Price'].mean()
    price_trend = (group['Modal_Price'].iloc[-1] - group['Modal_Price'].iloc[0]) / group['Modal_Price'].iloc[0] * 100
    
    # 6. Maximum Single-Day Drop
    max_drop = abs(price_changes.min()) * 100 if price_changes.min() < 0 else 0
    
    risk_metrics.append({
        'Commodity': commodity,
        'District': district,
        'Average_Price': avg_price,
        'Price_Volatility_Rs': price_volatility,
        'Price_CV_Percent': price_cv,
        'Crash_Count': crash_count,
        'Crash_Frequency_Percent': crash_frequency,
        'Stability_Score': stability_score,
        'Supply_CV_Percent': supply_cv,
        'Price_Trend_Percent': price_trend,
        'Max_Single_Day_Drop_Percent': max_drop,
        'Data_Points': len(group)
    })

risk_df = pd.DataFrame(risk_metrics)
print(f"\n✓ Risk metrics calculated for {len(risk_df)} Commodity-District pairs")
print(f"\nSample Risk Metrics:")
print(risk_df.head(3).to_string(index=False))

# ============================================================================
# STEP 3: K-MEANS CLUSTERING FOR RISK CLASSIFICATION
# ============================================================================
print("\n" + "=" * 80)
print("STEP 3: K-MEANS CLUSTERING FOR RISK CLASSIFICATION")
print("=" * 80)

print("""
🤖 K-MEANS CLUSTERING EXPLANATION:

What is K-Means?
• An AI algorithm that groups similar items together
• Like sorting mangoes: some are sweet (stable), some are sour (risky)
• Machine finds patterns in data automatically

How it works here:
• Looks at volatility, crashes, and stability
• Groups crops into 3 categories:
  1. 🟢 LOW RISK (Stable prices, predictable)
  2. 🟡 MEDIUM RISK (Some ups and downs)
  3. 🔴 HIGH RISK (Very unstable, dangerous)
""")

# Select features for clustering
clustering_features = ['Price_CV_Percent', 'Crash_Frequency_Percent', 
                       'Supply_CV_Percent', 'Max_Single_Day_Drop_Percent']

X_cluster = risk_df[clustering_features].fillna(0)

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_cluster)

# Apply K-Means (3 clusters: Low, Medium, High risk)
n_clusters = 3
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
risk_df['Risk_Cluster'] = kmeans.fit_predict(X_scaled)

# Determine which cluster represents which risk level
cluster_risk_levels = []
for i in range(n_clusters):
    cluster_data = risk_df[risk_df['Risk_Cluster'] == i]
    avg_cv = cluster_data['Price_CV_Percent'].mean()
    avg_crash = cluster_data['Crash_Frequency_Percent'].mean()
    risk_score = avg_cv + (avg_crash * 5)  # Weight crashes more heavily
    cluster_risk_levels.append((i, risk_score))

# Sort by risk score and assign labels
cluster_risk_levels.sort(key=lambda x: x[1])
cluster_mapping = {
    cluster_risk_levels[0][0]: 'Low Risk 🟢',
    cluster_risk_levels[1][0]: 'Medium Risk 🟡',
    cluster_risk_levels[2][0]: 'High Risk 🔴'
}

risk_df['Risk_Category'] = risk_df['Risk_Cluster'].map(cluster_mapping)

print(f"\n✓ K-Means clustering completed with {n_clusters} risk categories")
print(f"\nRisk Category Distribution:")
print(risk_df['Risk_Category'].value_counts().to_string())

print(f"\nCluster Characteristics:")
for category in ['Low Risk 🟢', 'Medium Risk 🟡', 'High Risk 🔴']:
    cluster_data = risk_df[risk_df['Risk_Category'] == category]
    print(f"\n{category}:")
    print(f"  • Average Price Volatility: {cluster_data['Price_CV_Percent'].mean():.2f}%")
    print(f"  • Average Crash Frequency: {cluster_data['Crash_Frequency_Percent'].mean():.2f}%")
    print(f"  • Average Stability Score: {cluster_data['Stability_Score'].mean():.2f}/100")
    print(f"  • Count: {len(cluster_data)} commodity-district pairs")

# ============================================================================
# STEP 4: ANOMALY DETECTION FOR PRICE CRASHES
# ============================================================================
print("\n" + "=" * 80)
print("STEP 4: DETECTING PRICE CRASH ANOMALIES")
print("=" * 80)

print("""
🚨 ANOMALY DETECTION EXPLANATION:

What are Anomalies?
• Unusual events that don't follow normal patterns
• Like sudden heavy rain in summer - unexpected!
• In our case: Sudden big price drops

How we detect them:
• Using Isolation Forest algorithm
• Finds days when prices behaved very differently
• Helps farmers prepare for sudden market shocks
""")

# Prepare data for anomaly detection
df_anomaly = df[['Date_crop', 'Commodity', 'District', 'Modal_Price', 
                 'Price_Change_Pct', 'Price_Volatility', 'Arrivals']].copy()

# Create anomaly features
anomaly_features = ['Price_Change_Pct', 'Price_Volatility']
X_anomaly = df_anomaly[anomaly_features].fillna(0)

# Apply Isolation Forest
iso_forest = IsolationForest(contamination=0.05, random_state=42)  # 5% as anomalies
df_anomaly['Anomaly'] = iso_forest.fit_predict(X_anomaly)
df_anomaly['Is_Anomaly'] = df_anomaly['Anomaly'] == -1  # -1 indicates anomaly

# Focus on negative anomalies (price crashes)
df_anomaly['Is_Price_Crash'] = (df_anomaly['Is_Anomaly']) & (df_anomaly['Price_Change_Pct'] < -5)

anomaly_count = df_anomaly['Is_Anomaly'].sum()
crash_count = df_anomaly['Is_Price_Crash'].sum()

print(f"\n✓ Anomaly detection completed")
print(f"  • Total anomalies detected: {anomaly_count:,} days ({(anomaly_count/len(df_anomaly)*100):.2f}%)")
print(f"  • Price crashes identified: {crash_count:,} days ({(crash_count/len(df_anomaly)*100):.2f}%)")

# Top 10 worst price crashes
worst_crashes = df_anomaly[df_anomaly['Is_Price_Crash']].nsmallest(10, 'Price_Change_Pct')
print(f"\n🚨 TOP 10 WORST PRICE CRASHES:")
print("   " + "-" * 70)
for idx, row in worst_crashes.iterrows():
    print(f"   {row['Date_crop'].strftime('%Y-%m-%d')} | {row['Commodity']:12} | {row['District']:15} | {row['Price_Change_Pct']:+.2f}%")

# ============================================================================
# STEP 5: ASSIGN COMPREHENSIVE RISK SCORES
# ============================================================================
print("\n" + "=" * 80)
print("STEP 5: ASSIGNING COMPREHENSIVE RISK SCORES")
print("=" * 80)

print("""
📊 RISK SCORE CALCULATION (0-100 scale):

Risk Score Components:
• 30% - Price Volatility (how much prices jump)
• 30% - Crash Frequency (how often big drops happen)
• 20% - Supply Volatility (arrival unpredictability)
• 20% - Maximum Single-Day Drop (biggest price fall)

Score Interpretation:
• 0-30:   Low Risk 🟢 (Safe to grow)
• 31-60:  Medium Risk 🟡 (Be careful, watch market)
• 61-100: High Risk 🔴 (Very risky, need protection)
""")

# Calculate composite risk score (0-100)
def calculate_risk_score(row):
    # Normalize each component to 0-100 scale
    volatility_score = min(row['Price_CV_Percent'], 50) / 50 * 100
    crash_score = min(row['Crash_Frequency_Percent'], 10) / 10 * 100
    supply_score = min(row['Supply_CV_Percent'], 100) / 100 * 100
    drop_score = min(row['Max_Single_Day_Drop_Percent'], 30) / 30 * 100
    
    # Weighted average
    risk_score = (volatility_score * 0.30 + 
                  crash_score * 0.30 + 
                  supply_score * 0.20 + 
                  drop_score * 0.20)
    
    return min(risk_score, 100)

risk_df['Risk_Score'] = risk_df.apply(calculate_risk_score, axis=1)

# Classify based on risk score
def classify_risk_level(score):
    if score < 30:
        return 'Low Risk 🟢'
    elif score < 60:
        return 'Medium Risk 🟡'
    else:
        return 'High Risk 🔴'

risk_df['Risk_Level'] = risk_df['Risk_Score'].apply(classify_risk_level)

print(f"\n✓ Risk scores calculated for all commodity-district pairs")

# ============================================================================
# STEP 6: AGGREGATE RISK SCORES BY COMMODITY AND DISTRICT
# ============================================================================
print("\n" + "=" * 80)
print("STEP 6: RISK RANKINGS")
print("=" * 80)

# Top risky commodities
commodity_risk = risk_df.groupby('Commodity').agg({
    'Risk_Score': 'mean',
    'Price_CV_Percent': 'mean',
    'Crash_Frequency_Percent': 'mean',
    'District': 'count'
}).round(2)
commodity_risk.columns = ['Risk_Score', 'Price_Volatility_%', 'Crash_Freq_%', 'Districts_Count']
commodity_risk = commodity_risk.sort_values('Risk_Score', ascending=False)

print(f"\n🌾 TOP 10 RISKIEST CROPS:")
print("   " + "-" * 70)
print(f"   {'Rank':<5} {'Commodity':<15} {'Risk Score':<12} {'Volatility':<12} {'Crashes':<10}")
print("   " + "-" * 70)
for i, (commodity, row) in enumerate(commodity_risk.head(10).iterrows(), 1):
    risk_emoji = '🔴' if row['Risk_Score'] >= 60 else ('🟡' if row['Risk_Score'] >= 30 else '🟢')
    print(f"   {i:<5} {commodity:<15} {row['Risk_Score']:>6.1f} {risk_emoji:<5} {row['Price_Volatility_%']:>6.1f}%     {row['Crash_Freq_%']:>5.2f}%")

print(f"\n🌾 TOP 10 SAFEST CROPS:")
print("   " + "-" * 70)
print(f"   {'Rank':<5} {'Commodity':<15} {'Risk Score':<12} {'Volatility':<12} {'Crashes':<10}")
print("   " + "-" * 70)
for i, (commodity, row) in enumerate(commodity_risk.tail(10).iterrows(), 1):
    risk_emoji = '🟢'
    print(f"   {i:<5} {commodity:<15} {row['Risk_Score']:>6.1f} {risk_emoji:<5} {row['Price_Volatility_%']:>6.1f}%     {row['Crash_Freq_%']:>5.2f}%")

# Top risky districts
district_risk = risk_df.groupby('District').agg({
    'Risk_Score': 'mean',
    'Price_CV_Percent': 'mean',
    'Crash_Frequency_Percent': 'mean',
    'Commodity': 'count'
}).round(2)
district_risk.columns = ['Risk_Score', 'Price_Volatility_%', 'Crash_Freq_%', 'Commodities_Count']
district_risk = district_risk.sort_values('Risk_Score', ascending=False)

print(f"\n📍 TOP 10 RISKIEST DISTRICTS:")
print("   " + "-" * 70)
print(f"   {'Rank':<5} {'District':<15} {'Risk Score':<12} {'Volatility':<12} {'Crashes':<10}")
print("   " + "-" * 70)
for i, (district, row) in enumerate(district_risk.head(10).iterrows(), 1):
    risk_emoji = '🔴' if row['Risk_Score'] >= 60 else ('🟡' if row['Risk_Score'] >= 30 else '🟢')
    print(f"   {i:<5} {district:<15} {row['Risk_Score']:>6.1f} {risk_emoji:<5} {row['Price_Volatility_%']:>6.1f}%     {row['Crash_Freq_%']:>5.2f}%")

print(f"\n📍 TOP 10 SAFEST DISTRICTS:")
print("   " + "-" * 70)
print(f"   {'Rank':<5} {'District':<15} {'Risk Score':<12} {'Volatility':<12} {'Crashes':<10}")
print("   " + "-" * 70)
for i, (district, row) in enumerate(district_risk.tail(10).iterrows(), 1):
    risk_emoji = '🟢'
    print(f"   {i:<5} {district:<15} {row['Risk_Score']:>6.1f} {risk_emoji:<5} {row['Price_Volatility_%']:>6.1f}%     {row['Crash_Freq_%']:>5.2f}%")

# ============================================================================
# STEP 7: VISUALIZATIONS
# ============================================================================
print("\n" + "=" * 80)
print("STEP 7: CREATING RISK ANALYSIS VISUALIZATIONS")
print("=" * 80)

fig = plt.figure(figsize=(20, 14))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# 1. Risk Category Distribution (Pie Chart)
ax1 = fig.add_subplot(gs[0, 0])
risk_counts = risk_df['Risk_Level'].value_counts()
colors_pie = ['#2ECC71', '#F39C12', '#E74C3C']  # Green, Yellow, Red
ax1.pie(risk_counts.values, labels=risk_counts.index, autopct='%1.1f%%',
        colors=colors_pie, startangle=90, textprops={'fontsize': 10, 'fontweight': 'bold'})
ax1.set_title('Risk Category Distribution', fontweight='bold', fontsize=12)

# 2. Top 10 Riskiest Commodities
ax2 = fig.add_subplot(gs[0, 1:])
top_risky = commodity_risk.head(10).sort_values('Risk_Score')
colors_bar = ['#E74C3C' if x >= 60 else '#F39C12' if x >= 30 else '#2ECC71' 
              for x in top_risky['Risk_Score']]
ax2.barh(top_risky.index, top_risky['Risk_Score'], color=colors_bar, alpha=0.8)
ax2.set_xlabel('Risk Score', fontweight='bold')
ax2.set_title('Top 10 Riskiest Commodities', fontweight='bold', fontsize=12)
ax2.axvline(x=30, color='orange', linestyle='--', linewidth=1, alpha=0.5)
ax2.axvline(x=60, color='red', linestyle='--', linewidth=1, alpha=0.5)
ax2.grid(True, alpha=0.3, axis='x')

# 3. Top 10 Riskiest Districts
ax3 = fig.add_subplot(gs[1, :])
top_risky_districts = district_risk.head(10).sort_values('Risk_Score')
colors_bar_d = ['#E74C3C' if x >= 60 else '#F39C12' if x >= 30 else '#2ECC71' 
                for x in top_risky_districts['Risk_Score']]
ax3.barh(top_risky_districts.index, top_risky_districts['Risk_Score'], color=colors_bar_d, alpha=0.8)
ax3.set_xlabel('Risk Score', fontweight='bold')
ax3.set_title('Top 10 Riskiest Districts', fontweight='bold', fontsize=12)
ax3.axvline(x=30, color='orange', linestyle='--', linewidth=1, alpha=0.5)
ax3.axvline(x=60, color='red', linestyle='--', linewidth=1, alpha=0.5)
ax3.grid(True, alpha=0.3, axis='x')

# 4. Risk Score vs Price Volatility Scatter
ax4 = fig.add_subplot(gs[2, 0])
for risk_level, color in zip(['Low Risk 🟢', 'Medium Risk 🟡', 'High Risk 🔴'], 
                              ['#2ECC71', '#F39C12', '#E74C3C']):
    subset = risk_df[risk_df['Risk_Level'] == risk_level]
    ax4.scatter(subset['Price_CV_Percent'], subset['Risk_Score'], 
                alpha=0.6, s=50, label=risk_level, color=color)
ax4.set_xlabel('Price Volatility (CV %)', fontweight='bold')
ax4.set_ylabel('Risk Score', fontweight='bold')
ax4.set_title('Risk Score vs Price Volatility', fontweight='bold', fontsize=12)
ax4.legend()
ax4.grid(True, alpha=0.3)

# 5. Crash Frequency Distribution
ax5 = fig.add_subplot(gs[2, 1])
ax5.hist(risk_df['Crash_Frequency_Percent'], bins=30, color='#E74C3C', alpha=0.7, edgecolor='black')
ax5.set_xlabel('Crash Frequency (%)', fontweight='bold')
ax5.set_ylabel('Count', fontweight='bold')
ax5.set_title('Distribution of Price Crash Frequency', fontweight='bold', fontsize=12)
ax5.axvline(x=risk_df['Crash_Frequency_Percent'].mean(), color='red', 
            linestyle='--', linewidth=2, label=f'Mean: {risk_df["Crash_Frequency_Percent"].mean():.2f}%')
ax5.legend()
ax5.grid(True, alpha=0.3)

# 6. K-Means Clustering Visualization
ax6 = fig.add_subplot(gs[2, 2])
for risk_level, color in zip(['Low Risk 🟢', 'Medium Risk 🟡', 'High Risk 🔴'], 
                              ['#2ECC71', '#F39C12', '#E74C3C']):
    subset = risk_df[risk_df['Risk_Category'] == risk_level]
    ax6.scatter(subset['Price_CV_Percent'], subset['Crash_Frequency_Percent'], 
                alpha=0.6, s=50, label=risk_level, color=color)
ax6.set_xlabel('Price Volatility (CV %)', fontweight='bold')
ax6.set_ylabel('Crash Frequency (%)', fontweight='bold')
ax6.set_title('K-Means Risk Clustering', fontweight='bold', fontsize=12)
ax6.legend()
ax6.grid(True, alpha=0.3)

plt.suptitle('AI-Based Crop Price Risk Analysis Dashboard', 
             fontsize=16, fontweight='bold', y=0.995)
plt.savefig('Risk_Analysis_Dashboard.png', dpi=300, bbox_inches='tight')
print("✓ Saved: Risk_Analysis_Dashboard.png")
plt.close()

# Save risk data
risk_df.to_csv('Risk_Analysis_Results.csv', index=False)
print("✓ Saved: Risk_Analysis_Results.csv")

commodity_risk.to_csv('Commodity_Risk_Rankings.csv')
print("✓ Saved: Commodity_Risk_Rankings.csv")

district_risk.to_csv('District_Risk_Rankings.csv')
print("✓ Saved: District_Risk_Rankings.csv")

# ============================================================================
# STEP 8: FARMER-FRIENDLY EXPLANATIONS
# ============================================================================
print("\n" + "=" * 80)
print("STEP 8: FARMER-FRIENDLY RISK GUIDE")
print("=" * 80)

print("""
╔═══════════════════════════════════════════════════════════════════════════╗
║                   🌾 FARMER'S GUIDE TO CROP RISK 🌾                       ║
╚═══════════════════════════════════════════════════════════════════════════╝

📚 WHAT IS RISK IN FARMING?

Risk means "uncertainty" - not knowing what price you'll get for your crop.
• HIGH RISK = Prices jump a lot, hard to plan
• LOW RISK = Prices steady, easy to plan

Think of it like weather:
• Low Risk Crop = Steady weather all year
• High Risk Crop = Sometimes hot, sometimes cold, unpredictable!

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🟢 LOW RISK CROPS (Score 0-30) - SAFE FOR FARMERS

What it means:
✓ Prices stay mostly steady throughout the year
✓ You can predict income better
✓ Market is stable, less chance of loss
✓ Good for farmers who need guaranteed income

Example: If price is ₹2000 today, it will likely be ₹1900-₹2100 next month

Best for:
• Small farmers who can't take losses
• Farmers with loans to repay
• Those who need predictable income

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🟡 MEDIUM RISK CROPS (Score 31-60) - BE CAREFUL

What it means:
⚠ Prices go up and down moderately
⚠ Some months good, some months bad
⚠ Need to watch market regularly
⚠ Can make good profit OR face some loss

Example: If price is ₹2000 today, it might be ₹1700-₹2400 next month

Best for:
• Farmers who can handle some uncertainty
• Those who watch market news regularly
• Farmers with some savings for bad times
• Can wait for good prices before selling

Tips:
• Don't sell all produce at once
• Store some for when prices are better
• Watch weather and festival seasons
• Consider government schemes for price support

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🔴 HIGH RISK CROPS (Score 61-100) - VERY DANGEROUS

What it means:
⛔ Prices change dramatically and unpredictably
⛔ Today ₹3000, next week ₹2000, next month ₹3500
⛔ Very hard to plan income
⛔ Can make HUGE profit OR HUGE loss

Example: If price is ₹2000 today, it might be ₹1200-₹3000 next month

Dangers:
• Price can crash suddenly (10-30% drop in a day)
• Supply changes affect prices a lot
• Weather impact is very high
• Market manipulation possible

Who should grow:
• Only farmers who can afford losses
• Those with good market connections
• Farmers with storage facilities
• Those who can wait for right selling time

Protection strategies:
🛡️ NEVER grow only high-risk crops
🛡️ Mix with low-risk crops (diversification)
🛡️ Get crop insurance (Pradhan Mantri Fasal Bima Yojana)
🛡️ Use contract farming if available
🛡️ Store produce, don't sell in panic
🛡️ Join farmer groups for better bargaining

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

💡 PRACTICAL ADVICE FOR FARMERS:

1. DIVERSIFY YOUR CROPS
   • Grow 2-3 different crops
   • Mix high-risk and low-risk crops
   • If one fails, others will support
   
2. WATCH FOR WARNING SIGNS
   🚨 Sudden price drop >10% in a day = DANGER
   🚨 Too much supply in market = Prices will fall
   🚨 Bad weather news = Prices might jump
   
3. USE TECHNOLOGY
   • Check mandi prices daily (mobile apps)
   • Watch weather forecasts
   • Get SMS alerts for price changes
   • Use this AI system regularly
   
4. FINANCIAL PLANNING
   • Save money during good price periods
   • Don't spend all income immediately
   • Keep emergency fund for 3-6 months
   • Take loans carefully, only for low-risk crops
   
5. SELLING STRATEGY
   • Don't sell everything on harvest day (lowest prices!)
   • Store for 1-2 months if possible
   • Sell in batches when prices are good
   • Use price prediction to decide timing

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🎯 HOW TO USE RISK SCORES:

Planning Next Season:
1. Check risk score of crop you want to grow
2. Check your district's risk score
3. If BOTH high = VERY DANGEROUS, think twice!
4. If crop high but district low = Medium risk
5. If both low = SAFE to grow

Example Decision:
❌ BAD: Growing high-risk crop in high-risk district
✓ OKAY: Growing medium-risk crop in low-risk district
✓✓ GOOD: Growing low-risk crop in low-risk district
⚠️ RISKY: Growing high-risk crop in low-risk district (only if you can afford loss)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📞 REMEMBER:

• This is AI-based guidance, not guarantee
• Always consult local agricultural officer
• Weather and government policies can change everything
• Your experience and local knowledge is valuable
• Don't make big decisions based on AI alone

🌾 Happy Farming! Plan Smart, Farm Safe! 🌾
""")

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("FINAL SUMMARY")
print("=" * 80)

print(f"""
✅ RISK ANALYSIS COMPLETED SUCCESSFULLY!

📊 ANALYSIS SUMMARY:

1. DATA PROCESSED:
   • {len(risk_df):,} Commodity-District combinations analyzed
   • {len(df):,} individual price records examined
   • {df['Commodity'].nunique()} commodities evaluated
   • {df['District'].nunique()} districts assessed

2. RISK CLASSIFICATION:
   • {len(risk_df[risk_df['Risk_Level'] == 'Low Risk 🟢'])} pairs in LOW RISK 🟢
   • {len(risk_df[risk_df['Risk_Level'] == 'Medium Risk 🟡'])} pairs in MEDIUM RISK 🟡
   • {len(risk_df[risk_df['Risk_Level'] == 'High Risk 🔴'])} pairs in HIGH RISK 🔴

3. ANOMALIES DETECTED:
   • {anomaly_count:,} unusual price movements identified
   • {crash_count:,} price crash events detected
   • Average crash severity: {df_anomaly[df_anomaly['Is_Price_Crash']]['Price_Change_Pct'].mean():.2f}%

4. TOP FINDINGS:
   • Riskiest Commodity: {commodity_risk.index[0]} (Score: {commodity_risk.iloc[0]['Risk_Score']:.1f})
   • Safest Commodity: {commodity_risk.index[-1]} (Score: {commodity_risk.iloc[-1]['Risk_Score']:.1f})
   • Riskiest District: {district_risk.index[0]} (Score: {district_risk.iloc[0]['Risk_Score']:.1f})
   • Safest District: {district_risk.index[-1]} (Score: {district_risk.iloc[-1]['Risk_Score']:.1f})

5. FILES GENERATED:
   ✓ Risk_Analysis_Dashboard.png (6 visualizations)
   ✓ Risk_Analysis_Results.csv (detailed risk data)
   ✓ Commodity_Risk_Rankings.csv (crop risk scores)
   ✓ District_Risk_Rankings.csv (district risk scores)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🎯 RECOMMENDATIONS FOR POLICY MAKERS:

1. HIGH RISK AREAS NEED:
   • Better storage facilities (reduce panic selling)
   • Crop insurance schemes
   • Minimum Support Price (MSP) guarantees
   • Direct market linkages for farmers

2. PRICE STABILIZATION:
   • Monitor high-risk commodities closely
   • Buffer stock for volatile crops
   • Import/export regulations during crashes
   • Farmer education on risk management

3. DISTRICT-SPECIFIC INTERVENTIONS:
   • High-risk districts need immediate support
   • Infrastructure for price information
   • Mandi reforms for transparency
   • Transport and cold storage facilities

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
""")

print("=" * 80)
print("🌾 RISK ANALYSIS COMPLETE! FARMERS CAN NOW MAKE INFORMED DECISIONS! 🌾")
print("=" * 80)
