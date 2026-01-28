"""
AgriSense AI - Machine Learning Model Training and Evaluation
This script trains and evaluates multiple ML models for crop price prediction
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("MACHINE LEARNING MODEL TRAINING FOR CROP PRICE PREDICTION")
print("=" * 80)

# ============================================================================
# STEP 1: LOAD FEATURE-ENGINEERED DATASET
# ============================================================================
print("\n" + "=" * 80)
print("STEP 1: LOADING FEATURE-ENGINEERED DATASET")
print("=" * 80)

df = pd.read_csv('Crop_Price_Features_Engineered.csv')
print(f"✓ Dataset loaded: {df.shape}")

# Convert date columns
df['Date_crop'] = pd.to_datetime(df['Date_crop'])
df['Date_weather'] = pd.to_datetime(df['Date_weather'])

print(f"  Total records: {len(df):,}")
print(f"  Total features: {df.shape[1]}")
print(f"  Date range: {df['Date_crop'].min()} to {df['Date_crop'].max()}")

# ============================================================================
# STEP 2: FEATURE SELECTION AND PREPARATION
# ============================================================================
print("\n" + "=" * 80)
print("STEP 2: FEATURE SELECTION AND PREPARATION")
print("=" * 80)

# Define target variable
target = 'Modal_Price'

# Define features to exclude (non-predictive or target-related)
exclude_cols = [
    'Date_crop', 'Date_weather', 'Modal_Price', 'Min_Price', 'Max_Price',
    'State', 'Market', 'Variety', 'District', 'Commodity', 'Month_Name',
    'Season', 'Rainfall_Category', 'Log_Modal_Price'  # Exclude log of target
]

# Select feature columns
feature_cols = [col for col in df.columns if col not in exclude_cols]

print(f"\n✓ Target variable: {target}")
print(f"✓ Number of features selected: {len(feature_cols)}")
print(f"\nFeature categories:")

# Categorize features for better understanding
lag_features = [col for col in feature_cols if 'Lag' in col]
ma_features = [col for col in feature_cols if 'MA' in col]
temporal_features = [col for col in feature_cols if any(x in col for x in ['Day', 'Week', 'Month', 'Quarter', 'Season'])]
weather_features = [col for col in feature_cols if any(x in col for x in ['Temp', 'Rainfall', 'Humidity', 'Wind'])]
price_change_features = [col for col in feature_cols if 'Change' in col or 'Volatility' in col]

print(f"  • Lag features: {len(lag_features)}")
print(f"  • Moving average features: {len(ma_features)}")
print(f"  • Temporal features: {len(temporal_features)}")
print(f"  • Weather features: {len(weather_features)}")
print(f"  • Price change features: {len(price_change_features)}")

# Prepare X and y
X = df[feature_cols].copy()
y = df[target].copy()

print(f"\n✓ Feature matrix X: {X.shape}")
print(f"✓ Target vector y: {y.shape}")

# Handle any remaining missing values
print(f"\nMissing values in X: {X.isnull().sum().sum()}")
print(f"Missing values in y: {y.isnull().sum()}")

if X.isnull().sum().sum() > 0:
    X = X.fillna(0)
    print(f"✓ Filled missing values with 0")

if y.isnull().sum() > 0:
    # Remove rows with missing target values
    valid_idx = y.notnull()
    X = X[valid_idx]
    y = y[valid_idx]
    print(f"✓ Removed rows with missing target values")

# ============================================================================
# STEP 3: TRAIN-TEST SPLIT (TIME-BASED)
# ============================================================================
print("\n" + "=" * 80)
print("STEP 3: TRAIN-TEST SPLIT (TIME-BASED)")
print("=" * 80)

print("""
🎯 TIME-BASED SPLIT STRATEGY:
   • For time series data, we MUST maintain temporal order
   • Random shuffling would cause data leakage (using future to predict past)
   • Using 80% earliest data for training, 20% latest data for testing
   • This simulates real-world scenario: predict future prices from past data
""")

# Sort by date to ensure temporal order (already sorted from feature engineering)
split_index = int(len(X) * 0.8)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]
y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

print(f"\n✓ Training set: {X_train.shape[0]:,} samples ({X_train.shape[0]/len(X)*100:.1f}%)")
print(f"✓ Test set: {X_test.shape[0]:,} samples ({X_test.shape[0]/len(X)*100:.1f}%)")

train_date_range = f"{df['Date_crop'].iloc[0]} to {df['Date_crop'].iloc[split_index-1]}"
test_date_range = f"{df['Date_crop'].iloc[split_index]} to {df['Date_crop'].iloc[-1]}"

print(f"\nTraining period: {train_date_range}")
print(f"Testing period: {test_date_range}")

# ============================================================================
# STEP 4: FEATURE SCALING
# ============================================================================
print("\n" + "=" * 80)
print("STEP 4: FEATURE SCALING")
print("=" * 80)

print("""
📊 WHY FEATURE SCALING:
   • Linear models sensitive to feature scales
   • Gradient-based algorithms converge faster with scaled features
   • Ensures all features contribute equally to the model
   • Using StandardScaler (mean=0, std=1)
""")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"✓ Features scaled using StandardScaler")
print(f"  Sample feature means after scaling: {X_train_scaled.mean(axis=0)[:5].round(4)}")
print(f"  Sample feature stds after scaling: {X_train_scaled.std(axis=0)[:5].round(4)}")

# ============================================================================
# STEP 5: MODEL TRAINING AND EVALUATION
# ============================================================================
print("\n" + "=" * 80)
print("STEP 5: MODEL TRAINING AND EVALUATION")
print("=" * 80)

# Dictionary to store results
results = {}

# Function to evaluate model
def evaluate_model(model, X_train, X_test, y_train, y_test, model_name):
    print(f"\n{'=' * 80}")
    print(f"TRAINING: {model_name}")
    print(f"{'=' * 80}")
    
    # Train model
    print(f"Training {model_name}...")
    model.fit(X_train, y_train)
    print(f"✓ Training completed")
    
    # Make predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Calculate metrics for training set
    train_r2 = r2_score(y_train, y_train_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    train_mae = mean_absolute_error(y_train, y_train_pred)
    train_mape = np.mean(np.abs((y_train - y_train_pred) / y_train)) * 100
    
    # Calculate metrics for test set
    test_r2 = r2_score(y_test, y_test_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_mape = np.mean(np.abs((y_test - y_test_pred) / y_test)) * 100
    
    print(f"\n📊 TRAINING SET METRICS:")
    print(f"   R² Score:  {train_r2:.4f}")
    print(f"   RMSE:      ₹{train_rmse:.2f}")
    print(f"   MAE:       ₹{train_mae:.2f}")
    print(f"   MAPE:      {train_mape:.2f}%")
    
    print(f"\n📊 TEST SET METRICS:")
    print(f"   R² Score:  {test_r2:.4f}")
    print(f"   RMSE:      ₹{test_rmse:.2f}")
    print(f"   MAE:       ₹{test_mae:.2f}")
    print(f"   MAPE:      {test_mape:.2f}%")
    
    # Check for overfitting
    overfit_score = train_r2 - test_r2
    print(f"\n🔍 OVERFITTING CHECK:")
    print(f"   R² Difference: {overfit_score:.4f}")
    if overfit_score > 0.1:
        print(f"   ⚠ Warning: Model may be overfitting (train R² >> test R²)")
    elif overfit_score > 0.05:
        print(f"   ℹ Slight overfitting detected")
    else:
        print(f"   ✓ Good generalization")
    
    return {
        'model': model,
        'train_r2': train_r2,
        'train_rmse': train_rmse,
        'train_mae': train_mae,
        'train_mape': train_mape,
        'test_r2': test_r2,
        'test_rmse': test_rmse,
        'test_mae': test_mae,
        'test_mape': test_mape,
        'predictions': y_test_pred
    }

# ============================================================================
# MODEL 1: LINEAR REGRESSION
# ============================================================================
print("\n" + "=" * 80)
print("MODEL 1: LINEAR REGRESSION")
print("=" * 80)

print("""
📌 LINEAR REGRESSION CHARACTERISTICS:
   • Simple, interpretable baseline model
   • Assumes linear relationship between features and target
   • Fast training and prediction
   • Works well with scaled features
   • Good for understanding feature importance (coefficients)
""")

lr_model = LinearRegression()
results['Linear Regression'] = evaluate_model(
    lr_model, X_train_scaled, X_test_scaled, y_train, y_test, 'Linear Regression'
)

# ============================================================================
# MODEL 2: RANDOM FOREST REGRESSOR
# ============================================================================
print("\n" + "=" * 80)
print("MODEL 2: RANDOM FOREST REGRESSOR")
print("=" * 80)

print("""
📌 RANDOM FOREST CHARACTERISTICS:
   • Ensemble of decision trees
   • Handles non-linear relationships well
   • Robust to outliers and missing values
   • Feature importance ranking available
   • Less prone to overfitting than single trees
   • No feature scaling required (but we use scaled features for consistency)
""")

rf_model = RandomForestRegressor(
    n_estimators=100,      # Number of trees
    max_depth=20,          # Maximum tree depth
    min_samples_split=5,   # Minimum samples to split node
    min_samples_leaf=2,    # Minimum samples in leaf
    random_state=42,
    n_jobs=-1,            # Use all CPU cores
    verbose=1
)

results['Random Forest'] = evaluate_model(
    rf_model, X_train, X_test, y_train, y_test, 'Random Forest Regressor'
)

# ============================================================================
# MODEL 3: GRADIENT BOOSTING REGRESSOR
# ============================================================================
print("\n" + "=" * 80)
print("MODEL 3: GRADIENT BOOSTING REGRESSOR")
print("=" * 80)

print("""
📌 GRADIENT BOOSTING CHARACTERISTICS:
   • Sequential ensemble method
   • Builds trees to correct previous trees' errors
   • Often achieves highest accuracy
   • Handles complex non-linear relationships
   • Provides feature importance
   • Requires careful tuning to avoid overfitting
""")

gb_model = GradientBoostingRegressor(
    n_estimators=100,          # Number of boosting stages
    learning_rate=0.1,         # Shrinks contribution of each tree
    max_depth=5,               # Maximum depth of trees
    min_samples_split=5,       # Minimum samples to split
    min_samples_leaf=2,        # Minimum samples in leaf
    subsample=0.8,             # Fraction of samples for each tree
    random_state=42,
    verbose=1
)

results['Gradient Boosting'] = evaluate_model(
    gb_model, X_train, X_test, y_train, y_test, 'Gradient Boosting Regressor'
)

# ============================================================================
# STEP 6: MODEL COMPARISON TABLE
# ============================================================================
print("\n" + "=" * 80)
print("STEP 6: MODEL PERFORMANCE COMPARISON")
print("=" * 80)

# Create comparison dataframe
comparison_data = []
for model_name, metrics in results.items():
    comparison_data.append({
        'Model': model_name,
        'Train R²': metrics['train_r2'],
        'Test R²': metrics['test_r2'],
        'Train RMSE': metrics['train_rmse'],
        'Test RMSE': metrics['test_rmse'],
        'Train MAE': metrics['train_mae'],
        'Test MAE': metrics['test_mae'],
        'Train MAPE%': metrics['train_mape'],
        'Test MAPE%': metrics['test_mape'],
        'Overfit (ΔR²)': metrics['train_r2'] - metrics['test_r2']
    })

comparison_df = pd.DataFrame(comparison_data)

print("\n" + "=" * 80)
print("📊 MODEL PERFORMANCE COMPARISON TABLE")
print("=" * 80)
print(comparison_df.to_string(index=False))

# Identify best model based on test R²
best_model_idx = comparison_df['Test R²'].idxmax()
best_model_name = comparison_df.loc[best_model_idx, 'Model']
best_r2 = comparison_df.loc[best_model_idx, 'Test R²']

print(f"\n🏆 BEST MODEL: {best_model_name}")
print(f"   Test R² Score: {best_r2:.4f}")
print(f"   Test RMSE: ₹{comparison_df.loc[best_model_idx, 'Test RMSE']:.2f}")
print(f"   Test MAE: ₹{comparison_df.loc[best_model_idx, 'Test MAE']:.2f}")

# Save comparison table
comparison_df.to_csv('Model_Performance_Comparison.csv', index=False)
print(f"\n✓ Comparison table saved: Model_Performance_Comparison.csv")

# ============================================================================
# STEP 7: VISUALIZATIONS
# ============================================================================
print("\n" + "=" * 80)
print("STEP 7: CREATING VISUALIZATIONS")
print("=" * 80)

# 7.1 Model Performance Comparison Chart
fig, axes = plt.subplots(2, 2, figsize=(16, 10))
fig.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')

# R² Score Comparison
ax1 = axes[0, 0]
x_pos = np.arange(len(comparison_df))
width = 0.35
ax1.bar(x_pos - width/2, comparison_df['Train R²'], width, label='Train R²', alpha=0.8, color='#2E86AB')
ax1.bar(x_pos + width/2, comparison_df['Test R²'], width, label='Test R²', alpha=0.8, color='#A23B72')
ax1.set_xlabel('Model')
ax1.set_ylabel('R² Score')
ax1.set_title('R² Score Comparison', fontweight='bold')
ax1.set_xticks(x_pos)
ax1.set_xticklabels(comparison_df['Model'], rotation=15, ha='right')
ax1.legend()
ax1.grid(True, alpha=0.3, axis='y')

# RMSE Comparison
ax2 = axes[0, 1]
ax2.bar(x_pos - width/2, comparison_df['Train RMSE'], width, label='Train RMSE', alpha=0.8, color='#F77F00')
ax2.bar(x_pos + width/2, comparison_df['Test RMSE'], width, label='Test RMSE', alpha=0.8, color='#D62828')
ax2.set_xlabel('Model')
ax2.set_ylabel('RMSE (₹)')
ax2.set_title('Root Mean Squared Error Comparison', fontweight='bold')
ax2.set_xticks(x_pos)
ax2.set_xticklabels(comparison_df['Model'], rotation=15, ha='right')
ax2.legend()
ax2.grid(True, alpha=0.3, axis='y')

# MAE Comparison
ax3 = axes[1, 0]
ax3.bar(x_pos - width/2, comparison_df['Train MAE'], width, label='Train MAE', alpha=0.8, color='#06A77D')
ax3.bar(x_pos + width/2, comparison_df['Test MAE'], width, label='Test MAE', alpha=0.8, color='#005F73')
ax3.set_xlabel('Model')
ax3.set_ylabel('MAE (₹)')
ax3.set_title('Mean Absolute Error Comparison', fontweight='bold')
ax3.set_xticks(x_pos)
ax3.set_xticklabels(comparison_df['Model'], rotation=15, ha='right')
ax3.legend()
ax3.grid(True, alpha=0.3, axis='y')

# Overfitting Analysis
ax4 = axes[1, 1]
colors = ['red' if x > 0.1 else 'orange' if x > 0.05 else 'green' for x in comparison_df['Overfit (ΔR²)']]
ax4.bar(x_pos, comparison_df['Overfit (ΔR²)'], alpha=0.8, color=colors)
ax4.axhline(y=0.05, color='orange', linestyle='--', linewidth=2, label='Acceptable (0.05)')
ax4.axhline(y=0.1, color='red', linestyle='--', linewidth=2, label='High (0.10)')
ax4.set_xlabel('Model')
ax4.set_ylabel('R² Difference (Train - Test)')
ax4.set_title('Overfitting Analysis', fontweight='bold')
ax4.set_xticks(x_pos)
ax4.set_xticklabels(comparison_df['Model'], rotation=15, ha='right')
ax4.legend()
ax4.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('Model_Performance_Comparison.png', dpi=300, bbox_inches='tight')
print("✓ Saved: Model_Performance_Comparison.png")
plt.close()

# 7.2 Actual vs Predicted for Best Model
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle(f'Actual vs Predicted Prices - Model Comparison', fontsize=16, fontweight='bold')

for idx, (model_name, metrics) in enumerate(results.items()):
    ax = axes[idx]
    
    # Sample points for better visualization
    sample_size = min(5000, len(y_test))
    sample_indices = np.random.choice(len(y_test), sample_size, replace=False)
    
    y_test_sample = y_test.iloc[sample_indices]
    y_pred_sample = metrics['predictions'][sample_indices]
    
    ax.scatter(y_test_sample, y_pred_sample, alpha=0.5, s=20)
    
    # Perfect prediction line
    min_val = min(y_test_sample.min(), y_pred_sample.min())
    max_val = max(y_test_sample.max(), y_pred_sample.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
    
    ax.set_xlabel('Actual Price (₹)')
    ax.set_ylabel('Predicted Price (₹)')
    ax.set_title(f'{model_name}\nR²={metrics["test_r2"]:.4f}', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('Actual_vs_Predicted_Comparison.png', dpi=300, bbox_inches='tight')
print("✓ Saved: Actual_vs_Predicted_Comparison.png")
plt.close()

# ============================================================================
# STEP 8: FEATURE IMPORTANCE (FOR TREE-BASED MODELS)
# ============================================================================
print("\n" + "=" * 80)
print("STEP 8: FEATURE IMPORTANCE ANALYSIS")
print("=" * 80)

# Random Forest Feature Importance
print("\n📊 TOP 20 MOST IMPORTANT FEATURES (Random Forest):")
rf_importance = pd.DataFrame({
    'Feature': feature_cols,
    'Importance': results['Random Forest']['model'].feature_importances_
}).sort_values('Importance', ascending=False).head(20)

for idx, row in rf_importance.iterrows():
    print(f"   {row['Feature']:40s} {row['Importance']:.6f}")

# Visualize feature importance
fig, axes = plt.subplots(1, 2, figsize=(16, 8))
fig.suptitle('Feature Importance Analysis', fontsize=16, fontweight='bold')

# Random Forest
ax1 = axes[0]
top_features_rf = rf_importance.head(15)
ax1.barh(range(len(top_features_rf)), top_features_rf['Importance'], color='#2E86AB', alpha=0.8)
ax1.set_yticks(range(len(top_features_rf)))
ax1.set_yticklabels(top_features_rf['Feature'])
ax1.set_xlabel('Importance Score')
ax1.set_title('Top 15 Features - Random Forest', fontweight='bold')
ax1.grid(True, alpha=0.3, axis='x')
ax1.invert_yaxis()

# Gradient Boosting
ax2 = axes[1]
gb_importance = pd.DataFrame({
    'Feature': feature_cols,
    'Importance': results['Gradient Boosting']['model'].feature_importances_
}).sort_values('Importance', ascending=False).head(15)

ax2.barh(range(len(gb_importance)), gb_importance['Importance'], color='#D62828', alpha=0.8)
ax2.set_yticks(range(len(gb_importance)))
ax2.set_yticklabels(gb_importance['Feature'])
ax2.set_xlabel('Importance Score')
ax2.set_title('Top 15 Features - Gradient Boosting', fontweight='bold')
ax2.grid(True, alpha=0.3, axis='x')
ax2.invert_yaxis()

plt.tight_layout()
plt.savefig('Feature_Importance_Analysis.png', dpi=300, bbox_inches='tight')
print("\n✓ Saved: Feature_Importance_Analysis.png")
plt.close()

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("FINAL SUMMARY AND RECOMMENDATIONS")
print("=" * 80)

print(f"""
🎯 KEY FINDINGS:

1. BEST PERFORMING MODEL:
   • {best_model_name}
   • Test R² = {best_r2:.4f} ({best_r2*100:.2f}% variance explained)
   • Test RMSE = ₹{comparison_df.loc[best_model_idx, 'Test RMSE']:.2f}
   • Test MAE = ₹{comparison_df.loc[best_model_idx, 'Test MAE']:.2f}

2. MODEL COMPARISON:
""")

for idx, row in comparison_df.iterrows():
    print(f"   {row['Model']:20s} → R²: {row['Test R²']:.4f}, RMSE: ₹{row['Test RMSE']:.2f}")

print(f"""
3. TOP PREDICTIVE FEATURES:
   • {rf_importance.iloc[0]['Feature']}
   • {rf_importance.iloc[1]['Feature']}
   • {rf_importance.iloc[2]['Feature']}
   • {rf_importance.iloc[3]['Feature']}
   • {rf_importance.iloc[4]['Feature']}

4. MODEL CHARACTERISTICS:
   • Linear Regression: Fast, interpretable, baseline performance
   • Random Forest: Robust, handles non-linearity, good generalization
   • Gradient Boosting: Highest accuracy, captures complex patterns

💡 RECOMMENDATIONS:

1. FOR PRODUCTION DEPLOYMENT:
   • Use {best_model_name} for best accuracy
   • Consider ensemble of top 2 models for robustness
   • Implement periodic model retraining (monthly)

2. FURTHER IMPROVEMENTS:
   • Hyperparameter tuning using GridSearch/RandomSearch
   • Try XGBoost or LightGBM for better performance
   • Implement deep learning models (LSTM) for time series
   • Add more external features (economic indicators, export data)

3. MODEL MONITORING:
   • Track prediction errors over time
   • Monitor feature drift and data quality
   • Set up alerts for anomalous predictions
   • Regular model validation on new data

4. BUSINESS APPLICATION:
   • Provide price predictions with confidence intervals
   • Alert farmers about expected price movements
   • Integrate with mobile/web application
   • Generate automated reports and insights
""")

print("=" * 80)
print("✅ MODEL TRAINING AND EVALUATION COMPLETED SUCCESSFULLY!")
print("=" * 80)

print("\n📁 Generated Files:")
print("   1. Model_Performance_Comparison.csv")
print("   2. Model_Performance_Comparison.png")
print("   3. Actual_vs_Predicted_Comparison.png")
print("   4. Feature_Importance_Analysis.png")

print("\n" + "=" * 80)
