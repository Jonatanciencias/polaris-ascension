"""
Phase 2: Neural Performance Predictor - Model Training

Entrena modelo ML para predecir performance de kernels GEMM
"""

import json
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import pickle

# Load dataset
print("=" * 80)
print("NEURAL PERFORMANCE PREDICTOR - TRAINING")
print("=" * 80)
print()

with open('neural_predictor_dataset.json', 'r') as f:
    dataset_json = json.load(f)

data = dataset_json['data']
print(f"Loaded {len(data)} samples")
print()

# Convert to DataFrame
df = pd.DataFrame(data)

# Feature engineering
print("Features available:")
print(df.columns.tolist())
print()

# Select features for training
feature_cols = ['M', 'N', 'K', 'tile_size', 'threads', 'vectorized']
target_col = 'gflops'

X = df[feature_cols].values
y = df[target_col].values

print(f"Training shape: X={X.shape}, y={y.shape}")
print()

# Additional engineered features
X_engineered = np.column_stack([
    X,  # Original features
    X[:, 0] * X[:, 1] * X[:, 2],  # Total FLOPs (M*N*K)
    X[:, 0] / X[:, 3],  # M / tile_size (num tiles in M)
    X[:, 1] / X[:, 3],  # N / tile_size (num tiles in N)  
    X[:, 2] / X[:, 3],  # K / tile_size (num tiles in K)
    X[:, 0] == X[:, 1],  # Is square (M == N)
    X[:, 4] / 256.0,  # Thread utilization (threads / max_threads)
])

feature_names = feature_cols + [
    'total_ops', 'tiles_m', 'tiles_n', 'tiles_k', 
    'is_square', 'thread_util'
]

print(f"Engineered features: {len(feature_names)}")
print(feature_names)
print()

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_engineered, y, test_size=0.25, random_state=42
)

print(f"Train set: {len(X_train)} samples")
print(f"Test set:  {len(X_test)} samples")
print()

# Train models
print("Training models...")
print()

models = {}

# Model 1: Random Forest
print("1. Random Forest Regressor")
rf_model = RandomForestRegressor(
    n_estimators=100,
    max_depth=10,
    min_samples_split=2,
    random_state=42,
    n_jobs=-1
)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)
rf_mae = mean_absolute_error(y_test, rf_pred)
rf_r2 = r2_score(y_test, rf_pred)

print(f"  MAE: {rf_mae:.2f} GFLOPS")
print(f"  R²:  {rf_r2:.4f}")

# Cross-validation
rf_cv_scores = cross_val_score(rf_model, X_engineered, y, cv=5, 
                                scoring='neg_mean_absolute_error')
print(f"  CV MAE: {-rf_cv_scores.mean():.2f} ± {rf_cv_scores.std():.2f}")
print()

models['random_forest'] = {
    'model': rf_model,
    'mae': rf_mae,
    'r2': rf_r2,
    'cv_mae': -rf_cv_scores.mean()
}

# Model 2: Gradient Boosting
print("2. Gradient Boosting Regressor")
gb_model = GradientBoostingRegressor(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    random_state=42
)
gb_model.fit(X_train, y_train)
gb_pred = gb_model.predict(X_test)
gb_mae = mean_absolute_error(y_test, gb_pred)
gb_r2 = r2_score(y_test, gb_pred)

print(f"  MAE: {gb_mae:.2f} GFLOPS")
print(f"  R²:  {gb_r2:.4f}")

gb_cv_scores = cross_val_score(gb_model, X_engineered, y, cv=5,
                               scoring='neg_mean_absolute_error')
print(f"  CV MAE: {-gb_cv_scores.mean():.2f} ± {gb_cv_scores.std():.2f}")
print()

models['gradient_boosting'] = {
    'model': gb_model,
    'mae': gb_mae,
    'r2': gb_r2,
    'cv_mae': -gb_cv_scores.mean()
}

# Select best model
best_model_name = min(models.keys(), key=lambda k: models[k]['cv_mae'])
best_model = models[best_model_name]['model']

print("=" * 80)
print(f"BEST MODEL: {best_model_name}")
print(f"  MAE: {models[best_model_name]['mae']:.2f} GFLOPS")
print(f"  R²:  {models[best_model_name]['r2']:.4f}")
print(f"  CV MAE: {models[best_model_name]['cv_mae']:.2f} GFLOPS")
print("=" * 80)
print()

# Feature importances (for Random Forest)
if best_model_name == 'random_forest':
    importances = best_model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    print("Feature Importances:")
    for i in range(min(10, len(feature_names))):
        idx = indices[i]
        print(f"  {i+1}. {feature_names[idx]:15s}: {importances[idx]:.4f}")
    print()

# Save model
model_file = 'neural_predictor_model.pkl'
with open(model_file, 'wb') as f:
    pickle.dump({
        'model': best_model,
        'model_name': best_model_name,
        'feature_names': feature_names,
        'feature_cols': feature_cols,
        'mae': models[best_model_name]['mae'],
        'r2': models[best_model_name]['r2'],
        'metadata': dataset_json['metadata']
    }, f)

print(f"✅ Model saved to: {model_file}")
print()

# Visualization
print("Creating visualizations...")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Plot 1: Actual vs Predicted
y_pred_all = best_model.predict(X_engineered)
axes[0].scatter(y, y_pred_all, alpha=0.6, s=50)
axes[0].plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
axes[0].set_xlabel('Actual GFLOPS', fontsize=12)
axes[0].set_ylabel('Predicted GFLOPS', fontsize=12)
axes[0].set_title(f'{best_model_name.replace("_", " ").title()}\nR² = {models[best_model_name]["r2"]:.4f}', 
                  fontsize=14)
axes[0].grid(True, alpha=0.3)

# Plot 2: Residuals
residuals = y - y_pred_all
axes[1].scatter(y_pred_all, residuals, alpha=0.6, s=50)
axes[1].axhline(y=0, color='r', linestyle='--', lw=2)
axes[1].set_xlabel('Predicted GFLOPS', fontsize=12)
axes[1].set_ylabel('Residual (Actual - Predicted)', fontsize=12)
axes[1].set_title(f'Residual Plot\nMAE = {models[best_model_name]["cv_mae"]:.2f} GFLOPS',
                 fontsize=14)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('neural_predictor_evaluation.png', dpi=100, bbox_inches='tight')
print("✅ Visualization saved to: neural_predictor_evaluation.png")
print()

# Test predictions
print("=" * 80)
print("TEST PREDICTIONS")
print("=" * 80)
print()

test_cases = [
    (512, 512, 512, 16, 256, 0, "tile16_16x16 @ 512"),
    (512, 512, 512, 20, 100, 1, "tile20_10x10_vec @ 512"),
    (1024, 1024, 1024, 16, 256, 0, "tile16_16x16 @ 1024"),
    (1024, 1024, 1024, 20, 100, 1, "tile20_10x10_vec @ 1024"),
    (2048, 2048, 2048, 16, 256, 0, "tile16_16x16 @ 2048"),
    (2048, 2048, 2048, 20, 100, 1, "tile20_10x10_vec @ 2048"),
]

print("Config                        | Predicted | Recommendation")
print("-" * 80)

for M, N, K, tile, threads, vec, name in test_cases:
    # Engineer features
    x = np.array([[M, N, K, tile, threads, vec,
                   M*N*K, M/tile, N/tile, K/tile, M==N, threads/256.0]])
    pred = best_model.predict(x)[0]
    print(f"{name:28s}  | {pred:7.1f}   |")

print()

# Kernel selection test
print("=" * 80)
print("AUTOMATIC KERNEL SELECTION TEST")
print("=" * 80)
print()

print("Size  | tile16  | tile20  | Best Choice | Expected Gain")
print("-" * 80)

for size in [512, 1024, 1280, 1536, 2048]:
    M = N = K = size
    
    # Predict tile16
    x16 = np.array([[M, N, K, 16, 256, 0,
                     M*N*K, M/16, N/16, K/16, True, 1.0]])
    pred16 = best_model.predict(x16)[0]
    
    # Predict tile20
    x20 = np.array([[M, N, K, 20, 100, 1,
                     M*N*K, M/20, N/20, K/20, True, 100/256]])
    pred20 = best_model.predict(x20)[0]
    
    # Select best
    if pred20 > pred16:
        best = "tile20_vec"
        gain = ((pred20 - pred16) / pred16) * 100
    else:
        best = "tile16"
        gain = 0.0
    
    print(f"{size:4d}  | {pred16:6.1f}  | {pred20:6.1f}  | {best:11s} | {gain:+5.1f}%")

print()
print("=" * 80)
print("✅ MODEL TRAINING COMPLETE!")
print("=" * 80)
