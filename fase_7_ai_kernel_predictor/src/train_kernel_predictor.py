#!/usr/bin/env python3
"""
FASE 7: AI KERNEL PREDICTOR - ML MODEL TRAINING
Entrena modelos de ML para predecir el mejor kernel GEMM por características

Fecha: 25 Enero 2026
"""

import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import joblib
import warnings
warnings.filterwarnings('ignore')

class AIKernelPredictor:
    """Predictor de ML para selección automática de kernels GEMM"""

    def __init__(self, project_root=None):
        self.project_root = Path(project_root or Path(__file__).parent.parent.parent)
        self.data_dir = self.project_root / "fase_7_ai_kernel_predictor" / "data"
        self.models_dir = self.project_root / "fase_7_ai_kernel_predictor" / "models"
        self.models_dir.mkdir(exist_ok=True)

        # Modelos disponibles
        self.models = {}
        self.scaler = StandardScaler()

        # Features para predicción
        self.feature_columns = [
            'log_matrix_size', 'optimization_level', 'memory_intensity',
            'compute_intensity', 'kernel_basic', 'kernel_gcn4_optimized',
            'kernel_simd', 'kernel_strassen', 'kernel_winograd'
        ]

    def load_dataset(self):
        """Carga el dataset de benchmarks procesado"""
        # Try multiple possible dataset locations
        possible_paths = [
            self.data_dir / "simple_benchmark_ml_dataset.csv",
            self.data_dir / "complete_benchmark_ml_dataset.csv",
            self.data_dir / "benchmark_ml_dataset.csv"
        ]

        dataset_path = None
        for path in possible_paths:
            if path.exists():
                dataset_path = path
                break

        if dataset_path is None:
            print("❌ No se encontró ningún dataset. Ejecutar primero simple_data_collect.py")
            return None

        df = pd.read_csv(dataset_path)
        print(f"📊 Dataset cargado: {len(df)} registros desde {dataset_path.name}")

        # Mostrar estadísticas básicas
        print("📈 Estadísticas del dataset:")
        print(f"   - Tamaños de matriz: {sorted(df['matrix_size'].unique())}")
        print(f"   - Performance: {df['gflops'].min():.1f} - {df['gflops'].max():.1f} GFLOPS")
        print(f"   - Tipos de kernel: {len([col for col in df.columns if col.startswith('kernel_')])}")

        return df

    def prepare_features_and_target(self, df):
        """Prepara features y target para ML"""
        # Features disponibles
        available_features = [col for col in self.feature_columns if col in df.columns]

        if not available_features:
            print("❌ No hay features disponibles en el dataset")
            return None, None

        X = df[available_features].copy()

        # Target: GFLOPS performance
        y = df['gflops'].copy()

        # Remove rows with invalid targets
        valid_mask = y > 0
        X = X[valid_mask]
        y = y[valid_mask]

        print(f"✅ Features preparados: {X.shape[1]} features, {len(X)} muestras")
        print(f"   Target range: {y.min():.1f} - {y.max():.1f} GFLOPS")

        return X, y

    def train_random_forest(self, X, y):
        """Entrena modelo Random Forest"""
        print("🌲 Entrenando Random Forest...")

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Train model
        rf_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )

        rf_model.fit(X_train_scaled, y_train)

        # Evaluate
        y_pred = rf_model.predict(X_test_scaled)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        print(f"   Random Forest - MAE: {mae:.3f}, R²: {r2:.3f}")
        # Cross-validation
        cv_scores = cross_val_score(rf_model, X_train_scaled, y_train,
                                  cv=5, scoring='r2')
        print(f"   Random Forest - CV R²: {cv_scores.mean():.3f}")
        self.models['random_forest'] = {
            'model': rf_model,
            'scaler': self.scaler,
            'metrics': {'mae': mae, 'r2': r2, 'cv_r2_mean': cv_scores.mean()},
            'features': list(X.columns)
        }

        return rf_model

    def train_xgboost(self, X, y):
        """Entrena modelo XGBoost"""
        print("🚀 Entrenando XGBoost...")

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Train model
        xgb_model = xgb.XGBRegressor(
            objective='reg:squarederror',
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42
        )

        xgb_model.fit(X_train_scaled, y_train)

        # Evaluate
        y_pred = xgb_model.predict(X_test_scaled)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        print(f"   XGBoost - MAE: {mae:.3f}, R²: {r2:.3f}")
        # Cross-validation
        cv_scores = cross_val_score(xgb_model, X_train_scaled, y_train,
                                  cv=5, scoring='r2')
        print(f"   XGBoost - CV R²: {cv_scores.mean():.3f}")
        self.models['xgboost'] = {
            'model': xgb_model,
            'scaler': self.scaler,
            'metrics': {'mae': mae, 'r2': r2, 'cv_r2_mean': cv_scores.mean()},
            'features': list(X.columns)
        }

        return xgb_model

    def train_all_models(self):
        """Entrena todos los modelos disponibles"""
        print("🤖 FASE 7: AI KERNEL PREDICTOR - MODEL TRAINING")
        print("=" * 60)

        # Load dataset
        df = self.load_dataset()
        if df is None:
            return False

        # Prepare features and target
        X, y = self.prepare_features_and_target(df)
        if X is None:
            return False

        # Train models
        self.train_random_forest(X, y)
        self.train_xgboost(X, y)

        # Compare models
        self.compare_models()

        # Save best model
        self.save_best_model()

        print("✅ Model training completado!")
        return True

    def compare_models(self):
        """Compara el rendimiento de todos los modelos"""
        print("\n📊 Comparación de Modelos:")
        print("-" * 40)

        results = []
        for name, model_data in self.models.items():
            metrics = model_data['metrics']
            results.append({
                'Model': name,
                'MAE': metrics['mae'],
                'R²': metrics['r2'],
                'CV R²': metrics['cv_r2_mean']
            })

        results_df = pd.DataFrame(results)
        print(results_df.to_string(index=False, float_format='%.3f'))

        return results_df

    def save_best_model(self):
        """Guarda el mejor modelo basado en R² score"""
        if not self.models:
            return

        # Find best model by CV R²
        best_model_name = max(self.models.keys(),
                            key=lambda x: self.models[x]['metrics']['cv_r2_mean'])

        best_model_data = self.models[best_model_name]

        # Save model
        model_path = self.models_dir / f"kernel_predictor_{best_model_name}.joblib"
        joblib.dump(best_model_data, model_path)

        # Save metadata
        metadata = {
            'best_model': best_model_name,
            'metrics': best_model_data['metrics'],
            'features': best_model_data['features'],
            'training_date': '2026-01-25',
            'phase': '7_ai_kernel_predictor',
            'cv_r2': best_model_data['metrics']['cv_r2_mean'],
            'mae': best_model_data['metrics']['mae']
        }

        metadata_path = self.models_dir / "model_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"💾 Mejor modelo guardado: {best_model_name}")
        print(f"   - R² CV: {best_model_data['metrics']['cv_r2_mean']:.3f}")
        print(f"   - MAE: {best_model_data['metrics']['mae']:.1f} GFLOPS")

    def predict_kernel_performance(self, matrix_size, kernel_type='auto'):
        """Predice el rendimiento para un tamaño de matriz dado"""
        # Load best model if available
        metadata_path = self.models_dir / "model_metadata.json"
        if not metadata_path.exists():
            print("❌ No hay modelo entrenado disponible")
            return None

        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        model_name = metadata['model_name']
        model_path = self.models_dir / f"kernel_predictor_{model_name}.joblib"

        if not model_path.exists():
            print("❌ Modelo no encontrado")
            return None

        model_data = joblib.load(model_path)
        model = model_data['model']
        scaler = model_data['scaler']
        features = model_data['features']

        # Create feature vector
        log_size = np.log2(matrix_size)

        # Create one-hot encoding for kernel type
        kernel_features = {
            'kernel_basic': 1 if kernel_type == 'basic' else 0,
            'kernel_gcn4_optimized': 1 if kernel_type == 'gcn4_optimized' else 0,
            'kernel_simd': 1 if kernel_type == 'simd' else 0,
            'kernel_strassen': 1 if kernel_type == 'strassen' else 0,
            'kernel_winograd': 1 if kernel_type == 'winograd' else 0
        }

        # Default values for missing features
        feature_vector = {
            'log_matrix_size': log_size,
            'optimization_level': 4,  # GCN4 level
            'memory_intensity': matrix_size ** 2 / 1e6,  # Rough estimate
            'compute_intensity': 5.0  # GFLOPS/W estimate
        }

        # Add kernel features
        feature_vector.update(kernel_features)

        # Create DataFrame with correct column order
        feature_df = pd.DataFrame([feature_vector])
        feature_df = feature_df[features]  # Reorder to match training

        # Scale features
        feature_scaled = scaler.transform(feature_df)

        # Predict
        prediction = model.predict(feature_scaled)[0]

        result = {
            'matrix_size': matrix_size,
            'kernel_type': kernel_type,
            'predicted_gflops': round(prediction, 1),
            'model_used': model_name,
            'confidence_metric': metadata['metrics']['cv_r2_mean']
        }

        return result

    def plot_feature_importance(self):
        """Grafica la importancia de las features"""
        if 'random_forest' not in self.models:
            print("❌ Modelo Random Forest no disponible para feature importance")
            return

        model_data = self.models['random_forest']
        model = model_data['model']
        features = model_data['features']

        # Get feature importance
        importance = model.feature_importances_

        # Plot
        plt.figure(figsize=(10, 6))
        plt.barh(features, importance)
        plt.xlabel('Importance')
        plt.ylabel('Features')
        plt.title('Feature Importance - Random Forest')
        plt.tight_layout()

        plot_path = self.models_dir / "feature_importance.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"📊 Feature importance plot saved: {plot_path}")

def main():
    """Función principal para training de modelos"""
    predictor = AIKernelPredictor()

    # Train all models
    success = predictor.train_all_models()

    if success:
        # Plot feature importance
        predictor.plot_feature_importance()

        # Test prediction
        print("\n🧪 Testing prediction...")
        test_sizes = [512, 1024, 2048, 4096]
        for size in test_sizes:
            result = predictor.predict_kernel_performance(size, 'gcn4_optimized')
            if result:
                print(f"   Matrix {size}x{size}: {result['predicted_gflops']} GFLOPS")

        print("\n🎉 AI Kernel Predictor training completado!")
        print("📁 Modelos guardados en: fase_7_ai_kernel_predictor/models/")
    else:
        print("❌ Error en training de modelos")

if __name__ == "__main__":
    main()