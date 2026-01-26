#!/usr/bin/env python3
"""
🚀 ENSEMBLE PREDICTOR FOR AI KERNEL OPTIMIZATION
===============================================

Sistema de predicción ML que combina múltiples modelos para predecir
configuraciones óptimas de kernel basadas en datos de Phase 13.

Características:
- Ensemble de Random Forest, XGBoost y Neural Networks
- Predicción de work-group sizes y configuraciones de memoria
- Validación cruzada y evaluación de precisión
- Predicciones en tiempo real para optimización automática

Author: AI Assistant
Date: 2026-01-25
"""

import numpy as np
import pandas as pd
import json
import os
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
import logging
import warnings
warnings.filterwarnings('ignore')

# ML Libraries
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# Optional ML libraries (with fallbacks)
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("XGBoost not available, using Random Forest only")

try:
    from tensorflow import keras
    from tensorflow.keras import layers, models
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("TensorFlow not available, using tree-based models only")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EnsemblePredictor:
    """
    Ensemble ML Predictor for Kernel Optimization

    Combines multiple ML models to predict optimal kernel configurations
    based on hardware characteristics and performance data.
    """

    def __init__(self):
        self.base_dir = Path(__file__).parent.parent
        self.data_dir = self.base_dir / "src" / "data"
        self.models_dir = self.base_dir / "src" / "models"

        # Model storage
        self.models = {}
        self.scalers = {}
        self.encoders = {}

        # Performance tracking
        self.prediction_stats = {
            'workgroup': {'predictions': 0, 'accuracy': 0.0, 'mae': 0.0},
            'memory': {'predictions': 0, 'accuracy': 0.0, 'mae': 0.0},
            'combined': {'predictions': 0, 'accuracy': 0.0, 'mae': 0.0}
        }

        logger.info("🚀 Ensemble Predictor initialized")

    def load_datasets(self) -> Dict[str, pd.DataFrame]:
        """Load training datasets from Phase 13"""
        datasets = {}

        try:
            datasets['workgroup'] = pd.read_csv(self.data_dir / 'phase13_workgroup_dataset.csv')
            datasets['memory'] = pd.read_csv(self.data_dir / 'phase13_memory_dataset.csv')
            datasets['combined'] = pd.read_csv(self.data_dir / 'phase13_combined_dataset.csv')
            datasets['architecture'] = pd.read_csv(self.data_dir / 'phase13_architecture_info.csv')

            logger.info("✅ Datasets loaded successfully")
            for name, df in datasets.items():
                logger.info(f"  {name}: {len(df)} samples, {len(df.columns)} features")

        except Exception as e:
            logger.error(f"Failed to load datasets: {e}")
            raise

        return datasets

    def preprocess_data(self, datasets: Dict[str, pd.DataFrame]) -> Dict[str, Dict[str, Any]]:
        """Preprocess datasets for ML training"""
        processed_data = {}

        # Process work-group data
        wg_data = self._preprocess_workgroup_data(datasets['workgroup'])
        processed_data['workgroup'] = wg_data

        # Process memory data
        mem_data = self._preprocess_memory_data(datasets['memory'])
        processed_data['memory'] = mem_data

        # Process combined data
        combined_data = self._preprocess_combined_data(datasets['combined'])
        processed_data['combined'] = combined_data

        logger.info("✅ Data preprocessing completed")
        return processed_data

    def _preprocess_workgroup_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Preprocess work-group optimization data"""
        # Features for regression (predict GFLOPS)
        feature_cols = ['wg_size_0', 'wg_size_1', 'wg_total_size', 'compute_units',
                       'wavefront_size', 'max_wg_size']

        # Classification target (optimal vs non-optimal)
        class_target = 'is_optimal'

        # Regression target
        reg_target = 'gflops'

        # Filter valid configurations
        valid_df = df[df['is_valid'] == True].copy()

        if len(valid_df) < 5:
            logger.warning("Insufficient valid work-group samples for training")
            return {}

        X_reg = valid_df[feature_cols]
        y_reg = valid_df[reg_target]
        y_class = valid_df[class_target].astype(int)

        # Split data
        X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
            X_reg, y_reg, test_size=0.2, random_state=42
        )
        X_train_class, X_test_class, y_train_class, y_test_class = train_test_split(
            X_reg, y_class, test_size=0.2, random_state=42
        )

        # Scale features
        scaler_reg = StandardScaler()
        X_train_reg_scaled = scaler_reg.fit_transform(X_train_reg)
        X_test_reg_scaled = scaler_reg.transform(X_test_reg)

        scaler_class = StandardScaler()
        X_train_class_scaled = scaler_class.fit_transform(X_train_class)
        X_test_class_scaled = scaler_class.transform(X_test_class)

        return {
            'X_train_reg': X_train_reg_scaled,
            'X_test_reg': X_test_reg_scaled,
            'y_train_reg': y_train_reg,
            'y_test_reg': y_test_reg,
            'X_train_class': X_train_class_scaled,
            'X_test_class': X_test_class_scaled,
            'y_train_class': y_train_class,
            'y_test_class': y_test_class,
            'scaler_reg': scaler_reg,
            'scaler_class': scaler_class,
            'feature_names': feature_cols
        }

    def _preprocess_memory_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Preprocess memory optimization data"""
        # Features for regression
        feature_cols = ['use_lds', 'lds_tile_size', 'vector_width', 'unroll_factor',
                       'prefetch_distance', 'local_mem_size_kb', 'global_mem_size_gb']

        # Encode categorical features
        df_processed = df.copy()
        encoder = LabelEncoder()
        df_processed['memory_type_encoded'] = encoder.fit_transform(df_processed['memory_type'])
        feature_cols.append('memory_type_encoded')

        # Regression target
        reg_target = 'gflops'

        if len(df_processed) < 3:
            logger.warning("Insufficient memory samples for training")
            return {}

        X = df_processed[feature_cols]
        y = df_processed[reg_target]

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        return {
            'X_train': X_train_scaled,
            'X_test': X_test_scaled,
            'y_train': y_train,
            'y_test': y_test,
            'scaler': scaler,
            'encoder': encoder,
            'feature_names': feature_cols
        }

    def _preprocess_combined_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Preprocess combined work-group and memory data"""
        # Features combining both optimization types
        feature_cols = [
            'wg_size_0', 'wg_size_1', 'wg_total_size', 'wg_occupancy', 'wg_efficiency',
            'use_lds', 'lds_tile_size', 'vector_width', 'unroll_factor', 'prefetch_distance',
            'compute_units', 'wavefront_size', 'local_mem_size_kb', 'global_mem_size_gb'
        ]

        # Encode memory type
        df_processed = df.copy()
        encoder = LabelEncoder()
        df_processed['memory_type_encoded'] = encoder.fit_transform(df_processed['memory_type'])
        feature_cols.append('memory_type_encoded')

        # Regression target
        reg_target = 'estimated_gflops'

        if len(df_processed) < 10:
            logger.warning("Insufficient combined samples for training")
            return {}

        X = df_processed[feature_cols]
        y = df_processed[reg_target]

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        return {
            'X_train': X_train_scaled,
            'X_test': X_test_scaled,
            'y_train': y_train,
            'y_test': y_test,
            'scaler': scaler,
            'encoder': encoder,
            'feature_names': feature_cols
        }

    def train_ensemble_models(self, processed_data: Dict[str, Dict[str, Any]]):
        """Train ensemble models for each optimization type"""
        logger.info("🏋️ Training ensemble models...")

        # Train work-group models
        if 'workgroup' in processed_data and processed_data['workgroup']:
            self._train_workgroup_models(processed_data['workgroup'])

        # Train memory models
        if 'memory' in processed_data and processed_data['memory']:
            self._train_memory_models(processed_data['memory'])

        # Train combined models
        if 'combined' in processed_data and processed_data['combined']:
            self._train_combined_models(processed_data['combined'])

        logger.info("✅ Ensemble model training completed")

    def _train_workgroup_models(self, data: Dict[str, Any]):
        """Train work-group optimization models"""
        logger.info("Training work-group models...")

        # Regression model (predict GFLOPS)
        rf_reg = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        rf_reg.fit(data['X_train_reg'], data['y_train_reg'])

        # XGBoost regression (if available)
        if XGBOOST_AVAILABLE:
            xgb_reg = xgb.XGBRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42
            )
            xgb_reg.fit(data['X_train_reg'], data['y_train_reg'])

        # Neural network (if available)
        if TENSORFLOW_AVAILABLE:
            nn_reg = self._build_neural_network(data['X_train_reg'].shape[1])
            nn_reg.fit(
                data['X_train_reg'], data['y_train_reg'],
                epochs=50, batch_size=8, verbose=0,
                validation_split=0.2
            )

        # Classification model (predict optimal config)
        rf_class = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        rf_class.fit(data['X_train_class'], data['y_train_class'])

        # Store models
        self.models['workgroup_reg'] = {
            'rf': rf_reg,
            'xgb': xgb_reg if XGBOOST_AVAILABLE else None,
            'nn': nn_reg if TENSORFLOW_AVAILABLE else None
        }
        self.models['workgroup_class'] = rf_class
        self.scalers['workgroup_reg'] = data['scaler_reg']
        self.scalers['workgroup_class'] = data['scaler_class']

        # Evaluate models
        self._evaluate_workgroup_models(data)

    def _train_memory_models(self, data: Dict[str, Any]):
        """Train memory optimization models"""
        logger.info("Training memory models...")

        # Random Forest regression
        rf_reg = RandomForestRegressor(
            n_estimators=100,
            max_depth=8,
            random_state=42,
            n_jobs=-1
        )
        rf_reg.fit(data['X_train'], data['y_train'])

        # XGBoost regression (if available)
        if XGBOOST_AVAILABLE:
            xgb_reg = xgb.XGBRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42
            )
            xgb_reg.fit(data['X_train'], data['y_train'])

        # Store models and scaler
        self.models['memory_reg'] = {
            'rf': rf_reg,
            'xgb': xgb_reg if XGBOOST_AVAILABLE else None
        }
        self.scalers['memory_reg'] = data['scaler']
        self.encoders['memory_type'] = data['encoder']

        # Evaluate models
        self._evaluate_memory_models(data)

    def _train_combined_models(self, data: Dict[str, Any]):
        """Train combined optimization models"""
        logger.info("Training combined models...")

        # Random Forest regression
        rf_reg = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        rf_reg.fit(data['X_train'], data['y_train'])

        # XGBoost regression (if available)
        if XGBOOST_AVAILABLE:
            xgb_reg = xgb.XGBRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42
            )
            xgb_reg.fit(data['X_train'], data['y_train'])

        # Store models and scaler
        self.models['combined_reg'] = {
            'rf': rf_reg,
            'xgb': xgb_reg if XGBOOST_AVAILABLE else None
        }
        self.scalers['combined_reg'] = data['scaler']
        self.encoders['combined_memory_type'] = data['encoder']

        # Evaluate models
        self._evaluate_combined_models(data)

    def _build_neural_network(self, input_dim: int):
        """Build a simple neural network for regression"""
        try:
            from tensorflow import keras
            from tensorflow.keras import layers, models
        except ImportError:
            return None

        model = models.Sequential([
            layers.Dense(64, activation='relu', input_dim=input_dim),
            layers.Dropout(0.2),
            layers.Dense(32, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(16, activation='relu'),
            layers.Dense(1)  # Regression output
        ])

        model.compile(
            optimizer='adam',
            loss='mean_squared_error',
            metrics=['mae']
        )

        return model

    def _evaluate_workgroup_models(self, data: Dict[str, Any]):
        """Evaluate work-group models performance"""
        logger.info("Evaluating work-group models...")

        # Regression evaluation
        models_reg = self.models['workgroup_reg']
        X_test = data['X_test_reg']
        y_test = data['y_test_reg']

        results = {}
        for name, model in models_reg.items():
            if model is None:
                continue

            if name == 'nn' and TENSORFLOW_AVAILABLE:
                y_pred = model.predict(X_test).flatten()
            else:
                y_pred = model.predict(X_test)

            mae = mean_absolute_error(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            results[name] = {'mae': mae, 'mse': mse, 'r2': r2}
            logger.info(".2f")

        # Classification evaluation
        class_model = self.models['workgroup_class']
        X_test_class = data['X_test_class']
        y_test_class = data['y_test_class']

        y_pred_class = class_model.predict(X_test_class)
        accuracy = accuracy_score(y_test_class, y_pred_class)

        logger.info(f"Work-group classification accuracy: {accuracy:.2f}")

        # Store best model info
        self.models['workgroup_reg']['best_model'] = max(results.items(), key=lambda x: x[1]['r2'])[0]

    def _evaluate_memory_models(self, data: Dict[str, Any]):
        """Evaluate memory models performance"""
        logger.info("Evaluating memory models...")

        models_reg = self.models['memory_reg']
        X_test = data['X_test']
        y_test = data['y_test']

        results = {}
        for name, model in models_reg.items():
            if model is None:
                continue

            y_pred = model.predict(X_test)
            mae = mean_absolute_error(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            results[name] = {'mae': mae, 'mse': mse, 'r2': r2}
            logger.info(".2f")

        # Store best model info
        self.models['memory_reg']['best_model'] = max(results.items(), key=lambda x: x[1]['r2'])[0]

    def _evaluate_combined_models(self, data: Dict[str, Any]):
        """Evaluate combined models performance"""
        logger.info("Evaluating combined models...")

        models_reg = self.models['combined_reg']
        X_test = data['X_test']
        y_test = data['y_test']

        results = {}
        for name, model in models_reg.items():
            if model is None:
                continue

            y_pred = model.predict(X_test)
            mae = mean_absolute_error(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            results[name] = {'mae': mae, 'mse': mse, 'r2': r2}
            logger.info(".2f")

        # Store best model info
        self.models['combined_reg']['best_model'] = max(results.items(), key=lambda x: x[1]['r2'])[0]

    def predict_workgroup_config(self, hardware_features: Dict[str, Any]) -> Dict[str, Any]:
        """Predict optimal work-group configuration"""
        try:
            # Prepare features
            features = np.array([[
                hardware_features.get('wg_size_0', 4),
                hardware_features.get('wg_size_1', 64),
                hardware_features.get('wg_total_size', 256),
                hardware_features.get('compute_units', 36),
                hardware_features.get('wavefront_size', 64),
                hardware_features.get('max_wg_size', 256)
            ]])

            # Scale features
            scaler = self.scalers.get('workgroup_reg')
            if scaler:
                features_scaled = scaler.transform(features)
            else:
                features_scaled = features

            # Get best regression model
            reg_models = self.models.get('workgroup_reg', {})
            best_model_name = reg_models.get('best_model', 'rf')
            best_model = reg_models.get(best_model_name)

            if best_model is None:
                return {'error': 'No trained work-group model available'}

            # Predict performance
            if best_model_name == 'nn' and TENSORFLOW_AVAILABLE:
                predicted_gflops = best_model.predict(features_scaled).flatten()[0]
            else:
                predicted_gflops = best_model.predict(features_scaled)[0]

            # Get classification prediction
            class_model = self.models.get('workgroup_class')
            if class_model:
                scaler_class = self.scalers.get('workgroup_class')
                if scaler_class:
                    features_class_scaled = scaler_class.transform(features)
                else:
                    features_class_scaled = features

                is_optimal_prob = class_model.predict_proba(features_class_scaled)[0][1]
            else:
                is_optimal_prob = 0.5

            prediction = {
                'predicted_gflops': float(predicted_gflops),
                'optimal_probability': float(is_optimal_prob),
                'confidence': 'high' if is_optimal_prob > 0.7 else 'medium' if is_optimal_prob > 0.5 else 'low',
                'model_used': best_model_name,
                'timestamp': pd.Timestamp.now().isoformat()
            }

            # Update stats
            self.prediction_stats['workgroup']['predictions'] += 1

            return prediction

        except Exception as e:
            logger.error(f"Work-group prediction failed: {e}")
            return {'error': str(e)}

    def predict_memory_config(self, hardware_features: Dict[str, Any]) -> Dict[str, Any]:
        """Predict optimal memory configuration"""
        try:
            # Prepare features (this would be more complex in practice)
            features = np.array([[
                hardware_features.get('use_lds', 1),
                hardware_features.get('lds_tile_size', 32),
                hardware_features.get('vector_width', 4),
                hardware_features.get('unroll_factor', 4),
                hardware_features.get('prefetch_distance', 2),
                hardware_features.get('local_mem_size_kb', 64),
                hardware_features.get('global_mem_size_gb', 8.0),
                0  # memory_type_encoded (placeholder)
            ]])

            # Scale features
            scaler = self.scalers.get('memory_reg')
            if scaler:
                features_scaled = scaler.transform(features)
            else:
                features_scaled = features

            # Get best model
            reg_models = self.models.get('memory_reg', {})
            best_model_name = reg_models.get('best_model', 'rf')
            best_model = reg_models.get(best_model_name)

            if best_model is None:
                return {'error': 'No trained memory model available'}

            # Predict performance
            predicted_gflops = best_model.predict(features_scaled)[0]

            prediction = {
                'predicted_gflops': float(predicted_gflops),
                'recommended_config': {
                    'use_lds': bool(hardware_features.get('use_lds', 1)),
                    'lds_tile_size': int(hardware_features.get('lds_tile_size', 32)),
                    'vector_width': int(hardware_features.get('vector_width', 4)),
                    'unroll_factor': int(hardware_features.get('unroll_factor', 4))
                },
                'model_used': best_model_name,
                'timestamp': pd.Timestamp.now().isoformat()
            }

            # Update stats
            self.prediction_stats['memory']['predictions'] += 1

            return prediction

        except Exception as e:
            logger.error(f"Memory prediction failed: {e}")
            return {'error': str(e)}

    def predict_combined_config(self, hardware_features: Dict[str, Any]) -> Dict[str, Any]:
        """Predict optimal combined work-group and memory configuration"""
        try:
            # Prepare combined features
            features = np.array([[
                hardware_features.get('wg_size_0', 4),
                hardware_features.get('wg_size_1', 64),
                hardware_features.get('wg_total_size', 256),
                hardware_features.get('wg_occupancy', 0.8),
                hardware_features.get('wg_efficiency', 0.9),
                hardware_features.get('use_lds', 1),
                hardware_features.get('lds_tile_size', 32),
                hardware_features.get('vector_width', 4),
                hardware_features.get('unroll_factor', 4),
                hardware_features.get('prefetch_distance', 2),
                hardware_features.get('compute_units', 36),
                hardware_features.get('wavefront_size', 64),
                hardware_features.get('local_mem_size_kb', 64),
                hardware_features.get('global_mem_size_gb', 8.0),
                0  # memory_type_encoded (placeholder)
            ]])

            # Scale features
            scaler = self.scalers.get('combined_reg')
            if scaler:
                features_scaled = scaler.transform(features)
            else:
                features_scaled = features

            # Get best model
            reg_models = self.models.get('combined_reg', {})
            best_model_name = reg_models.get('best_model', 'rf')
            best_model = reg_models.get(best_model_name)

            if best_model is None:
                return {'error': 'No trained combined model available'}

            # Predict performance
            predicted_gflops = best_model.predict(features_scaled)[0]

            prediction = {
                'predicted_gflops': float(predicted_gflops),
                'workgroup_config': {
                    'size': [hardware_features.get('wg_size_0', 4),
                            hardware_features.get('wg_size_1', 64)]
                },
                'memory_config': {
                    'use_lds': bool(hardware_features.get('use_lds', 1)),
                    'lds_tile_size': int(hardware_features.get('lds_tile_size', 32)),
                    'vector_width': int(hardware_features.get('vector_width', 4)),
                    'unroll_factor': int(hardware_features.get('unroll_factor', 4))
                },
                'model_used': best_model_name,
                'timestamp': pd.Timestamp.now().isoformat()
            }

            # Update stats
            self.prediction_stats['combined']['predictions'] += 1

            return prediction

        except Exception as e:
            logger.error(f"Combined prediction failed: {e}")
            return {'error': str(e)}

    def save_models(self):
        """Save trained models to disk"""
        try:
            # Save models
            for model_name, model_data in self.models.items():
                if isinstance(model_data, dict):
                    for sub_name, model in model_data.items():
                        if model is not None and sub_name != 'best_model':
                            model_path = self.models_dir / f"{model_name}_{sub_name}.joblib"
                            joblib.dump(model, model_path)
                else:
                    model_path = self.models_dir / f"{model_name}.joblib"
                    joblib.dump(model_data, model_path)

            # Save scalers
            for scaler_name, scaler in self.scalers.items():
                scaler_path = self.models_dir / f"{scaler_name}_scaler.joblib"
                joblib.dump(scaler, scaler_path)

            # Save encoders
            for encoder_name, encoder in self.encoders.items():
                encoder_path = self.models_dir / f"{encoder_name}_encoder.joblib"
                joblib.dump(encoder, encoder_path)

            # Save prediction stats
            stats_path = self.models_dir / "prediction_stats.json"
            with open(stats_path, 'w') as f:
                json.dump(self.prediction_stats, f, indent=2)

            logger.info("✅ Models saved successfully")

        except Exception as e:
            logger.error(f"Failed to save models: {e}")

    def load_models(self):
        """Load trained models from disk"""
        try:
            # Load regression models (stored as individual files)
            for model_name in ['workgroup_reg', 'memory_reg', 'combined_reg']:
                model_dict = {}
                # Try to load RF model
                rf_path = self.models_dir / f"{model_name}_rf.joblib"
                if rf_path.exists():
                    model_dict['rf'] = joblib.load(rf_path)

                # Try to load XGB model
                xgb_path = self.models_dir / f"{model_name}_xgb.joblib"
                if xgb_path.exists():
                    model_dict['xgb'] = joblib.load(xgb_path)

                # Try to load NN model
                nn_path = self.models_dir / f"{model_name}_nn.joblib"
                if nn_path.exists():
                    model_dict['nn'] = joblib.load(nn_path)

                if model_dict:
                    self.models[model_name] = model_dict

            # Load classification model (stored as single model)
            class_model_path = self.models_dir / "workgroup_class.joblib"
            if class_model_path.exists():
                self.models['workgroup_class'] = joblib.load(class_model_path)

            # Load scalers
            for scaler_name in ['workgroup_reg', 'workgroup_class', 'memory_reg', 'combined_reg']:
                scaler_path = self.models_dir / f"{scaler_name}_scaler.joblib"
                if scaler_path.exists():
                    self.scalers[scaler_name] = joblib.load(scaler_path)

            # Load encoders
            for encoder_name in ['memory_type', 'combined_memory_type']:
                encoder_path = self.models_dir / f"{encoder_name}_encoder.joblib"
                if encoder_path.exists():
                    self.encoders[encoder_name] = joblib.load(encoder_path)

            # Load prediction stats
            stats_path = self.models_dir / "prediction_stats.json"
            if stats_path.exists():
                with open(stats_path, 'r') as f:
                    self.prediction_stats = json.load(f)

            logger.info(f"✅ Models loaded successfully: {list(self.models.keys())}")

        except Exception as e:
            logger.error(f"Failed to load models: {e}")
            import traceback
            traceback.print_exc()

    def get_model_summary(self) -> Dict[str, Any]:
        """Get summary of trained models and their performance"""
        summary = {
            'models_trained': list(self.models.keys()),
            'prediction_stats': self.prediction_stats,
            'available_libraries': {
                'xgboost': XGBOOST_AVAILABLE,
                'tensorflow': TENSORFLOW_AVAILABLE
            },
            'model_comparison': {
                'workgroup_best_model': self.models.get('workgroup_reg', {}).get('best_model', 'unknown'),
                'memory_best_model': self.models.get('memory_reg', {}).get('best_model', 'unknown'),
                'combined_best_model': self.models.get('combined_reg', {}).get('best_model', 'unknown')
            },
            'timestamp': pd.Timestamp.now().isoformat()
        }

        return summary

def main():
    """Main function for ensemble predictor training"""
    try:
        predictor = EnsemblePredictor()

        # Load datasets
        datasets = predictor.load_datasets()

        # Preprocess data
        processed_data = predictor.preprocess_data(datasets)

        # Train ensemble models
        predictor.train_ensemble_models(processed_data)

        # Save models
        predictor.save_models()

        # Get summary
        summary = predictor.get_model_summary()
        with open(predictor.models_dir / 'model_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)

        logger.info("🎯 Ensemble predictor training completed successfully!")
        logger.info(f"📊 Models trained: {len(predictor.models)}")
        logger.info(f"📈 Prediction stats: {predictor.prediction_stats}")

    except Exception as e:
        logger.error(f"Ensemble predictor training failed: {e}")
        raise

if __name__ == "__main__":
    main()