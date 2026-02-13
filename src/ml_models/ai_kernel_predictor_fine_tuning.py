#!/usr/bin/env python3
"""
🎯 AI KERNEL PREDICTOR FINE-TUNING
===================================

Script para re-entrenar el AI Kernel Predictor usando el dataset recopilado.
FASE 9.3: Fine-tuning del predictor ML para mejorar accuracy en selección de técnicas.
"""

import sys
import numpy as np
import pandas as pd
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
import matplotlib.pyplot as plt

# import seaborn as sns  # Not available in this environment


class AIKernelPredictorFineTuner:
    """
    Fine-tuner para el AI Kernel Predictor usando dataset recopilado.
    """

    def __init__(self, dataset_path: str = None):
        """
        Inicializa el fine-tuner.

        Args:
            dataset_path: Ruta al dataset de entrenamiento
        """
        self.dataset_path = dataset_path or "ml_training_dataset_*.csv"
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_columns = [
            "matrix_size",
            "matrix_type",
            "technique",
            "sparsity_a",
            "sparsity_b",
            "rank_ratio_a",
            "rank_ratio_b",
            "computational_intensity",
            "memory_usage_mb",
        ]
        self.target_column = "gflops_achieved"

        print("🎯 AI KERNEL PREDICTOR FINE-TUNING")
        print("=" * 50)

    def load_latest_dataset(self) -> pd.DataFrame:
        """Carga el dataset más reciente."""
        import glob

        if "*" in self.dataset_path:
            files = glob.glob(self.dataset_path)
            if not files:
                raise FileNotFoundError(f"No se encontraron archivos: {self.dataset_path}")

            # Tomar el más reciente
            latest_file = max(files, key=lambda x: Path(x).stat().st_mtime)
            print(f"📂 Cargando dataset más reciente: {latest_file}")
        else:
            latest_file = self.dataset_path

        df = pd.read_csv(latest_file)

        print(f"✅ Dataset cargado: {len(df)} registros")
        print(f"   Columnas: {len(df.columns)}")
        print(f"   Técnicas: {df['technique'].unique()}")
        print(f"   Tipos de matriz: {df['matrix_type'].unique()}")

        return df

    def clean_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """Limpia el dataset removiendo outliers y datos inválidos."""
        print("\n🧹 LIMPIANDO DATASET...")

        # Remover filas con valores NaN o infinitos
        initial_count = len(df)
        df = df.replace([np.inf, -np.inf], np.nan).dropna()
        print(f"   Removidos NaN/infinitos: {initial_count - len(df)} filas")

        # Remover outliers usando IQR method (solo para GFLOPS > 0)
        valid_data = df[df["gflops_achieved"] > 0].copy()
        if len(valid_data) > 0:
            Q1 = valid_data["gflops_achieved"].quantile(0.25)
            Q3 = valid_data["gflops_achieved"].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            outliers = valid_data[
                (valid_data["gflops_achieved"] < lower_bound)
                | (valid_data["gflops_achieved"] > upper_bound)
            ]

            if len(outliers) > 0:
                print(f"   Removidos outliers: {len(outliers)} filas")
                print(f"   Rango válido GFLOPS: [{lower_bound:.3f}, {upper_bound:.3f}]")
                df = df[~df.index.isin(outliers.index)]

        # Asegurar que success=True
        success_count = len(df[df["success"] == True])
        if success_count < len(df):
            print(f"   Manteniendo solo ejecuciones exitosas: {success_count}/{len(df)}")
            df = df[df["success"] == True]

        print(f"✅ Dataset limpio: {len(df)} registros")

        return df

    def analyze_dataset(self, df: pd.DataFrame):
        """Analiza el dataset recopilado."""
        print("\n📊 ANÁLISIS DEL DATASET")
        print("=" * 30)

        # Estadísticas generales
        print(f"Total de registros: {len(df)}")
        print(f"Técnicas evaluadas: {df['technique'].nunique()}")
        print(f"Tipos de matriz: {df['matrix_type'].nunique()}")
        print(f"Tamaños de matriz: {sorted(df['matrix_size'].unique())}")

        # Performance por técnica
        print("\n🎯 PERFORMANCE POR TÉCNICA:")
        technique_stats = (
            df.groupby("technique")
            .agg(
                {
                    "gflops_achieved": ["mean", "max", "min", "std", "count"],
                    "relative_error": ["mean", "max"],
                    "execution_time": ["mean", "max"],
                }
            )
            .round(3)
        )

        for technique in technique_stats.index:
            stats = technique_stats.loc[technique]
            print(f"   {technique}:")
            print(f"     GFLOPS promedio: {stats['gflops_achieved']['mean']:.3f}")
            print(f"     GFLOPS máximo: {stats['gflops_achieved']['max']:.3f}")
            print(f"     Tasa de éxito: {stats['gflops_achieved']['count']:.1f}%")
        # Performance por tipo de matriz
        print("\n🏷️  PERFORMANCE POR TIPO DE MATRIZ:")
        matrix_stats = (
            df.groupby(["matrix_type", "technique"])["gflops_achieved"]
            .agg(["mean", "max"])
            .round(3)
        )
        for (matrix_type, technique), stats in matrix_stats.iterrows():
            print(
                f"   {matrix_type}_{technique}: {stats['mean']:.3f} GFLOPS (max: {stats['max']:.3f})"
            )

        # Correlaciones
        numeric_cols = [
            "matrix_size",
            "gflops_achieved",
            "relative_error",
            "sparsity_a",
            "sparsity_b",
            "rank_ratio_a",
            "rank_ratio_b",
            "computational_intensity",
        ]
        correlations = df[numeric_cols].corr()["gflops_achieved"].sort_values(ascending=False)
        print("\n📈 CORRELACIONES CON GFLOPS:")
        for feature, corr in correlations.items():
            if feature != "gflops_achieved":
                print(f"   {feature}: {corr:.3f}")

    def prepare_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Prepara features y target para entrenamiento."""
        print("\n🔧 PREPARANDO FEATURES PARA ENTRENAMIENTO...")

        # Codificar variables categóricas
        df_processed = df.copy()

        for col in ["matrix_type", "technique"]:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
            df_processed[col] = self.label_encoders[col].fit_transform(df_processed[col])

        # Seleccionar features y target
        X = df_processed[self.feature_columns]
        y = df_processed[self.target_column]

        # Escalar features
        X_scaled = self.scaler.fit_transform(X)

        print(f"   Features: {X.shape[1]} dimensiones")
        print(f"   Registros: {X.shape[0]}")
        print(f"   Target range: [{y.min():.3f}, {y.max():.3f}] GFLOPS")

        return X_scaled, y.values

    def train_model(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Entrena el modelo de ML."""
        print("\n🤖 ENTRENANDO MODELO...")

        # Dividir datos
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Probar diferentes modelos
        models = {
            "RandomForest": RandomForestRegressor(
                n_estimators=100, max_depth=10, random_state=42, n_jobs=-1
            ),
            "GradientBoosting": GradientBoostingRegressor(
                n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42
            ),
        }

        best_model = None
        best_score = -np.inf
        best_metrics = {}

        for name, model in models.items():
            print(f"\n   Probando {name}...")

            # Cross-validation
            cv_scores = cross_val_score(model, X_train, y_train, cv=3, scoring="r2", n_jobs=-1)
            cv_r2 = cv_scores.mean()

            # Entrenar modelo completo
            model.fit(X_train, y_train)

            # Predecir en test set
            y_pred = model.predict(X_test)

            # Métricas
            r2 = r2_score(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(np.mean((y_test - y_pred) ** 2))

            print(f"   CV R²: {cv_r2:.3f}")
            print(f"   Test R²: {r2:.3f}")
            print(f"   Test MAE: {mae:.3f} GFLOPS")
            if r2 > best_score:
                best_score = r2
                best_model = model
                best_metrics = {
                    "model_name": name,
                    "cv_r2": cv_r2,
                    "test_r2": r2,
                    "test_mae": mae,
                    "test_rmse": rmse,
                    "y_test": y_test,
                    "y_pred": y_pred,
                }

        self.model = best_model
        print(f"\n🏆 MEJOR MODELO: {best_metrics['model_name']}")
        print(f"   Mejor R²: {best_score:.3f}")
        return best_metrics

    def save_model(self, filename: str = "ai_kernel_predictor_finetuned.pkl"):
        """Guarda el modelo entrenado."""
        if self.model is None:
            raise ValueError("No hay modelo entrenado para guardar")

        model_data = {
            "model": self.model,
            "scaler": self.scaler,
            "label_encoders": self.label_encoders,
            "feature_columns": self.feature_columns,
            "target_column": self.target_column,
            "timestamp": pd.Timestamp.now(),
        }

        joblib.dump(model_data, filename)
        print(f"💾 Modelo guardado: {filename}")

    def create_performance_report(self, df: pd.DataFrame, metrics: Dict[str, Any]):
        """Crea reporte de performance."""
        print("\n📋 REPORTE DE PERFORMANCE")
        print("=" * 25)

        # Resumen del dataset
        print(f"Dataset final: {len(df)} registros")
        print(f"Features utilizadas: {len(self.feature_columns)}")
        print(f"Modelo: {metrics['model_name']}")

        # Métricas de accuracy
        print("\n🎯 MÉTRICAS DE ACCURACY:")
        print(f"   R² Score: {metrics['test_r2']:.3f}")
        print(f"   MAE: {metrics['test_mae']:.3f} GFLOPS")
        print(f"   RMSE: {metrics['test_rmse']:.3f} GFLOPS")
        # Análisis por técnica
        print("\n🏷️  PREDICCIÓN POR TÉCNICA:")
        technique_performance = (
            df.groupby("technique").agg({"gflops_achieved": ["mean", "std", "count"]}).round(3)
        )

        for technique in technique_performance.index:
            stats = technique_performance.loc[technique]
            print(
                f"   {technique}: {stats['gflops_achieved']['mean']:.3f} ± {stats['gflops_achieved']['std']:.3f} GFLOPS ({int(stats['gflops_achieved']['count'])} muestras)"
            )

        # Próximos pasos
        print("\n🎯 PRÓXIMOS PASOS:")
        print("   1. Integrar modelo fine-tuned en Breakthrough Selector")
        print("   2. Validar mejoras en benchmark integrado")
        print("   3. Ejecutar FASE 9.4: Optimización híbrida avanzada")
        print("   4. Apuntar a superar 890.3+ GFLOPS objetivo")

    def run_fine_tuning(self):
        """Ejecuta el proceso completo de fine-tuning."""
        try:
            # Cargar datos
            df = self.load_latest_dataset()

            # Limpiar datos
            df_clean = self.clean_dataset(df)

            # Analizar datos
            self.analyze_dataset(df_clean)

            # Preparar features
            X, y = self.prepare_features(df_clean)

            # Entrenar modelo
            metrics = self.train_model(X, y)

            # Guardar modelo
            self.save_model()

            # Crear reporte
            self.create_performance_report(df_clean, metrics)

            print("\n✅ FINE-TUNING COMPLETADO EXITOSAMENTE")
            print("🎯 El AI Kernel Predictor está listo para mejorar la selección de técnicas!")

            return True

        except Exception as e:
            print(f"❌ Error en fine-tuning: {e}")
            import traceback

            traceback.print_exc()
            return False


def main():
    """Función principal."""
    tuner = AIKernelPredictorFineTuner()
    success = tuner.run_fine_tuning()

    if success:
        print("\n🚀 FASE 9.3 COMPLETADA: AI Kernel Predictor fine-tuned")
        print("   Próximo: Integrar en sistema y validar mejoras")
    else:
        print("\n❌ FASE 9.3 FALLIDA: Revisar errores y reintentar")
        sys.exit(1)


if __name__ == "__main__":
    main()
