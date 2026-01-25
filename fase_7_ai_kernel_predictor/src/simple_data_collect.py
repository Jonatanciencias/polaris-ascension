#!/usr/bin/env python3
"""
FASE 7: AI KERNEL PREDICTOR - SIMPLE DATA COLLECTION
Script simplificado para procesar archivos de benchmark históricos

Fecha: 25 Enero 2026
"""

import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
import re

def collect_simple_benchmark_data():
    """Recopila datos de benchmark de manera simplificada"""
    print("🚀 FASE 7: Simple Benchmark Data Collection")
    print("=" * 50)

    project_root = Path(__file__).parent.parent.parent
    data_dir = project_root / "fase_7_ai_kernel_predictor" / "data"
    data_dir.mkdir(exist_ok=True)

    # Buscar archivos de benchmark
    benchmark_files = list(project_root.glob("**/*benchmark*.json"))
    print(f"📁 Encontrados {len(benchmark_files)} archivos de benchmark")

    all_records = []

    for file_path in benchmark_files:
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)

            # Procesar archivos con formato nested (results -> size -> kernel)
            if 'results' in data:
                records = process_nested_benchmark(data, file_path)
                all_records.extend(records)

        except Exception as e:
            print(f"⚠️  Error procesando {file_path.name}: {e}")
            continue

    print(f"📊 Recopilados {len(all_records)} registros válidos")

    if all_records:
        # Crear DataFrame
        df = pd.DataFrame(all_records)

        # Crear features para ML
        df = create_ml_features(df)

        # Guardar dataset
        output_path = data_dir / "simple_benchmark_ml_dataset.csv"
        df.to_csv(output_path, index=False)

        print(f"💾 Dataset guardado: {output_path}")
        print(f"📈 Registros: {len(df)}")

        # Mostrar resumen
        print("\n📊 Resumen del Dataset:")
        print(f"   - Tamaños de matriz únicos: {sorted(df['matrix_size'].unique())}")
        print(f"   - Performance range: {df['gflops'].min():.1f} - {df['gflops'].max():.1f} GFLOPS")
        print(f"   - Tipos de kernel: {df['kernel_type'].value_counts().to_dict()}")

        return df

    return None

def process_nested_benchmark(data, file_path):
    """Procesa archivos con formato nested benchmark_info/results"""
    records = []

    results = data.get('results', {})
    benchmark_info = data.get('benchmark_info', {})

    for size_key, size_data in results.items():
        if isinstance(size_data, dict):
            for kernel_key, kernel_data in size_data.items():
                if isinstance(kernel_data, dict) and 'gflops' in kernel_data:
                    # Extraer tamaño de matriz
                    size_match = re.search(r'(\d+)x\1x\1', size_key)
                    matrix_size = int(size_match.group(1)) if size_match else 0

                    if matrix_size > 0:
                        record = {
                            'source_file': file_path.name,
                            'matrix_size': matrix_size,
                            'gflops': kernel_data.get('gflops', 0),
                            'kernel_type': map_kernel_type_simple(kernel_key),
                            'optimization_level': infer_opt_level_simple(benchmark_info),
                            'execution_time': kernel_data.get('avg_time_ms', 0) / 1000,  # ms to s
                            'timestamp': benchmark_info.get('timestamp', 'unknown')
                        }

                        if record['gflops'] > 0:
                            records.append(record)

    return records

def map_kernel_type_simple(kernel_key):
    """Mapeo simple de tipos de kernel"""
    key = kernel_key.lower()

    if 'gcn4' in key or 'optimized' in key:
        return 'gcn4_optimized'
    elif 'simd' in key:
        return 'simd'
    elif 'basic' in key:
        return 'basic'
    elif 'strassen' in key:
        return 'strassen'
    elif 'winograd' in key:
        return 'winograd'
    else:
        return 'unknown'

def infer_opt_level_simple(benchmark_info):
    """Inferir nivel de optimización de manera simple"""
    phase = str(benchmark_info.get('phase', '')).lower()

    if 'phase 5' in phase:
        return 5
    elif 'phase 4' in phase:
        return 4
    elif 'phase 3' in phase:
        return 3
    elif 'phase 2' in phase:
        return 2
    else:
        return 4  # Default para benchmarks optimizados

def create_ml_features(df):
    """Crear features para ML training"""
    # Logarithmic matrix size
    df['log_matrix_size'] = np.log2(df['matrix_size'])

    # Memory intensity (rough estimate)
    df['memory_intensity'] = df['matrix_size'] ** 2 / df['execution_time'] if df['execution_time'].gt(0).any() else 0

    # Compute intensity (GFLOPS/W, assuming 150W default)
    df['compute_intensity'] = df['gflops'] / 150

    # One-hot encoding for kernel types
    kernel_dummies = pd.get_dummies(df['kernel_type'], prefix='kernel')
    df = pd.concat([df, kernel_dummies], axis=1)

    # Optimization level as numeric
    df['optimization_level'] = df['optimization_level'].astype(int)

    return df

if __name__ == "__main__":
    dataset = collect_simple_benchmark_data()

    if dataset is not None:
        print("\n✅ Data collection completada exitosamente!")
        print("🎯 Listo para training de ML models")
    else:
        print("\n❌ No se pudo crear el dataset")