#!/usr/bin/env python3
"""
Script simple para recalibrar el selector inteligente con datos de hardware
"""

import json
import pandas as pd
from pathlib import Path

def recalibrate_selector_with_hardware_data(benchmark_data_path: Path):
    """Recalibrar el selector inteligente con datos reales del hardware"""
    print("🔧 RECALIBRANDO SELECTOR CON DATOS DE HARDWARE")
    print("=" * 50)

    # Cargar datos de benchmark
    with open(benchmark_data_path, 'r') as f:
        data = json.load(f)

    # Convertir a DataFrame para análisis
    results_df = pd.DataFrame(data['benchmark_results'])

    # Filtrar datos válidos
    valid_results = results_df[results_df['execution_time'].notna() & (results_df['execution_time'] != 'inf')]

    if len(valid_results) == 0:
        print("❌ No hay datos válidos para recalibración")
        return

    print(f"📊 Datos válidos para recalibración: {len(valid_results)}")

    # Análisis de performance por técnica
    technique_performance = valid_results.groupby('technique').agg({
        'gflops': ['mean', 'std', 'count'],
        'execution_time': ['mean', 'std']
    }).round(3)

    print("\n📈 Performance por técnica:")
    print(technique_performance)

    # Encontrar la mejor técnica para cada tipo de matriz
    matrix_best_techniques = {}
    for matrix_name in valid_results['matrix_name'].unique():
        matrix_data = valid_results[valid_results['matrix_name'] == matrix_name]
        if len(matrix_data) > 0:
            best_technique = matrix_data.loc[matrix_data['gflops'].idxmax(), 'technique']
            matrix_best_techniques[matrix_name] = best_technique

    print("\n🎯 Mejores técnicas por tipo de matriz:")
    for matrix, technique in matrix_best_techniques.items():
        print(f"   {matrix}: {technique}")

    # Generar nuevos pesos basados en datos reales
    new_weights = {}

    for technique in valid_results['technique'].unique():
        tech_data = valid_results[valid_results['technique'] == technique]

        # Calcular score basado en performance relativa
        avg_gflops = tech_data['gflops'].mean()
        max_gflops = valid_results['gflops'].max()

        if max_gflops > 0:
            relative_score = avg_gflops / max_gflops
            new_weights[technique] = max(0.1, relative_score)  # Mínimo 0.1
        else:
            new_weights[technique] = 0.5  # Valor por defecto

    # Normalizar pesos
    total_weight = sum(new_weights.values())
    if total_weight > 0:
        new_weights = {k: v/total_weight for k, v in new_weights.items()}

    print("\n⚖️  Nuevos pesos calculados:")
    for technique, weight in new_weights.items():
        print(".3f")

    # Guardar nuevos pesos
    # Convertir las claves de tuplas a strings para JSON
    perf_dict = {}
    for technique in technique_performance.index:
        perf_dict[technique] = {}
        for col in technique_performance.columns:
            perf_dict[technique][str(col)] = technique_performance.loc[technique, col]

    weights_data = {
        'hardware': data['hardware_profile']['gpu_name'],
        'calibration_date': data['benchmark_results'][0]['timestamp'] if data['benchmark_results'] else None,
        'new_weights': new_weights,
        'technique_performance': perf_dict,
        'data_points': len(valid_results)
    }

    weights_path = Path("models/hardware_calibrated_weights.json")
    with open(weights_path, 'w') as f:
        json.dump(weights_data, f, indent=2, default=str)

    print(f"\n💾 Nuevos pesos guardados en: {weights_path}")

    return new_weights

if __name__ == "__main__":
    benchmark_path = Path("benchmark_data/hardware_benchmark_results.json")
    if benchmark_path.exists():
        recalibrate_selector_with_hardware_data(benchmark_path)
        print("\n🎉 RECALIBRACIÓN COMPLETADA")
        print("   El selector inteligente ahora está optimizado para RX580")
    else:
        print(f"❌ Archivo de benchmark no encontrado: {benchmark_path}")