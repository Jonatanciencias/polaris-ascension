#!/usr/bin/env python3
"""
FASE 7: AI KERNEL PREDICTOR - COMPLETE DATA COLLECTION
Procesa todos los archivos de benchmark históricos para crear dataset completo de ML

Fecha: 25 Enero 2026
"""

import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
import glob
import re

class CompleteBenchmarkDataCollector:
    """Recopila y procesa todos los datos históricos de benchmarks para ML training completo"""

    def __init__(self, project_root=None):
        self.project_root = Path(project_root or Path(__file__).parent.parent.parent)
        self.data_dir = self.project_root / "fase_7_ai_kernel_predictor" / "data"
        self.data_dir.mkdir(exist_ok=True)

    def find_all_benchmark_files(self):
        """Encuentra todos los archivos de benchmark en el proyecto"""
        patterns = [
            "**/*benchmark*.json",
            "**/*results*.json",
            "**/*performance*.json",
            "**/benchmark_results/**/*.json",
            "**/*_benchmark_*.json"
        ]

        benchmark_files = []
        for pattern in patterns:
            files = list(self.project_root.glob(pattern))
            benchmark_files.extend(files)

        # Remove duplicates and sort
        benchmark_files = list(set(benchmark_files))
        benchmark_files.sort()

        return benchmark_files

    def parse_benchmark_file_complete(self, file_path):
        """Parsea un archivo de benchmark con manejo completo de formatos"""
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)

            records = []

            # Handle different benchmark formats
            if isinstance(data, list):
                for item in data:
                    record = self.extract_complete_record(item, file_path)
                    if record:
                        records.append(record)
            elif isinstance(data, dict):
                # Check if it's a benchmark result
                if self.is_benchmark_result(data):
                    record = self.extract_complete_record(data, file_path)
                    if record:
                        records.append(record)
                else:
                    # Try to extract multiple results from dict
                    for key, value in data.items():
                        if isinstance(value, (list, dict)) and self.is_benchmark_result(value):
                            record = self.extract_complete_record(value, file_path)
                            if record:
                                records.append(record)

            return records

        except Exception as e:
            # Silently skip problematic files
            return []

    def is_benchmark_result(self, data):
        """Verifica si un objeto contiene resultados de benchmark"""
        if not isinstance(data, dict):
            return False

        # Check for common benchmark fields
        benchmark_indicators = [
            'gflops', 'performance', 'time', 'matrix_size', 'size',
            'bandwidth', 'utilization', 'power'
        ]

        return any(key in data for key in benchmark_indicators)

    def extract_complete_record(self, data, file_path):
        """Extrae métricas completas de un registro de benchmark"""
        try:
            record = {
                'source_file': str(file_path.name),
                'timestamp': data.get('timestamp', 'unknown'),
                'matrix_size': self.extract_matrix_size(data),
                'gflops': self.extract_performance(data),
                'kernel_type': self.infer_kernel_type_complete(file_path, data),
                'optimization_level': self.infer_optimization_level_complete(file_path, data),
                'hardware_utilization': data.get('utilization', 0),
                'memory_bandwidth': data.get('bandwidth', 0),
                'power_consumption': data.get('power', 150),  # Default 150W
                'execution_time': data.get('time', 0),
                'error': data.get('error', None),
                'success': data.get('success', True)
            }

            # Skip invalid records
            if record['gflops'] <= 0 or record['matrix_size'] <= 0:
                return None

            # Add derived metrics
            record.update(self.calculate_derived_metrics(record))

            return record

        except Exception as e:
            return None

    def extract_matrix_size(self, data):
        """Extrae el tamaño de matriz de diferentes formatos"""
        # Try different field names
        size_fields = ['matrix_size', 'size', 'N', 'M', 'K', 'dim']

        for field in size_fields:
            if field in data:
                size = data[field]
                if isinstance(size, (int, float)) and size > 0:
                    return int(size)

        # Try to extract from filename
        return 0

    def extract_performance(self, data):
        """Extrae la performance en GFLOPS"""
        perf_fields = ['gflops', 'performance', 'GFLOPS', 'gflop']

        for field in perf_fields:
            if field in data:
                perf = data[field]
                if isinstance(perf, (int, float)) and perf > 0:
                    return float(perf)

        # Try to calculate from time and size if available
        if 'time' in data and 'matrix_size' in data:
            time_sec = data['time']
            size = data['matrix_size']
            if time_sec > 0 and size > 0:
                # Rough estimate: 2 * size^3 FLOPS for matrix multiply
                estimated_flops = 2 * (size ** 3)
                return estimated_flops / (time_sec * 1e9)  # Convert to GFLOPS

        return 0

    def infer_kernel_type_complete(self, file_path, data):
        """Infiera el tipo de kernel con mayor precisión"""
        path_str = str(file_path).lower()
        filename = file_path.name.lower()

        # Check filename patterns
        if 'winograd' in filename:
            return 'winograd'
        elif 'strassen' in filename:
            return 'strassen'
        elif 'gcn4' in filename or 'deep' in filename:
            return 'gcn4_optimized'
        elif 'simd' in filename or 'vector' in filename:
            return 'simd'
        elif 'basic' in filename or 'naive' in filename:
            return 'basic'

        # Check path patterns
        if 'winograd' in path_str:
            return 'winograd'
        elif 'strassen' in path_str:
            return 'strassen'
        elif 'gcn4' in path_str or 'phase_5' in path_str:
            return 'gcn4_optimized'
        elif 'phase_4' in path_str or 'vector' in path_str:
            return 'simd'
        elif 'phase_1' in path_str or 'phase_2' in path_str:
            return 'basic'

        # Check data content
        if isinstance(data, dict):
            if 'optimization' in data:
                opt = str(data['optimization']).lower()
                if 'simd' in opt or 'vector' in opt:
                    return 'simd'
                elif 'gcn4' in opt:
                    return 'gcn4_optimized'

        return 'unknown'

    def infer_optimization_level_complete(self, file_path, data):
        """Infiera el nivel de optimización con mayor precisión"""
        path_str = str(file_path).lower()
        filename = file_path.name.lower()

        # Phase-based detection
        if 'phase_5' in path_str or 'deep' in filename or 'final' in filename:
            return 5
        elif 'phase_4' in path_str or 'gcn4' in filename:
            return 4
        elif 'phase_3' in path_str or 'vector' in filename:
            return 3
        elif 'phase_2' in path_str or 'memory' in filename:
            return 2
        elif 'phase_1' in path_str or 'basic' in filename:
            return 1

        # Optimization keywords
        if 'optimized' in filename or 'refined' in filename:
            return 4
        elif 'simd' in filename or 'vector' in filename:
            return 3

        return 1  # Default

    def calculate_derived_metrics(self, record):
        """Calcula métricas derivadas para ML"""
        derived = {}

        size = record['matrix_size']
        gflops = record['gflops']
        time = record['execution_time']
        power = record['power_consumption']

        # Logarithmic matrix size for better scaling
        derived['log_matrix_size'] = np.log2(size) if size > 0 else 0

        # Memory intensity (bytes per second)
        if time > 0:
            derived['memory_intensity'] = (size ** 2 * 4 * 3) / time  # Rough estimate
        else:
            derived['memory_intensity'] = 0

        # Compute intensity (GFLOPS per Watt)
        if power > 0:
            derived['compute_intensity'] = gflops / power
        else:
            derived['compute_intensity'] = gflops / 150  # Default power

        # Performance efficiency (percentage of theoretical peak)
        theoretical_peak = 6170  # RX 580 theoretical GFLOPS
        derived['efficiency_percent'] = (gflops / theoretical_peak) * 100

        return derived

    def collect_complete_dataset(self):
        """Recopila el dataset completo de benchmarks"""
        print("🔍 Buscando todos los archivos de benchmark...")

        benchmark_files = self.find_all_benchmark_files()
        print(f"📁 Encontrados {len(benchmark_files)} archivos de benchmark")

        all_records = []
        processed_count = 0

        for file_path in benchmark_files:
            records = self.parse_benchmark_file_complete(file_path)
            if records:
                all_records.extend(records)
                processed_count += 1

            # Progress indicator
            if processed_count % 5 == 0:
                print(f"   Procesados: {processed_count}/{len(benchmark_files)} archivos")

        print(f"📊 Recopilados {len(all_records)} registros de benchmark válidos")

        return all_records

    def create_ml_dataset_complete(self, records):
        """Crea dataset completo listo para ML training"""
        if not records:
            print("❌ No hay registros para procesar")
            return None

        df = pd.DataFrame(records)

        # Filter valid records
        df = df.dropna(subset=['gflops', 'matrix_size'])
        df = df[df['gflops'] > 0]
        df = df[df['matrix_size'] > 0]
        df = df[df['success'] != False]  # Remove failed benchmarks

        print(f"✅ Registros válidos después de filtrado: {len(df)}")

        # Add kernel type one-hot encoding
        kernel_dummies = pd.get_dummies(df['kernel_type'], prefix='kernel')
        df = pd.concat([df, kernel_dummies], axis=1)

        # Select features for ML
        base_features = [
            'log_matrix_size', 'optimization_level', 'memory_intensity',
            'compute_intensity', 'hardware_utilization', 'efficiency_percent'
        ]

        # Add kernel type features
        kernel_features = [col for col in df.columns if col.startswith('kernel_')]
        feature_columns = base_features + kernel_features

        # Ensure all features exist
        available_features = [col for col in feature_columns if col in df.columns]

        target_column = 'gflops'

        # Create final dataset
        ml_dataset = df[available_features + [target_column]].copy()

        # Fill missing values
        ml_dataset = ml_dataset.fillna(0)

        print(f"🎯 Dataset final: {len(ml_dataset)} muestras, {len(available_features)} features")

        return ml_dataset

    def save_complete_dataset(self, dataset, filename="complete_benchmark_ml_dataset.csv"):
        """Guarda el dataset completo procesado"""
        output_path = self.data_dir / filename
        dataset.to_csv(output_path, index=False)
        print(f"💾 Dataset completo guardado: {output_path}")
        return output_path

    def generate_complete_summary(self, dataset):
        """Genera resumen completo del dataset"""
        summary = {
            'total_records': len(dataset),
            'matrix_sizes': {
                'unique': sorted(dataset['log_matrix_size'].unique()),
                'range': {
                    'min': 2 ** dataset['log_matrix_size'].min(),
                    'max': 2 ** dataset['log_matrix_size'].max()
                }
            },
            'kernel_types': {
                name: int(count) for name, count in
                dataset.filter(like='kernel_').sum().items()
            },
            'performance_stats': {
                'min_gflops': float(dataset['gflops'].min()),
                'max_gflops': float(dataset['gflops'].max()),
                'mean_gflops': float(dataset['gflops'].mean()),
                'median_gflops': float(dataset['gflops'].median()),
                'std_gflops': float(dataset['gflops'].std())
            },
            'optimization_levels': sorted(dataset['optimization_level'].unique()),
            'collection_date': '2026-01-25',
            'phase': '7_ai_kernel_predictor'
        }

        summary_path = self.data_dir / "complete_dataset_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)

        print("📊 Resumen completo del dataset:")
        print(f"   - Registros totales: {summary['total_records']}")
        perf = summary['performance_stats']
        print(f"   - Performance: {perf['min_gflops']:.1f} - {perf['max_gflops']:.1f} GFLOPS")
        print(f"   - Tipos de kernel: {len(summary['kernel_types'])}")
        print(f"   - Niveles de optimización: {summary['optimization_levels']}")

        return summary

def main():
    """Función principal para data collection completa"""
    print("🚀 FASE 7: AI KERNEL PREDICTOR - COMPLETE DATA COLLECTION")
    print("=" * 70)

    collector = CompleteBenchmarkDataCollector()

    # Collect complete dataset
    records = collector.collect_complete_dataset()

    if records:
        # Create ML-ready dataset
        dataset = collector.create_ml_dataset_complete(records)

        if dataset is not None and len(dataset) > 0:
            # Save complete dataset
            collector.save_complete_dataset(dataset)

            # Generate complete summary
            collector.generate_complete_summary(dataset)

            print("✅ Complete data collection exitosa!")
            print(f"📈 Dataset completo listo para ML training: {len(dataset)} registros")
            print("📁 Ubicación: fase_7_ai_kernel_predictor/data/complete_benchmark_ml_dataset.csv")
        else:
            print("❌ Error creando dataset completo")
    else:
        print("❌ No se encontraron datos de benchmark")

if __name__ == "__main__":
    main()