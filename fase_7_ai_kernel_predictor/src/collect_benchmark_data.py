#!/usr/bin/env python3
"""
FASE 7: AI KERNEL PREDICTOR - DATA COLLECTION SCRIPT
Recopila datos históricos de benchmarks para training del ML predictor

Fecha: 25 Enero 2026
"""

import os
import json
import pandas as pd
import numpy as np
from pathlib import Path

def main():
    """Función principal para data collection"""
    print("🚀 FASE 7: AI KERNEL PREDICTOR - DATA COLLECTION")
    print("=" * 60)

    # Simple test - find benchmark files
    project_root = Path(__file__).parent.parent.parent
    print(f"Project root: {project_root}")

    # Find JSON files
    json_files = list(project_root.glob("**/*.json"))
    print(f"Found {len(json_files)} JSON files")

    benchmark_count = 0
    for file_path in json_files:
        if 'benchmark' in file_path.name.lower() or 'result' in file_path.name.lower():
            print(f"  📊 {file_path.name}")
            benchmark_count += 1

    print(f"\n📈 Total benchmark files found: {benchmark_count}")

    # Create sample dataset for testing
    sample_data = {
        'matrix_size': [512, 1024, 2048, 512, 1024, 2048],
        'kernel_type': ['basic', 'simd', 'gcn4', 'winograd', 'gcn4', 'winograd'],
        'gflops': [60.0, 285.0, 691.5, 890.3, 855.6, 1023.8],
        'optimization_level': [1, 3, 4, 5, 5, 6]
    }

    df = pd.DataFrame(sample_data)
    df['log_matrix_size'] = np.log2(df['matrix_size'])

    # Save sample dataset
    data_dir = project_root / "fase_7_ai_kernel_predictor" / "data"
    data_dir.mkdir(exist_ok=True)
    output_path = data_dir / "sample_ml_dataset.csv"
    df.to_csv(output_path, index=False)

    print(f"💾 Sample dataset saved: {output_path}")
    print("✅ Data collection preparation completada!")

if __name__ == "__main__":
    main()