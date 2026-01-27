#!/usr/bin/env python3
"""
📊 GENERADOR RÁPIDO DE DATASET DE ENTRENAMIENTO
==============================================

Versión optimizada que genera un dataset útil rápidamente.
"""

import numpy as np
import pandas as pd
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

def generate_training_dataset(output_path: Path, n_samples: int = 500) -> pd.DataFrame:
    """Genera un dataset de entrenamiento optimizado"""
    print(f"🚀 Generando dataset de {n_samples} muestras...")

    np.random.seed(42)
    data = []

    techniques = ['low_rank', 'cw', 'ai_predictor', 'tensor_core', 'quantum', 'neuromorphic']
    sizes = [256, 512, 1024]

    for i in range(n_samples):
        # Características de la matriz
        size = np.random.choice(sizes)
        sparsity = np.random.beta(2, 5)  # Más probable matrices densas
        condition = 10 ** np.random.uniform(0, 8)  # Condition number variable

        # Técnica óptima (simulada basada en características)
        if sparsity > 0.7:
            optimal_tech = 'neuromorphic' if np.random.random() > 0.3 else 'low_rank'
        elif condition > 1e6:
            optimal_tech = 'cw' if np.random.random() > 0.4 else 'tensor_core'
        elif size > 512:
            optimal_tech = 'tensor_core' if np.random.random() > 0.3 else 'cw'
        else:
            optimal_tech = np.random.choice(['ai_predictor', 'low_rank', 'cw'])

        # Performance de la técnica óptima
        base_perf = {
            'low_rank': 120, 'cw': 180, 'ai_predictor': 150,
            'tensor_core': 200, 'quantum': 50, 'neuromorphic': 80
        }

        perf = base_perf[optimal_tech]
        perf *= np.random.normal(1.0, 0.1)  # Variabilidad

        # Ajustes por características
        if size > 512:
            perf *= 1.3
        if sparsity > 0.5:
            perf *= 0.9

        data.append({
            'matrix_size': size,
            'sparsity': sparsity,
            'condition_number': condition,
            'optimal_technique': optimal_tech,
            'best_performance': perf,
            'memory_footprint_mb': size * size * 4 * 2 / (1024**2),  # Aproximado
            'structure_type': 'sparse' if sparsity > 0.5 else 'dense'
        })

    df = pd.DataFrame(data)

    # Guardar
    output_path.parent.mkdir(exist_ok=True)
    df.to_csv(output_path, index=False)

    # Metadata
    metadata = {
        'creation_timestamp': time.time(),
        'n_samples': len(df),
        'techniques': techniques,
        'sizes': sizes,
        'optimal_distribution': df['optimal_technique'].value_counts().to_dict()
    }

    json_path = output_path.with_suffix('.json')
    with open(json_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"✅ Dataset generado: {len(df)} muestras")
    print(f"   Distribución: {metadata['optimal_distribution']}")

    return df

def main():
    """Función principal"""
    project_root = Path(__file__).parent.parent
    data_dir = project_root / "data"

    # Generar dataset rápido
    dataset = generate_training_dataset(data_dir / "training_dataset.csv", n_samples=500)

    print("\n📊 Dataset creado exitosamente")
    print(f"   Archivo: {data_dir / 'training_dataset.csv'}")
    print(f"   Muestras: {len(dataset)}")

    return dataset

if __name__ == "__main__":
    main()