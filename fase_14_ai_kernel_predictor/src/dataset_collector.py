#!/usr/bin/env python3
"""
🚀 DATASET COLLECTOR FOR AI KERNEL PREDICTOR
===========================================

Recopila y procesa datos de optimización para entrenar el sistema ML
que predice configuraciones óptimas de kernel.

Incorpora resultados de Phase 13 (GCN Architecture Tuning) con:
- Work-group optimization results (8 configurations)
- Memory access optimization results (5 configurations)
- Hardware specifications y performance metrics

Author: AI Assistant
Date: 2026-01-25
"""

import numpy as np
import pandas as pd
import json
import os
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DatasetCollector:
    """
    Dataset Collector for AI Kernel Predictor

    Collects optimization results from Phase 13 and integrates them
    with existing ML system data for training improved predictors.
    """

    def __init__(self):
        self.base_dir = Path(__file__).parent.parent
        self.data_dir = self.base_dir / "src" / "data"
        self.models_dir = self.base_dir / "src" / "models"

        # Create directories
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)

        # Phase 13 results paths
        self.phase13_workgroup_path = Path("../../fase_13_gcn_architecture/src/results/workgroup_optimization_results.json")
        self.phase13_memory_path = Path("../../fase_13_gcn_architecture/src/results/memory_optimization_results.json")
        self.phase13_arch_path = Path("../../fase_13_gcn_architecture/src/results/gcn_architecture_analysis.json")

        logger.info("🚀 Dataset Collector initialized")

    def collect_phase13_data(self) -> Dict[str, pd.DataFrame]:
        """
        Collect all Phase 13 optimization results

        Returns:
            Dictionary with workgroup, memory, and combined datasets
        """
        logger.info("📊 Collecting Phase 13 optimization data...")

        # Collect work-group data
        workgroup_data = self._collect_workgroup_data()

        # Collect memory optimization data
        memory_data = self._collect_memory_data()

        # Collect architecture info
        arch_info = self._collect_architecture_info()

        # Combine datasets
        combined_data = self._combine_datasets(workgroup_data, memory_data, arch_info)

        datasets = {
            'workgroup': workgroup_data,
            'memory': memory_data,
            'combined': combined_data,
            'architecture': arch_info
        }

        logger.info(f"✅ Collected {len(workgroup_data)} work-group samples, "
                   f"{len(memory_data)} memory samples, "
                   f"{len(combined_data)} combined samples")

        return datasets

    def _collect_workgroup_data(self) -> pd.DataFrame:
        """Collect work-group optimization results from Phase 13"""
        try:
            with open(self.phase13_workgroup_path, 'r') as f:
                data = json.load(f)

            records = []
            for config in data['tested_configs']:
                record = {
                    # Features
                    'wg_size_0': config['size'][0],
                    'wg_size_1': config['size'][1],
                    'wg_total_size': config['size'][0] * config['size'][1],

                    # Hardware context (from architecture analysis)
                    'compute_units': 36,  # Polaris 10
                    'wavefront_size': 64,
                    'max_wg_size': 256,

                    # Performance metrics
                    'gflops': config['gflops'],
                    'kernel_time_ms': config['kernel_time_ms'],
                    'occupancy': config['occupancy'],
                    'efficiency': config['efficiency'],

                    # Labels
                    'is_valid': config['valid'],
                    'performance_rank': 0,  # Will be calculated
                    'is_optimal': False,  # Will be set

                    # Metadata
                    'optimization_type': 'workgroup',
                    'matrix_size': 1024,  # Test size used
                    'timestamp': data['timestamp']
                }
                records.append(record)

            df = pd.DataFrame(records)

            # Calculate performance ranks
            df = df.sort_values('gflops', ascending=False)
            df['performance_rank'] = range(1, len(df) + 1)

            # Mark optimal configuration
            optimal_size = data['optimal_config']['size']
            df['is_optimal'] = (df['wg_size_0'] == optimal_size[0]) & (df['wg_size_1'] == optimal_size[1])

            logger.info(f"✅ Collected {len(df)} work-group optimization records")
            return df

        except Exception as e:
            logger.error(f"Failed to collect work-group data: {e}")
            return pd.DataFrame()

    def _collect_memory_data(self) -> pd.DataFrame:
        """Collect memory optimization results from Phase 13"""
        try:
            with open(self.phase13_memory_path, 'r') as f:
                data = json.load(f)

            records = []
            for config_data in data['tested_configs']:
                config = config_data['config']
                perf = config_data['performance']

                record = {
                    # Memory configuration features
                    'memory_type': config['kernel_type'],
                    'use_lds': config['use_lds'],
                    'lds_tile_size': config['lds_tile_size'],
                    'vector_width': config['vector_width'],
                    'unroll_factor': config['unroll_factor'],
                    'prefetch_distance': config.get('prefetch_distance', 0),

                    # Hardware context
                    'local_mem_size_kb': 64,
                    'global_mem_size_gb': 8.0,
                    'memory_channels': 4,

                    # Performance metrics
                    'gflops': perf['gflops'],
                    'memory_bandwidth_gbs': perf['memory_bandwidth_gbs'],
                    'kernel_time_ms': perf['kernel_time_ms'],
                    'coalescing_efficiency': perf['coalescing_efficiency'],
                    'cache_hit_rate': perf['cache_hit_rate'],
                    'lds_utilization': perf['lds_utilization'],
                    'bandwidth_efficiency': perf['bandwidth_efficiency'],

                    # Labels
                    'performance_rank': 0,  # Will be calculated
                    'is_optimal': False,  # Will be set

                    # Metadata
                    'optimization_type': 'memory',
                    'matrix_size': 1024,
                    'timestamp': data['timestamp']
                }
                records.append(record)

            df = pd.DataFrame(records)

            # Calculate performance ranks
            df = df.sort_values('gflops', ascending=False)
            df['performance_rank'] = range(1, len(df) + 1)

            # Mark optimal configuration
            optimal_type = data['optimal_config']['kernel_type']
            df['is_optimal'] = (df['memory_type'] == optimal_type)

            logger.info(f"✅ Collected {len(df)} memory optimization records")
            return df

        except Exception as e:
            logger.error(f"Failed to collect memory data: {e}")
            return pd.DataFrame()

    def _collect_architecture_info(self) -> pd.DataFrame:
        """Collect architecture information for context features"""
        try:
            with open(self.phase13_arch_path, 'r') as f:
                data = json.load(f)

            arch = data['architecture_info']
            perf = data['performance_metrics']

            record = {
                # Hardware specifications
                'device_name': arch['device_name'],
                'compute_units': arch['compute_units'],
                'wavefront_size': arch['wavefront_size'],
                'max_work_group_size': arch['max_work_group_size'],
                'max_work_item_sizes': arch['max_work_item_sizes'],
                'local_mem_size_kb': arch['local_mem_size_kb'],
                'global_mem_size_gb': arch['global_mem_size_gb'],
                'max_clock_frequency_mhz': arch['max_clock_frequency_mhz'],
                'opencl_version': arch['opencl_version'],

                # Baseline performance
                'baseline_gflops': perf['gflops_baseline'],
                'baseline_memory_bandwidth': perf['memory_bandwidth_gbs'],
                'baseline_compute_utilization': perf['compute_utilization'],
                'baseline_memory_utilization': perf['memory_utilization'],

                # Metadata
                'timestamp': data['timestamp']
            }

            df = pd.DataFrame([record])
            logger.info("✅ Collected architecture information")
            return df

        except Exception as e:
            logger.error(f"Failed to collect architecture info: {e}")
            return pd.DataFrame()

    def _combine_datasets(self, workgroup_df: pd.DataFrame,
                         memory_df: pd.DataFrame,
                         arch_df: pd.DataFrame) -> pd.DataFrame:
        """Combine work-group and memory datasets for joint prediction"""
        try:
            # Create combined records by pairing work-group and memory configs
            combined_records = []

            # Get architecture info
            arch_info = arch_df.iloc[0].to_dict() if not arch_df.empty else {}

            for _, wg_row in workgroup_df.iterrows():
                for _, mem_row in memory_df.iterrows():
                    # Skip invalid combinations
                    if not wg_row['is_valid']:
                        continue

                    record = {
                        # Work-group features
                        'wg_size_0': wg_row['wg_size_0'],
                        'wg_size_1': wg_row['wg_size_1'],
                        'wg_total_size': wg_row['wg_total_size'],
                        'wg_occupancy': wg_row['occupancy'],
                        'wg_efficiency': wg_row['efficiency'],

                        # Memory features
                        'memory_type': mem_row['memory_type'],
                        'use_lds': mem_row['use_lds'],
                        'lds_tile_size': mem_row['lds_tile_size'],
                        'vector_width': mem_row['vector_width'],
                        'unroll_factor': mem_row['unroll_factor'],
                        'prefetch_distance': mem_row['prefetch_distance'],

                        # Hardware context
                        'compute_units': arch_info.get('compute_units', 36),
                        'wavefront_size': arch_info.get('wavefront_size', 64),
                        'local_mem_size_kb': arch_info.get('local_mem_size_kb', 64),
                        'global_mem_size_gb': arch_info.get('global_mem_size_gb', 8.0),

                        # Combined performance (estimated)
                        'estimated_gflops': (wg_row['gflops'] + mem_row['gflops']) * 0.8,  # Rough estimate
                        'wg_contribution': wg_row['gflops'],
                        'mem_contribution': mem_row['gflops'],

                        # Labels
                        'is_wg_optimal': wg_row['is_optimal'],
                        'is_mem_optimal': mem_row['is_optimal'],
                        'is_combined_optimal': wg_row['is_optimal'] and mem_row['is_optimal'],

                        # Metadata
                        'optimization_type': 'combined',
                        'matrix_size': 1024,
                        'timestamp': wg_row['timestamp']
                    }
                    combined_records.append(record)

            df = pd.DataFrame(combined_records)

            # Calculate combined performance ranks
            df = df.sort_values('estimated_gflops', ascending=False)
            df['combined_rank'] = range(1, len(df) + 1)

            logger.info(f"✅ Created {len(df)} combined configuration records")
            return df

        except Exception as e:
            logger.error(f"Failed to combine datasets: {e}")
            return pd.DataFrame()

    def save_datasets(self, datasets: Dict[str, pd.DataFrame]):
        """Save collected datasets to CSV files"""
        try:
            # Save individual datasets
            datasets['workgroup'].to_csv(self.data_dir / 'phase13_workgroup_dataset.csv', index=False)
            datasets['memory'].to_csv(self.data_dir / 'phase13_memory_dataset.csv', index=False)
            datasets['combined'].to_csv(self.data_dir / 'phase13_combined_dataset.csv', index=False)
            datasets['architecture'].to_csv(self.data_dir / 'phase13_architecture_info.csv', index=False)

            # Create summary statistics
            summary = self._create_dataset_summary(datasets)
            with open(self.data_dir / 'dataset_summary.json', 'w') as f:
                json.dump(summary, f, indent=2)

            logger.info("✅ Datasets saved to CSV files")

        except Exception as e:
            logger.error(f"Failed to save datasets: {e}")
            raise

    def _create_dataset_summary(self, datasets: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Create summary statistics of collected datasets"""
        summary = {
            'collection_timestamp': pd.Timestamp.now().isoformat(),
            'datasets': {}
        }

        for name, df in datasets.items():
            if df.empty:
                continue

            summary['datasets'][name] = {
                'num_samples': len(df),
                'num_features': len(df.columns),
                'feature_names': df.columns.tolist(),
                'performance_range': {
                    'min': df.get('gflops', df.get('estimated_gflops', pd.Series())).min(),
                    'max': df.get('gflops', df.get('estimated_gflops', pd.Series())).max(),
                    'mean': df.get('gflops', df.get('estimated_gflops', pd.Series())).mean()
                } if 'gflops' in df.columns or 'estimated_gflops' in df.columns else None
            }

        return summary

    def validate_datasets(self, datasets: Dict[str, pd.DataFrame]) -> Dict[str, bool]:
        """Validate dataset integrity and completeness"""
        validation_results = {}

        # Check work-group dataset
        wg_df = datasets.get('workgroup', pd.DataFrame())
        validation_results['workgroup'] = self._validate_workgroup_dataset(wg_df)

        # Check memory dataset
        mem_df = datasets.get('memory', pd.DataFrame())
        validation_results['memory'] = self._validate_memory_dataset(mem_df)

        # Check combined dataset
        combined_df = datasets.get('combined', pd.DataFrame())
        validation_results['combined'] = self._validate_combined_dataset(combined_df)

        # Overall validation
        validation_results['overall'] = all(validation_results.values())

        logger.info(f"✅ Dataset validation: {sum(validation_results.values())}/{len(validation_results)} passed")
        return validation_results

    def _validate_workgroup_dataset(self, df: pd.DataFrame) -> bool:
        """Validate work-group dataset"""
        if df.empty:
            return False

        required_columns = ['wg_size_0', 'wg_size_1', 'gflops', 'is_valid']
        return all(col in df.columns for col in required_columns)

    def _validate_memory_dataset(self, df: pd.DataFrame) -> bool:
        """Validate memory dataset"""
        if df.empty:
            return False

        required_columns = ['memory_type', 'gflops', 'memory_bandwidth_gbs']
        return all(col in df.columns for col in required_columns)

    def _validate_combined_dataset(self, df: pd.DataFrame) -> bool:
        """Validate combined dataset"""
        if df.empty:
            return False

        required_columns = ['wg_size_0', 'memory_type', 'estimated_gflops']
        return all(col in df.columns for col in required_columns)

    def generate_feature_importance_analysis(self, datasets: Dict[str, pd.DataFrame]):
        """Generate preliminary feature importance analysis"""
        try:
            analysis = {}

            # Work-group feature correlations
            wg_df = datasets.get('workgroup', pd.DataFrame())
            if not wg_df.empty:
                numeric_cols = wg_df.select_dtypes(include=[np.number]).columns
                correlations = wg_df[numeric_cols].corr()['gflops'].abs().sort_values(ascending=False)
                analysis['workgroup_features'] = correlations.head(10).to_dict()

            # Memory feature correlations
            mem_df = datasets.get('memory', pd.DataFrame())
            if not mem_df.empty:
                numeric_cols = mem_df.select_dtypes(include=[np.number]).columns
                correlations = mem_df[numeric_cols].corr()['gflops'].abs().sort_values(ascending=False)
                analysis['memory_features'] = correlations.head(10).to_dict()

            # Save analysis
            with open(self.data_dir / 'preliminary_feature_analysis.json', 'w') as f:
                json.dump(analysis, f, indent=2)

            logger.info("✅ Preliminary feature importance analysis completed")

        except Exception as e:
            logger.error(f"Failed to generate feature analysis: {e}")

def main():
    """Main function for dataset collection"""
    try:
        collector = DatasetCollector()

        # Collect Phase 13 data
        datasets = collector.collect_phase13_data()

        # Validate datasets
        validation = collector.validate_datasets(datasets)
        if not validation['overall']:
            logger.error("Dataset validation failed!")
            return

        # Save datasets
        collector.save_datasets(datasets)

        # Generate preliminary analysis
        collector.generate_feature_importance_analysis(datasets)

        logger.info("🎯 Dataset collection completed successfully!")
        logger.info(f"📊 Collected {len(datasets['workgroup'])} work-group, "
                   f"{len(datasets['memory'])} memory, "
                   f"{len(datasets['combined'])} combined samples")

    except Exception as e:
        logger.error(f"Dataset collection failed: {e}")
        raise

if __name__ == "__main__":
    main()