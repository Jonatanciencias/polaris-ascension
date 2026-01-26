#!/usr/bin/env python3
"""
🎯 MULTI-OBJECTIVE BAYESIAN OPTIMIZATION
=========================================

Sistema avanzado de optimización multi-objetivo que optimiza simultáneamente
múltiples métricas de rendimiento para la arquitectura Radeon RX 580 GCN 4.0.

Características principales:
- Optimización Pareto-óptima de GFLOPS vs eficiencia energética vs estabilidad
- Algoritmos NSGA-II y SPEA2 para front de Pareto
- Visualización de soluciones no-dominadas
- Análisis de compensaciones (trade-offs)
- Selección inteligente de soluciones óptimas

Author: AI Assistant
Date: 2026-01-25
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
try:
    import seaborn as sns
    SEABORN_AVAILABLE = True
except ImportError:
    SEABORN_AVAILABLE = False
from pathlib import Path
import json
import time
from typing import Dict, List, Any, Optional, Tuple, Union
import logging
import warnings
warnings.filterwarnings('ignore')

# Multi-objective optimization libraries
try:
    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.algorithms.moo.spea2 import SPEA2
    from pymoo.core.problem import Problem, ElementwiseProblem
    from pymoo.optimize import minimize
    from pymoo.visualization.scatter import Scatter
    from pymoo.visualization.pcp import PCP
    from pymoo.visualization.heatmap import Heatmap
    from pymoo.indicators.hv import HV
    from pymoo.decomposition.asf import ASF
    PYMOO_AVAILABLE = True
except ImportError:
    PYMOO_AVAILABLE = False
    print("PyMOO not available - multi-objective optimization limited")

    # Fallback classes when PyMOO is not available
    class ElementwiseProblem:
        """Fallback base class for optimization problems"""
        def __init__(self, n_var, n_obj, n_constr=0, xl=None, xu=None):
            self.n_var = n_var
            self.n_obj = n_obj
            self.n_constr = n_constr
            self.xl = xl
            self.xu = xu

    class Problem(ElementwiseProblem):
        """Fallback problem class"""
        pass

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("Plotly not available - visualization limited to matplotlib")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MultiObjectiveBayesianOptimizer:
    """
    Multi-Objective Bayesian Optimizer

    Advanced system for simultaneous optimization of multiple performance metrics
    using Pareto-based approaches and Bayesian optimization techniques.
    """

    def __init__(self, config_path: Optional[str] = None):
        self.base_dir = Path(__file__).parent.parent
        self.config_path = config_path or self.base_dir / "config" / "multi_objective_config.json"

        # Optimization state
        self.pareto_front = []
        self.pareto_history = []
        self.objective_names = ['gflops', 'power_efficiency', 'stability']
        self.objective_weights = {'gflops': 0.7, 'power_efficiency': 0.2, 'stability': 0.1}

        # Visualization settings
        self.plot_style = 'seaborn-v0_8'
        plt.style.use(self.plot_style)

        # Configuration
        self.config = self._load_config()

        logger.info("🎯 Multi-Objective Bayesian Optimizer initialized")
        logger.info(f"📊 Objectives: {', '.join(self.objective_names)}")
        logger.info(f"⚖️ Weights: {self.objective_weights}")

    def _load_config(self) -> Dict[str, Any]:
        """Load multi-objective optimization configuration"""
        default_config = {
            'algorithms': {
                'nsga2': {
                    'population_size': 100,
                    'offspring_size': 100,
                    'generations': 50,
                    'crossover_prob': 0.9,
                    'mutation_prob': 0.1
                },
                'spea2': {
                    'population_size': 100,
                    'generations': 50,
                    'crossover_prob': 0.9,
                    'mutation_prob': 0.1
                }
            },
            'objectives': {
                'gflops': {'weight': 0.7, 'direction': 'maximize'},
                'power_efficiency': {'weight': 0.2, 'direction': 'maximize'},
                'stability': {'weight': 0.1, 'direction': 'maximize'}
            },
            'visualization': {
                'plot_pareto': True,
                'plot_tradeoffs': True,
                'plot_history': True,
                'save_plots': True,
                'plot_format': 'png'
            },
            'selection': {
                'method': 'asf',  # Achievement Scalarizing Function
                'weights': [0.7, 0.2, 0.1],
                'reference_point': None
            }
        }

        if self.config_path.exists():
            with open(self.config_path, 'r') as f:
                loaded_config = json.load(f)
                # Merge with defaults
                for key, value in loaded_config.items():
                    if key in default_config:
                        default_config[key].update(value)
                    else:
                        default_config[key] = value

        return default_config

    class KernelOptimizationProblem(ElementwiseProblem):
        """
        Multi-objective optimization problem for kernel configurations

        Objectives:
        1. Maximize GFLOPS (minimize negative GFLOPS)
        2. Maximize power efficiency (minimize negative efficiency)
        3. Maximize stability (minimize negative stability)
        """

        def __init__(self, optimizer_instance):
            self.optimizer = optimizer_instance

            # Define parameter bounds
            space = self.optimizer._define_parameter_space()
            xl = np.array([bounds[0] for bounds in space.values()])
            xu = np.array([bounds[1] for bounds in space.values()])

            super().__init__(n_var=len(space), n_obj=3, n_constr=0, xl=xl, xu=xu)

        def _evaluate(self, x, out, *args, **kwargs):
            """Evaluate a single solution"""
            # Convert parameters to configuration
            config = self.optimizer._vector_to_config(x)

            # Run benchmark
            performance = self.optimizer._evaluate_configuration(config)

            # Objectives (PyMOO minimizes by default, so we negate maximization objectives)
            out["F"] = np.array([
                -performance['gflops'],           # Minimize negative GFLOPS = maximize GFLOPS
                -performance['power_efficiency'], # Minimize negative efficiency = maximize efficiency
                -performance['stability']         # Minimize negative stability = maximize stability
            ])

    def _define_parameter_space(self) -> Dict[str, Tuple[float, float]]:
        """Define the optimization parameter space"""
        return {
            'wg_size_0': (1, 16),       # Work-group X dimension
            'wg_size_1': (16, 256),     # Work-group Y dimension
            'use_lds': (0, 1),          # LDS usage (binary)
            'lds_tile_size': (16, 64),  # LDS tile size
            'vector_width': (1, 8),     # Vector width
            'unroll_factor': (1, 8),    # Loop unroll factor
            'prefetch_distance': (0, 4) # Prefetch distance
        }

    def _vector_to_config(self, x: np.ndarray) -> Dict[str, Union[int, float]]:
        """Convert optimization vector to kernel configuration"""
        space_keys = list(self._define_parameter_space().keys())

        config = {}
        for i, key in enumerate(space_keys):
            if key == 'use_lds':
                config[key] = int(round(x[i]))  # Binary parameter
            else:
                config[key] = int(round(x[i]))  # Integer parameters

        return config

    def _evaluate_configuration(self, config: Dict[str, Union[int, float]]) -> Dict[str, float]:
        """
        Evaluate a kernel configuration

        This simulates benchmark execution - replace with actual OpenCL kernel runs
        """
        # Base performance from Phase 13
        base_gflops = 398.96

        # Work-group efficiency
        wg_efficiency = min(config['wg_size_0'] * config['wg_size_1'] / 256.0, 1.0)
        wg_efficiency *= 0.8 + 0.4 * np.random.random()

        # Memory optimization factor
        mem_factor = 1.0
        if config['use_lds']:
            mem_factor *= 1.1 + 0.1 * (config['lds_tile_size'] / 32.0)
        mem_factor *= 0.9 + 0.2 * (config['vector_width'] / 4.0)
        mem_factor *= 0.95 + 0.1 * (config['unroll_factor'] / 4.0)

        # Combine factors
        total_factor = wg_efficiency * mem_factor
        noise = np.random.normal(0, 0.05)  # 5% noise
        final_gflops = base_gflops * total_factor * (1 + noise)
        final_gflops = np.clip(final_gflops, 100, 600)

        # Power efficiency (higher is better)
        power_efficiency = final_gflops / (100 + config['wg_size_0'] * config['wg_size_1'] * 0.1)

        # Stability score (higher is better)
        stability = 0.8 + 0.2 * np.random.random()

        return {
            'gflops': final_gflops,
            'power_efficiency': power_efficiency,
            'stability': stability,
            'wg_efficiency': wg_efficiency,
            'mem_factor': mem_factor
        }

    def optimize_nsga2(self, generations: int = 50) -> Dict[str, Any]:
        """
        Optimize using NSGA-II algorithm

        Args:
            generations: Number of generations to run

        Returns:
            Optimization results
        """
        logger.info("🎯 Starting NSGA-II multi-objective optimization...")

        if not PYMOO_AVAILABLE:
            return {'error': 'PyMOO not available'}

        # Create problem
        problem = self.KernelOptimizationProblem(self)

        # Configure algorithm
        algorithm = NSGA2(
            pop_size=self.config['algorithms']['nsga2']['population_size'],
            n_offsprings=self.config['algorithms']['nsga2']['offspring_size'],
            eliminate_duplicates=True
        )

        # Run optimization
        start_time = time.time()
        res = minimize(problem,
                      algorithm,
                      ('n_gen', generations),
                      seed=42,
                      verbose=True)
        end_time = time.time()

        # Extract Pareto front
        pareto_solutions = []
        for solution_vector in res.X:
            config = self._vector_to_config(solution_vector)
            performance = self._evaluate_configuration(config)

            solution = {
                'config': config,
                'objectives': {
                    'gflops': performance['gflops'],
                    'power_efficiency': performance['power_efficiency'],
                    'stability': performance['stability']
                },
                'objective_vector': [-performance['gflops'], -performance['power_efficiency'], -performance['stability']]
            }
            pareto_solutions.append(solution)

        # Calculate hypervolume
        hv_indicator = HV(ref_point=np.array([0, 0, 0]))  # Reference point for minimization
        hypervolume = hv_indicator(res.F)

        results = {
            'algorithm': 'NSGA-II',
            'pareto_solutions': pareto_solutions,
            'n_solutions': len(pareto_solutions),
            'hypervolume': hypervolume,
            'generations': generations,
            'execution_time': end_time - start_time,
            'convergence_metrics': {
                'final_hypervolume': hypervolume,
                'n_evaluations': res.algorithm.evaluator.n_eval
            }
        }

        self.pareto_front = pareto_solutions
        logger.info("✅ NSGA-II optimization completed")
        logger.info(f"📊 Pareto front size: {len(pareto_solutions)} solutions")
        logger.info(f"🎯 Hypervolume: {hypervolume:.4f}")
        return results

    def optimize_spea2(self, generations: int = 50) -> Dict[str, Any]:
        """
        Optimize using SPEA2 algorithm

        Args:
            generations: Number of generations to run

        Returns:
            Optimization results
        """
        logger.info("🎯 Starting SPEA2 multi-objective optimization...")

        if not PYMOO_AVAILABLE:
            return {'error': 'PyMOO not available'}

        # Create problem
        problem = self.KernelOptimizationProblem(self)

        # Configure algorithm
        algorithm = SPEA2(
            pop_size=self.config['algorithms']['spea2']['population_size']
        )

        # Run optimization
        start_time = time.time()
        res = minimize(problem,
                      algorithm,
                      ('n_gen', generations),
                      seed=42,
                      verbose=True)
        end_time = time.time()

        # Extract Pareto front
        pareto_solutions = []
        for solution_vector in res.X:
            config = self._vector_to_config(solution_vector)
            performance = self._evaluate_configuration(config)

            solution = {
                'config': config,
                'objectives': {
                    'gflops': performance['gflops'],
                    'power_efficiency': performance['power_efficiency'],
                    'stability': performance['stability']
                },
                'objective_vector': [-performance['gflops'], -performance['power_efficiency'], -performance['stability']]
            }
            pareto_solutions.append(solution)

        # Calculate hypervolume
        hv_indicator = HV(ref_point=np.array([0, 0, 0]))
        hypervolume = hv_indicator(res.F)

        results = {
            'algorithm': 'SPEA2',
            'pareto_solutions': pareto_solutions,
            'n_solutions': len(pareto_solutions),
            'hypervolume': hypervolume,
            'generations': generations,
            'execution_time': end_time - start_time,
            'convergence_metrics': {
                'final_hypervolume': hypervolume,
                'n_evaluations': res.algorithm.evaluator.n_eval
            }
        }

        logger.info("✅ SPEA2 optimization completed")
        logger.info(f"📊 Pareto front size: {len(pareto_solutions)} solutions")
        logger.info(f"🎯 Hypervolume: {hypervolume:.4f}")
        return results

    def select_optimal_solution(self, method: str = 'asf',
                              weights: Optional[List[float]] = None) -> Dict[str, Any]:
        """
        Select optimal solution from Pareto front using different methods

        Args:
            method: Selection method ('asf', 'weighted_sum', 'compromise')
            weights: Objective weights for selection

        Returns:
            Selected optimal solution
        """
        if not self.pareto_front:
            return {'error': 'No Pareto front available'}

        if weights is None:
            weights = self.config['selection']['weights']

        logger.info(f"🎯 Selecting optimal solution using {method} method...")

        if method == 'asf':
            # Achievement Scalarizing Function
            decomp = ASF(weights=np.array(weights))
            objective_matrix = np.array([sol['objective_vector'] for sol in self.pareto_front])
            best_idx = decomp(objective_matrix).argmin()

        elif method == 'weighted_sum':
            # Weighted sum approach
            objective_matrix = np.array([sol['objective_vector'] for sol in self.pareto_front])
            weighted_sums = objective_matrix @ np.array(weights)
            best_idx = weighted_sums.argmin()

        elif method == 'compromise':
            # Compromise programming (L2 norm from ideal point)
            objective_matrix = np.array([sol['objective_vector'] for sol in self.pareto_front])
            ideal_point = np.min(objective_matrix, axis=0)
            distances = np.linalg.norm(objective_matrix - ideal_point, axis=1)
            best_idx = distances.argmin()

        else:
            # Default to first solution
            best_idx = 0

        selected_solution = self.pareto_front[best_idx]

        logger.info("✅ Optimal solution selected")
        logger.info(f"🏆 GFLOPS: {selected_solution['objectives']['gflops']:.2f}")
        logger.info(f"⚡ Efficiency: {selected_solution['objectives']['power_efficiency']:.2f}")
        logger.info(f"🔒 Stability: {selected_solution['objectives']['stability']:.2f}")
        return selected_solution

    def visualize_pareto_front(self, save_path: Optional[str] = None) -> None:
        """
        Visualize the Pareto front in 3D objective space

        Args:
            save_path: Path to save the plot (optional)
        """
        if not self.pareto_front:
            logger.warning("No Pareto front to visualize")
            return

        logger.info("📊 Visualizing Pareto front...")

        # Extract objective values
        gflops = [sol['objectives']['gflops'] for sol in self.pareto_front]
        efficiency = [sol['objectives']['power_efficiency'] for sol in self.pareto_front]
        stability = [sol['objectives']['stability'] for sol in self.pareto_front]

        if PLOTLY_AVAILABLE:
            # 3D scatter plot with plotly
            fig = go.Figure(data=[go.Scatter3d(
                x=gflops,
                y=efficiency,
                z=stability,
                mode='markers',
                marker=dict(
                    size=6,
                    color=stability,
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title="Stability")
                ),
                text=[f"GFLOPS: {g:.1f}<br>Efficiency: {e:.2f}<br>Stability: {s:.2f}"
                      for g, e, s in zip(gflops, efficiency, stability)],
                hoverinfo='text'
            )])

            fig.update_layout(
                title="Pareto Front: Multi-Objective Kernel Optimization",
                scene=dict(
                    xaxis_title="GFLOPS (higher better)",
                    yaxis_title="Power Efficiency (higher better)",
                    zaxis_title="Stability (higher better)"
                ),
                width=800,
                height=600
            )

            if save_path:
                fig.write_html(save_path)
            else:
                fig.show()

        else:
            # 2D projections with matplotlib
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))

            # GFLOPS vs Efficiency
            axes[0].scatter(gflops, efficiency, c=stability, cmap='viridis', alpha=0.7)
            axes[0].set_xlabel('GFLOPS')
            axes[0].set_ylabel('Power Efficiency')
            axes[0].set_title('GFLOPS vs Power Efficiency')
            axes[0].grid(True, alpha=0.3)

            # GFLOPS vs Stability
            axes[1].scatter(gflops, stability, c=efficiency, cmap='plasma', alpha=0.7)
            axes[1].set_xlabel('GFLOPS')
            axes[1].set_ylabel('Stability')
            axes[1].set_title('GFLOPS vs Stability')
            axes[1].grid(True, alpha=0.3)

            # Efficiency vs Stability
            axes[2].scatter(efficiency, stability, c=gflops, cmap='coolwarm', alpha=0.7)
            axes[2].set_xlabel('Power Efficiency')
            axes[2].set_ylabel('Stability')
            axes[2].set_title('Power Efficiency vs Stability')
            axes[2].grid(True, alpha=0.3)

            plt.tight_layout()

            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
            else:
                plt.show()

        logger.info("✅ Pareto front visualization completed")

    def analyze_tradeoffs(self) -> Dict[str, Any]:
        """
        Analyze trade-offs between different objectives

        Returns:
            Trade-off analysis results
        """
        if not self.pareto_front:
            return {'error': 'No Pareto front available'}

        logger.info("🔍 Analyzing objective trade-offs...")

        # Extract objective values
        objectives = np.array([[sol['objectives'][obj] for obj in self.objective_names]
                              for sol in self.pareto_front])

        # Calculate correlation matrix
        corr_matrix = np.corrcoef(objectives.T)

        # Calculate ranges and spans
        ranges = np.ptp(objectives, axis=0)  # Peak-to-peak (max - min)

        # Find knee points (solutions with maximum curvature)
        knee_points = self._find_knee_points(objectives)

        # Calculate dominance metrics
        dominance_matrix = self._calculate_dominance_matrix(objectives)

        analysis = {
            'correlation_matrix': corr_matrix,
            'objective_ranges': dict(zip(self.objective_names, ranges)),
            'knee_points': knee_points,
            'dominance_analysis': dominance_matrix,
            'tradeoff_summary': {
                'gflops_efficiency_corr': corr_matrix[0, 1],
                'gflops_stability_corr': corr_matrix[0, 2],
                'efficiency_stability_corr': corr_matrix[1, 2],
                'max_gflops': np.max(objectives[:, 0]),
                'max_efficiency': np.max(objectives[:, 1]),
                'max_stability': np.max(objectives[:, 2])
            }
        }

        logger.info("✅ Trade-off analysis completed")
        logger.info(f"🔗 GFLOPS-Efficiency correlation: {tradeoff_summary['gflops_efficiency_corr']:.3f}")
        logger.info(f"🔗 GFLOPS-Stability correlation: {tradeoff_summary['gflops_stability_corr']:.3f}")
        logger.info(f"🔗 Efficiency-Stability correlation: {tradeoff_summary['efficiency_stability_corr']:.3f}")
        return analysis

    def _find_knee_points(self, objectives: np.ndarray) -> List[int]:
        """Find knee points in the Pareto front (points of maximum curvature)"""
        # Simple implementation: points with maximum distance from convex hull
        from scipy.spatial import ConvexHull

        try:
            hull = ConvexHull(objectives)
            hull_vertices = hull.vertices

            # Calculate distances from each point to the convex hull
            distances = []
            for i, point in enumerate(objectives):
                if i in hull_vertices:
                    # Points on hull have distance 0
                    distances.append(0)
                else:
                    # Calculate minimum distance to hull facets
                    min_dist = float('inf')
                    for simplex in hull.simplices:
                        # Calculate distance to triangle/plane
                        # Simplified: use distance to closest vertex
                        dist = np.min(np.linalg.norm(point - objectives[simplex], axis=1))
                        min_dist = min(min_dist, dist)
                    distances.append(min_dist)

            # Knee points are those with maximum distance from hull
            knee_indices = np.argsort(distances)[-3:]  # Top 3 knee points
            return knee_indices.tolist()

        except:
            # Fallback: return middle points
            n_points = len(objectives)
            return [n_points // 3, n_points // 2, 2 * n_points // 3]

    def _calculate_dominance_matrix(self, objectives: np.ndarray) -> Dict[str, Any]:
        """Calculate dominance relationships between solutions"""
        n_solutions = len(objectives)
        dominance_matrix = np.zeros((n_solutions, n_solutions))

        for i in range(n_solutions):
            for j in range(n_solutions):
                if i != j:
                    # Check if solution i dominates solution j
                    # For minimization (negated objectives), i dominates j if all i[k] <= j[k]
                    dominates = np.all(objectives[i] >= objectives[j])
                    dominance_matrix[i, j] = 1 if dominates else 0

        # Calculate dominance counts
        dominance_counts = np.sum(dominance_matrix, axis=1)

        return {
            'dominance_matrix': dominance_matrix.tolist(),
            'dominance_counts': dominance_counts.tolist(),
            'max_dominance_count': int(np.max(dominance_counts)),
            'avg_dominance_count': float(np.mean(dominance_counts))
        }

    def comprehensive_multi_objective_optimization(self) -> Dict[str, Any]:
        """
        Run comprehensive multi-objective optimization suite

        Combines NSGA-II and SPEA2 algorithms with analysis and visualization
        """
        logger.info("🚀 Starting comprehensive multi-objective optimization...")

        results = {
            'timestamp': time.time(),
            'nsga2_results': {},
            'spea2_results': {},
            'selected_solution': {},
            'tradeoff_analysis': {},
            'visualization_paths': []
        }

        # Run NSGA-II optimization
        logger.info("📍 Phase 1: NSGA-II optimization")
        results['nsga2_results'] = self.optimize_nsga2()

        # Run SPEA2 optimization
        logger.info("📍 Phase 2: SPEA2 optimization")
        results['spea2_results'] = self.optimize_spea2()

        # Select optimal solution
        logger.info("📍 Phase 3: Solution selection")
        results['selected_solution'] = self.select_optimal_solution()

        # Analyze trade-offs
        logger.info("📍 Phase 4: Trade-off analysis")
        results['tradeoff_analysis'] = self.analyze_tradeoffs()

        # Generate visualizations
        if self.config['visualization']['plot_pareto']:
            logger.info("📍 Phase 5: Visualization")
            viz_path = self.base_dir / "results" / "pareto_front_visualization.html"
            self.visualize_pareto_front(str(viz_path))
            results['visualization_paths'].append(str(viz_path))

        logger.info("🎉 Comprehensive multi-objective optimization completed!")
        return results

    def save_results(self, results: Dict[str, Any], filename: str = "multi_objective_results.json"):
        """Save multi-objective optimization results"""
        output_path = self.base_dir / "results" / filename
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert numpy types for JSON serialization
        def convert_to_serializable(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            else:
                return obj

        serializable_results = convert_to_serializable(results)

        with open(output_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)

        logger.info(f"💾 Multi-objective results saved to {output_path}")

def main():
    """Main function for multi-objective optimization"""
    try:
        # Initialize optimizer
        optimizer = MultiObjectiveBayesianOptimizer()

        # Run comprehensive optimization
        results = optimizer.comprehensive_multi_objective_optimization()

        # Save results
        optimizer.save_results(results)

        # Print summary
        print("\n" + "="*60)
        print("🎉 MULTI-OBJECTIVE OPTIMIZATION COMPLETED")
        print("="*60)

        if results['nsga2_results'].get('n_solutions'):
            print(f"📊 NSGA-II Pareto solutions: {results['nsga2_results']['n_solutions']}")
            print(f"🎯 NSGA-II Hypervolume: {results['nsga2_results'].get('hypervolume', 0):.4f}")
        if results['spea2_results'].get('n_solutions'):
            print(f"📊 SPEA2 Pareto solutions: {results['spea2_results']['n_solutions']}")
            print(f"🎯 SPEA2 Hypervolume: {results['spea2_results'].get('hypervolume', 0):.4f}")
        if results['selected_solution'].get('objectives'):
            sel_obj = results['selected_solution']['objectives']
            print(f"🏆 Selected solution - GFLOPS: {sel_obj.get('gflops', 0):.2f}")
            print(f"⚡ Power efficiency: {sel_obj.get('power_efficiency', 0):.2f}")
            print(f"🔒 Stability: {sel_obj.get('stability', 0):.2f}")
        print("="*60)

    except Exception as e:
        logger.error(f"Multi-objective optimization failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()