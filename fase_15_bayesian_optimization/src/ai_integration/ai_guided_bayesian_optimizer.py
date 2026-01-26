#!/usr/bin/env python3
"""
🚀 AI-GUIDED BAYESIAN OPTIMIZER
================================

Sistema de optimización bayesiana que aprovecha las predicciones del
AI Kernel Predictor para inicialización inteligente y optimización
multi-objetivo de configuraciones de kernel.

Características principales:
- Integración con AI Kernel Predictor (17.7% MAPE accuracy)
- Optimización multi-objetivo (GFLOPS, eficiencia energética, estabilidad)
- Ejecución paralela para aceleración
- Cuantificación de incertidumbre en resultados
- Estrategias de muestreo adaptativo

Author: AI Assistant
Date: 2026-01-25
"""

import numpy as np
import pandas as pd
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
import warnings
warnings.filterwarnings('ignore')

# Bayesian Optimization libraries
try:
    from bayes_opt import BayesianOptimization
    from bayes_opt.util import UtilityFunction
    BAYESIAN_OPT_AVAILABLE = True
except ImportError:
    BAYESIAN_OPT_AVAILABLE = False
    print("BayesianOptimization not available, using fallback implementation")

try:
    from skopt import gp_minimize, forest_minimize
    from skopt.space import Real, Integer, Categorical
    from skopt.utils import use_named_args
    SKOPT_AVAILABLE = True
except ImportError:
    SKOPT_AVAILABLE = False
    print("scikit-optimize not available, using basic optimization")

# Multi-objective optimization
try:
    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.core.problem import Problem
    from pymoo.optimize import minimize
    from pymoo.visualization.scatter import Scatter
    PYMOO_AVAILABLE = True
except ImportError:
    PYMOO_AVAILABLE = False
    print("pymoo not available, multi-objective optimization limited")

# Local imports
import sys
sys.path.append(str(Path(__file__).parent.parent.parent / "fase_14_ai_kernel_predictor" / "src"))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AIGuidedBayesianOptimizer:
    """
    AI-Guided Bayesian Optimizer

    Advanced Bayesian optimization system that leverages AI Kernel Predictor
    predictions for intelligent initialization and multi-objective optimization.
    """

    def __init__(self, config_path: Optional[str] = None):
        self.base_dir = Path(__file__).parent.parent
        self.config_path = config_path or self.base_dir / "config" / "bayesian_config.json"

        # AI Integration
        self.ai_predictor = None
        self.ai_accuracy = 0.177  # 17.7% MAPE from Phase 14

        # Optimization state
        self.optimization_history = []
        self.current_best = None
        self.convergence_history = []

        # Multi-objective state
        self.pareto_front = []
        self.objective_weights = {'gflops': 0.7, 'efficiency': 0.2, 'stability': 0.1}

        # Parallel execution
        self.max_workers = min(4, os.cpu_count() or 2)
        self.executor = ThreadPoolExecutor(max_workers=self.max_workers)

        # Configuration
        self.config = self._load_config()

        logger.info("🚀 AI-Guided Bayesian Optimizer initialized")
        logger.info(f"📊 AI Prediction Accuracy: {self.ai_accuracy:.1%} MAPE")
        logger.info(f"🔄 Parallel Workers: {self.max_workers}")

    def _load_config(self) -> Dict[str, Any]:
        """Load optimization configuration"""
        default_config = {
            'optimization': {
                'max_iterations': 50,
                'initial_points': 10,
                'acquisition_function': 'ucb',
                'kappa': 2.576,  # 99% confidence
                'xi': 0.01
            },
            'multi_objective': {
                'algorithm': 'nsga2',
                'population_size': 100,
                'generations': 50,
                'objectives': ['gflops', 'power_efficiency', 'stability']
            },
            'parallel': {
                'batch_size': 4,
                'max_concurrent': 4,
                'timeout_seconds': 300
            },
            'uncertainty': {
                'quantification_method': 'gaussian_process',
                'confidence_level': 0.95,
                'risk_tolerance': 0.05
            },
            'ai_integration': {
                'use_ai_initialization': True,
                'ai_confidence_threshold': 0.7,
                'prediction_weight': 0.8,
                'exploration_bonus': 0.2
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

    def initialize_ai_predictor(self):
        """Initialize connection to AI Kernel Predictor"""
        try:
            from ensemble_predictor import EnsemblePredictor
            self.ai_predictor = EnsemblePredictor()
            self.ai_predictor.load_models()
            logger.info("✅ AI Kernel Predictor integrated successfully")
        except Exception as e:
            logger.warning(f"Failed to initialize AI predictor: {e}")
            logger.warning("Proceeding with Bayesian optimization only")
            self.ai_predictor = None

    def define_optimization_space(self) -> Dict[str, Tuple[float, float]]:
        """
        Define the optimization parameter space for Radeon RX 580

        Based on Phase 13 GCN architecture analysis and AI predictions
        """
        # Work-group dimensions (from Phase 13 analysis)
        wg_space = {
            'wg_size_0': (1, 16),      # Work-group X dimension
            'wg_size_1': (16, 256),    # Work-group Y dimension
        }

        # Memory configuration (from Phase 13 memory optimization)
        memory_space = {
            'use_lds': (0, 1),         # LDS usage (binary)
            'lds_tile_size': (16, 64), # LDS tile size
            'vector_width': (1, 8),    # Vector width
            'unroll_factor': (1, 8),   # Loop unroll factor
            'prefetch_distance': (0, 4) # Prefetch distance
        }

        # Combine spaces
        optimization_space = {**wg_space, **memory_space}

        logger.info(f"🎯 Optimization space defined: {len(optimization_space)} parameters")
        for param, bounds in optimization_space.items():
            logger.info(f"   {param}: {bounds}")

        return optimization_space

    def ai_guided_initialization(self, space: Dict[str, Tuple[float, float]],
                               n_initial: int = 10) -> List[Dict[str, float]]:
        """
        Generate initial points using AI predictions for intelligent initialization

        Args:
            space: Parameter space bounds
            n_initial: Number of initial points to generate

        Returns:
            List of initial parameter configurations
        """
        initial_points = []

        if self.ai_predictor is None:
            # Fallback: Random initialization
            logger.info("🔄 AI predictor not available, using random initialization")
            for i in range(n_initial):
                point = {}
                for param, (low, high) in space.items():
                    if param == 'use_lds':
                        point[param] = np.random.choice([0, 1])
                    else:
                        point[param] = np.random.uniform(low, high)
                initial_points.append(point)
            return initial_points

        # AI-guided initialization
        logger.info("🤖 Generating AI-guided initial points...")

        ai_suggestions = []
        confidence_scores = []

        # Generate diverse initial points with AI guidance
        for i in range(n_initial * 2):  # Generate more candidates
            # Create candidate configuration
            candidate = {}
            for param, (low, high) in space.items():
                if param == 'use_lds':
                    candidate[param] = np.random.choice([0, 1])
                else:
                    candidate[param] = np.random.uniform(low, high)

            # Get AI prediction for this configuration
            try:
                prediction = self.ai_predictor.predict_combined_config(candidate)
                if 'error' not in prediction:
                    ai_suggestions.append((candidate, prediction))
                    confidence_scores.append(prediction.get('optimal_probability', 0.5))
            except Exception as e:
                logger.debug(f"AI prediction failed for candidate {i}: {e}")

        # Select top N candidates based on AI confidence
        if ai_suggestions:
            # Sort by AI confidence
            sorted_suggestions = sorted(zip(ai_suggestions, confidence_scores),
                                      key=lambda x: x[1], reverse=True)

            # Select top candidates, ensuring diversity
            selected_candidates = []
            for (candidate, prediction), confidence in sorted_suggestions[:n_initial]:
                # Add some exploration bonus for diverse configurations
                exploration_bonus = np.random.random() * self.config['ai_integration']['exploration_bonus']
                adjusted_confidence = confidence + exploration_bonus

                candidate['_ai_confidence'] = confidence
                candidate['_adjusted_confidence'] = adjusted_confidence
                selected_candidates.append(candidate)

            initial_points = selected_candidates
            logger.info(f"✅ Generated {len(initial_points)} AI-guided initial points")
            logger.info(f"📊 Average AI confidence: {np.mean(confidence_scores):.2f}")

        else:
            # Fallback to random if AI fails
            logger.warning("⚠️ AI predictions failed, falling back to random initialization")
            for i in range(n_initial):
                point = {}
                for param, (low, high) in space.items():
                    if param == 'use_lds':
                        point[param] = np.random.choice([0, 1])
                    else:
                        point[param] = np.random.uniform(low, high)
                initial_points.append(point)

        return initial_points

    def objective_function(self, **params) -> float:
        """
        Objective function for Bayesian optimization

        Evaluates a kernel configuration and returns performance metrics.
        This is a simplified version - in practice, this would run actual benchmarks.
        """
        # Convert parameters to integers where needed
        config = {
            'wg_size_0': int(params['wg_size_0']),
            'wg_size_1': int(params['wg_size_1']),
            'use_lds': int(params['use_lds']),
            'lds_tile_size': int(params['lds_tile_size']),
            'vector_width': int(params['vector_width']),
            'unroll_factor': int(params['unroll_factor']),
            'prefetch_distance': int(params['prefetch_distance'])
        }

        # Simulate benchmark execution (replace with actual OpenCL benchmark)
        performance = self._simulate_benchmark(config)

        # Store in history
        result = {
            'config': config,
            'performance': performance,
            'timestamp': time.time(),
            'ai_guided': '_ai_confidence' in params
        }
        self.optimization_history.append(result)

        # Update current best
        if self.current_best is None or performance['gflops'] > self.current_best['performance']['gflops']:
            self.current_best = result

        logger.info(f"🔬 Evaluated config: WG({config['wg_size_0']},{config['wg_size_1']}) "
                   f"LDS({config['use_lds']}) -> {performance['gflops']:.2f} GFLOPS")

        return performance['gflops']

    def _simulate_benchmark(self, config: Dict[str, int]) -> Dict[str, float]:
        """
        Simulate benchmark execution (replace with actual OpenCL kernel execution)

        This is a simplified simulation based on Phase 13 results and AI predictions.
        """
        # Base performance from Phase 13
        base_gflops = 398.96

        # Work-group efficiency factor
        wg_efficiency = min(config['wg_size_0'] * config['wg_size_1'] / 256.0, 1.0)
        wg_efficiency *= 0.8 + 0.4 * np.random.random()  # Add some noise

        # Memory optimization factor
        mem_factor = 1.0
        if config['use_lds']:
            mem_factor *= 1.1 + 0.1 * (config['lds_tile_size'] / 32.0)
        mem_factor *= 0.9 + 0.2 * (config['vector_width'] / 4.0)
        mem_factor *= 0.95 + 0.1 * (config['unroll_factor'] / 4.0)

        # Combine factors with some realistic noise
        total_factor = wg_efficiency * mem_factor
        noise = np.random.normal(0, 0.05)  # 5% noise
        final_gflops = base_gflops * total_factor * (1 + noise)

        # Ensure reasonable bounds
        final_gflops = np.clip(final_gflops, 100, 600)

        # Calculate additional metrics
        power_efficiency = final_gflops / (100 + config['wg_size_0'] * config['wg_size_1'] * 0.1)
        stability = 0.8 + 0.2 * np.random.random()  # Stability score

        return {
            'gflops': final_gflops,
            'power_efficiency': power_efficiency,
            'stability': stability,
            'wg_efficiency': wg_efficiency,
            'mem_factor': mem_factor
        }

    def optimize_single_objective(self, max_iterations: int = 50) -> Dict[str, Any]:
        """
        Perform single-objective Bayesian optimization (maximize GFLOPS)

        Args:
            max_iterations: Maximum number of optimization iterations

        Returns:
            Optimization results and best configuration
        """
        logger.info("🎯 Starting single-objective Bayesian optimization...")

        # Define optimization space
        space = self.define_optimization_space()

        # Generate AI-guided initial points
        n_initial = self.config['optimization']['initial_points']
        initial_points = self.ai_guided_initialization(space, n_initial)

        # Initialize Bayesian optimizer
        if BAYESIAN_OPT_AVAILABLE:
            optimizer = BayesianOptimization(
                f=self.objective_function,
                pbounds=space,
                verbose=1,
                random_state=42
            )

            # Set initial points
            for point in initial_points:
                # Remove AI metadata for optimization
                clean_point = {k: v for k, v in point.items() if not k.startswith('_')}
                optimizer.probe(clean_point, lazy=True)

            # Run optimization
            optimizer.maximize(
                init_points=0,  # Already set initial points
                n_iter=max_iterations,
                acq=self.config['optimization']['acquisition_function'],
                kappa=self.config['optimization']['kappa'],
                xi=self.config['optimization']['xi']
            )

            # Extract results
            best_config = optimizer.max
            best_performance = best_config['target']

        else:
            # Fallback implementation using random search with AI guidance
            logger.warning("BayesianOptimization not available, using AI-guided random search")

            best_config = None
            best_performance = 0

            for i in range(max_iterations):
                # Sample from AI-guided distribution
                if i < len(initial_points):
                    candidate = initial_points[i]
                else:
                    candidate = {}
                    for param, (low, high) in space.items():
                        if param == 'use_lds':
                            candidate[param] = np.random.choice([0, 1])
                        else:
                            candidate[param] = np.random.uniform(low, high)

                # Clean candidate
                clean_candidate = {k: v for k, v in candidate.items() if not k.startswith('_')}

                # Evaluate
                performance = self.objective_function(**clean_candidate)

                if performance > best_performance:
                    best_performance = performance
                    best_config = {'params': clean_candidate, 'target': performance}

        # Compile results
        results = {
            'best_config': best_config,
            'best_performance': best_performance,
            'optimization_history': self.optimization_history,
            'convergence_history': self.convergence_history,
            'total_evaluations': len(self.optimization_history),
            'ai_guided_initialization': self.ai_predictor is not None,
            'optimization_method': 'bayesian' if BAYESIAN_OPT_AVAILABLE else 'ai_guided_random'
        }

        logger.info("✅ Single-objective optimization completed")
        logger.info(f"🏆 Best performance: {best_performance:.2f} GFLOPS")
        return results

    def optimize_multi_objective(self, generations: int = 50) -> Dict[str, Any]:
        """
        Perform multi-objective Bayesian optimization

        Optimizes GFLOPS, power efficiency, and stability simultaneously.
        """
        logger.info("🎯 Starting multi-objective Bayesian optimization...")

        if not PYMOO_AVAILABLE:
            logger.warning("PyMOO not available, skipping multi-objective optimization")
            return {'error': 'PyMOO not available'}

        # Define multi-objective problem
        class KernelOptimizationProblem(Problem):
            def __init__(self, optimizer):
                self.optimizer = optimizer
                super().__init__(n_var=7, n_obj=3, n_constr=0, xl=None, xu=None)
                # Set bounds
                space = optimizer.define_optimization_space()
                self.xl = np.array([bounds[0] for bounds in space.values()])
                self.xu = np.array([bounds[1] for bounds in space.values()])

            def _evaluate(self, x, out, *args, **kwargs):
                # Evaluate multiple objectives for each solution
                objectives = []
                for solution in x:
                    config = {
                        'wg_size_0': int(solution[0]),
                        'wg_size_1': int(solution[1]),
                        'use_lds': int(solution[2]),
                        'lds_tile_size': int(solution[3]),
                        'vector_width': int(solution[4]),
                        'unroll_factor': int(solution[5]),
                        'prefetch_distance': int(solution[6])
                    }

                    performance = self.optimizer._simulate_benchmark(config)

                    # Objectives: maximize GFLOPS, maximize efficiency, maximize stability
                    # (PyMOO minimizes by default, so we negate)
                    obj_gflops = -performance['gflops']
                    obj_efficiency = -performance['power_efficiency']
                    obj_stability = -performance['stability']

                    objectives.append([obj_gflops, obj_efficiency, obj_stability])

                out["F"] = np.array(objectives)

        # Create and run optimization
        problem = KernelOptimizationProblem(self)

        algorithm = NSGA2(pop_size=self.config['multi_objective']['population_size'])

        res = minimize(problem,
                      algorithm,
                      ('n_gen', generations),
                      seed=42,
                      verbose=True)

        # Extract Pareto front
        pareto_front = []
        for solution in res.X:
            config = {
                'wg_size_0': int(solution[0]),
                'wg_size_1': int(solution[1]),
                'use_lds': int(solution[2]),
                'lds_tile_size': int(solution[3]),
                'vector_width': int(solution[4]),
                'unroll_factor': int(solution[5]),
                'prefetch_distance': int(solution[6])
            }

            performance = self._simulate_benchmark(config)

            pareto_front.append({
                'config': config,
                'objectives': {
                    'gflops': performance['gflops'],
                    'power_efficiency': performance['power_efficiency'],
                    'stability': performance['stability']
                }
            })

        results = {
            'pareto_front': pareto_front,
            'n_solutions': len(pareto_front),
            'hypervolume': None,  # Could be calculated with additional metrics
            'generations': generations,
            'algorithm': 'NSGA-II'
        }

        logger.info("✅ Multi-objective optimization completed")
        logger.info(f"📊 Pareto front size: {len(pareto_front)} solutions")

        return results

    def parallel_optimization(self, n_parallel: int = 4, max_iterations: int = 25) -> Dict[str, Any]:
        """
        Perform parallel Bayesian optimization experiments

        Args:
            n_parallel: Number of parallel optimization runs
            max_iterations: Iterations per optimization run

        Returns:
            Results from all parallel optimizations
        """
        logger.info(f"🔄 Starting parallel optimization with {n_parallel} concurrent runs...")

        def single_optimization_run(run_id: int) -> Dict[str, Any]:
            """Single optimization run for parallel execution"""
            logger.info(f"🏃 Run {run_id}: Starting optimization...")

            # Create separate optimizer instance for this run
            optimizer = AIGuidedBayesianOptimizer()
            optimizer.initialize_ai_predictor()

            # Run optimization
            results = optimizer.optimize_single_objective(max_iterations)

            results['run_id'] = run_id
            logger.info(f"✅ Run {run_id}: Completed - Best: {results['best_performance']:.2f} GFLOPS")

            return results

        # Execute parallel runs
        futures = []
        for run_id in range(n_parallel):
            future = self.executor.submit(single_optimization_run, run_id)
            futures.append(future)

        # Collect results
        parallel_results = []
        for future in as_completed(futures):
            try:
                result = future.result(timeout=self.config['parallel']['timeout_seconds'])
                parallel_results.append(result)
            except Exception as e:
                logger.error(f"Parallel optimization failed: {e}")

        # Analyze parallel results
        if parallel_results:
            best_overall = max(parallel_results, key=lambda x: x['best_performance'])
            avg_performance = np.mean([r['best_performance'] for r in parallel_results])
            std_performance = np.std([r['best_performance'] for r in parallel_results])

            analysis = {
                'parallel_results': parallel_results,
                'best_overall': best_overall,
                'average_performance': avg_performance,
                'performance_std': std_performance,
                'speedup_factor': n_parallel,  # Theoretical speedup
                'total_evaluations': sum(r['total_evaluations'] for r in parallel_results)
            }

            logger.info("✅ Parallel optimization completed")
            logger.info(f"🏆 Best overall: {best_overall:.2f} GFLOPS")
            logger.info(f"📊 Average performance: {avg_performance:.2f} GFLOPS")
            return analysis
        else:
            return {'error': 'All parallel runs failed'}

    def uncertainty_quantification(self, config: Dict[str, Any], n_samples: int = 100) -> Dict[str, Any]:
        """
        Quantify uncertainty in optimization results

        Args:
            config: Configuration to analyze
            n_samples: Number of uncertainty samples

        Returns:
            Uncertainty analysis results
        """
        logger.info("🔬 Performing uncertainty quantification...")

        # Generate multiple performance samples for the configuration
        performances = []
        for _ in range(n_samples):
            perf = self._simulate_benchmark(config)
            performances.append(perf['gflops'])

        performances = np.array(performances)

        # Calculate uncertainty metrics
        mean_performance = np.mean(performances)
        std_performance = np.std(performances)
        confidence_interval_95 = 1.96 * std_performance / np.sqrt(n_samples)

        # Risk assessment
        percentile_5 = np.percentile(performances, 5)
        percentile_95 = np.percentile(performances, 95)
        risk_of_failure = np.mean(performances < mean_performance * 0.9)  # Below 90% of mean

        uncertainty_results = {
            'mean_performance': mean_performance,
            'std_performance': std_performance,
            'confidence_interval_95': confidence_interval_95,
            'percentile_5': percentile_5,
            'percentile_95': percentile_95,
            'risk_of_failure': risk_of_failure,
            'cv_efficiency': std_performance / mean_performance,  # Coefficient of variation
            'n_samples': n_samples
        }

        logger.info("✅ Uncertainty quantification completed")
        logger.info(f"📊 Mean performance: {mean_performance:.2f} GFLOPS")
        logger.info(f"📈 Uncertainty: ±{confidence_interval_95:.2f} GFLOPS")
        return uncertainty_results

    def comprehensive_optimization(self) -> Dict[str, Any]:
        """
        Run comprehensive optimization suite

        Combines single-objective, multi-objective, and parallel optimization
        with uncertainty quantification.
        """
        logger.info("🚀 Starting comprehensive Bayesian optimization suite...")

        results = {
            'timestamp': time.time(),
            'single_objective': {},
            'multi_objective': {},
            'parallel': {},
            'uncertainty_analysis': {},
            'final_recommendations': {}
        }

        # 1. Single-objective optimization
        logger.info("📍 Phase 1: Single-objective optimization")
        results['single_objective'] = self.optimize_single_objective()

        # 2. Multi-objective optimization
        logger.info("📍 Phase 2: Multi-objective optimization")
        results['multi_objective'] = self.optimize_multi_objective()

        # 3. Parallel optimization validation
        logger.info("📍 Phase 3: Parallel optimization validation")
        results['parallel'] = self.parallel_optimization(n_parallel=3)

        # 4. Uncertainty quantification for best configuration
        if results['single_objective'].get('best_config'):
            best_config = results['single_objective']['best_config']['params']
            logger.info("📍 Phase 4: Uncertainty quantification")
            results['uncertainty_analysis'] = self.uncertainty_quantification(best_config)

        # 5. Generate final recommendations
        results['final_recommendations'] = self._generate_recommendations(results)

        logger.info("🎉 Comprehensive optimization completed!")
        return results

    def _generate_recommendations(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate final optimization recommendations"""
        recommendations = {
            'best_single_config': None,
            'pareto_solutions': [],
            'confidence_assessment': {},
            'performance_projection': {},
            'risk_assessment': {}
        }

        # Best single-objective configuration
        if results['single_objective'].get('best_config'):
            best_single = results['single_objective']['best_config']
            recommendations['best_single_config'] = {
                'config': best_single['params'],
                'performance': best_single['target'],
                'improvement_over_baseline': best_single['target'] - 398.96
            }

        # Pareto-optimal solutions
        if results['multi_objective'].get('pareto_front'):
            recommendations['pareto_solutions'] = results['multi_objective']['pareto_front'][:5]  # Top 5

        # Confidence assessment
        if results['uncertainty_analysis']:
            uncertainty = results['uncertainty_analysis']
            recommendations['confidence_assessment'] = {
                'mean_performance': uncertainty['mean_performance'],
                'confidence_interval': f"±{uncertainty['confidence_interval_95']:.2f}",
                'risk_level': 'Low' if uncertainty['risk_of_failure'] < 0.1 else 'Medium' if uncertainty['risk_of_failure'] < 0.2 else 'High'
            }

        # Performance projection
        baseline = 398.96
        if recommendations['best_single_config']:
            achieved = recommendations['best_single_config']['performance']
            improvement = (achieved - baseline) / baseline * 100

            recommendations['performance_projection'] = {
                'baseline_gflops': baseline,
                'achieved_gflops': achieved,
                'improvement_percent': improvement,
                'target_achieved': improvement >= 15.0  # 15% target
            }

        return recommendations

    def save_results(self, results: Dict[str, Any], filename: str = "bayesian_optimization_results.json"):
        """Save optimization results to file"""
        output_path = self.base_dir / "results" / filename
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert numpy types to native Python types for JSON serialization
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

        logger.info(f"💾 Results saved to {output_path}")

def main():
    """Main function for AI-guided Bayesian optimization"""
    try:
        # Initialize optimizer
        optimizer = AIGuidedBayesianOptimizer()

        # Initialize AI predictor integration
        optimizer.initialize_ai_predictor()

        # Run comprehensive optimization
        results = optimizer.comprehensive_optimization()

        # Save results
        optimizer.save_results(results)

        # Print summary
        print("\n" + "="*60)
        print("🎉 BAYESIAN OPTIMIZATION COMPLETED")
        print("="*60)

        if results['single_objective'].get('best_performance'):
            best_perf = results['single_objective']['best_performance']
            improvement = (best_perf - 398.96) / 398.96 * 100
            print(f"🏆 Best performance: {best_perf:.2f} GFLOPS")
            print(f"📈 Improvement: +{improvement:.1f}% over baseline")
        if results['multi_objective'].get('n_solutions'):
            print(f"📊 Pareto solutions found: {results['multi_objective']['n_solutions']}")

        if results['parallel'].get('best_overall'):
            parallel_best = results['parallel']['best_overall']['best_performance']
            print(f"🔄 Parallel best: {parallel_best:.2f} GFLOPS")
        print("="*60)

    except Exception as e:
        logger.error(f"Bayesian optimization failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()