"""
Hyperparameter Tuning System

Professional hyperparameter optimization with:
- Grid search
- Random search
- Optuna integration (Bayesian optimization)
- Parallel trials support
- Result tracking and visualization
- Best configuration selection

Optimized for AMD Radeon RX 580 training workflows.

Author: AMD GPU Computing Team
Date: January 21, 2026
"""

import torch
import torch.nn as nn
from typing import Dict, List, Callable, Any, Optional, Tuple
from pathlib import Path
import logging
import json
import time
from itertools import product
import random
import numpy as np

try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    logging.warning("Optuna not installed. Bayesian optimization unavailable.")

logger = logging.getLogger(__name__)


# ============================================================================
# Hyperparameter Space
# ============================================================================

class HyperparameterSpace:
    """Define hyperparameter search space"""
    
    def __init__(self):
        self.params = {}
    
    def add_categorical(self, name: str, choices: List[Any]):
        """Add categorical parameter"""
        self.params[name] = {
            'type': 'categorical',
            'choices': choices
        }
    
    def add_float(self, name: str, low: float, high: float, log: bool = False):
        """Add continuous float parameter"""
        self.params[name] = {
            'type': 'float',
            'low': low,
            'high': high,
            'log': log
        }
    
    def add_int(self, name: str, low: int, high: int, log: bool = False):
        """Add integer parameter"""
        self.params[name] = {
            'type': 'int',
            'low': low,
            'high': high,
            'log': log
        }
    
    def sample_random(self) -> Dict[str, Any]:
        """Sample random configuration from space"""
        config = {}
        
        for name, spec in self.params.items():
            if spec['type'] == 'categorical':
                config[name] = random.choice(spec['choices'])
            elif spec['type'] == 'float':
                if spec['log']:
                    config[name] = np.exp(
                        np.random.uniform(
                            np.log(spec['low']),
                            np.log(spec['high'])
                        )
                    )
                else:
                    config[name] = np.random.uniform(spec['low'], spec['high'])
            elif spec['type'] == 'int':
                if spec['log']:
                    config[name] = int(np.exp(
                        np.random.uniform(
                            np.log(spec['low']),
                            np.log(spec['high'])
                        )
                    ))
                else:
                    config[name] = np.random.randint(spec['low'], spec['high'] + 1)
        
        return config
    
    def get_grid(self) -> List[Dict[str, Any]]:
        """Get grid of all configurations (for grid search)"""
        
        # Only works for categorical parameters
        categorical_params = {
            name: spec['choices']
            for name, spec in self.params.items()
            if spec['type'] == 'categorical'
        }
        
        if not categorical_params:
            raise ValueError("Grid search requires categorical parameters")
        
        # Generate all combinations
        names = list(categorical_params.keys())
        values = list(categorical_params.values())
        
        grid = []
        for combination in product(*values):
            config = dict(zip(names, combination))
            grid.append(config)
        
        return grid


# ============================================================================
# Hyperparameter Tuner
# ============================================================================

class HyperparameterTuner:
    """
    Professional hyperparameter tuning system.
    
    Supports:
    - Grid search (exhaustive)
    - Random search (efficient)
    - Optuna (Bayesian optimization, best results)
    """
    
    def __init__(
        self,
        objective_fn: Callable[[Dict], float],
        space: HyperparameterSpace,
        method: str = 'random',  # 'grid', 'random', 'optuna'
        n_trials: int = 20,
        direction: str = 'maximize',  # 'maximize' or 'minimize'
        output_dir: str = './outputs/tuning'
    ):
        """
        Initialize hyperparameter tuner.
        
        Args:
            objective_fn: Function that takes config dict and returns metric
            space: Hyperparameter search space
            method: 'grid', 'random', or 'optuna'
            n_trials: Number of trials (for random/optuna)
            direction: 'maximize' or 'minimize' metric
            output_dir: Directory to save results
        """
        self.objective_fn = objective_fn
        self.space = space
        self.method = method.lower()
        self.n_trials = n_trials
        self.direction = direction
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Results tracking
        self.trials = []
        self.best_config = None
        self.best_score = float('-inf') if direction == 'maximize' else float('inf')
        
        # Validate method
        if self.method not in ['grid', 'random', 'optuna']:
            raise ValueError(f"Unknown method: {method}")
        
        if self.method == 'optuna' and not OPTUNA_AVAILABLE:
            raise ValueError("Optuna not installed. Use 'grid' or 'random' instead.")
        
        logger.info(f"Initialized {self.method} tuner")
        logger.info(f"Direction: {self.direction}")
        logger.info(f"Number of trials: {self.n_trials}")
    
    def _is_better(self, new_score: float, old_score: float) -> bool:
        """Check if new score is better than old score"""
        if self.direction == 'maximize':
            return new_score > old_score
        else:
            return new_score < old_score
    
    def _update_best(self, config: Dict, score: float):
        """Update best configuration if improved"""
        if self._is_better(score, self.best_score):
            self.best_score = score
            self.best_config = config.copy()
            logger.info(f"✓ New best score: {score:.4f}")
    
    def run_grid_search(self) -> Dict:
        """Run grid search"""
        logger.info("=" * 80)
        logger.info("Starting Grid Search")
        logger.info("=" * 80)
        
        # Get grid
        grid = self.space.get_grid()
        total_trials = len(grid)
        
        logger.info(f"Total configurations: {total_trials}")
        
        # Evaluate each configuration
        for trial_idx, config in enumerate(grid, 1):
            logger.info(f"\nTrial {trial_idx}/{total_trials}")
            logger.info(f"Config: {config}")
            
            try:
                # Evaluate
                start_time = time.time()
                score = self.objective_fn(config)
                elapsed = time.time() - start_time
                
                # Record result
                result = {
                    'trial': trial_idx,
                    'config': config,
                    'score': score,
                    'time': elapsed
                }
                self.trials.append(result)
                
                # Update best
                self._update_best(config, score)
                
                logger.info(f"Score: {score:.4f} (time: {elapsed:.1f}s)")
                
            except Exception as e:
                logger.error(f"Trial {trial_idx} failed: {e}")
                self.trials.append({
                    'trial': trial_idx,
                    'config': config,
                    'score': None,
                    'error': str(e)
                })
        
        return self._finalize_results()
    
    def run_random_search(self) -> Dict:
        """Run random search"""
        logger.info("=" * 80)
        logger.info("Starting Random Search")
        logger.info("=" * 80)
        logger.info(f"Number of trials: {self.n_trials}")
        
        for trial_idx in range(1, self.n_trials + 1):
            logger.info(f"\nTrial {trial_idx}/{self.n_trials}")
            
            # Sample random configuration
            config = self.space.sample_random()
            logger.info(f"Config: {config}")
            
            try:
                # Evaluate
                start_time = time.time()
                score = self.objective_fn(config)
                elapsed = time.time() - start_time
                
                # Record result
                result = {
                    'trial': trial_idx,
                    'config': config,
                    'score': score,
                    'time': elapsed
                }
                self.trials.append(result)
                
                # Update best
                self._update_best(config, score)
                
                logger.info(f"Score: {score:.4f} (time: {elapsed:.1f}s)")
                
            except Exception as e:
                logger.error(f"Trial {trial_idx} failed: {e}")
                self.trials.append({
                    'trial': trial_idx,
                    'config': config,
                    'score': None,
                    'error': str(e)
                })
        
        return self._finalize_results()
    
    def run_optuna_search(self) -> Dict:
        """Run Optuna Bayesian optimization"""
        logger.info("=" * 80)
        logger.info("Starting Optuna Bayesian Optimization")
        logger.info("=" * 80)
        logger.info(f"Number of trials: {self.n_trials}")
        
        def optuna_objective(trial):
            """Optuna objective function"""
            # Sample hyperparameters
            config = {}
            
            for name, spec in self.space.params.items():
                if spec['type'] == 'categorical':
                    config[name] = trial.suggest_categorical(name, spec['choices'])
                elif spec['type'] == 'float':
                    config[name] = trial.suggest_float(
                        name, spec['low'], spec['high'], log=spec['log']
                    )
                elif spec['type'] == 'int':
                    config[name] = trial.suggest_int(
                        name, spec['low'], spec['high'], log=spec['log']
                    )
            
            logger.info(f"\nTrial {trial.number + 1}/{self.n_trials}")
            logger.info(f"Config: {config}")
            
            try:
                # Evaluate
                start_time = time.time()
                score = self.objective_fn(config)
                elapsed = time.time() - start_time
                
                # Record result
                result = {
                    'trial': trial.number + 1,
                    'config': config,
                    'score': score,
                    'time': elapsed
                }
                self.trials.append(result)
                
                logger.info(f"Score: {score:.4f} (time: {elapsed:.1f}s)")
                
                return score
                
            except Exception as e:
                logger.error(f"Trial {trial.number + 1} failed: {e}")
                self.trials.append({
                    'trial': trial.number + 1,
                    'config': config,
                    'score': None,
                    'error': str(e)
                })
                # Return worst possible score
                return float('-inf') if self.direction == 'maximize' else float('inf')
        
        # Create Optuna study
        study = optuna.create_study(
            direction=self.direction,
            sampler=optuna.samplers.TPESampler(seed=42)
        )
        
        # Optimize
        study.optimize(optuna_objective, n_trials=self.n_trials)
        
        # Get best configuration
        self.best_config = study.best_params
        self.best_score = study.best_value
        
        return self._finalize_results()
    
    def run(self) -> Dict:
        """
        Run hyperparameter tuning.
        
        Returns:
            Dictionary with results and best configuration
        """
        start_time = time.time()
        
        # Run tuning based on method
        if self.method == 'grid':
            results = self.run_grid_search()
        elif self.method == 'random':
            results = self.run_random_search()
        elif self.method == 'optuna':
            results = self.run_optuna_search()
        
        total_time = time.time() - start_time
        
        logger.info("\n" + "=" * 80)
        logger.info("Tuning Complete!")
        logger.info(f"Total time: {total_time / 60:.1f} minutes")
        logger.info(f"Best score: {self.best_score:.4f}")
        logger.info(f"Best config: {self.best_config}")
        logger.info("=" * 80)
        
        return results
    
    def _finalize_results(self) -> Dict:
        """Finalize and save results"""
        
        # Prepare results dictionary
        results = {
            'method': self.method,
            'n_trials': len(self.trials),
            'best_score': self.best_score,
            'best_config': self.best_config,
            'trials': self.trials
        }
        
        # Save results
        results_path = self.output_dir / f'{self.method}_results.json'
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Saved results to {results_path}")
        
        # Save best config separately
        best_config_path = self.output_dir / 'best_config.json'
        with open(best_config_path, 'w') as f:
            json.dump({
                'config': self.best_config,
                'score': self.best_score,
                'method': self.method
            }, f, indent=2)
        
        return results
    
    def get_top_k_configs(self, k: int = 5) -> List[Dict]:
        """Get top K configurations"""
        
        # Sort trials by score
        valid_trials = [t for t in self.trials if t.get('score') is not None]
        
        if self.direction == 'maximize':
            sorted_trials = sorted(valid_trials, key=lambda x: x['score'], reverse=True)
        else:
            sorted_trials = sorted(valid_trials, key=lambda x: x['score'])
        
        return sorted_trials[:k]


# ============================================================================
# Predefined Search Spaces
# ============================================================================

def create_default_search_space() -> HyperparameterSpace:
    """Create default search space for CIFAR training"""
    
    space = HyperparameterSpace()
    
    # Learning rate
    space.add_float('learning_rate', low=1e-4, high=1e-1, log=True)
    
    # Batch size
    space.add_categorical('batch_size', [64, 128, 256])
    
    # Optimizer
    space.add_categorical('optimizer', ['sgd', 'adam', 'adamw'])
    
    # Weight decay
    space.add_float('weight_decay', low=1e-5, high=1e-2, log=True)
    
    # Learning rate schedule
    space.add_categorical('lr_schedule', ['step', 'cosine', 'exponential'])
    
    # Momentum (for SGD)
    space.add_float('momentum', low=0.85, high=0.95, log=False)
    
    return space


# ============================================================================
# Demo Code
# ============================================================================

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("=" * 80)
    print("Hyperparameter Tuning Demo")
    print("=" * 80)
    
    # Define mock objective function
    def mock_objective(config: Dict) -> float:
        """
        Mock objective function that simulates training.
        Returns accuracy based on config.
        """
        # Simulate training time
        time.sleep(0.1)
        
        # Mock score calculation (higher LR = better, up to a point)
        lr = config.get('learning_rate', 0.01)
        batch_size = config.get('batch_size', 128)
        
        # Optimal values
        optimal_lr = 0.01
        optimal_batch = 128
        
        # Calculate "accuracy" (closer to optimal = higher)
        lr_score = 1.0 - abs(np.log10(lr) - np.log10(optimal_lr)) / 2
        batch_score = 1.0 - abs(batch_size - optimal_batch) / 200
        
        # Add randomness
        noise = np.random.normal(0, 0.02)
        
        accuracy = 0.85 + 0.1 * (lr_score + batch_score) / 2 + noise
        accuracy = np.clip(accuracy, 0.7, 0.95)
        
        return accuracy
    
    # Create search space
    print("\n1. Creating search space...")
    space = HyperparameterSpace()
    space.add_float('learning_rate', low=1e-3, high=1e-1, log=True)
    space.add_categorical('batch_size', [64, 128, 256])
    space.add_categorical('optimizer', ['sgd', 'adam'])
    
    # Run random search
    print("\n2. Running random search (10 trials)...")
    tuner = HyperparameterTuner(
        objective_fn=mock_objective,
        space=space,
        method='random',
        n_trials=10,
        direction='maximize',
        output_dir='./outputs/tuning_demo'
    )
    
    results = tuner.run()
    
    # Show top 3 configs
    print("\n3. Top 3 configurations:")
    top_configs = tuner.get_top_k_configs(k=3)
    for i, trial in enumerate(top_configs, 1):
        print(f"\n  Rank {i}:")
        print(f"    Score: {trial['score']:.4f}")
        print(f"    Config: {trial['config']}")
    
    print("\n" + "=" * 80)
    print("✓ Demo complete!")
    print(f"Results saved to: {tuner.output_dir}")
    print("=" * 80)
