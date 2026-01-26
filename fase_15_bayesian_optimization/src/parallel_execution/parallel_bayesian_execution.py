#!/usr/bin/env python3
"""
🔄 PARALLEL BAYESIAN EXECUTION ENGINE
=====================================

Sistema de ejecución paralela para optimización bayesiana que acelera
el proceso de auto-tuning mediante ejecución concurrente de múltiples
experimentos de optimización.

Características principales:
- Ejecución paralela de optimizaciones Bayesianas
- Gestión inteligente de recursos GPU/CPU
- Balanceo de carga adaptativo
- Recolección distribuida de resultados
- Sincronización y coordinación de workers
- Tolerancia a fallos y recuperación

Author: AI Assistant
Date: 2026-01-25
"""

import numpy as np
import pandas as pd
import json
import time
import threading
import multiprocessing
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from queue import Queue, PriorityQueue
import logging
import signal
import os
import psutil
import warnings
warnings.filterwarnings('ignore')

# Parallel processing libraries
try:
    import dask
    import dask.distributed
    DASK_AVAILABLE = True
except ImportError:
    DASK_AVAILABLE = False
    print("Dask not available - limited parallel capabilities")

try:
    import joblib
    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False
    print("Joblib not available - using basic multiprocessing")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ParallelBayesianExecution:
    """
    Parallel Bayesian Execution Engine

    Advanced parallel execution system for Bayesian optimization that
    distributes optimization tasks across multiple CPU/GPU resources.
    """

    def __init__(self, config_path: Optional[str] = None):
        self.base_dir = Path(__file__).parent.parent
        self.config_path = config_path or self.base_dir / "config" / "parallel_config.json"

        # Execution state
        self.active_workers = 0
        self.completed_tasks = 0
        self.failed_tasks = 0
        self.execution_queue = PriorityQueue()
        self.results_queue = Queue()
        self.worker_status = {}

        # Resource monitoring
        self.resource_monitor = ResourceMonitor()
        self.system_limits = self._get_system_limits()

        # Synchronization
        self.stop_event = threading.Event()
        self.pause_event = threading.Event()
        self.pause_event.set()  # Start unpaused

        # Configuration
        self.config = self._load_config()

        # Signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        logger.info("🔄 Parallel Bayesian Execution Engine initialized")
        logger.info(f"🖥️ Available CPU cores: {self.system_limits['cpu_cores']}")
        logger.info(f"🧠 Available memory: {self.system_limits['memory_gb']:.1f} GB")
        logger.info(f"🎯 Max concurrent workers: {self.config['execution']['max_concurrent_workers']}")

    def _load_config(self) -> Dict[str, Any]:
        """Load parallel execution configuration"""
        default_config = {
            'execution': {
                'max_concurrent_workers': min(4, multiprocessing.cpu_count()),
                'worker_timeout_seconds': 300,
                'task_priority_levels': 3,
                'load_balancing': True,
                'adaptive_scaling': True
            },
            'resources': {
                'cpu_per_worker': 1,
                'memory_per_worker_gb': 2.0,
                'gpu_per_worker': 0,  # 0 = CPU only, 1 = GPU required
                'max_memory_usage_percent': 80.0
            },
            'scheduling': {
                'algorithm': 'priority_queue',  # 'priority_queue', 'round_robin', 'adaptive'
                'batch_size': 4,
                'queue_size_limit': 100,
                'preemption_enabled': False
            },
            'fault_tolerance': {
                'max_retries': 3,
                'retry_delay_seconds': 5,
                'failure_threshold_percent': 20.0,
                'auto_recovery': True
            },
            'monitoring': {
                'log_interval_seconds': 10,
                'performance_tracking': True,
                'resource_alerts': True,
                'alert_threshold_percent': 90.0
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

    def _get_system_limits(self) -> Dict[str, Any]:
        """Get system resource limits"""
        return {
            'cpu_cores': multiprocessing.cpu_count(),
            'memory_gb': psutil.virtual_memory().total / (1024**3),
            'available_memory_gb': psutil.virtual_memory().available / (1024**3),
            'cpu_percent': psutil.cpu_percent(interval=1),
            'memory_percent': psutil.virtual_memory().percent
        }

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        logger.info(f"🛑 Received signal {signum}, initiating graceful shutdown...")
        self.stop_event.set()

    class OptimizationTask:
        """Represents a single optimization task"""

        def __init__(self, task_id: str, config: Dict[str, Any], priority: int = 1,
                     dependencies: Optional[List[str]] = None):
            self.task_id = task_id
            self.config = config
            self.priority = priority
            self.dependencies = dependencies or []
            self.status = 'pending'
            self.start_time = None
            self.end_time = None
            self.worker_id = None
            self.attempts = 0
            self.max_attempts = 3
            self.result = None
            self.error = None

        def __lt__(self, other):
            # Priority queue comparison (higher priority = lower number)
            return self.priority < other.priority

        def to_dict(self) -> Dict[str, Any]:
            return {
                'task_id': self.task_id,
                'config': self.config,
                'priority': self.priority,
                'dependencies': self.dependencies,
                'status': self.status,
                'start_time': self.start_time,
                'end_time': self.end_time,
                'worker_id': self.worker_id,
                'attempts': self.attempts,
                'result': self.result,
                'error': self.error
            }

    def create_optimization_tasks(self, n_tasks: int = 10,
                                task_configs: Optional[List[Dict[str, Any]]] = None) -> List[OptimizationTask]:
        """
        Create optimization tasks for parallel execution

        Args:
            n_tasks: Number of tasks to create
            task_configs: Specific configurations for tasks (optional)

        Returns:
            List of optimization tasks
        """
        tasks = []

        if task_configs:
            # Use provided configurations
            for i, config in enumerate(task_configs):
                task = self.OptimizationTask(
                    task_id=f"task_{i:03d}",
                    config=config,
                    priority=config.get('priority', 1)
                )
                tasks.append(task)
        else:
            # Generate diverse optimization tasks
            for i in range(n_tasks):
                # Vary optimization parameters for diversity
                config = {
                    'optimization': {
                        'max_iterations': np.random.randint(20, 50),
                        'initial_points': np.random.randint(5, 15),
                        'acquisition_function': np.random.choice(['ucb', 'ei', 'poi']),
                        'kappa': np.random.uniform(1.5, 3.0)
                    },
                    'exploration': {
                        'exploration_weight': np.random.uniform(0.1, 0.5),
                        'diversity_bonus': np.random.uniform(0.0, 0.3)
                    },
                    'priority': np.random.randint(1, 4)  # Priority levels 1-3
                }

                task = self.OptimizationTask(
                    task_id=f"task_{i:03d}",
                    config=config,
                    priority=config['priority']
                )
                tasks.append(task)

        logger.info(f"📋 Created {len(tasks)} optimization tasks")
        return tasks

    def submit_tasks(self, tasks: List[OptimizationTask]) -> None:
        """Submit tasks to the execution queue"""
        for task in tasks:
            if self.execution_queue.qsize() < self.config['scheduling']['queue_size_limit']:
                self.execution_queue.put(task)
                logger.debug(f"📤 Submitted task {task.task_id} (priority {task.priority})")
            else:
                logger.warning("⚠️ Task queue full, dropping task")

        logger.info(f"✅ Submitted {len(tasks)} tasks to execution queue")

    def execute_parallel_optimization(self, tasks: List[OptimizationTask],
                                    execution_mode: str = 'thread_pool') -> Dict[str, Any]:
        """
        Execute optimization tasks in parallel

        Args:
            tasks: List of optimization tasks
            execution_mode: 'thread_pool', 'process_pool', or 'dask'

        Returns:
            Execution results and statistics
        """
        logger.info(f"🚀 Starting parallel optimization with {execution_mode} mode...")

        # Submit tasks
        self.submit_tasks(tasks)

        # Start resource monitoring
        monitor_thread = threading.Thread(target=self._monitor_resources)
        monitor_thread.daemon = True
        monitor_thread.start()

        # Execute based on mode
        if execution_mode == 'thread_pool':
            results = self._execute_thread_pool()
        elif execution_mode == 'process_pool':
            results = self._execute_process_pool()
        elif execution_mode == 'dask' and DASK_AVAILABLE:
            results = self._execute_dask()
        else:
            logger.warning(f"⚠️ Unsupported execution mode {execution_mode}, falling back to thread_pool")
            results = self._execute_thread_pool()

        # Compile final results
        execution_stats = {
            'total_tasks': len(tasks),
            'completed_tasks': self.completed_tasks,
            'failed_tasks': self.failed_tasks,
            'execution_time': time.time() - time.time(),  # Would need to track start time
            'success_rate': self.completed_tasks / len(tasks) if tasks else 0,
            'average_task_time': 0,  # Calculate from results
            'resource_utilization': self.resource_monitor.get_summary(),
            'execution_mode': execution_mode
        }

        logger.info("✅ Parallel optimization execution completed")
        logger.info(f"📊 Completed: {self.completed_tasks}, Failed: {self.failed_tasks}")

        return {
            'execution_stats': execution_stats,
            'task_results': results,
            'system_info': self.system_limits
        }

    def _execute_thread_pool(self) -> List[Dict[str, Any]]:
        """Execute tasks using thread pool"""
        results = []

        with ThreadPoolExecutor(max_workers=self.config['execution']['max_concurrent_workers']) as executor:
            futures = {}

            while not self.execution_queue.empty() and not self.stop_event.is_set():
                # Submit available tasks
                while (len(futures) < self.config['execution']['max_concurrent_workers']
                       and not self.execution_queue.empty()):

                    task = self.execution_queue.get()
                    if self._check_resource_availability():
                        future = executor.submit(self._execute_single_task, task)
                        futures[future] = task
                        task.status = 'running'
                        task.start_time = time.time()
                        self.active_workers += 1
                        logger.debug(f"🏃 Started task {task.task_id}")

                # Wait for completed tasks
                for future in as_completed(futures, timeout=1.0):
                    task = futures[future]
                    try:
                        result = future.result(timeout=self.config['execution']['worker_timeout_seconds'])
                        task.result = result
                        task.status = 'completed'
                        task.end_time = time.time()
                        self.completed_tasks += 1
                        results.append(task.to_dict())
                        logger.debug(f"✅ Completed task {task.task_id}")
                    except Exception as e:
                        task.error = str(e)
                        task.status = 'failed'
                        task.attempts += 1
                        self.failed_tasks += 1

                        if task.attempts < task.max_attempts:
                            # Retry task
                            logger.warning(f"⚠️ Task {task.task_id} failed (attempt {task.attempts}), retrying...")
                            self.execution_queue.put(task)
                        else:
                            logger.error(f"❌ Task {task.task_id} failed permanently: {e}")
                            results.append(task.to_dict())

                    self.active_workers -= 1
                    del futures[future]

                    if self.stop_event.is_set():
                        break

                # Brief pause to prevent busy waiting
                time.sleep(0.1)

        return results

    def _execute_process_pool(self) -> List[Dict[str, Any]]:
        """Execute tasks using process pool (better for CPU-intensive tasks)"""
        results = []

        with ProcessPoolExecutor(max_workers=self.config['execution']['max_concurrent_workers']) as executor:
            futures = {}

            while not self.execution_queue.empty() and not self.stop_event.is_set():
                # Submit available tasks
                while (len(futures) < self.config['execution']['max_concurrent_workers']
                       and not self.execution_queue.empty()):

                    task = self.execution_queue.get()
                    if self._check_resource_availability():
                        future = executor.submit(self._execute_single_task_process, task)
                        futures[future] = task
                        task.status = 'running'
                        task.start_time = time.time()
                        self.active_workers += 1

                # Collect results (similar to thread pool)
                for future in as_completed(futures, timeout=1.0):
                    task = futures[future]
                    try:
                        result = future.result(timeout=self.config['execution']['worker_timeout_seconds'])
                        task.result = result
                        task.status = 'completed'
                        task.end_time = time.time()
                        self.completed_tasks += 1
                        results.append(task.to_dict())
                    except Exception as e:
                        task.error = str(e)
                        task.status = 'failed'
                        task.attempts += 1
                        self.failed_tasks += 1

                        if task.attempts < task.max_attempts:
                            self.execution_queue.put(task)
                        else:
                            results.append(task.to_dict())

                    self.active_workers -= 1
                    del futures[future]

                    if self.stop_event.is_set():
                        break

                time.sleep(0.1)

        return results

    def _execute_dask(self) -> List[Dict[str, Any]]:
        """Execute tasks using Dask distributed computing"""
        try:
            # Initialize Dask client
            client = dask.distributed.Client(processes=False, threads_per_worker=1,
                                           n_workers=self.config['execution']['max_concurrent_workers'])

            futures = []
            results = []

            # Submit all tasks
            while not self.execution_queue.empty():
                task = self.execution_queue.get()
                future = client.submit(self._execute_single_task, task)
                futures.append((future, task))

            # Collect results
            for future, task in futures:
                try:
                    result = future.result(timeout=self.config['execution']['worker_timeout_seconds'])
                    task.result = result
                    task.status = 'completed'
                    results.append(task.to_dict())
                    self.completed_tasks += 1
                except Exception as e:
                    task.error = str(e)
                    task.status = 'failed'
                    results.append(task.to_dict())
                    self.failed_tasks += 1

            client.close()
            return results

        except Exception as e:
            logger.error(f"Dask execution failed: {e}")
            return []

    def _execute_single_task(self, task: OptimizationTask) -> Dict[str, Any]:
        """Execute a single optimization task"""
        try:
            # Import here to avoid circular imports
            from ai_integration.ai_guided_bayesian_optimizer import AIGuidedBayesianOptimizer

            # Create optimizer instance
            optimizer = AIGuidedBayesianOptimizer()

            # Initialize AI predictor
            optimizer.initialize_ai_predictor()

            # Configure optimizer with task settings
            task_config = task.config
            max_iter = task_config.get('optimization', {}).get('max_iterations', 25)

            # Run optimization
            result = optimizer.optimize_single_objective(max_iterations=max_iter)

            # Add task metadata
            result['task_id'] = task.task_id
            result['execution_time'] = time.time() - task.start_time if task.start_time else 0

            return result

        except Exception as e:
            logger.error(f"Task {task.task_id} execution failed: {e}")
            raise

    def _execute_single_task_process(self, task: OptimizationTask) -> Dict[str, Any]:
        """Execute a single task in a separate process (for CPU isolation)"""
        # Set process-specific random seed for reproducibility
        np.random.seed(int(time.time() * 1000) % 2**32)

        return self._execute_single_task(task)

    def _check_resource_availability(self) -> bool:
        """Check if system resources are available for new tasks"""
        current_limits = self._get_system_limits()

        # Check CPU availability
        cpu_available = (self.active_workers * self.config['resources']['cpu_per_worker']
                        < current_limits['cpu_cores'] * 0.8)  # Leave 20% headroom

        # Check memory availability
        memory_per_worker = self.config['resources']['memory_per_worker_gb']
        memory_available = (current_limits['available_memory_gb']
                          > memory_per_worker * 1.2)  # 20% safety margin

        # Check memory usage threshold
        memory_ok = current_limits['memory_percent'] < self.config['resources']['max_memory_usage_percent']

        available = cpu_available and memory_available and memory_ok

        if not available:
            logger.debug("⚠️ Resource constraints detected, pausing task submission")
            logger.debug(f"   CPU: {cpu_available}, Memory: {memory_available}, Usage: {memory_ok}")

        return available

    def _monitor_resources(self) -> None:
        """Monitor system resources during execution"""
        while not self.stop_event.is_set():
            if self.config['monitoring']['resource_alerts']:
                limits = self._get_system_limits()

                # Alert on high resource usage
                if limits['cpu_percent'] > self.config['monitoring']['alert_threshold_percent']:
                    logger.warning(f"⚠️ High CPU usage: {limits['cpu_percent']:.1f}%")
                if limits['memory_percent'] > self.config['monitoring']['alert_threshold_percent']:
                    logger.warning(f"⚠️ High memory usage: {limits['memory_percent']:.1f}%")
            time.sleep(self.config['monitoring']['log_interval_seconds'])

    def adaptive_load_balancing(self, current_stats: Dict[str, Any]) -> None:
        """
        Adaptive load balancing based on current execution statistics

        Args:
            current_stats: Current execution statistics
        """
        if not self.config['execution']['adaptive_scaling']:
            return

        success_rate = current_stats.get('success_rate', 0)
        avg_task_time = current_stats.get('average_task_time', 0)
        active_workers = current_stats.get('active_workers', 0)

        # Adjust worker count based on performance
        if success_rate < 0.7:  # Low success rate
            # Reduce workers to avoid resource contention
            new_workers = max(1, self.config['execution']['max_concurrent_workers'] - 1)
            logger.info(f"📉 Low success rate ({success_rate:.2f}), reducing workers to {new_workers}")
            self.config['execution']['max_concurrent_workers'] = new_workers

        elif success_rate > 0.9 and avg_task_time < 60:  # High performance
            # Increase workers if resources allow
            if self._check_resource_availability():
                new_workers = min(self.system_limits['cpu_cores'],
                                self.config['execution']['max_concurrent_workers'] + 1)
                logger.info(f"📈 High performance, increasing workers to {new_workers}")
                self.config['execution']['max_concurrent_workers'] = new_workers

    def get_execution_status(self) -> Dict[str, Any]:
        """Get current execution status"""
        return {
            'active_workers': self.active_workers,
            'completed_tasks': self.completed_tasks,
            'failed_tasks': self.failed_tasks,
            'queue_size': self.execution_queue.qsize(),
            'system_resources': self._get_system_limits(),
            'is_running': not self.stop_event.is_set(),
            'is_paused': not self.pause_event.is_set()
        }

    def pause_execution(self) -> None:
        """Pause task execution"""
        self.pause_event.clear()
        logger.info("⏸️ Execution paused")

    def resume_execution(self) -> None:
        """Resume task execution"""
        self.pause_event.set()
        logger.info("▶️ Execution resumed")

    def stop_execution(self) -> None:
        """Stop execution gracefully"""
        self.stop_event.set()
        logger.info("🛑 Execution stopping...")

    def save_execution_results(self, results: Dict[str, Any],
                             filename: str = "parallel_execution_results.json") -> None:
        """Save parallel execution results"""
        output_path = self.base_dir / "results" / filename
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)

        logger.info(f"💾 Parallel execution results saved to {output_path}")

class ResourceMonitor:
    """Monitor system resources during parallel execution"""

    def __init__(self):
        self.cpu_history = []
        self.memory_history = []
        self.start_time = time.time()

    def update(self) -> None:
        """Update resource measurements"""
        self.cpu_history.append(psutil.cpu_percent())
        self.memory_history.append(psutil.virtual_memory().percent)

        # Keep only last 100 measurements
        if len(self.cpu_history) > 100:
            self.cpu_history = self.cpu_history[-100:]
            self.memory_history = self.memory_history[-100:]

    def get_summary(self) -> Dict[str, Any]:
        """Get resource usage summary"""
        if not self.cpu_history:
            return {'error': 'No resource data available'}

        return {
            'avg_cpu_percent': np.mean(self.cpu_history),
            'max_cpu_percent': np.max(self.cpu_history),
            'avg_memory_percent': np.mean(self.memory_history),
            'max_memory_percent': np.max(self.memory_history),
            'monitoring_duration': time.time() - self.start_time,
            'samples_collected': len(self.cpu_history)
        }

def main():
    """Main function for parallel Bayesian execution"""
    try:
        # Initialize parallel executor
        executor = ParallelBayesianExecution()

        # Create optimization tasks
        tasks = executor.create_optimization_tasks(n_tasks=8)

        # Execute parallel optimization
        results = executor.execute_parallel_optimization(tasks, execution_mode='thread_pool')

        # Save results
        executor.save_execution_results(results)

        # Print summary
        print("\n" + "="*60)
        print("🎉 PARALLEL BAYESIAN EXECUTION COMPLETED")
        print("="*60)

        stats = results['execution_stats']
        print(f"📊 Total tasks: {stats['total_tasks']}")
        print(f"✅ Completed: {stats['completed_tasks']}")
        print(f"❌ Failed: {stats['failed_tasks']}")
        print(f"⏱️ Average time: {stats['average_task_time']:.2f}s")
        print(f"📈 Success rate: {stats['success_rate']:.1f}%")
        print(f"🔧 Resource utilization: {stats['resource_utilization']:.1f}%")
        print("="*60)

    except Exception as e:
        logger.error(f"Parallel execution failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()