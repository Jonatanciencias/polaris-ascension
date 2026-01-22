"""
Load Balancing System for Distributed Inference
==============================================

This module provides intelligent load balancing strategies for
distributing inference tasks across multiple GPU workers.

Strategies:
----------
1. Round Robin - Simple rotation through workers
2. Least Loaded - Send to worker with fewest active tasks
3. GPU Match - Match task to GPU with best capabilities
4. Latency-Based - Prefer workers with lowest latency
5. Memory-Aware - Consider available GPU memory

Performance Considerations:
--------------------------
- Batch size optimization per GPU
- Network bandwidth estimation
- Task complexity estimation
- Worker capability profiling

Version: 0.6.0-dev
License: MIT
"""

import time
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging
from collections import deque
import heapq

logger = logging.getLogger(__name__)


class LoadBalanceStrategy(Enum):
    """Load balancing strategies."""
    ROUND_ROBIN = "round_robin"
    LEAST_LOADED = "least_loaded"
    GPU_MATCH = "gpu_match"
    LATENCY_BASED = "latency_based"
    MEMORY_AWARE = "memory_aware"
    ADAPTIVE = "adaptive"


@dataclass
class WorkerLoad:
    """
    Tracks load metrics for a worker.
    
    Attributes:
        worker_id: Worker identifier
        active_tasks: Number of currently running tasks
        queue_length: Number of queued tasks
        avg_latency_ms: Average task latency
        gpu_memory_used_gb: GPU memory in use
        gpu_memory_total_gb: Total GPU memory
        gpu_utilization: GPU utilization (0-1)
        last_update: Last update timestamp
    """
    worker_id: str
    active_tasks: int = 0
    queue_length: int = 0
    avg_latency_ms: float = 0.0
    gpu_memory_used_gb: float = 0.0
    gpu_memory_total_gb: float = 8.0
    gpu_utilization: float = 0.0
    last_update: float = field(default_factory=time.time)
    
    @property
    def memory_available_gb(self) -> float:
        """Get available GPU memory."""
        return self.gpu_memory_total_gb - self.gpu_memory_used_gb
    
    @property
    def load_score(self) -> float:
        """
        Calculate overall load score (0-1, higher = more loaded).
        
        Combines multiple metrics:
        - Active tasks weight: 0.4
        - GPU utilization weight: 0.3
        - Memory usage weight: 0.2
        - Queue length weight: 0.1
        """
        # Normalize active tasks (assume max 10 concurrent)
        task_score = min(self.active_tasks / 10.0, 1.0)
        
        # GPU utilization is already 0-1
        gpu_score = self.gpu_utilization
        
        # Memory usage ratio
        memory_score = self.gpu_memory_used_gb / self.gpu_memory_total_gb
        
        # Queue length (assume max 20 queued)
        queue_score = min(self.queue_length / 20.0, 1.0)
        
        # Weighted combination
        return (
            0.4 * task_score +
            0.3 * gpu_score +
            0.2 * memory_score +
            0.1 * queue_score
        )


@dataclass
class TaskRequirements:
    """
    Requirements for an inference task.
    
    Attributes:
        min_vram_gb: Minimum VRAM required
        preferred_gpu_family: Preferred GPU family (Polaris, Vega, etc.)
        batch_size: Batch size for inference
        priority: Task priority (0-10, higher = more important)
        timeout_seconds: Maximum execution time
    """
    min_vram_gb: float = 2.0
    preferred_gpu_family: Optional[str] = None
    batch_size: int = 1
    priority: int = 5
    timeout_seconds: float = 300.0


class RoundRobinBalancer:
    """
    Simple round-robin load balancer.
    
    Rotates through workers in sequence. Simple and fair,
    but doesn't consider actual load.
    
    Example:
        balancer = RoundRobinBalancer(["worker1", "worker2", "worker3"])
        
        for task in tasks:
            worker_id = balancer.select_worker()
            send_task_to_worker(worker_id, task)
    """
    
    def __init__(self, worker_ids: List[str]):
        """
        Initialize round-robin balancer.
        
        Args:
            worker_ids: List of available worker IDs
        """
        self.worker_ids = worker_ids.copy()
        self.current_index = 0
    
    def select_worker(self, task_requirements: Optional[TaskRequirements] = None) -> Optional[str]:
        """
        Select next worker in rotation.
        
        Args:
            task_requirements: Task requirements (ignored in round-robin)
        
        Returns:
            Selected worker ID or None if no workers
        """
        if not self.worker_ids:
            return None
        
        worker_id = self.worker_ids[self.current_index]
        self.current_index = (self.current_index + 1) % len(self.worker_ids)
        
        return worker_id
    
    def add_worker(self, worker_id: str):
        """Add worker to rotation."""
        if worker_id not in self.worker_ids:
            self.worker_ids.append(worker_id)
    
    def remove_worker(self, worker_id: str):
        """Remove worker from rotation."""
        if worker_id in self.worker_ids:
            self.worker_ids.remove(worker_id)
            # Adjust index if needed
            if self.current_index >= len(self.worker_ids) and self.worker_ids:
                self.current_index = 0


class LeastLoadedBalancer:
    """
    Least-loaded load balancer.
    
    Always selects the worker with the lowest current load.
    Provides better resource utilization than round-robin.
    
    Example:
        balancer = LeastLoadedBalancer()
        balancer.add_worker("worker1", WorkerLoad(worker_id="worker1"))
        balancer.add_worker("worker2", WorkerLoad(worker_id="worker2"))
        
        # Select least loaded worker
        worker_id = balancer.select_worker()
    """
    
    def __init__(self):
        """Initialize least-loaded balancer."""
        self.worker_loads: Dict[str, WorkerLoad] = {}
    
    def add_worker(self, worker_id: str, load: WorkerLoad):
        """
        Add worker with initial load.
        
        Args:
            worker_id: Worker identifier
            load: Worker load metrics
        """
        self.worker_loads[worker_id] = load
    
    def remove_worker(self, worker_id: str):
        """Remove worker."""
        if worker_id in self.worker_loads:
            del self.worker_loads[worker_id]
    
    def update_load(self, worker_id: str, load: WorkerLoad):
        """
        Update worker load metrics.
        
        Args:
            worker_id: Worker identifier
            load: Updated load metrics
        """
        if worker_id in self.worker_loads:
            self.worker_loads[worker_id] = load
    
    def select_worker(self, task_requirements: Optional[TaskRequirements] = None) -> Optional[str]:
        """
        Select worker with lowest load.
        
        Args:
            task_requirements: Task requirements (considered if provided)
        
        Returns:
            Selected worker ID or None if no workers
        """
        if not self.worker_loads:
            return None
        
        # Filter workers by requirements
        eligible_workers = self._filter_by_requirements(task_requirements)
        
        if not eligible_workers:
            # Fall back to all workers if none match requirements
            eligible_workers = list(self.worker_loads.keys())
        
        # Select worker with lowest load score
        best_worker = min(
            eligible_workers,
            key=lambda wid: self.worker_loads[wid].load_score
        )
        
        return best_worker
    
    def _filter_by_requirements(
        self,
        requirements: Optional[TaskRequirements]
    ) -> List[str]:
        """Filter workers matching task requirements."""
        if requirements is None:
            return list(self.worker_loads.keys())
        
        eligible = []
        
        for worker_id, load in self.worker_loads.items():
            # Check memory requirement
            if load.memory_available_gb < requirements.min_vram_gb:
                continue
            
            eligible.append(worker_id)
        
        return eligible
    
    def get_load_distribution(self) -> Dict[str, float]:
        """
        Get load scores for all workers.
        
        Returns:
            Dictionary of worker_id -> load_score
        """
        return {
            worker_id: load.load_score
            for worker_id, load in self.worker_loads.items()
        }


class GPUMatchBalancer:
    """
    GPU capability matching balancer.
    
    Matches tasks to GPUs based on hardware capabilities:
    - Matches task complexity to GPU performance
    - Considers GPU architecture features
    - Optimizes for specific model types
    
    Example:
        balancer = GPUMatchBalancer()
        balancer.add_worker("rx580", gpu_family="Polaris", vram_gb=8.0)
        balancer.add_worker("vega56", gpu_family="Vega", vram_gb=8.0)
        
        # Task requiring FP16 goes to Vega
        requirements = TaskRequirements(preferred_gpu_family="Vega")
        worker_id = balancer.select_worker(requirements)
    """
    
    def __init__(self):
        """Initialize GPU match balancer."""
        self.worker_capabilities: Dict[str, Dict[str, Any]] = {}
        self.worker_loads: Dict[str, WorkerLoad] = {}
    
    def add_worker(
        self,
        worker_id: str,
        gpu_family: str,
        vram_gb: float,
        compute_units: int = 36,
        fp16_support: bool = False
    ):
        """
        Add worker with GPU capabilities.
        
        Args:
            worker_id: Worker identifier
            gpu_family: GPU family (Polaris, Vega, etc.)
            vram_gb: VRAM in GB
            compute_units: Number of compute units
            fp16_support: Whether GPU supports FP16
        """
        self.worker_capabilities[worker_id] = {
            'gpu_family': gpu_family,
            'vram_gb': vram_gb,
            'compute_units': compute_units,
            'fp16_support': fp16_support,
        }
        
        self.worker_loads[worker_id] = WorkerLoad(
            worker_id=worker_id,
            gpu_memory_total_gb=vram_gb
        )
    
    def remove_worker(self, worker_id: str):
        """Remove worker."""
        if worker_id in self.worker_capabilities:
            del self.worker_capabilities[worker_id]
        if worker_id in self.worker_loads:
            del self.worker_loads[worker_id]
    
    def update_load(self, worker_id: str, load: WorkerLoad):
        """Update worker load."""
        if worker_id in self.worker_loads:
            self.worker_loads[worker_id] = load
    
    def select_worker(self, task_requirements: Optional[TaskRequirements] = None) -> Optional[str]:
        """
        Select best-matching worker for task.
        
        Args:
            task_requirements: Task requirements
        
        Returns:
            Selected worker ID or None if no workers
        """
        if not self.worker_capabilities:
            return None
        
        if task_requirements is None:
            # No requirements, use least loaded
            return min(
                self.worker_loads.keys(),
                key=lambda wid: self.worker_loads[wid].load_score
            )
        
        # Score each worker
        scores = {}
        
        for worker_id in self.worker_capabilities.keys():
            score = self._calculate_match_score(worker_id, task_requirements)
            scores[worker_id] = score
        
        # Select worker with highest score
        if scores:
            best_worker = max(scores.keys(), key=lambda wid: scores[wid])
            return best_worker
        
        return None
    
    def _calculate_match_score(
        self,
        worker_id: str,
        requirements: TaskRequirements
    ) -> float:
        """
        Calculate how well worker matches task requirements.
        
        Returns score 0-1, higher is better match.
        """
        caps = self.worker_capabilities[worker_id]
        load = self.worker_loads[worker_id]
        
        score = 1.0
        
        # Memory check (required)
        if load.memory_available_gb < requirements.min_vram_gb:
            return 0.0  # Can't run this task
        
        # GPU family preference
        if requirements.preferred_gpu_family:
            if caps['gpu_family'] == requirements.preferred_gpu_family:
                score *= 1.2
            else:
                score *= 0.8
        
        # Current load penalty
        score *= (1.0 - load.load_score * 0.5)
        
        return score


class AdaptiveBalancer:
    """
    Adaptive load balancer that learns from task execution.
    
    Monitors task performance on different workers and
    adapts strategy based on observed patterns.
    
    Features:
    - Learns worker performance characteristics
    - Adapts to changing conditions
    - Combines multiple strategies
    - Tracks task completion times
    
    Example:
        balancer = AdaptiveBalancer()
        balancer.add_worker("worker1", WorkerLoad(worker_id="worker1"))
        
        # Record task execution
        worker_id = balancer.select_worker()
        result = execute_task(worker_id, task)
        balancer.record_task_completion(worker_id, task, result.latency_ms)
    """
    
    def __init__(self, adaptation_rate: float = 0.1):
        """
        Initialize adaptive balancer.
        
        Args:
            adaptation_rate: How quickly to adapt (0-1)
        """
        self.adaptation_rate = adaptation_rate
        self.worker_loads: Dict[str, WorkerLoad] = {}
        
        # Performance history
        self.task_history: Dict[str, deque] = {}  # worker_id -> deque of latencies
        self.max_history_size = 100
        
        # Strategy weights (learned over time)
        self.strategy_weights = {
            'least_loaded': 0.5,
            'latency': 0.3,
            'memory': 0.2,
        }
    
    def add_worker(self, worker_id: str, load: WorkerLoad):
        """Add worker."""
        self.worker_loads[worker_id] = load
        self.task_history[worker_id] = deque(maxlen=self.max_history_size)
    
    def remove_worker(self, worker_id: str):
        """Remove worker."""
        if worker_id in self.worker_loads:
            del self.worker_loads[worker_id]
        if worker_id in self.task_history:
            del self.task_history[worker_id]
    
    def update_load(self, worker_id: str, load: WorkerLoad):
        """Update worker load."""
        if worker_id in self.worker_loads:
            self.worker_loads[worker_id] = load
    
    def select_worker(self, task_requirements: Optional[TaskRequirements] = None) -> Optional[str]:
        """
        Select worker using adaptive strategy.
        
        Args:
            task_requirements: Task requirements
        
        Returns:
            Selected worker ID
        """
        if not self.worker_loads:
            return None
        
        # Calculate composite scores for each worker
        scores = {}
        
        for worker_id in self.worker_loads.keys():
            score = self._calculate_adaptive_score(worker_id, task_requirements)
            scores[worker_id] = score
        
        # Select worker with best score
        best_worker = max(scores.keys(), key=lambda wid: scores[wid])
        return best_worker
    
    def _calculate_adaptive_score(
        self,
        worker_id: str,
        requirements: Optional[TaskRequirements]
    ) -> float:
        """Calculate adaptive score for worker."""
        load = self.worker_loads[worker_id]
        
        # Component scores
        load_score = 1.0 - load.load_score  # Invert so higher = better
        
        # Latency score (based on history)
        if self.task_history[worker_id]:
            avg_latency = sum(self.task_history[worker_id]) / len(self.task_history[worker_id])
            # Normalize to 0-1 (assume 1000ms is slow)
            latency_score = max(0.0, 1.0 - avg_latency / 1000.0)
        else:
            latency_score = 0.5  # Neutral for unknown workers
        
        # Memory score
        memory_score = load.memory_available_gb / load.gpu_memory_total_gb
        
        # Combine with learned weights
        composite_score = (
            self.strategy_weights['least_loaded'] * load_score +
            self.strategy_weights['latency'] * latency_score +
            self.strategy_weights['memory'] * memory_score
        )
        
        # Apply requirements filter
        if requirements and load.memory_available_gb < requirements.min_vram_gb:
            composite_score = 0.0
        
        return composite_score
    
    def record_task_completion(
        self,
        worker_id: str,
        latency_ms: float,
        success: bool = True
    ):
        """
        Record task completion for learning.
        
        Args:
            worker_id: Worker that executed task
            latency_ms: Task latency in milliseconds
            success: Whether task succeeded
        """
        if worker_id not in self.task_history:
            return
        
        # Record latency (penalize failures)
        recorded_latency = latency_ms if success else latency_ms * 2.0
        self.task_history[worker_id].append(recorded_latency)
        
        # Adapt strategy weights based on performance
        self._adapt_weights(worker_id, latency_ms, success)
    
    def _adapt_weights(self, worker_id: str, latency_ms: float, success: bool):
        """Adapt strategy weights based on performance."""
        # Simple adaptation: increase weight of latency if tasks are slow
        if latency_ms > 500.0:  # Slow task
            self.strategy_weights['latency'] += self.adaptation_rate * 0.1
            self.strategy_weights['least_loaded'] -= self.adaptation_rate * 0.05
            self.strategy_weights['memory'] -= self.adaptation_rate * 0.05
        
        # Normalize weights
        total = sum(self.strategy_weights.values())
        for key in self.strategy_weights:
            self.strategy_weights[key] /= total


# Demo code
if __name__ == "__main__":
    print("=" * 70)
    print("Load Balancing System Demo")
    print("=" * 70)
    
    # Create sample workers
    workers = [
        WorkerLoad("worker1", active_tasks=2, gpu_memory_used_gb=3.0, gpu_utilization=0.4),
        WorkerLoad("worker2", active_tasks=5, gpu_memory_used_gb=6.0, gpu_utilization=0.8),
        WorkerLoad("worker3", active_tasks=1, gpu_memory_used_gb=2.0, gpu_utilization=0.2),
    ]
    
    print("\nWorker States:")
    print("-" * 70)
    for worker in workers:
        print(f"{worker.worker_id}:")
        print(f"  Active tasks: {worker.active_tasks}")
        print(f"  GPU utilization: {worker.gpu_utilization:.1%}")
        print(f"  Memory: {worker.gpu_memory_used_gb:.1f}/{worker.gpu_memory_total_gb:.1f} GB")
        print(f"  Load score: {worker.load_score:.2f}")
    
    # Test least-loaded balancer
    print("\nLeast-Loaded Balancer:")
    print("-" * 70)
    balancer = LeastLoadedBalancer()
    for worker in workers:
        balancer.add_worker(worker.worker_id, worker)
    
    for i in range(5):
        selected = balancer.select_worker()
        print(f"Task {i+1} → {selected}")
    
    # Test GPU match balancer
    print("\nGPU Match Balancer:")
    print("-" * 70)
    gpu_balancer = GPUMatchBalancer()
    gpu_balancer.add_worker("rx580", "Polaris", 8.0, 36, False)
    gpu_balancer.add_worker("vega56", "Vega", 8.0, 56, True)
    
    # Task preferring Vega
    requirements = TaskRequirements(preferred_gpu_family="Vega", min_vram_gb=4.0)
    selected = gpu_balancer.select_worker(requirements)
    print(f"Vega-preferred task → {selected}")
    
    print("\n" + "=" * 70)
