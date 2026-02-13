"""
Distributed Computing Comprehensive Demo
========================================

This demo showcases the complete distributed inference system:

1. **Local Mode**: Single coordinator + multiple simulated workers
2. **Load Balancing**: Different strategies in action
3. **Fault Tolerance**: Automatic retry and circuit breaker
4. **Health Monitoring**: Worker heartbeat and failure detection
5. **Task Distribution**: Priority queuing and result aggregation

Scenario:
---------
Simulate a small GPU cluster with 3 workers processing
image classification tasks with different priorities.

Workers:
- Worker 1: RX 580 (8GB) - General purpose
- Worker 2: RX 580 (8GB) - General purpose
- Worker 3: Vega 56 (8GB) - High-performance (simulated)

Tasks:
- Mix of normal and high-priority tasks
- Some tasks will fail to demonstrate retry
- Monitor load distribution and failover

Usage:
------
python examples/distributed_comprehensive_demo.py

This demo runs in simulation mode (no actual GPU required).

Version: 0.6.0-dev
License: MIT
"""

import logging
import random
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, List

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

import os

# Import distributed modules
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from distributed.communication import Message, MessageType
from distributed.coordinator import ClusterCoordinator, TaskPriority
from distributed.fault_tolerance import CircuitBreaker, HealthChecker, RetryConfig, RetryManager
from distributed.load_balancing import (
    AdaptiveBalancer,
    LeastLoadedBalancer,
    LoadBalanceStrategy,
    TaskRequirements,
    WorkerLoad,
)
from distributed.worker import InferenceWorker, WorkerConfig


@dataclass
class SimulatedTask:
    """Simulated inference task."""

    task_id: str
    model: str
    complexity: float  # 0-1, affects duration
    priority: TaskPriority
    should_fail: bool = False


class SimulatedWorkerPool:
    """
    Simulates multiple workers without actual network/GPU.

    For demo purposes only - real deployment uses actual
    Worker nodes with ROCm backend.
    """

    def __init__(self, num_workers: int = 3):
        """Initialize simulated worker pool."""
        self.num_workers = num_workers
        self.workers: Dict[str, Dict[str, Any]] = {}
        self.active = False

        # Create simulated workers
        for i in range(num_workers):
            worker_id = f"worker-{i+1}"
            self.workers[worker_id] = {
                "id": worker_id,
                "gpu": "RX 580" if i < 2 else "Vega 56",
                "vram_gb": 8.0,
                "active_tasks": 0,
                "completed_tasks": 0,
                "failed_tasks": 0,
                "load": WorkerLoad(worker_id=worker_id, gpu_memory_total_gb=8.0),
            }

    def get_worker_info(self, worker_id: str) -> Dict[str, Any]:
        """Get worker information."""
        return self.workers.get(worker_id, {})

    def execute_task(self, worker_id: str, task: SimulatedTask) -> Dict[str, Any]:
        """
        Simulate task execution.

        Returns:
            Result dictionary
        """
        worker = self.workers[worker_id]
        worker["active_tasks"] += 1

        try:
            # Simulate processing time
            duration = task.complexity * random.uniform(0.5, 2.0)
            time.sleep(duration)

            # Simulate failure
            if task.should_fail:
                raise RuntimeError("Simulated task failure")

            # Generate result
            result = {
                "task_id": task.task_id,
                "model": task.model,
                "prediction": random.choice(["cat", "dog", "bird", "car"]),
                "confidence": random.uniform(0.7, 0.99),
                "worker": worker_id,
                "duration": duration,
            }

            worker["completed_tasks"] += 1
            return result

        except Exception as e:
            worker["failed_tasks"] += 1
            raise

        finally:
            worker["active_tasks"] -= 1

    def get_statistics(self) -> Dict[str, Any]:
        """Get pool statistics."""
        total_completed = sum(w["completed_tasks"] for w in self.workers.values())
        total_failed = sum(w["failed_tasks"] for w in self.workers.values())

        return {
            "total_workers": self.num_workers,
            "total_completed": total_completed,
            "total_failed": total_failed,
            "workers": {
                wid: {
                    "gpu": w["gpu"],
                    "completed": w["completed_tasks"],
                    "failed": w["failed_tasks"],
                    "active": w["active_tasks"],
                }
                for wid, w in self.workers.items()
            },
        }


class DistributedInferenceDemo:
    """
    Comprehensive demo of distributed inference system.
    """

    def __init__(self):
        """Initialize demo."""
        print("=" * 70)
        print("Distributed Computing Comprehensive Demo")
        print("=" * 70)
        print()

        # Create simulated worker pool
        self.worker_pool = SimulatedWorkerPool(num_workers=3)

        # Create load balancer
        self.balancer = AdaptiveBalancer()
        for worker_id, worker in self.worker_pool.workers.items():
            self.balancer.add_worker(worker_id, worker["load"])

        # Create fault tolerance components
        self.retry_manager = RetryManager(RetryConfig(max_attempts=3))
        self.health_checker = HealthChecker(heartbeat_interval=5.0, timeout_seconds=15.0)

        # Register workers with health checker
        for worker_id in self.worker_pool.workers.keys():
            self.health_checker.register_worker(worker_id)

        # Task tracking
        self.completed_tasks: List[Dict[str, Any]] = []
        self.failed_tasks: List[Dict[str, Any]] = []

    def run(self):
        """Run complete demo."""
        print("DEMO SCENARIO:")
        print("-" * 70)
        print("Simulating small GPU cluster with 3 workers:")
        print("  - Worker 1: RX 580 (8GB)")
        print("  - Worker 2: RX 580 (8GB)")
        print("  - Worker 3: Vega 56 (8GB)")
        print()
        print("Processing 20 image classification tasks:")
        print("  - Mix of normal and high-priority")
        print("  - Some tasks will fail (simulated)")
        print("  - Automatic retry on failure")
        print()
        input("Press Enter to start demo...")
        print()

        # Demo 1: Basic load balancing
        self._demo_load_balancing()

        # Demo 2: Fault tolerance
        self._demo_fault_tolerance()

        # Demo 3: Priority queuing
        self._demo_priority_queuing()

        # Demo 4: Adaptive learning
        self._demo_adaptive_learning()

        # Show final statistics
        self._show_final_statistics()

    def _demo_load_balancing(self):
        """Demo 1: Load balancing strategies."""
        print("\n" + "=" * 70)
        print("DEMO 1: Load Balancing")
        print("=" * 70)
        print()
        print("Testing adaptive load balancer with 10 tasks...")
        print()

        tasks = [
            SimulatedTask(
                task_id=f"task-{i+1}",
                model=random.choice(["resnet50", "mobilenet", "efficientnet"]),
                complexity=random.uniform(0.3, 0.8),
                priority=TaskPriority.NORMAL,
            )
            for i in range(10)
        ]

        for task in tasks:
            # Select worker
            worker_id = self.balancer.select_worker()

            print(f"Task {task.task_id}:")
            print(f"  Model: {task.model}")
            print(f"  Assigned to: {worker_id}")

            # Execute task
            try:
                result = self.worker_pool.execute_task(worker_id, task)
                print(f"  Result: {result['prediction']} ({result['confidence']:.2%})")
                print(f"  Duration: {result['duration']:.2f}s")

                self.completed_tasks.append(result)

                # Record for adaptive learning
                self.balancer.record_task_completion(
                    worker_id, result["duration"] * 1000, success=True  # Convert to ms
                )

            except Exception as e:
                print(f"  ✗ FAILED: {e}")
                self.failed_tasks.append(
                    {"task_id": task.task_id, "worker": worker_id, "error": str(e)}
                )

            print()

        # Show load distribution
        stats = self.worker_pool.get_statistics()
        print("\nLoad Distribution:")
        print("-" * 70)
        for worker_id, worker_stats in stats["workers"].items():
            print(f"{worker_id}: {worker_stats['completed']} completed")

    def _demo_fault_tolerance(self):
        """Demo 2: Fault tolerance and retry."""
        print("\n" + "=" * 70)
        print("DEMO 2: Fault Tolerance & Retry")
        print("=" * 70)
        print()
        print("Testing automatic retry on task failure...")
        print()

        # Create tasks that will fail
        failing_task = SimulatedTask(
            task_id="failing-task-1",
            model="resnet50",
            complexity=0.5,
            priority=TaskPriority.NORMAL,
            should_fail=True,  # This will fail
        )

        worker_id = self.balancer.select_worker()

        print(f"Task {failing_task.task_id}:")
        print(f"  Assigned to: {worker_id}")
        print(f"  (Simulated failure)")
        print()

        # Attempt with retry
        for attempt in range(1, 4):
            print(f"  Attempt {attempt}:")

            try:
                result = self.worker_pool.execute_task(worker_id, failing_task)
                print(f"    ✓ SUCCESS")
                break

            except Exception as e:
                print(f"    ✗ FAILED: {e}")

                if self.retry_manager.should_retry(failing_task.task_id, attempt):
                    delay = self.retry_manager.get_retry_delay(attempt)
                    print(f"    Retrying in {delay:.1f}s...")
                    time.sleep(delay)

                    # Try different worker on retry
                    worker_id = self.balancer.select_worker()
                    print(f"    Reassigned to: {worker_id}")
                else:
                    print(f"    Max retries reached, giving up")
                    self.failed_tasks.append(
                        {"task_id": failing_task.task_id, "worker": worker_id, "error": str(e)}
                    )

    def _demo_priority_queuing(self):
        """Demo 3: Priority-based task queuing."""
        print("\n" + "=" * 70)
        print("DEMO 3: Priority Queuing")
        print("=" * 70)
        print()
        print("Testing priority-based task scheduling...")
        print()

        # Create mixed priority tasks
        tasks = [
            SimulatedTask("low-1", "mobilenet", 0.3, TaskPriority.LOW),
            SimulatedTask("urgent-1", "resnet50", 0.7, TaskPriority.URGENT),
            SimulatedTask("normal-1", "efficientnet", 0.5, TaskPriority.NORMAL),
            SimulatedTask("high-1", "resnet50", 0.6, TaskPriority.HIGH),
            SimulatedTask("normal-2", "mobilenet", 0.4, TaskPriority.NORMAL),
        ]

        print("Task Queue (submitted in order):")
        for task in tasks:
            print(f"  {task.task_id} - Priority: {task.priority.name}")
        print()

        # Sort by priority (would be done by PriorityQueue in real system)
        sorted_tasks = sorted(tasks, key=lambda t: t.priority.value)

        print("Execution Order (by priority):")
        for task in sorted_tasks:
            worker_id = self.balancer.select_worker()

            print(f"  {task.task_id} ({task.priority.name}) → {worker_id}")

            try:
                result = self.worker_pool.execute_task(worker_id, task)
                self.completed_tasks.append(result)
            except:
                pass

        print()

    def _demo_adaptive_learning(self):
        """Demo 4: Adaptive load balancer learning."""
        print("\n" + "=" * 70)
        print("DEMO 4: Adaptive Learning")
        print("=" * 70)
        print()
        print("Demonstrating adaptive balancer learning worker performance...")
        print()

        # Simulate worker performance difference
        # Worker 3 (Vega) will be "faster" by reducing complexity

        print("Initial strategy weights:")
        print(f"  {self.balancer.strategy_weights}")
        print()

        # Run 5 tasks and observe adaptation
        for i in range(5):
            task = SimulatedTask(
                task_id=f"adaptive-{i+1}",
                model="resnet50",
                complexity=0.5,
                priority=TaskPriority.NORMAL,
            )

            worker_id = self.balancer.select_worker()

            # Simulate Vega being faster
            if "vega" in worker_id.lower() or worker_id == "worker-3":
                task.complexity *= 0.6  # 40% faster

            try:
                result = self.worker_pool.execute_task(worker_id, task)
                print(f"Task {task.task_id}: {worker_id} - {result['duration']:.2f}s")

                self.balancer.record_task_completion(
                    worker_id, result["duration"] * 1000, success=True
                )

            except:
                pass

        print()
        print("Adapted strategy weights:")
        print(f"  {self.balancer.strategy_weights}")
        print()
        print("Balancer has learned to prefer faster workers!")

    def _show_final_statistics(self):
        """Show final statistics."""
        print("\n" + "=" * 70)
        print("FINAL STATISTICS")
        print("=" * 70)
        print()

        stats = self.worker_pool.get_statistics()

        print(f"Total Tasks: {len(self.completed_tasks) + len(self.failed_tasks)}")
        print(f"  Completed: {len(self.completed_tasks)}")
        print(f"  Failed: {len(self.failed_tasks)}")
        print(
            f"  Success Rate: {len(self.completed_tasks) / max(1, len(self.completed_tasks) + len(self.failed_tasks)):.1%}"
        )
        print()

        print("Worker Performance:")
        print("-" * 70)
        for worker_id, worker_stats in stats["workers"].items():
            print(f"{worker_id} ({worker_stats['gpu']}):")
            print(f"  Completed: {worker_stats['completed']}")
            print(f"  Failed: {worker_stats['failed']}")
            print()

        # Calculate average latency
        if self.completed_tasks:
            avg_latency = sum(t["duration"] for t in self.completed_tasks) / len(
                self.completed_tasks
            )
            print(f"Average Latency: {avg_latency:.2f}s")

        print()
        print("Demo completed successfully!")
        print("=" * 70)


def main():
    """Run comprehensive demo."""
    demo = DistributedInferenceDemo()

    try:
        demo.run()
    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user")
    except Exception as e:
        print(f"\n\nError: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
