"""
Cluster Coordinator for Distributed Inference
=============================================

The Coordinator is the central manager for a distributed inference cluster.
It handles:
- Worker registration and discovery
- Task distribution and load balancing
- Health monitoring and failover
- Result aggregation

Architecture:
------------
                ┌──────────────────┐
                │   Coordinator    │
                │  (This Module)   │
                └────────┬─────────┘
                         │
         ┌───────────────┼───────────────┐
         │               │               │
    ┌────▼────┐    ┌────▼────┐    ┌────▼────┐
    │ Worker1 │    │ Worker2 │    │ Worker3 │
    │ (RX 580)│    │ (RX 580)│    │ (Vega)  │
    └─────────┘    └─────────┘    └─────────┘

Communication Flow:
------------------
1. Workers register with coordinator on startup
2. Coordinator assigns tasks based on load balancing strategy
3. Workers send heartbeats to coordinator
4. Workers return results to coordinator
5. Coordinator aggregates results for client

Version: 0.6.0-dev
License: MIT
"""

import time
import uuid
import logging
import threading
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
from queue import Queue, Empty, PriorityQueue

# Import our distributed modules
try:
    from .communication import Message, MessageType, MessageRouter, ConnectionPool
    from .load_balancing import (
        LoadBalanceStrategy, WorkerLoad, TaskRequirements,
        LeastLoadedBalancer, GPUMatchBalancer, AdaptiveBalancer
    )
    from .fault_tolerance import (
        RetryManager, RetryConfig, CircuitBreaker,
        HealthChecker, FailureType
    )
except ImportError:
    # For standalone execution
    import sys
    import os
    sys.path.insert(0, os.path.dirname(__file__))
    from communication import Message, MessageType, MessageRouter, ConnectionPool
    from load_balancing import (
        LoadBalanceStrategy, WorkerLoad, TaskRequirements,
        LeastLoadedBalancer, GPUMatchBalancer, AdaptiveBalancer
    )
    from fault_tolerance import (
        RetryManager, RetryConfig, CircuitBreaker,
        HealthChecker, FailureType
    )

logger = logging.getLogger(__name__)


class TaskPriority(Enum):
    """Task priority levels."""
    LOW = 3
    NORMAL = 2
    HIGH = 1
    URGENT = 0


@dataclass
class WorkerInfo:
    """
    Information about a registered worker.
    
    Attributes:
        worker_id: Unique worker identifier
        address: Network address (host:port)
        gpu_name: GPU model name
        gpu_memory_gb: Total GPU memory
        capabilities: Worker capabilities dict
        status: Current worker status
        registered_at: Registration timestamp
        last_heartbeat: Last heartbeat timestamp
    """
    worker_id: str
    address: str
    gpu_name: str = "Unknown"
    gpu_memory_gb: float = 8.0
    capabilities: Dict[str, Any] = field(default_factory=dict)
    status: str = "ready"
    registered_at: float = field(default_factory=time.time)
    last_heartbeat: float = field(default_factory=time.time)
    
    @property
    def uptime_seconds(self) -> float:
        """Get worker uptime."""
        return time.time() - self.registered_at


@dataclass
class Task:
    """
    Distributed inference task.
    
    Attributes:
        task_id: Unique task identifier
        payload: Task data (model, input, config)
        requirements: Task requirements
        priority: Task priority
        created_at: Creation timestamp
        assigned_worker: Worker assigned to task
        started_at: Execution start timestamp
        completed_at: Completion timestamp
        result: Task result
        error: Error message if failed
    """
    task_id: str
    payload: Dict[str, Any]
    requirements: Optional[TaskRequirements] = None
    priority: TaskPriority = TaskPriority.NORMAL
    created_at: float = field(default_factory=time.time)
    assigned_worker: Optional[str] = None
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    result: Optional[Any] = None
    error: Optional[str] = None
    
    @property
    def is_completed(self) -> bool:
        """Check if task completed."""
        return self.completed_at is not None
    
    @property
    def duration_seconds(self) -> Optional[float]:
        """Get task duration."""
        if self.started_at and self.completed_at:
            return self.completed_at - self.started_at
        return None
    
    def __lt__(self, other):
        """Compare by priority for priority queue."""
        return self.priority.value < other.priority.value


class ClusterCoordinator:
    """
    Central coordinator for distributed inference cluster.
    
    Manages workers, distributes tasks, monitors health, and
    aggregates results.
    
    Example:
        # Start coordinator
        coordinator = ClusterCoordinator(
            bind_address="tcp://0.0.0.0:5555",
            strategy=LoadBalanceStrategy.ADAPTIVE
        )
        
        coordinator.start()
        
        # Submit tasks
        task_id = coordinator.submit_task({
            "model": "resnet50",
            "input": image_data,
        })
        
        # Wait for result
        result = coordinator.get_result(task_id, timeout=30.0)
        
        coordinator.stop()
    """
    
    def __init__(
        self,
        bind_address: str = "tcp://0.0.0.0:5555",
        strategy: LoadBalanceStrategy = LoadBalanceStrategy.ADAPTIVE,
        heartbeat_interval: float = 10.0,
        heartbeat_timeout: float = 30.0,
        enable_retry: bool = True,
        max_retries: int = 3,
    ):
        """
        Initialize cluster coordinator.
        
        Args:
            bind_address: ZeroMQ bind address
            strategy: Load balancing strategy
            heartbeat_interval: Expected heartbeat interval
            heartbeat_timeout: Heartbeat timeout threshold
            enable_retry: Enable automatic retry on failure
            max_retries: Maximum retry attempts
        """
        self.bind_address = bind_address
        self.strategy = strategy
        self.running = False
        
        # Communication
        self.router: Optional[MessageRouter] = None
        self.conn_pool: Optional[ConnectionPool] = None
        
        # Workers
        self.workers: Dict[str, WorkerInfo] = {}
        self.worker_loads: Dict[str, WorkerLoad] = {}
        
        # Load balancing
        if strategy == LoadBalanceStrategy.ADAPTIVE:
            self.balancer = AdaptiveBalancer()
        elif strategy == LoadBalanceStrategy.GPU_MATCH:
            self.balancer = GPUMatchBalancer()
        else:  # Default to least loaded
            self.balancer = LeastLoadedBalancer()
        
        # Fault tolerance
        self.health_checker = HealthChecker(heartbeat_interval, heartbeat_timeout)
        
        if enable_retry:
            retry_config = RetryConfig(max_attempts=max_retries)
            self.retry_manager = RetryManager(retry_config)
        else:
            self.retry_manager = None
        
        # Task management
        self.pending_tasks: PriorityQueue = PriorityQueue()
        self.active_tasks: Dict[str, Task] = {}
        self.completed_tasks: Dict[str, Task] = {}
        
        # Threading
        self.lock = threading.RLock()
        self.coordinator_thread: Optional[threading.Thread] = None
        self.heartbeat_thread: Optional[threading.Thread] = None
        
        # Statistics
        self.stats = {
            'tasks_submitted': 0,
            'tasks_completed': 0,
            'tasks_failed': 0,
            'workers_registered': 0,
            'workers_failed': 0,
        }
    
    def start(self):
        """Start coordinator service."""
        if self.running:
            logger.warning("Coordinator already running")
            return
        
        logger.info(f"Starting coordinator on {self.bind_address}")
        
        # Initialize communication
        try:
            self.router = MessageRouter(self.bind_address, is_server=True)
            self.conn_pool = ConnectionPool()
        except Exception as e:
            logger.error(f"Failed to initialize communication: {e}")
            raise
        
        self.running = True
        
        # Start coordinator thread
        self.coordinator_thread = threading.Thread(
            target=self._coordinator_loop,
            daemon=True
        )
        self.coordinator_thread.start()
        
        # Start heartbeat monitoring thread
        self.heartbeat_thread = threading.Thread(
            target=self._heartbeat_loop,
            daemon=True
        )
        self.heartbeat_thread.start()
        
        logger.info("Coordinator started successfully")
    
    def stop(self):
        """Stop coordinator service."""
        logger.info("Stopping coordinator")
        self.running = False
        
        # Wait for threads
        if self.coordinator_thread:
            self.coordinator_thread.join(timeout=5.0)
        if self.heartbeat_thread:
            self.heartbeat_thread.join(timeout=5.0)
        
        # Close communication
        if self.router:
            self.router.close()
        if self.conn_pool:
            self.conn_pool.close()
        
        logger.info("Coordinator stopped")
    
    def submit_task(
        self,
        payload: Dict[str, Any],
        requirements: Optional[TaskRequirements] = None,
        priority: TaskPriority = TaskPriority.NORMAL
    ) -> str:
        """
        Submit task for execution.
        
        Args:
            payload: Task data dictionary
            requirements: Task requirements
            priority: Task priority
        
        Returns:
            Task ID
        """
        task_id = str(uuid.uuid4())
        
        task = Task(
            task_id=task_id,
            payload=payload,
            requirements=requirements,
            priority=priority
        )
        
        with self.lock:
            self.pending_tasks.put(task)
            self.stats['tasks_submitted'] += 1
        
        logger.info(f"Task submitted: {task_id} (priority={priority.value})")
        return task_id
    
    def get_result(self, task_id: str, timeout: Optional[float] = None) -> Optional[Any]:
        """
        Get task result (blocking).
        
        Args:
            task_id: Task identifier
            timeout: Maximum wait time (None = wait forever)
        
        Returns:
            Task result or None if timeout/error
        """
        start_time = time.time()
        
        while True:
            with self.lock:
                # Check completed tasks
                if task_id in self.completed_tasks:
                    task = self.completed_tasks[task_id]
                    if task.error:
                        raise RuntimeError(f"Task failed: {task.error}")
                    return task.result
            
            # Check timeout
            if timeout is not None:
                elapsed = time.time() - start_time
                if elapsed >= timeout:
                    return None
            
            # Wait a bit
            time.sleep(0.1)
    
    def get_worker_stats(self) -> Dict[str, Any]:
        """Get worker statistics."""
        with self.lock:
            healthy = [wid for wid in self.workers if self.health_checker.is_healthy(wid)]
            
            return {
                'total_workers': len(self.workers),
                'healthy_workers': len(healthy),
                'unhealthy_workers': len(self.workers) - len(healthy),
                'workers': {
                    wid: {
                        'status': worker.status,
                        'uptime': worker.uptime_seconds,
                        'gpu': worker.gpu_name,
                        'load': self.worker_loads.get(wid, WorkerLoad(wid)).load_score
                    }
                    for wid, worker in self.workers.items()
                }
            }
    
    def get_task_stats(self) -> Dict[str, Any]:
        """Get task statistics."""
        with self.lock:
            return {
                'submitted': self.stats['tasks_submitted'],
                'completed': self.stats['tasks_completed'],
                'failed': self.stats['tasks_failed'],
                'pending': self.pending_tasks.qsize(),
                'active': len(self.active_tasks),
                'success_rate': (
                    self.stats['tasks_completed'] / max(1, self.stats['tasks_submitted'])
                )
            }
    
    def _coordinator_loop(self):
        """Main coordinator event loop."""
        logger.info("Coordinator loop started")
        
        while self.running:
            try:
                # Receive messages from workers
                sender_id, message = self.router.receive(timeout=0.1)
                
                if message:
                    self._handle_message(sender_id, message)
                
                # Assign pending tasks
                self._assign_tasks()
                
            except Exception as e:
                logger.error(f"Error in coordinator loop: {e}")
                time.sleep(0.1)
        
        logger.info("Coordinator loop stopped")
    
    def _heartbeat_loop(self):
        """Heartbeat monitoring loop."""
        logger.info("Heartbeat monitor started")
        
        while self.running:
            try:
                # Check for unhealthy workers
                unhealthy = self.health_checker.get_unhealthy_workers()
                
                for worker_id in unhealthy:
                    self._handle_worker_failure(worker_id)
                
                time.sleep(5.0)
                
            except Exception as e:
                logger.error(f"Error in heartbeat loop: {e}")
                time.sleep(1.0)
        
        logger.info("Heartbeat monitor stopped")
    
    def _handle_message(self, sender_id: str, message: Message):
        """Handle incoming message from worker."""
        
        if message.type == MessageType.REGISTER:
            self._handle_worker_registration(sender_id, message)
        
        elif message.type == MessageType.HEARTBEAT:
            self._handle_heartbeat(sender_id, message)
        
        elif message.type == MessageType.RESULT:
            self._handle_task_result(sender_id, message)
        
        elif message.type == MessageType.ERROR:
            self._handle_task_error(sender_id, message)
        
        else:
            logger.warning(f"Unknown message type from {sender_id}: {message.type}")
    
    def _handle_worker_registration(self, worker_id: str, message: Message):
        """Handle worker registration."""
        logger.info(f"Worker registered: {worker_id}")
        
        # Extract worker info
        data = message.payload or {}
        
        worker_info = WorkerInfo(
            worker_id=worker_id,
            address=data.get('address', 'unknown'),
            gpu_name=data.get('gpu_name', 'Unknown'),
            gpu_memory_gb=data.get('gpu_memory_gb', 8.0),
            capabilities=data.get('capabilities', {})
        )
        
        with self.lock:
            self.workers[worker_id] = worker_info
            self.worker_loads[worker_id] = WorkerLoad(
                worker_id=worker_id,
                gpu_memory_total_gb=worker_info.gpu_memory_gb
            )
            
            # Register with health checker
            self.health_checker.register_worker(worker_id)
            
            # Add to load balancer
            if hasattr(self.balancer, 'add_worker'):
                self.balancer.add_worker(worker_id, self.worker_loads[worker_id])
            
            self.stats['workers_registered'] += 1
        
        # Send ACK
        ack = Message(type=MessageType.ACK, payload={'status': 'registered'})
        self.router.send(worker_id, ack)
    
    def _handle_heartbeat(self, worker_id: str, message: Message):
        """Handle worker heartbeat."""
        self.health_checker.record_heartbeat(worker_id)
        
        # Update worker load
        if message.payload:
            load_data = message.payload
            with self.lock:
                if worker_id in self.worker_loads:
                    load = self.worker_loads[worker_id]
                    load.active_tasks = load_data.get('active_tasks', 0)
                    load.gpu_utilization = load_data.get('gpu_utilization', 0.0)
                    load.gpu_memory_used_gb = load_data.get('gpu_memory_used', 0.0)
                    load.last_update = time.time()
                    
                    # Update balancer
                    if hasattr(self.balancer, 'update_load'):
                        self.balancer.update_load(worker_id, load)
    
    def _handle_task_result(self, worker_id: str, message: Message):
        """Handle task completion result."""
        data = message.payload or {}
        task_id = data.get('task_id')
        
        if not task_id:
            logger.warning("Received result without task_id")
            return
        
        with self.lock:
            if task_id not in self.active_tasks:
                logger.warning(f"Received result for unknown task: {task_id}")
                return
            
            task = self.active_tasks[task_id]
            task.completed_at = time.time()
            task.result = data.get('result')
            
            # Move to completed
            del self.active_tasks[task_id]
            self.completed_tasks[task_id] = task
            
            self.stats['tasks_completed'] += 1
            
            # Update load
            if worker_id in self.worker_loads:
                self.worker_loads[worker_id].active_tasks -= 1
        
        logger.info(f"Task completed: {task_id} by {worker_id}")
        
        # Record success for adaptive balancer
        if isinstance(self.balancer, AdaptiveBalancer):
            duration_ms = (task.duration_seconds or 0) * 1000
            self.balancer.record_task_completion(worker_id, duration_ms, success=True)
    
    def _handle_task_error(self, worker_id: str, message: Message):
        """Handle task execution error."""
        data = message.payload or {}
        task_id = data.get('task_id')
        error_msg = data.get('error', 'Unknown error')
        
        if not task_id:
            logger.warning("Received error without task_id")
            return
        
        with self.lock:
            if task_id not in self.active_tasks:
                logger.warning(f"Received error for unknown task: {task_id}")
                return
            
            task = self.active_tasks[task_id]
            
            # Try retry if enabled
            if self.retry_manager and self.retry_manager.should_retry(task_id, 1):
                logger.info(f"Retrying failed task: {task_id}")
                # Put back in queue
                del self.active_tasks[task_id]
                self.pending_tasks.put(task)
            else:
                # Mark as failed
                task.completed_at = time.time()
                task.error = error_msg
                del self.active_tasks[task_id]
                self.completed_tasks[task_id] = task
                self.stats['tasks_failed'] += 1
                
                logger.error(f"Task failed: {task_id} - {error_msg}")
    
    def _handle_worker_failure(self, worker_id: str):
        """Handle worker failure."""
        logger.warning(f"Worker failed: {worker_id}")
        
        with self.lock:
            # Mark worker as failed
            if worker_id in self.workers:
                self.workers[worker_id].status = "error"
            
            # Reassign active tasks from this worker
            failed_tasks = [
                task for task in self.active_tasks.values()
                if task.assigned_worker == worker_id
            ]
            
            for task in failed_tasks:
                del self.active_tasks[task.task_id]
                
                # Reset assignment
                task.assigned_worker = None
                task.started_at = None
                
                # Put back in queue
                self.pending_tasks.put(task)
                
                logger.info(f"Reassigning task {task.task_id} from failed worker")
            
            self.stats['workers_failed'] += 1
    
    def _assign_tasks(self):
        """Assign pending tasks to available workers."""
        try:
            # Get task if available (non-blocking)
            task = self.pending_tasks.get_nowait()
        except Empty:
            return
        
        # Select worker using load balancer
        worker_id = self.balancer.select_worker(task.requirements)
        
        if not worker_id:
            # No workers available, put back in queue
            self.pending_tasks.put(task)
            return
        
        # Check if worker is healthy
        if not self.health_checker.is_healthy(worker_id):
            # Worker not healthy, put back and try again
            self.pending_tasks.put(task)
            return
        
        # Assign task
        with self.lock:
            task.assigned_worker = worker_id
            task.started_at = time.time()
            self.active_tasks[task.task_id] = task
            
            # Update worker load
            if worker_id in self.worker_loads:
                self.worker_loads[worker_id].active_tasks += 1
        
        # Send task to worker
        message = Message(
            type=MessageType.TASK,
            payload={
                'task_id': task.task_id,
                'payload': task.payload
            }
        )
        
        self.router.send(worker_id, message)
        
        logger.info(f"Task {task.task_id} assigned to {worker_id}")


# Demo code
if __name__ == "__main__":
    print("=" * 70)
    print("Cluster Coordinator Demo")
    print("=" * 70)
    
    # This is just a structure demo
    # Actual usage requires running workers
    
    coordinator = ClusterCoordinator(
        bind_address="tcp://127.0.0.1:5555",
        strategy=LoadBalanceStrategy.ADAPTIVE
    )
    
    print(f"\nCoordinator initialized:")
    print(f"  Bind address: {coordinator.bind_address}")
    print(f"  Strategy: {coordinator.strategy.value}")
    print(f"  Retry enabled: {coordinator.retry_manager is not None}")
    
    print("\nTo actually run the coordinator:")
    print("  coordinator.start()")
    print("  task_id = coordinator.submit_task({'model': 'resnet50', 'input': data})")
    print("  result = coordinator.get_result(task_id)")
    print("  coordinator.stop()")
    
    print("\n" + "=" * 70)
