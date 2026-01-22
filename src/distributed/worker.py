"""
Worker Node for Distributed Inference
=====================================

Worker nodes execute inference tasks assigned by the coordinator.
Each worker:
- Registers with coordinator on startup
- Sends periodic heartbeats
- Executes assigned tasks
- Reports results back to coordinator

Architecture:
------------
    ┌────────────────────────────────┐
    │      Coordinator (Remote)       │
    └────────────┬───────────────────┘
                 │
                 │ (Tasks & Heartbeats)
                 │
    ┌────────────▼───────────────────┐
    │     Worker Node (This)         │
    │  ┌──────────────────────────┐  │
    │  │   Inference Engine       │  │
    │  │  (ROCm Backend)          │  │
    │  └──────────────────────────┘  │
    │            │                    │
    │  ┌─────────▼─────────┐         │
    │  │   GPU (RX 580)    │         │
    │  │   8GB VRAM        │         │
    │  └───────────────────┘         │
    └────────────────────────────────┘

Worker Lifecycle:
----------------
1. Initialize: Load models, connect to GPU
2. Register: Announce availability to coordinator
3. Heartbeat: Send periodic health updates
4. Execute: Process assigned tasks
5. Report: Send results back to coordinator
6. Shutdown: Clean up resources

Version: 0.6.0-dev
License: MIT
"""

import time
import uuid
import logging
import threading
import traceback
from typing import Dict, Optional, Any, Callable
from dataclasses import dataclass
from queue import Queue, Empty

# Import distributed modules
try:
    from .communication import Message, MessageType, ZMQSocket
except ImportError:
    import sys
    import os
    sys.path.insert(0, os.path.dirname(__file__))
    from communication import Message, MessageType, ZMQSocket

logger = logging.getLogger(__name__)


@dataclass
class WorkerConfig:
    """
    Worker node configuration.
    
    Attributes:
        worker_id: Unique worker identifier (auto-generated if None)
        coordinator_address: Coordinator ZeroMQ address
        gpu_id: GPU device ID to use
        heartbeat_interval: Seconds between heartbeats
        max_concurrent_tasks: Maximum concurrent tasks
        task_timeout: Maximum task execution time
        auto_reconnect: Automatically reconnect on disconnect
    """
    worker_id: Optional[str] = None
    coordinator_address: str = "tcp://localhost:5555"
    gpu_id: int = 0
    heartbeat_interval: float = 10.0
    max_concurrent_tasks: int = 4
    task_timeout: float = 300.0
    auto_reconnect: bool = True


class InferenceWorker:
    """
    Worker node for distributed inference.
    
    Connects to coordinator, executes inference tasks,
    and reports results.
    
    Example:
        # Create worker
        config = WorkerConfig(
            coordinator_address="tcp://192.168.1.100:5555",
            gpu_id=0
        )
        
        worker = InferenceWorker(config)
        
        # Register inference handler
        @worker.register_handler
        def handle_inference(payload):
            model = payload['model']
            input_data = payload['input']
            # Run inference
            result = run_model(model, input_data)
            return result
        
        # Start worker
        worker.start()
        
        # Worker runs until stopped
        try:
            worker.wait()
        except KeyboardInterrupt:
            worker.stop()
    """
    
    def __init__(
        self,
        config: WorkerConfig,
        inference_handler: Optional[Callable] = None
    ):
        """
        Initialize worker node.
        
        Args:
            config: Worker configuration
            inference_handler: Function to handle inference tasks
        """
        self.config = config
        
        # Generate worker ID if not provided
        if self.config.worker_id is None:
            self.config.worker_id = f"worker-{uuid.uuid4().hex[:8]}"
        
        self.worker_id = self.config.worker_id
        self.running = False
        
        # Communication
        self.socket: Optional[ZMQSocket] = None
        
        # Task handling
        self.inference_handler = inference_handler
        self.active_tasks: Dict[str, threading.Thread] = {}
        self.task_queue: Queue = Queue()
        
        # Threading
        self.lock = threading.RLock()
        self.worker_thread: Optional[threading.Thread] = None
        self.heartbeat_thread: Optional[threading.Thread] = None
        
        # GPU info (will be detected)
        self.gpu_info = self._detect_gpu_info()
        
        # Statistics
        self.stats = {
            'tasks_completed': 0,
            'tasks_failed': 0,
            'uptime_start': None,
        }
    
    def _detect_gpu_info(self) -> Dict[str, Any]:
        """Detect GPU information."""
        info = {
            'gpu_name': 'Unknown',
            'gpu_memory_gb': 8.0,
            'gpu_family': 'Unknown',
            'compute_units': 36,
        }
        
        try:
            # Try to detect using rocm-smi or similar
            # For now, default to RX 580 info
            info['gpu_name'] = 'Radeon RX 580'
            info['gpu_memory_gb'] = 8.0
            info['gpu_family'] = 'Polaris'
            info['compute_units'] = 36
            
        except Exception as e:
            logger.warning(f"Could not detect GPU info: {e}")
        
        return info
    
    def register_handler(self, handler: Callable):
        """
        Register inference handler function.
        
        Handler should accept payload dict and return result.
        
        Args:
            handler: Inference handler function
        
        Example:
            @worker.register_handler
            def handle_task(payload):
                return process_inference(payload)
        """
        self.inference_handler = handler
        return handler
    
    def start(self):
        """Start worker node."""
        if self.running:
            logger.warning("Worker already running")
            return
        
        logger.info(f"Starting worker {self.worker_id}")
        
        # Connect to coordinator
        try:
            self.socket = ZMQSocket(
                self.config.coordinator_address,
                identity=self.worker_id.encode()
            )
        except Exception as e:
            logger.error(f"Failed to connect to coordinator: {e}")
            raise
        
        self.running = True
        self.stats['uptime_start'] = time.time()
        
        # Register with coordinator
        self._register_with_coordinator()
        
        # Start worker thread
        self.worker_thread = threading.Thread(
            target=self._worker_loop,
            daemon=True
        )
        self.worker_thread.start()
        
        # Start heartbeat thread
        self.heartbeat_thread = threading.Thread(
            target=self._heartbeat_loop,
            daemon=True
        )
        self.heartbeat_thread.start()
        
        logger.info(f"Worker {self.worker_id} started successfully")
    
    def stop(self):
        """Stop worker node."""
        logger.info(f"Stopping worker {self.worker_id}")
        self.running = False
        
        # Wait for threads
        if self.worker_thread:
            self.worker_thread.join(timeout=5.0)
        if self.heartbeat_thread:
            self.heartbeat_thread.join(timeout=5.0)
        
        # Close socket
        if self.socket:
            self.socket.close()
        
        logger.info(f"Worker {self.worker_id} stopped")
    
    def wait(self):
        """Wait for worker to finish (blocks until stop called)."""
        if self.worker_thread:
            self.worker_thread.join()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get worker statistics."""
        with self.lock:
            uptime = 0.0
            if self.stats['uptime_start']:
                uptime = time.time() - self.stats['uptime_start']
            
            return {
                'worker_id': self.worker_id,
                'uptime_seconds': uptime,
                'tasks_completed': self.stats['tasks_completed'],
                'tasks_failed': self.stats['tasks_failed'],
                'active_tasks': len(self.active_tasks),
                'gpu_info': self.gpu_info,
            }
    
    def _register_with_coordinator(self):
        """Send registration message to coordinator."""
        message = Message(
            type=MessageType.REGISTER,
            payload={
                'worker_id': self.worker_id,
                'address': self.config.coordinator_address,
                'gpu_name': self.gpu_info['gpu_name'],
                'gpu_memory_gb': self.gpu_info['gpu_memory_gb'],
                'capabilities': {
                    'gpu_family': self.gpu_info['gpu_family'],
                    'compute_units': self.gpu_info['compute_units'],
                    'max_concurrent': self.config.max_concurrent_tasks,
                }
            }
        )
        
        self.socket.send_message(message)
        logger.info("Registration message sent")
        
        # Wait for ACK
        ack = self.socket.recv_message(timeout=5.0)
        if ack and ack.type == MessageType.ACK:
            logger.info("Registration acknowledged by coordinator")
        else:
            logger.warning("No registration ACK received")
    
    def _worker_loop(self):
        """Main worker event loop."""
        logger.info("Worker loop started")
        
        while self.running:
            try:
                # Receive messages from coordinator
                message = self.socket.recv_message(timeout=0.1)
                
                if message:
                    self._handle_message(message)
                
            except Exception as e:
                logger.error(f"Error in worker loop: {e}")
                time.sleep(0.1)
        
        logger.info("Worker loop stopped")
    
    def _heartbeat_loop(self):
        """Heartbeat transmission loop."""
        logger.info("Heartbeat loop started")
        
        while self.running:
            try:
                self._send_heartbeat()
                time.sleep(self.config.heartbeat_interval)
            except Exception as e:
                logger.error(f"Error sending heartbeat: {e}")
                time.sleep(1.0)
        
        logger.info("Heartbeat loop stopped")
    
    def _send_heartbeat(self):
        """Send heartbeat to coordinator."""
        # Get current load
        with self.lock:
            active_count = len(self.active_tasks)
        
        # Get GPU metrics (simplified)
        gpu_utilization = min(1.0, active_count / self.config.max_concurrent_tasks)
        gpu_memory_used = self.gpu_info['gpu_memory_gb'] * gpu_utilization
        
        message = Message(
            type=MessageType.HEARTBEAT,
            payload={
                'worker_id': self.worker_id,
                'active_tasks': active_count,
                'gpu_utilization': gpu_utilization,
                'gpu_memory_used': gpu_memory_used,
            }
        )
        
        self.socket.send_message(message)
    
    def _handle_message(self, message: Message):
        """Handle message from coordinator."""
        
        if message.type == MessageType.TASK:
            self._handle_task(message)
        
        elif message.type == MessageType.SHUTDOWN:
            logger.info("Shutdown requested by coordinator")
            self.stop()
        
        else:
            logger.warning(f"Unknown message type: {message.type}")
    
    def _handle_task(self, message: Message):
        """Handle task assignment."""
        data = message.payload or {}
        task_id = data.get('task_id')
        payload = data.get('payload', {})
        
        if not task_id:
            logger.warning("Received task without task_id")
            return
        
        # Check if we can accept more tasks
        with self.lock:
            if len(self.active_tasks) >= self.config.max_concurrent_tasks:
                logger.warning(f"Max concurrent tasks reached, rejecting {task_id}")
                self._send_error(task_id, "Worker at capacity")
                return
        
        logger.info(f"Received task: {task_id}")
        
        # Start task execution in separate thread
        task_thread = threading.Thread(
            target=self._execute_task,
            args=(task_id, payload),
            daemon=True
        )
        
        with self.lock:
            self.active_tasks[task_id] = task_thread
        
        task_thread.start()
    
    def _execute_task(self, task_id: str, payload: Dict[str, Any]):
        """Execute inference task."""
        logger.info(f"Executing task: {task_id}")
        start_time = time.time()
        
        try:
            # Check if handler is registered
            if self.inference_handler is None:
                raise RuntimeError("No inference handler registered")
            
            # Execute inference
            result = self.inference_handler(payload)
            
            # Send result
            duration = time.time() - start_time
            logger.info(f"Task completed: {task_id} ({duration:.2f}s)")
            
            self._send_result(task_id, result)
            
            with self.lock:
                self.stats['tasks_completed'] += 1
        
        except Exception as e:
            # Send error
            duration = time.time() - start_time
            error_msg = f"{type(e).__name__}: {str(e)}"
            logger.error(f"Task failed: {task_id} ({duration:.2f}s) - {error_msg}")
            
            self._send_error(task_id, error_msg)
            
            with self.lock:
                self.stats['tasks_failed'] += 1
        
        finally:
            # Remove from active tasks
            with self.lock:
                if task_id in self.active_tasks:
                    del self.active_tasks[task_id]
    
    def _send_result(self, task_id: str, result: Any):
        """Send task result to coordinator."""
        message = Message(
            type=MessageType.RESULT,
            payload={
                'task_id': task_id,
                'result': result,
                'worker_id': self.worker_id,
            }
        )
        
        self.socket.send_message(message)
    
    def _send_error(self, task_id: str, error_msg: str):
        """Send task error to coordinator."""
        message = Message(
            type=MessageType.ERROR,
            payload={
                'task_id': task_id,
                'error': error_msg,
                'worker_id': self.worker_id,
            }
        )
        
        self.socket.send_message(message)


# Demo code
if __name__ == "__main__":
    print("=" * 70)
    print("Inference Worker Demo")
    print("=" * 70)
    
    # Create worker
    config = WorkerConfig(
        coordinator_address="tcp://localhost:5555",
        gpu_id=0,
        heartbeat_interval=5.0
    )
    
    worker = InferenceWorker(config)
    
    # Register simple handler
    @worker.register_handler
    def handle_inference(payload):
        """Simple demo handler."""
        import time
        import random
        
        model = payload.get('model', 'unknown')
        print(f"  Processing inference for model: {model}")
        
        # Simulate inference
        time.sleep(random.uniform(0.5, 2.0))
        
        return {
            'prediction': 'cat',
            'confidence': 0.95,
            'model': model,
        }
    
    print(f"\nWorker Configuration:")
    print(f"  Worker ID: {worker.worker_id}")
    print(f"  Coordinator: {config.coordinator_address}")
    print(f"  GPU: {worker.gpu_info['gpu_name']}")
    print(f"  Max concurrent tasks: {config.max_concurrent_tasks}")
    
    print("\nTo run the worker:")
    print("  1. Start coordinator first")
    print("  2. worker.start()")
    print("  3. Worker will register and wait for tasks")
    print("  4. Press Ctrl+C to stop")
    
    # Uncomment to actually run
    # worker.start()
    # try:
    #     worker.wait()
    # except KeyboardInterrupt:
    #     worker.stop()
    
    print("\n" + "=" * 70)
