"""
Legacy GPU AI Platform - Distributed Computing Layer
====================================================

This module provides distributed computing capabilities for clusters
of legacy AMD GPUs, enabling organizations in emerging countries to
build cost-effective AI infrastructure.

Vision:
------
In Latin America and other emerging regions, organizations may have
access to multiple older GPUs (university labs, refurbished equipment,
community centers). This layer enables them to work together as a
unified compute resource.

Architecture:
------------
                    ┌─────────────────┐
                    │  Coordinator    │
                    │  (Master Node)  │
                    └────────┬────────┘
                             │
            ┌────────────────┼────────────────┐
            │                │                │
     ┌──────▼──────┐  ┌──────▼──────┐  ┌──────▼──────┐
     │   Worker    │  │   Worker    │  │   Worker    │
     │  RX 580 #1  │  │  RX 580 #2  │  │  RX 570 #3  │
     └─────────────┘  └─────────────┘  └─────────────┘

Communication:
-------------
- ZeroMQ for lightweight, high-performance messaging
- MessagePack for efficient serialization
- Heartbeat system for worker health monitoring

Use Cases:
---------
1. University Computer Labs
   - Pool GPUs from multiple workstations
   - Enable larger batch sizes for training
   
2. Community AI Centers
   - Distributed inference for local services
   - Cost-effective image classification at scale
   
3. Educational Clusters
   - Students learn distributed AI concepts
   - Real-world cluster management experience

Version: 0.5.0-dev (Planned for v0.7.0)
License: MIT
"""

__version__ = "0.5.0-dev"
__all__ = [
    "Coordinator",
    "Worker", 
    "ClusterConfig",
    "create_local_cluster",
]

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from enum import Enum
import time


class WorkerStatus(Enum):
    """Status of a worker node."""
    OFFLINE = "offline"
    STARTING = "starting"
    READY = "ready"
    BUSY = "busy"
    ERROR = "error"


class TaskStatus(Enum):
    """Status of a distributed task."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class WorkerInfo:
    """Information about a worker node."""
    worker_id: str
    host: str
    port: int
    gpu_name: str
    gpu_family: str
    vram_gb: float
    status: WorkerStatus = WorkerStatus.OFFLINE
    current_task: Optional[str] = None
    last_heartbeat: float = 0.0


@dataclass
class ClusterConfig:
    """Configuration for a distributed cluster."""
    coordinator_host: str = "localhost"
    coordinator_port: int = 5555
    heartbeat_interval: float = 5.0  # seconds
    task_timeout: float = 300.0  # 5 minutes
    max_retries: int = 3
    load_balance_strategy: str = "round_robin"  # or "least_loaded", "gpu_match"


class Coordinator:
    """
    Cluster coordinator (master node).
    
    Manages worker registration, task distribution, and result aggregation.
    
    Example:
        config = ClusterConfig(coordinator_port=5555)
        coordinator = Coordinator(config)
        coordinator.start()
        
        # Distribute inference task across workers
        results = coordinator.distribute_inference(
            model_path="mobilenetv2.onnx",
            inputs=[img1, img2, img3, img4],
        )
        
    Note: Full implementation requires pyzmq and msgpack dependencies.
    These are optional dependencies (install with `pip install legacy-gpu-ai[distributed]`)
    """
    
    def __init__(self, config: Optional[ClusterConfig] = None):
        """
        Initialize coordinator.
        
        Args:
            config: Cluster configuration
        """
        self.config = config or ClusterConfig()
        self.workers: Dict[str, WorkerInfo] = {}
        self.pending_tasks: List[Dict[str, Any]] = []
        self.completed_tasks: Dict[str, Any] = {}
        self._running = False
        
    def start(self):
        """
        Start the coordinator.
        
        Note: Requires pyzmq. Install with: pip install legacy-gpu-ai[distributed]
        """
        try:
            import zmq
            self._context = zmq.Context()
            self._socket = self._context.socket(zmq.ROUTER)
            self._socket.bind(f"tcp://*:{self.config.coordinator_port}")
            self._running = True
            print(f"Coordinator started on port {self.config.coordinator_port}")
        except ImportError:
            raise ImportError(
                "Distributed features require pyzmq. "
                "Install with: pip install legacy-gpu-ai[distributed]"
            )
    
    def stop(self):
        """Stop the coordinator."""
        self._running = False
        if hasattr(self, '_socket'):
            self._socket.close()
        if hasattr(self, '_context'):
            self._context.term()
    
    def register_worker(self, worker_info: WorkerInfo) -> bool:
        """
        Register a new worker node.
        
        Args:
            worker_info: Worker information
            
        Returns:
            True if registration successful
        """
        self.workers[worker_info.worker_id] = worker_info
        worker_info.status = WorkerStatus.READY
        worker_info.last_heartbeat = time.time()
        return True
    
    def get_cluster_status(self) -> dict:
        """
        Get current cluster status.
        
        Returns:
            dict with cluster information
        """
        ready_workers = sum(
            1 for w in self.workers.values() 
            if w.status == WorkerStatus.READY
        )
        busy_workers = sum(
            1 for w in self.workers.values()
            if w.status == WorkerStatus.BUSY
        )
        total_vram = sum(w.vram_gb for w in self.workers.values())
        
        return {
            "total_workers": len(self.workers),
            "ready_workers": ready_workers,
            "busy_workers": busy_workers,
            "total_vram_gb": total_vram,
            "pending_tasks": len(self.pending_tasks),
            "completed_tasks": len(self.completed_tasks),
            "coordinator": {
                "host": self.config.coordinator_host,
                "port": self.config.coordinator_port,
            },
        }
    
    def distribute_inference(
        self,
        model_path: str,
        inputs: List[Any],
        batch_size: int = 1
    ) -> List[Any]:
        """
        Distribute inference task across workers.
        
        Args:
            model_path: Path to model (must be accessible by all workers)
            inputs: List of inputs to process
            batch_size: Batch size per worker
            
        Returns:
            List of results in same order as inputs
            
        Note: Placeholder implementation for v0.7.0
        """
        # Placeholder - actual implementation in v0.7.0
        if not self.workers:
            raise RuntimeError("No workers available. Register workers first.")
            
        results = []
        print(f"[Placeholder] Would distribute {len(inputs)} inputs across {len(self.workers)} workers")
        
        return results


class Worker:
    """
    Worker node for distributed inference.
    
    Example:
        worker = Worker(
            coordinator_host="192.168.1.100",
            coordinator_port=5555
        )
        worker.start()  # Blocks and processes tasks
    """
    
    def __init__(
        self,
        coordinator_host: str = "localhost",
        coordinator_port: int = 5555,
        worker_id: Optional[str] = None
    ):
        """
        Initialize worker.
        
        Args:
            coordinator_host: Coordinator hostname or IP
            coordinator_port: Coordinator port
            worker_id: Unique worker ID (auto-generated if not provided)
        """
        self.coordinator_host = coordinator_host
        self.coordinator_port = coordinator_port
        self.worker_id = worker_id or self._generate_id()
        self.status = WorkerStatus.OFFLINE
        self._model = None
        
    def _generate_id(self) -> str:
        """Generate unique worker ID."""
        import socket
        hostname = socket.gethostname()
        return f"worker-{hostname}-{int(time.time())}"
    
    def _detect_local_gpu(self) -> dict:
        """Detect local GPU information."""
        try:
            from src.core.gpu import GPUManager
            gpu = GPUManager()
            info = gpu.get_info()
            return {
                "name": info.get("device_name", "Unknown"),
                "family": self._classify_gpu(info.get("device_name", "")),
                "vram_gb": info.get("memory_total_gb", 0),
            }
        except Exception:
            return {"name": "Unknown", "family": "unknown", "vram_gb": 0}
    
    def _classify_gpu(self, name: str) -> str:
        """Classify GPU family from name."""
        name_lower = name.lower()
        if "580" in name_lower or "570" in name_lower or "480" in name_lower:
            return "polaris"
        elif "vega" in name_lower:
            return "vega"
        elif "5700" in name_lower or "5600" in name_lower:
            return "navi"
        return "unknown"
    
    def start(self):
        """
        Start the worker and connect to coordinator.
        
        Blocks until stopped.
        """
        try:
            import zmq
        except ImportError:
            raise ImportError(
                "Distributed features require pyzmq. "
                "Install with: pip install legacy-gpu-ai[distributed]"
            )
            
        gpu_info = self._detect_local_gpu()
        self.status = WorkerStatus.STARTING
        
        print(f"Worker {self.worker_id} starting...")
        print(f"GPU: {gpu_info['name']} ({gpu_info['family']})")
        print(f"Connecting to {self.coordinator_host}:{self.coordinator_port}")
        
        # Placeholder - full implementation in v0.7.0
        self.status = WorkerStatus.READY
        print("Worker ready (placeholder mode)")
    
    def stop(self):
        """Stop the worker."""
        self.status = WorkerStatus.OFFLINE


def create_local_cluster(num_workers: int = 2) -> tuple:
    """
    Create a local cluster for testing.
    
    Useful for development and testing distributed features
    on a single machine.
    
    Args:
        num_workers: Number of simulated workers
        
    Returns:
        Tuple of (coordinator, list of workers)
    """
    config = ClusterConfig(coordinator_port=5555)
    coordinator = Coordinator(config)
    
    workers = []
    for i in range(num_workers):
        worker = Worker(
            coordinator_port=5555,
            worker_id=f"local-worker-{i}"
        )
        workers.append(worker)
    
    return coordinator, workers


def cluster_status_report(coordinator: Coordinator) -> str:
    """Generate a formatted cluster status report."""
    status = coordinator.get_cluster_status()
    
    report = f"""
╔══════════════════════════════════════════════════════════════════╗
║         Legacy GPU AI Platform - Cluster Status                  ║
╠══════════════════════════════════════════════════════════════════╣
║ Coordinator: {status['coordinator']['host']}:{status['coordinator']['port']:<38} ║
╠══════════════════════════════════════════════════════════════════╣
║ Workers:                                                         ║
║   • Total: {status['total_workers']:<54} ║
║   • Ready: {status['ready_workers']:<54} ║
║   • Busy: {status['busy_workers']:<55} ║
║   • Total VRAM: {status['total_vram_gb']:.1f} GB{' '*45} ║
╠══════════════════════════════════════════════════════════════════╣
║ Tasks:                                                           ║
║   • Pending: {status['pending_tasks']:<52} ║
║   • Completed: {status['completed_tasks']:<50} ║
╚══════════════════════════════════════════════════════════════════╝
"""
    return report
