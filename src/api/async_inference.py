"""
Async Inference Queue for Production Deployment

Provides asynchronous batch inference with queue management:
- Non-blocking inference requests
- Batch processing for throughput
- Request prioritization
- Progress tracking
- Result caching

Usage:
    from src.api.async_inference import AsyncInferenceQueue
    
    # Create queue
    queue = AsyncInferenceQueue(model, batch_size=8)
    
    # Submit request
    job_id = await queue.submit(image_data, priority=1)
    
    # Get result
    result = await queue.get_result(job_id)

Author: AMD GPU Computing Team
Date: January 21, 2026
"""

import asyncio
import uuid
from typing import Any, Dict, List, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import logging
from collections import deque
import numpy as np

logger = logging.getLogger(__name__)


class JobStatus(Enum):
    """Job status in queue"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class JobPriority(Enum):
    """Job priority levels"""
    LOW = 0
    NORMAL = 1
    HIGH = 2
    URGENT = 3


@dataclass
class InferenceJob:
    """Inference job in queue"""
    job_id: str
    data: Any
    priority: JobPriority = JobPriority.NORMAL
    status: JobStatus = JobStatus.PENDING
    result: Optional[Any] = None
    error: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __lt__(self, other):
        """Compare by priority (for priority queue)"""
        return self.priority.value > other.priority.value
    
    @property
    def elapsed_time(self) -> Optional[float]:
        """Get elapsed time in seconds"""
        if self.completed_at and self.started_at:
            return (self.completed_at - self.started_at).total_seconds()
        return None
    
    @property
    def wait_time(self) -> float:
        """Get wait time in seconds"""
        if self.started_at:
            return (self.started_at - self.created_at).total_seconds()
        return (datetime.now() - self.created_at).total_seconds()


class AsyncInferenceQueue:
    """
    Asynchronous inference queue with batch processing.
    
    Features:
    - Non-blocking inference requests
    - Automatic batch formation for throughput
    - Priority-based scheduling
    - Progress tracking and result caching
    - Graceful shutdown
    
    Example:
        >>> queue = AsyncInferenceQueue(model, batch_size=8, max_wait_time=0.1)
        >>> job_id = await queue.submit(image_data)
        >>> result = await queue.get_result(job_id)
    """
    
    def __init__(
        self,
        inference_fn: Callable,
        batch_size: int = 8,
        max_wait_time: float = 0.1,
        max_queue_size: int = 1000,
        result_ttl: int = 3600,
        num_workers: int = 1
    ):
        """
        Initialize async inference queue.
        
        Args:
            inference_fn: Function to run inference (must accept batch)
            batch_size: Maximum batch size
            max_wait_time: Maximum wait time to form batch (seconds)
            max_queue_size: Maximum queue size
            result_ttl: Result time-to-live (seconds)
            num_workers: Number of worker threads
        """
        self.inference_fn = inference_fn
        self.batch_size = batch_size
        self.max_wait_time = max_wait_time
        self.max_queue_size = max_queue_size
        self.result_ttl = result_ttl
        self.num_workers = num_workers
        
        # Queue and jobs
        self.queue: asyncio.PriorityQueue = asyncio.PriorityQueue(maxsize=max_queue_size)
        self.jobs: Dict[str, InferenceJob] = {}
        self.results: Dict[str, Any] = {}
        
        # Statistics
        self.stats = {
            'total_submitted': 0,
            'total_completed': 0,
            'total_failed': 0,
            'total_batches': 0,
            'avg_batch_size': 0.0,
            'avg_latency': 0.0
        }
        
        # Workers
        self.workers: List[asyncio.Task] = []
        self.running = False
        
        logger.info(
            f"Initialized AsyncInferenceQueue "
            f"(batch_size={batch_size}, max_wait={max_wait_time}s)"
        )
    
    async def start(self):
        """Start worker threads"""
        if self.running:
            return
        
        self.running = True
        
        # Start workers
        for i in range(self.num_workers):
            worker = asyncio.create_task(self._worker(i))
            self.workers.append(worker)
        
        # Start cleanup task
        self.cleanup_task = asyncio.create_task(self._cleanup_results())
        
        logger.info(f"Started {self.num_workers} workers")
    
    async def stop(self):
        """Stop workers gracefully"""
        if not self.running:
            return
        
        self.running = False
        
        # Cancel workers
        for worker in self.workers:
            worker.cancel()
        
        # Cancel cleanup
        if hasattr(self, 'cleanup_task'):
            self.cleanup_task.cancel()
        
        # Wait for all to finish
        await asyncio.gather(*self.workers, return_exceptions=True)
        
        logger.info("Workers stopped")
    
    async def submit(
        self,
        data: Any,
        priority: JobPriority = JobPriority.NORMAL,
        metadata: Optional[Dict] = None
    ) -> str:
        """
        Submit inference job to queue.
        
        Args:
            data: Input data for inference
            priority: Job priority
            metadata: Optional metadata
            
        Returns:
            Job ID
            
        Raises:
            asyncio.QueueFull: If queue is full
        """
        # Create job
        job_id = str(uuid.uuid4())
        job = InferenceJob(
            job_id=job_id,
            data=data,
            priority=priority,
            metadata=metadata or {}
        )
        
        # Add to queue
        await self.queue.put((priority.value, job))
        
        # Store job
        self.jobs[job_id] = job
        
        # Update stats
        self.stats['total_submitted'] += 1
        
        logger.debug(f"Submitted job {job_id} (priority={priority.name})")
        
        return job_id
    
    async def get_result(
        self,
        job_id: str,
        timeout: Optional[float] = None
    ) -> Any:
        """
        Get job result (blocking).
        
        Args:
            job_id: Job ID
            timeout: Maximum wait time (None = infinite)
            
        Returns:
            Inference result
            
        Raises:
            KeyError: If job not found
            asyncio.TimeoutError: If timeout exceeded
            RuntimeError: If job failed
        """
        if job_id not in self.jobs:
            raise KeyError(f"Job {job_id} not found")
        
        job = self.jobs[job_id]
        
        # Wait for completion
        start_time = asyncio.get_event_loop().time()
        
        while job.status not in [JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED]:
            await asyncio.sleep(0.01)
            
            # Check timeout
            if timeout and (asyncio.get_event_loop().time() - start_time) > timeout:
                raise asyncio.TimeoutError(f"Job {job_id} timed out")
        
        # Check status
        if job.status == JobStatus.FAILED:
            raise RuntimeError(f"Job failed: {job.error}")
        
        if job.status == JobStatus.CANCELLED:
            raise RuntimeError(f"Job cancelled")
        
        return job.result
    
    def get_status(self, job_id: str) -> Dict[str, Any]:
        """
        Get job status.
        
        Args:
            job_id: Job ID
            
        Returns:
            Status dictionary
        """
        if job_id not in self.jobs:
            raise KeyError(f"Job {job_id} not found")
        
        job = self.jobs[job_id]
        
        return {
            'job_id': job_id,
            'status': job.status.value,
            'priority': job.priority.name,
            'created_at': job.created_at.isoformat(),
            'started_at': job.started_at.isoformat() if job.started_at else None,
            'completed_at': job.completed_at.isoformat() if job.completed_at else None,
            'elapsed_time': job.elapsed_time,
            'wait_time': job.wait_time,
            'metadata': job.metadata
        }
    
    async def cancel(self, job_id: str) -> bool:
        """
        Cancel job.
        
        Args:
            job_id: Job ID
            
        Returns:
            True if cancelled
        """
        if job_id not in self.jobs:
            return False
        
        job = self.jobs[job_id]
        
        # Can only cancel pending jobs
        if job.status == JobStatus.PENDING:
            job.status = JobStatus.CANCELLED
            logger.info(f"Cancelled job {job_id}")
            return True
        
        return False
    
    async def _worker(self, worker_id: int):
        """Worker thread to process jobs"""
        logger.info(f"Worker {worker_id} started")
        
        while self.running:
            try:
                # Collect batch
                batch = await self._collect_batch()
                
                if not batch:
                    await asyncio.sleep(0.01)
                    continue
                
                # Process batch
                await self._process_batch(batch)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Worker {worker_id} error: {e}")
                await asyncio.sleep(0.1)
        
        logger.info(f"Worker {worker_id} stopped")
    
    async def _collect_batch(self) -> List[InferenceJob]:
        """Collect batch of jobs from queue"""
        batch = []
        deadline = asyncio.get_event_loop().time() + self.max_wait_time
        
        while len(batch) < self.batch_size:
            try:
                # Calculate remaining wait time
                timeout = max(0.01, deadline - asyncio.get_event_loop().time())
                
                # Get job from queue
                priority, job = await asyncio.wait_for(
                    self.queue.get(),
                    timeout=timeout
                )
                
                # Skip if cancelled
                if job.status == JobStatus.CANCELLED:
                    continue
                
                batch.append(job)
                
            except asyncio.TimeoutError:
                break
        
        return batch
    
    async def _process_batch(self, batch: List[InferenceJob]):
        """Process batch of jobs"""
        if not batch:
            return
        
        logger.debug(f"Processing batch of {len(batch)} jobs")
        
        try:
            # Mark as processing
            for job in batch:
                job.status = JobStatus.PROCESSING
                job.started_at = datetime.now()
            
            # Collect input data
            batch_data = [job.data for job in batch]
            
            # Run inference
            results = await asyncio.to_thread(self.inference_fn, batch_data)
            
            # Store results
            for job, result in zip(batch, results):
                job.result = result
                job.status = JobStatus.COMPLETED
                job.completed_at = datetime.now()
                
                # Cache result
                self.results[job.job_id] = result
            
            # Update stats
            self.stats['total_completed'] += len(batch)
            self.stats['total_batches'] += 1
            
            # Update avg batch size
            total = self.stats['total_batches']
            self.stats['avg_batch_size'] = (
                (self.stats['avg_batch_size'] * (total - 1) + len(batch)) / total
            )
            
            # Update avg latency
            avg_latency = np.mean([job.elapsed_time for job in batch if job.elapsed_time])
            self.stats['avg_latency'] = (
                (self.stats['avg_latency'] * (total - 1) + avg_latency) / total
            )
            
            logger.debug(f"Batch processed successfully")
            
        except Exception as e:
            logger.error(f"Batch processing failed: {e}")
            
            # Mark all as failed
            for job in batch:
                job.status = JobStatus.FAILED
                job.error = str(e)
                job.completed_at = datetime.now()
            
            self.stats['total_failed'] += len(batch)
    
    async def _cleanup_results(self):
        """Cleanup old results periodically"""
        while self.running:
            try:
                await asyncio.sleep(60)  # Check every minute
                
                now = datetime.now()
                expired = []
                
                for job_id, job in self.jobs.items():
                    if job.completed_at:
                        age = (now - job.completed_at).total_seconds()
                        if age > self.result_ttl:
                            expired.append(job_id)
                
                # Remove expired
                for job_id in expired:
                    del self.jobs[job_id]
                    if job_id in self.results:
                        del self.results[job_id]
                
                if expired:
                    logger.info(f"Cleaned up {len(expired)} expired results")
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Cleanup error: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get queue statistics"""
        return {
            'queue_size': self.queue.qsize(),
            'active_jobs': sum(1 for j in self.jobs.values() if j.status == JobStatus.PROCESSING),
            'pending_jobs': sum(1 for j in self.jobs.values() if j.status == JobStatus.PENDING),
            'total_jobs': len(self.jobs),
            **self.stats
        }


if __name__ == "__main__":
    # Demo
    async def demo():
        print("Async Inference Queue Demo")
        print("=" * 60)
        
        # Mock inference function
        def mock_inference(batch):
            import time
            time.sleep(0.1)  # Simulate inference
            return [f"result_{i}" for i in range(len(batch))]
        
        # Create queue
        queue = AsyncInferenceQueue(
            mock_inference,
            batch_size=4,
            max_wait_time=0.2
        )
        
        # Start workers
        await queue.start()
        
        # Submit jobs
        job_ids = []
        for i in range(10):
            priority = JobPriority.HIGH if i < 3 else JobPriority.NORMAL
            job_id = await queue.submit(f"data_{i}", priority=priority)
            job_ids.append(job_id)
            print(f"Submitted job {i}: {job_id}")
        
        # Wait for results
        print("\nWaiting for results...")
        for i, job_id in enumerate(job_ids):
            result = await queue.get_result(job_id, timeout=5.0)
            print(f"Job {i} result: {result}")
        
        # Show stats
        print("\nQueue Statistics:")
        stats = queue.get_stats()
        for key, value in stats.items():
            print(f"  {key}: {value}")
        
        # Stop
        await queue.stop()
        
        print("\nDemo complete!")
    
    asyncio.run(demo())
