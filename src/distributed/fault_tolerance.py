"""
Fault Tolerance System for Distributed Inference
===============================================

This module provides comprehensive fault tolerance mechanisms
for distributed GPU computing systems.

Features:
---------
1. **Task Retry Logic** - Automatic retry with exponential backoff
2. **Worker Failure Detection** - Heartbeat monitoring and health checks
3. **Failover Management** - Transparent task migration
4. **State Persistence** - Checkpoint and recovery
5. **Circuit Breaker** - Prevent cascade failures

Design Philosophy:
-----------------
- Fail fast for unrecoverable errors
- Retry with backoff for transient failures
- Isolate failures to prevent cascade
- Maintain visibility into failure patterns

Version: 0.6.0-dev
License: MIT
"""

import time
import logging
from typing import Optional, Dict, Any, Callable, List
from dataclasses import dataclass, field
from enum import Enum
from collections import deque
import pickle
import json

logger = logging.getLogger(__name__)


class FailureType(Enum):
    """Types of failures in distributed system."""
    NETWORK_ERROR = "network_error"
    WORKER_CRASH = "worker_crash"
    TIMEOUT = "timeout"
    OOM_ERROR = "oom_error"  # Out of memory
    COMPUTATION_ERROR = "computation_error"
    INVALID_INPUT = "invalid_input"
    UNKNOWN = "unknown"


class RetryStrategy(Enum):
    """Retry strategies for failed tasks."""
    NONE = "none"  # Don't retry
    FIXED = "fixed"  # Fixed delay between retries
    EXPONENTIAL = "exponential"  # Exponential backoff
    LINEAR = "linear"  # Linear backoff


@dataclass
class RetryConfig:
    """
    Configuration for retry behavior.
    
    Attributes:
        strategy: Retry strategy to use
        max_attempts: Maximum retry attempts
        initial_delay_seconds: Initial delay before first retry
        max_delay_seconds: Maximum delay between retries
        backoff_multiplier: Multiplier for exponential backoff
        jitter: Add random jitter to avoid thundering herd
    """
    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL
    max_attempts: int = 3
    initial_delay_seconds: float = 1.0
    max_delay_seconds: float = 60.0
    backoff_multiplier: float = 2.0
    jitter: bool = True


@dataclass
class TaskAttempt:
    """
    Record of a task execution attempt.
    
    Attributes:
        task_id: Task identifier
        worker_id: Worker that executed attempt
        attempt_number: Attempt number (1-indexed)
        start_time: When attempt started
        end_time: When attempt ended (None if still running)
        success: Whether attempt succeeded
        failure_type: Type of failure (if failed)
        error_message: Error details
    """
    task_id: str
    worker_id: str
    attempt_number: int
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    success: bool = False
    failure_type: Optional[FailureType] = None
    error_message: Optional[str] = None
    
    @property
    def duration_seconds(self) -> Optional[float]:
        """Get attempt duration."""
        if self.end_time is None:
            return None
        return self.end_time - self.start_time


class RetryManager:
    """
    Manages task retry logic with configurable strategies.
    
    Implements exponential backoff, jitter, and failure tracking.
    
    Example:
        config = RetryConfig(
            strategy=RetryStrategy.EXPONENTIAL,
            max_attempts=3,
            initial_delay_seconds=1.0
        )
        
        retry_mgr = RetryManager(config)
        
        for attempt in range(config.max_attempts):
            try:
                result = execute_task()
                retry_mgr.record_success(task_id)
                break
            except Exception as e:
                if retry_mgr.should_retry(task_id, attempt + 1):
                    delay = retry_mgr.get_retry_delay(attempt + 1)
                    time.sleep(delay)
                else:
                    raise
    """
    
    def __init__(self, config: RetryConfig):
        """
        Initialize retry manager.
        
        Args:
            config: Retry configuration
        """
        self.config = config
        self.task_attempts: Dict[str, List[TaskAttempt]] = {}
    
    def should_retry(self, task_id: str, current_attempt: int) -> bool:
        """
        Check if task should be retried.
        
        Args:
            task_id: Task identifier
            current_attempt: Current attempt number (1-indexed)
        
        Returns:
            True if task should be retried
        """
        if self.config.strategy == RetryStrategy.NONE:
            return False
        
        return current_attempt < self.config.max_attempts
    
    def get_retry_delay(self, attempt_number: int) -> float:
        """
        Calculate delay before next retry.
        
        Args:
            attempt_number: Attempt number (1-indexed)
        
        Returns:
            Delay in seconds
        """
        if self.config.strategy == RetryStrategy.FIXED:
            delay = self.config.initial_delay_seconds
        
        elif self.config.strategy == RetryStrategy.LINEAR:
            delay = self.config.initial_delay_seconds * attempt_number
        
        elif self.config.strategy == RetryStrategy.EXPONENTIAL:
            delay = min(
                self.config.initial_delay_seconds * (self.config.backoff_multiplier ** (attempt_number - 1)),
                self.config.max_delay_seconds
            )
        
        else:
            delay = 0.0
        
        # Add jitter to avoid thundering herd
        if self.config.jitter:
            import random
            jitter = random.uniform(0, delay * 0.1)
            delay += jitter
        
        return delay
    
    def record_attempt(
        self,
        task_id: str,
        worker_id: str,
        attempt_number: int,
        success: bool,
        failure_type: Optional[FailureType] = None,
        error_message: Optional[str] = None
    ):
        """
        Record task attempt.
        
        Args:
            task_id: Task identifier
            worker_id: Worker that executed attempt
            attempt_number: Attempt number
            success: Whether attempt succeeded
            failure_type: Type of failure (if failed)
            error_message: Error details
        """
        if task_id not in self.task_attempts:
            self.task_attempts[task_id] = []
        
        attempt = TaskAttempt(
            task_id=task_id,
            worker_id=worker_id,
            attempt_number=attempt_number,
            end_time=time.time(),
            success=success,
            failure_type=failure_type,
            error_message=error_message
        )
        
        self.task_attempts[task_id].append(attempt)
    
    def get_task_history(self, task_id: str) -> List[TaskAttempt]:
        """Get attempt history for task."""
        return self.task_attempts.get(task_id, [])
    
    def clear_task(self, task_id: str):
        """Clear attempt history for completed task."""
        if task_id in self.task_attempts:
            del self.task_attempts[task_id]


class CircuitBreaker:
    """
    Circuit breaker pattern to prevent cascade failures.
    
    Three states:
    - CLOSED: Normal operation, requests pass through
    - OPEN: Failure threshold exceeded, requests fail fast
    - HALF_OPEN: Testing if service recovered
    
    Example:
        breaker = CircuitBreaker(
            failure_threshold=5,
            timeout_seconds=60.0
        )
        
        if breaker.is_available():
            try:
                result = call_service()
                breaker.record_success()
            except Exception as e:
                breaker.record_failure()
                raise
        else:
            raise ServiceUnavailableError("Circuit breaker open")
    """
    
    class State(Enum):
        CLOSED = "closed"
        OPEN = "open"
        HALF_OPEN = "half_open"
    
    def __init__(
        self,
        failure_threshold: int = 5,
        timeout_seconds: float = 60.0,
        half_open_max_calls: int = 1
    ):
        """
        Initialize circuit breaker.
        
        Args:
            failure_threshold: Consecutive failures to open circuit
            timeout_seconds: How long to keep circuit open
            half_open_max_calls: Max calls in half-open state
        """
        self.failure_threshold = failure_threshold
        self.timeout_seconds = timeout_seconds
        self.half_open_max_calls = half_open_max_calls
        
        self.state = self.State.CLOSED
        self.failure_count = 0
        self.last_failure_time: Optional[float] = None
        self.half_open_calls = 0
    
    def is_available(self) -> bool:
        """
        Check if service is available.
        
        Returns:
            True if requests should be allowed
        """
        if self.state == self.State.CLOSED:
            return True
        
        elif self.state == self.State.OPEN:
            # Check if timeout expired
            if self.last_failure_time is not None:
                elapsed = time.time() - self.last_failure_time
                if elapsed >= self.timeout_seconds:
                    # Transition to half-open
                    self.state = self.State.HALF_OPEN
                    self.half_open_calls = 0
                    logger.info("Circuit breaker transitioning to HALF_OPEN")
                    return True
            return False
        
        elif self.state == self.State.HALF_OPEN:
            # Allow limited calls in half-open state
            return self.half_open_calls < self.half_open_max_calls
        
        return False
    
    def record_success(self):
        """Record successful call."""
        if self.state == self.State.HALF_OPEN:
            # Success in half-open -> close circuit
            self.state = self.State.CLOSED
            self.failure_count = 0
            self.half_open_calls = 0
            logger.info("Circuit breaker closed after successful call")
        
        elif self.state == self.State.CLOSED:
            # Reset failure count on success
            self.failure_count = 0
    
    def record_failure(self):
        """Record failed call."""
        self.last_failure_time = time.time()
        
        if self.state == self.State.HALF_OPEN:
            # Failure in half-open -> reopen circuit
            self.state = self.State.OPEN
            logger.warning("Circuit breaker reopened after failure in HALF_OPEN")
        
        elif self.state == self.State.CLOSED:
            self.failure_count += 1
            if self.failure_count >= self.failure_threshold:
                # Open circuit
                self.state = self.State.OPEN
                logger.warning(
                    f"Circuit breaker opened after {self.failure_count} failures"
                )
        
        if self.state == self.State.HALF_OPEN:
            self.half_open_calls += 1
    
    def reset(self):
        """Manually reset circuit breaker."""
        self.state = self.State.CLOSED
        self.failure_count = 0
        self.half_open_calls = 0
        self.last_failure_time = None


class HealthChecker:
    """
    Worker health monitoring with heartbeat tracking.
    
    Monitors worker liveness and automatically marks
    unhealthy workers as unavailable.
    
    Example:
        health = HealthChecker(heartbeat_interval=10.0, timeout=30.0)
        health.register_worker("worker1")
        
        # Worker sends heartbeat
        health.record_heartbeat("worker1")
        
        # Check health
        if health.is_healthy("worker1"):
            send_task_to_worker("worker1", task)
    """
    
    def __init__(
        self,
        heartbeat_interval: float = 10.0,
        timeout_seconds: float = 30.0
    ):
        """
        Initialize health checker.
        
        Args:
            heartbeat_interval: Expected heartbeat interval
            timeout_seconds: Time without heartbeat before unhealthy
        """
        self.heartbeat_interval = heartbeat_interval
        self.timeout_seconds = timeout_seconds
        
        self.worker_heartbeats: Dict[str, float] = {}
        self.worker_health: Dict[str, bool] = {}
    
    def register_worker(self, worker_id: str):
        """Register worker for health monitoring."""
        self.worker_heartbeats[worker_id] = time.time()
        self.worker_health[worker_id] = True
        logger.info(f"Registered worker for health monitoring: {worker_id}")
    
    def unregister_worker(self, worker_id: str):
        """Unregister worker."""
        if worker_id in self.worker_heartbeats:
            del self.worker_heartbeats[worker_id]
        if worker_id in self.worker_health:
            del self.worker_health[worker_id]
    
    def record_heartbeat(self, worker_id: str):
        """
        Record worker heartbeat.
        
        Args:
            worker_id: Worker identifier
        """
        current_time = time.time()
        
        was_unhealthy = not self.worker_health.get(worker_id, True)
        
        self.worker_heartbeats[worker_id] = current_time
        self.worker_health[worker_id] = True
        
        if was_unhealthy:
            logger.info(f"Worker recovered: {worker_id}")
    
    def is_healthy(self, worker_id: str) -> bool:
        """
        Check if worker is healthy.
        
        Args:
            worker_id: Worker identifier
        
        Returns:
            True if worker is healthy
        """
        if worker_id not in self.worker_heartbeats:
            return False
        
        # Check timeout
        last_heartbeat = self.worker_heartbeats[worker_id]
        elapsed = time.time() - last_heartbeat
        
        if elapsed > self.timeout_seconds:
            # Mark as unhealthy
            if self.worker_health.get(worker_id, True):
                self.worker_health[worker_id] = False
                logger.warning(
                    f"Worker unhealthy (no heartbeat for {elapsed:.1f}s): {worker_id}"
                )
            return False
        
        return self.worker_health.get(worker_id, True)
    
    def get_healthy_workers(self) -> List[str]:
        """Get list of healthy workers."""
        return [
            worker_id
            for worker_id in self.worker_heartbeats.keys()
            if self.is_healthy(worker_id)
        ]
    
    def get_unhealthy_workers(self) -> List[str]:
        """Get list of unhealthy workers."""
        return [
            worker_id
            for worker_id in self.worker_heartbeats.keys()
            if not self.is_healthy(worker_id)
        ]


class CheckpointManager:
    """
    Checkpoint management for fault recovery.
    
    Periodically saves task state to enable recovery
    after worker failures.
    
    Example:
        checkpoint_mgr = CheckpointManager(checkpoint_dir="/tmp/checkpoints")
        
        # Save checkpoint
        state = {"model": model, "step": 100}
        checkpoint_mgr.save_checkpoint(task_id, state)
        
        # Recover from checkpoint
        state = checkpoint_mgr.load_checkpoint(task_id)
        if state:
            resume_from_step(state["step"])
    """
    
    def __init__(self, checkpoint_dir: str = "/tmp/checkpoints"):
        """
        Initialize checkpoint manager.
        
        Args:
            checkpoint_dir: Directory for checkpoint files
        """
        self.checkpoint_dir = checkpoint_dir
        
        # Create directory if needed
        import os
        os.makedirs(checkpoint_dir, exist_ok=True)
    
    def save_checkpoint(
        self,
        task_id: str,
        state: Dict[str, Any],
        format: str = "pickle"
    ) -> bool:
        """
        Save task checkpoint.
        
        Args:
            task_id: Task identifier
            state: State dictionary to save
            format: Serialization format (pickle or json)
        
        Returns:
            True if checkpoint saved successfully
        """
        import os
        
        filepath = os.path.join(self.checkpoint_dir, f"{task_id}.ckpt")
        
        try:
            if format == "pickle":
                with open(filepath, 'wb') as f:
                    pickle.dump(state, f)
            elif format == "json":
                with open(filepath, 'w') as f:
                    json.dump(state, f)
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            logger.info(f"Saved checkpoint for task {task_id}")
            return True
        
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
            return False
    
    def load_checkpoint(
        self,
        task_id: str,
        format: str = "pickle"
    ) -> Optional[Dict[str, Any]]:
        """
        Load task checkpoint.
        
        Args:
            task_id: Task identifier
            format: Serialization format
        
        Returns:
            State dictionary or None if not found
        """
        import os
        
        filepath = os.path.join(self.checkpoint_dir, f"{task_id}.ckpt")
        
        if not os.path.exists(filepath):
            return None
        
        try:
            if format == "pickle":
                with open(filepath, 'rb') as f:
                    state = pickle.load(f)
            elif format == "json":
                with open(filepath, 'r') as f:
                    state = json.load(f)
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            logger.info(f"Loaded checkpoint for task {task_id}")
            return state
        
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            return None
    
    def delete_checkpoint(self, task_id: str):
        """Delete checkpoint file."""
        import os
        
        filepath = os.path.join(self.checkpoint_dir, f"{task_id}.ckpt")
        
        if os.path.exists(filepath):
            try:
                os.remove(filepath)
                logger.info(f"Deleted checkpoint for task {task_id}")
            except Exception as e:
                logger.error(f"Failed to delete checkpoint: {e}")


# Demo code
if __name__ == "__main__":
    print("=" * 70)
    print("Fault Tolerance System Demo")
    print("=" * 70)
    
    # Demo retry manager
    print("\nRetry Manager with Exponential Backoff:")
    print("-" * 70)
    
    config = RetryConfig(
        strategy=RetryStrategy.EXPONENTIAL,
        max_attempts=4,
        initial_delay_seconds=1.0,
        jitter=False
    )
    
    retry_mgr = RetryManager(config)
    
    for attempt in range(1, 5):
        delay = retry_mgr.get_retry_delay(attempt)
        print(f"Attempt {attempt}: Delay = {delay:.2f}s")
    
    # Demo circuit breaker
    print("\nCircuit Breaker:")
    print("-" * 70)
    
    breaker = CircuitBreaker(failure_threshold=3, timeout_seconds=5.0)
    print(f"Initial state: {breaker.state.value}")
    
    # Simulate failures
    for i in range(3):
        breaker.record_failure()
        print(f"After failure {i+1}: {breaker.state.value}, available={breaker.is_available()}")
    
    # Wait for timeout
    print("Waiting for timeout...")
    time.sleep(5.1)
    print(f"After timeout: available={breaker.is_available()}, state={breaker.state.value}")
    
    # Demo health checker
    print("\nHealth Checker:")
    print("-" * 70)
    
    health = HealthChecker(heartbeat_interval=5.0, timeout_seconds=10.0)
    health.register_worker("worker1")
    health.register_worker("worker2")
    
    # Worker 1 sends heartbeat
    health.record_heartbeat("worker1")
    
    print(f"Worker 1 healthy: {health.is_healthy('worker1')}")
    print(f"Worker 2 healthy: {health.is_healthy('worker2')}")
    
    # Simulate timeout for worker 2
    print("\nSimulating timeout for worker2...")
    time.sleep(1.0)
    print(f"Worker 2 healthy: {health.is_healthy('worker2')}")
    
    print("\n" + "=" * 70)
