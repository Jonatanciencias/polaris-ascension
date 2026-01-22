"""
ZeroMQ Communication Layer for Distributed Computing
====================================================

This module provides high-performance messaging infrastructure for
distributed inference using ZeroMQ.

Features:
--------
- Request-Reply pattern for task distribution
- Publish-Subscribe for broadcast messages
- Push-Pull for pipeline workflows
- Async I/O support
- Automatic reconnection
- Message serialization with MessagePack

Architecture:
------------
    Coordinator (ROUTER socket)
         │
         ├─────────┬─────────┬─────────┐
         │         │         │         │
    Worker1   Worker2   Worker3   Worker4
    (DEALER)  (DEALER)  (DEALER)  (DEALER)

Message Types:
-------------
1. REGISTER - Worker registration
2. HEARTBEAT - Keep-alive ping
3. TASK - Inference task
4. RESULT - Task result
5. ERROR - Error notification
6. SHUTDOWN - Graceful shutdown

Version: 0.6.0-dev
License: MIT
"""

import json
import time
import uuid
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field, asdict
from enum import Enum
import logging

logger = logging.getLogger(__name__)

# Optional ZeroMQ support
try:
    import zmq
    ZMQ_AVAILABLE = True
except ImportError:
    ZMQ_AVAILABLE = False
    logger.warning(
        "ZeroMQ not available. Distributed mode disabled. "
        "Install with: pip install pyzmq"
    )

# Optional MessagePack support
try:
    import msgpack
    MSGPACK_AVAILABLE = True
except ImportError:
    MSGPACK_AVAILABLE = False
    logger.warning(
        "MessagePack not available. Using JSON (slower). "
        "Install with: pip install msgpack"
    )


class MessageType(Enum):
    """Types of messages in distributed system."""
    REGISTER = "register"
    HEARTBEAT = "heartbeat"
    TASK = "task"
    RESULT = "result"
    ERROR = "error"
    SHUTDOWN = "shutdown"
    ACK = "ack"


@dataclass
class Message:
    """
    Message structure for distributed communication.
    
    Attributes:
        type: Type of message
        payload: Message data
        sender: ID of sender
        timestamp: Message timestamp
        msg_id: Unique message identifier
    """
    type: MessageType
    payload: Optional[Dict[str, Any]] = None
    sender: Optional[str] = None
    timestamp: float = field(default_factory=time.time)
    msg_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary."""
        data = asdict(self)
        data['type'] = self.type.value
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Message":
        """Create message from dictionary."""
        data['type'] = MessageType(data['type'])
        return cls(**data)
    
    def serialize(self) -> bytes:
        """Serialize message to bytes."""
        data = self.to_dict()
        
        if MSGPACK_AVAILABLE:
            return msgpack.packb(data, use_bin_type=True)
        else:
            return json.dumps(data).encode('utf-8')
    
    @classmethod
    def deserialize(cls, data: bytes) -> "Message":
        """Deserialize message from bytes."""
        if MSGPACK_AVAILABLE:
            decoded = msgpack.unpackb(data, raw=False)
        else:
            decoded = json.loads(data.decode('utf-8'))
        
        return cls.from_dict(decoded)


class ZMQSocket:
    """
    Wrapper around ZeroMQ socket with automatic reconnection.
    
    Provides higher-level API on top of raw ZMQ sockets with:
    - Automatic reconnection on failure
    - Message serialization/deserialization
    - Timeout handling
    - Error recovery
    
    Example:
        socket = ZMQSocket(zmq.Context(), zmq.DEALER)
        socket.connect("tcp://localhost:5555")
        
        msg = Message(
            msg_id=str(uuid.uuid4()),
            msg_type=MessageType.TASK,
            sender_id="worker1",
            payload={"data": "..."}
        )
        
        socket.send_message(msg)
        reply = socket.recv_message(timeout=5000)
    """
    
    def __init__(
        self,
        context: "zmq.Context",
        socket_type: int,
        identity: Optional[str] = None
    ):
        """
        Initialize ZMQ socket wrapper.
        
        Args:
            context: ZeroMQ context
            socket_type: Socket type (DEALER, ROUTER, etc.)
            identity: Socket identity (for DEALER/ROUTER)
        """
        if not ZMQ_AVAILABLE:
            raise ImportError("ZeroMQ not available")
        
        self.context = context
        self.socket_type = socket_type
        self.identity = identity
        self.socket = None
        self._address = None
        self._create_socket()
    
    def _create_socket(self):
        """Create the actual ZMQ socket."""
        self.socket = self.context.socket(self.socket_type)
        
        # Set socket options
        self.socket.setsockopt(zmq.LINGER, 0)  # Don't block on close
        
        if self.identity:
            self.socket.setsockopt_string(zmq.IDENTITY, self.identity)
    
    def connect(self, address: str):
        """
        Connect to remote endpoint.
        
        Args:
            address: ZMQ address (e.g., "tcp://localhost:5555")
        """
        self._address = address
        self.socket.connect(address)
        logger.info(f"Connected to {address}")
    
    def bind(self, address: str):
        """
        Bind to local endpoint.
        
        Args:
            address: ZMQ address (e.g., "tcp://*:5555")
        """
        self._address = address
        self.socket.bind(address)
        logger.info(f"Bound to {address}")
    
    def send_message(self, message: Message, timeout: int = 5000):
        """
        Send a message.
        
        Args:
            message: Message to send
            timeout: Send timeout in milliseconds
        """
        self.socket.setsockopt(zmq.SNDTIMEO, timeout)
        
        try:
            serialized = message.serialize()
            self.socket.send(serialized)
            logger.debug(f"Sent message: {message.msg_type.value}")
        except zmq.error.Again:
            logger.error(f"Send timeout after {timeout}ms")
            raise TimeoutError(f"Failed to send message within {timeout}ms")
    
    def recv_message(self, timeout: int = 5000) -> Optional[Message]:
        """
        Receive a message.
        
        Args:
            timeout: Receive timeout in milliseconds
        
        Returns:
            Received message or None on timeout
        """
        self.socket.setsockopt(zmq.RCVTIMEO, timeout)
        
        try:
            data = self.socket.recv()
            message = Message.deserialize(data)
            logger.debug(f"Received message: {message.msg_type.value}")
            return message
        except zmq.error.Again:
            logger.debug(f"Receive timeout after {timeout}ms")
            return None
    
    def send_multipart(self, parts: List[bytes]):
        """Send multipart message."""
        self.socket.send_multipart(parts)
    
    def recv_multipart(self, timeout: int = 5000) -> Optional[List[bytes]]:
        """Receive multipart message."""
        self.socket.setsockopt(zmq.RCVTIMEO, timeout)
        
        try:
            return self.socket.recv_multipart()
        except zmq.error.Again:
            return None
    
    def close(self):
        """Close the socket."""
        if self.socket:
            self.socket.close()
            logger.info("Socket closed")


class MessageRouter:
    """
    Routes messages between coordinator and workers.
    
    Implements ROUTER-DEALER pattern for request-reply
    with multiple workers.
    
    Example:
        router = MessageRouter(zmq.Context())
        router.bind("tcp://*:5555")
        
        # Process incoming messages
        while True:
            sender_id, message = router.receive()
            if message:
                # Process message
                reply = create_reply(message)
                router.send(sender_id, reply)
    """
    
    def __init__(self, context: "zmq.Context", identity: str = "router"):
        """
        Initialize message router.
        
        Args:
            context: ZeroMQ context
            identity: Router identity
        """
        if not ZMQ_AVAILABLE:
            raise ImportError("ZeroMQ not available")
        
        self.context = context
        self.identity = identity
        self.socket = context.socket(zmq.ROUTER)
        self.socket.setsockopt_string(zmq.IDENTITY, identity)
        self.socket.setsockopt(zmq.LINGER, 0)
    
    def bind(self, address: str):
        """
        Bind router to address.
        
        Args:
            address: ZMQ address
        """
        self.socket.bind(address)
        logger.info(f"Router bound to {address}")
    
    def send(self, receiver_id: str, message: Message):
        """
        Send message to specific worker.
        
        Args:
            receiver_id: Worker ID
            message: Message to send
        """
        parts = [
            receiver_id.encode('utf-8'),
            b'',  # Delimiter
            message.serialize()
        ]
        self.socket.send_multipart(parts)
        logger.debug(f"Sent to {receiver_id}: {message.msg_type.value}")
    
    def receive(self, timeout: int = 1000) -> tuple[Optional[str], Optional[Message]]:
        """
        Receive message from any worker.
        
        Args:
            timeout: Receive timeout in milliseconds
        
        Returns:
            Tuple of (sender_id, message) or (None, None) on timeout
        """
        self.socket.setsockopt(zmq.RCVTIMEO, timeout)
        
        try:
            parts = self.socket.recv_multipart()
            
            if len(parts) >= 3:
                sender_id = parts[0].decode('utf-8')
                message_data = parts[2]
                message = Message.deserialize(message_data)
                
                logger.debug(f"Received from {sender_id}: {message.msg_type.value}")
                return sender_id, message
            else:
                logger.warning("Invalid message format")
                return None, None
        
        except zmq.error.Again:
            return None, None
    
    def close(self):
        """Close the router socket."""
        self.socket.close()


class ConnectionPool:
    """
    Manages pool of connections to workers.
    
    Provides connection pooling, health checking, and
    automatic reconnection for worker connections.
    
    Example:
        pool = ConnectionPool(zmq.Context())
        pool.add_worker("worker1", "tcp://192.168.1.100:5556")
        pool.add_worker("worker2", "tcp://192.168.1.101:5556")
        
        # Send task to healthy worker
        worker_id = pool.get_healthy_worker()
        pool.send_to_worker(worker_id, task_message)
    """
    
    def __init__(self, context: "zmq.Context"):
        """
        Initialize connection pool.
        
        Args:
            context: ZeroMQ context
        """
        if not ZMQ_AVAILABLE:
            raise ImportError("ZeroMQ not available")
        
        self.context = context
        self.workers: Dict[str, ZMQSocket] = {}
        self.worker_addresses: Dict[str, str] = {}
        self.worker_health: Dict[str, bool] = {}
        self.last_heartbeat: Dict[str, float] = {}
    
    def add_worker(self, worker_id: str, address: str):
        """
        Add worker to pool.
        
        Args:
            worker_id: Worker identifier
            address: Worker address
        """
        socket = ZMQSocket(self.context, zmq.DEALER, identity=f"pool-{worker_id}")
        socket.connect(address)
        
        self.workers[worker_id] = socket
        self.worker_addresses[worker_id] = address
        self.worker_health[worker_id] = True
        self.last_heartbeat[worker_id] = time.time()
        
        logger.info(f"Added worker {worker_id} at {address}")
    
    def remove_worker(self, worker_id: str):
        """
        Remove worker from pool.
        
        Args:
            worker_id: Worker identifier
        """
        if worker_id in self.workers:
            self.workers[worker_id].close()
            del self.workers[worker_id]
            del self.worker_addresses[worker_id]
            del self.worker_health[worker_id]
            del self.last_heartbeat[worker_id]
            
            logger.info(f"Removed worker {worker_id}")
    
    def send_to_worker(
        self,
        worker_id: str,
        message: Message,
        timeout: int = 5000
    ) -> bool:
        """
        Send message to specific worker.
        
        Args:
            worker_id: Worker identifier
            message: Message to send
            timeout: Send timeout
        
        Returns:
            True if sent successfully
        """
        if worker_id not in self.workers:
            logger.error(f"Worker {worker_id} not in pool")
            return False
        
        try:
            self.workers[worker_id].send_message(message, timeout)
            return True
        except Exception as e:
            logger.error(f"Failed to send to {worker_id}: {e}")
            self.worker_health[worker_id] = False
            return False
    
    def get_healthy_workers(self) -> List[str]:
        """
        Get list of healthy workers.
        
        Returns:
            List of healthy worker IDs
        """
        return [
            worker_id
            for worker_id, healthy in self.worker_health.items()
            if healthy
        ]
    
    def check_heartbeats(self, timeout: float = 30.0):
        """
        Check worker heartbeats and mark stale workers as unhealthy.
        
        Args:
            timeout: Heartbeat timeout in seconds
        """
        current_time = time.time()
        
        for worker_id in list(self.workers.keys()):
            time_since_heartbeat = current_time - self.last_heartbeat[worker_id]
            
            if time_since_heartbeat > timeout:
                logger.warning(
                    f"Worker {worker_id} heartbeat timeout "
                    f"({time_since_heartbeat:.1f}s)"
                )
                self.worker_health[worker_id] = False
    
    def update_heartbeat(self, worker_id: str):
        """
        Update heartbeat timestamp for worker.
        
        Args:
            worker_id: Worker identifier
        """
        if worker_id in self.workers:
            self.last_heartbeat[worker_id] = time.time()
            self.worker_health[worker_id] = True
    
    def close_all(self):
        """Close all worker connections."""
        for worker_id in list(self.workers.keys()):
            self.workers[worker_id].close()
        
        self.workers.clear()
        self.worker_addresses.clear()
        self.worker_health.clear()
        self.last_heartbeat.clear()


# Demo code
if __name__ == "__main__":
    print("=" * 70)
    print("ZeroMQ Communication Layer Demo")
    print("=" * 70)
    
    if not ZMQ_AVAILABLE:
        print("\n❌ ZeroMQ not available")
        print("Install with: pip install pyzmq")
    else:
        print("\n✅ ZeroMQ available")
        print("Version:", zmq.zmq_version())
    
    if not MSGPACK_AVAILABLE:
        print("\n⚠️  MessagePack not available (will use JSON)")
        print("Install with: pip install msgpack")
    else:
        print("\n✅ MessagePack available")
    
    print("\n" + "=" * 70)
    print("Message Example:")
    print("-" * 70)
    
    msg = Message(
        msg_id=str(uuid.uuid4()),
        msg_type=MessageType.TASK,
        sender_id="coordinator",
        receiver_id="worker1",
        payload={"model": "mobilenet.onnx", "input": "image.jpg"}
    )
    
    print(f"Message ID: {msg.msg_id}")
    print(f"Type: {msg.msg_type.value}")
    print(f"Sender: {msg.sender_id}")
    print(f"Receiver: {msg.receiver_id}")
    print(f"Payload: {msg.payload}")
    
    # Test serialization
    serialized = msg.serialize()
    print(f"\nSerialized size: {len(serialized)} bytes")
    
    deserialized = Message.deserialize(serialized)
    print(f"Deserialized type: {deserialized.msg_type.value}")
    
    print("\n" + "=" * 70)
