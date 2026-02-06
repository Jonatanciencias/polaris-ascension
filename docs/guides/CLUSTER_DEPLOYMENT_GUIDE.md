# 🚀 Cluster Deployment Guide
## Session 33 - Distributed Computing Deployment

**Date**: Enero 22, 2026  
**Version**: 0.7.0-dev  
**Target**: Production deployment of distributed cluster

---

## 📋 Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Prerequisites](#prerequisites)
4. [Single Node Deployment](#single-node-deployment)
5. [Multi-Node Cluster](#multi-node-cluster)
6. [Docker Deployment](#docker-deployment)
7. [Load Balancing](#load-balancing)
8. [Monitoring](#monitoring)
9. [Troubleshooting](#troubleshooting)
10. [Security](#security)

---

## Overview

The Legacy GPU AI platform supports three deployment modes:

### 1. **Standalone Mode**
- Single GPU, local processing
- Best for: Development, testing, single-machine deployments
- Latency: Minimal (no network overhead)

### 2. **LAN Cluster Mode**
- Multiple machines in local network
- Best for: Labs, offices, small datacenters
- Latency: Low (<5ms typically)

### 3. **WAN Distributed Mode**
- Machines across internet
- Best for: Multi-site deployments, cloud-edge hybrid
- Latency: Higher (varies by network)

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      CLIENT LAYER                            │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐                   │
│  │   CLI    │  │ REST API │  │  Web UI  │                   │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘                   │
│       │             │             │                          │
│       └─────────────┼─────────────┘                          │
│                     │                                        │
└─────────────────────┼────────────────────────────────────────┘
                      │
                      ↓
┌─────────────────────────────────────────────────────────────┐
│                  COORDINATOR LAYER                           │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐   │
│  │           Cluster Coordinator                        │   │
│  │  - Task queue (priority-based)                       │   │
│  │  - Worker registry                                   │   │
│  │  - Load balancing (5 strategies)                     │   │
│  │  - Health monitoring                                 │   │
│  │  - Fault tolerance                                   │   │
│  └──────────────────┬───────────────────────────────────┘   │
│                     │ ZMQ (tcp://coordinator:5555)          │
└─────────────────────┼────────────────────────────────────────┘
                      │
        ┌─────────────┼─────────────┬─────────────┐
        │             │             │             │
        ↓             ↓             ↓             ↓
┌───────────────┐ ┌───────────────┐ ┌───────────────┐
│   WORKER 1    │ │   WORKER 2    │ │   WORKER N    │
│               │ │               │ │               │
│ RX 580 8GB    │ │ Vega 56 8GB   │ │ RX 5700 8GB   │
│ 4 tasks/node  │ │ 6 tasks/node  │ │ 8 tasks/node  │
│               │ │               │ │               │
│ [Inference]   │ │ [Inference]   │ │ [Inference]   │
└───────────────┘ └───────────────┘ └───────────────┘
```

---

## Prerequisites

### System Requirements

#### Coordinator Node
- **CPU**: 2+ cores
- **RAM**: 4GB+
- **Network**: Stable connection
- **OS**: Linux (Ubuntu 20.04+ recommended)

#### Worker Nodes
- **GPU**: AMD Polaris/Vega/Navi (RX 580, Vega 56, RX 5700, etc.)
- **VRAM**: 4GB+ minimum
- **RAM**: 8GB+
- **Drivers**: ROCm 5.0+ or OpenCL
- **OS**: Linux (Ubuntu 20.04+ recommended)

### Software Dependencies

```bash
# Python 3.10+
python3 --version

# Required packages
pip install legacygpu-ai onnxruntime pyzmq msgpack-python

# Optional: For API server
pip install fastapi uvicorn

# Optional: For monitoring
pip install prometheus-client
```

---

## Single Node Deployment

### Quick Start (All-in-One)

1. **Start API Server**:
```bash
# Terminal 1: Start API with cluster support
legacygpu api start --host 0.0.0.0 --port 8000
```

2. **Start Worker**:
```bash
# Terminal 2: Start worker (connects to coordinator)
legacygpu worker start --coordinator tcp://localhost:5555
```

3. **Submit Task**:
```bash
# Terminal 3: Submit inference task
legacygpu task submit --model resnet50 --input image.jpg --wait
```

### Configuration File

Create `cluster_config.yaml`:

```yaml
coordinator:
  bind_address: "tcp://0.0.0.0:5555"
  balancing_strategy: "ADAPTIVE"
  max_queue_size: 1000
  heartbeat_interval: 5.0
  heartbeat_timeout: 15.0

worker:
  coordinator_address: "tcp://localhost:5555"
  max_concurrent_tasks: 4
  heartbeat_interval: 5.0

api:
  host: "0.0.0.0"
  port: 8000
  enable_cors: true
```

---

## Multi-Node Cluster

### Network Setup

#### 1. **Choose Coordinator Address**

The coordinator must be accessible from all workers.

**Option A: Fixed IP**
```bash
# On coordinator machine
export COORDINATOR_IP=$(hostname -I | awk '{print $1}')
echo "Coordinator IP: $COORDINATOR_IP"
```

**Option B: DNS Name**
```bash
# If you have DNS
export COORDINATOR_HOST="cluster-coordinator.local"
```

#### 2. **Configure Firewall**

```bash
# On coordinator machine - allow ZMQ port
sudo ufw allow 5555/tcp comment "Cluster Coordinator"

# On all machines - allow internal communication
sudo ufw allow from 192.168.1.0/24  # Adjust to your subnet
```

### Deployment Steps

#### Step 1: Start Coordinator

On the **coordinator machine**:

```bash
# Method 1: CLI (foreground)
legacygpu cluster start \
  --bind-address tcp://0.0.0.0:5555 \
  --strategy ADAPTIVE

# Method 2: Via API server
legacygpu api start --port 8000
# Coordinator starts automatically
```

#### Step 2: Start Workers

On **each worker machine**:

```bash
# Replace COORDINATOR_IP with actual IP
export COORDINATOR_IP=192.168.1.100

legacygpu worker start \
  --coordinator tcp://${COORDINATOR_IP}:5555 \
  --worker-id worker-$(hostname) \
  --max-tasks 4
```

#### Step 3: Verify Cluster

```bash
# Check cluster status
legacygpu cluster status

# Or via API
curl http://${COORDINATOR_IP}:8000/cluster/health | jq
```

Expected output:
```json
{
  "status": "HEALTHY",
  "total_workers": 3,
  "healthy_workers": 3,
  "pending_tasks": 0,
  "running_tasks": 0
}
```

#### Step 4: Submit Tasks

```bash
# Submit task to cluster
legacygpu task submit \
  --model resnet50 \
  --input image.jpg \
  --priority HIGH \
  --wait

# Batch submission
for i in {1..10}; do
  legacygpu task submit --model yolo --input frame_$i.jpg
done
```

### Example: 3-Node Lab Cluster

```yaml
# Lab Cluster Configuration

# Node 1: Coordinator + Worker
hostname: lab-gpu-1
ip: 192.168.1.100
gpu: RX 580 8GB
role: coordinator + worker

# Node 2: Worker
hostname: lab-gpu-2
ip: 192.168.1.101
gpu: Vega 56 8GB
role: worker

# Node 3: Worker
hostname: lab-gpu-3
ip: 192.168.1.102
gpu: RX 5700 8GB
role: worker
```

**Setup scripts**:

```bash
# On Node 1 (Coordinator)
#!/bin/bash
# setup_coordinator.sh

export BIND_IP=192.168.1.100

# Start API server (includes coordinator)
legacygpu api start --host 0.0.0.0 --port 8000 &

# Wait for coordinator to start
sleep 2

# Start local worker
legacygpu worker start \
  --coordinator tcp://localhost:5555 \
  --worker-id worker-lab-1 \
  --max-tasks 4 &

echo "Coordinator and Worker 1 started"
```

```bash
# On Node 2 & 3 (Workers)
#!/bin/bash
# setup_worker.sh

COORDINATOR_IP=192.168.1.100
WORKER_ID=worker-$(hostname)

legacygpu worker start \
  --coordinator tcp://${COORDINATOR_IP}:5555 \
  --worker-id ${WORKER_ID} \
  --max-tasks 6

echo "Worker ${WORKER_ID} started"
```

---

## Docker Deployment

### Dockerfile

```dockerfile
# Dockerfile
FROM ubuntu:22.04

# Install ROCm (for AMD GPUs)
RUN apt-get update && apt-get install -y \\
    wget gnupg2 software-properties-common \\
    && wget -q -O - https://repo.radeon.com/rocm/rocm.gpg.key | apt-key add - \\
    && echo 'deb [arch=amd64] https://repo.radeon.com/rocm/apt/debian/ ubuntu main' | tee /etc/apt/sources.list.d/rocm.list \\
    && apt-get update \\
    && apt-get install -y rocm-dev

# Install Python
RUN apt-get install -y python3.10 python3-pip

# Install Legacy GPU AI
RUN pip3 install legacygpu-ai[distributed]

# Expose ports
EXPOSE 5555 8000

CMD ["bash"]
```

### Docker Compose

```yaml
# docker-compose.yml
version: '3.8'

services:
  coordinator:
    build: .
    image: legacygpu-cluster:latest
    container_name: cluster-coordinator
    ports:
      - "8000:8000"  # API
      - "5555:5555"  # Coordinator
    command: legacygpu api start --host 0.0.0.0 --port 8000
    networks:
      - cluster-net
    restart: unless-stopped

  worker1:
    build: .
    image: legacygpu-cluster:latest
    container_name: cluster-worker-1
    devices:
      - /dev/kfd:/dev/kfd
      - /dev/dri:/dev/dri
    environment:
      - COORDINATOR_ADDRESS=tcp://coordinator:5555
    command: >
      legacygpu worker start
        --coordinator tcp://coordinator:5555
        --worker-id worker-1
        --max-tasks 4
    depends_on:
      - coordinator
    networks:
      - cluster-net
    restart: unless-stopped

  worker2:
    build: .
    image: legacygpu-cluster:latest
    container_name: cluster-worker-2
    devices:
      - /dev/kfd:/dev/kfd
      - /dev/dri:/dev/dri
    environment:
      - COORDINATOR_ADDRESS=tcp://coordinator:5555
    command: >
      legacygpu worker start
        --coordinator tcp://coordinator:5555
        --worker-id worker-2
        --max-tasks 4
    depends_on:
      - coordinator
    networks:
      - cluster-net
    restart: unless-stopped

networks:
  cluster-net:
    driver: bridge
```

### Launch Docker Cluster

```bash
# Build images
docker-compose build

# Start cluster
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f coordinator
docker-compose logs -f worker1

# Stop cluster
docker-compose down
```

---

## Load Balancing

### Available Strategies

#### 1. **ROUND_ROBIN** (Default)
- Simple rotation through workers
- Good for: Homogeneous cluster, equal task sizes
- Pro: Fair distribution
- Con: Ignores worker load

```bash
legacygpu cluster start --strategy ROUND_ROBIN
```

#### 2. **LEAST_LOADED**
- Selects worker with lowest current load
- Good for: Variable task sizes
- Pro: Better utilization
- Con: Requires real-time load tracking

```bash
legacygpu cluster start --strategy LEAST_LOADED
```

#### 3. **GPU_MATCH**
- Matches task requirements to GPU capabilities
- Good for: Mixed GPU types (RX 580, Vega 56, etc.)
- Pro: Hardware-aware
- Con: Requires capability metadata

```bash
legacygpu cluster start --strategy GPU_MATCH
```

#### 4. **LATENCY_BASED**
- Selects worker with lowest historical latency
- Good for: Latency-sensitive applications
- Pro: Optimizes for speed
- Con: May create hot spots

```bash
legacygpu cluster start --strategy LATENCY
```

#### 5. **ADAPTIVE** (Recommended)
- Learns optimal worker selection over time
- Good for: Production deployments
- Pro: Adapts to changing conditions
- Con: Needs warm-up period

```bash
legacygpu cluster start --strategy ADAPTIVE
```

### Change Strategy at Runtime

```bash
# Via CLI
curl -X PUT http://coordinator:8000/cluster/config/balancing \\
  -H "Content-Type: application/json" \\
  -d '{"strategy": "ADAPTIVE"}'

# Response
{
  "status": "success",
  "new_strategy": "ADAPTIVE"
}
```

---

## Monitoring

### Built-in Monitoring

#### 1. **CLI Monitoring**

```bash
# Cluster status
legacygpu cluster status

# Worker list
legacygpu cluster workers

# Task list
legacygpu task list
```

#### 2. **REST API Monitoring**

```bash
# Health check
curl http://coordinator:8000/cluster/health | jq

# Metrics
curl http://coordinator:8000/cluster/metrics | jq

# Worker details
curl http://coordinator:8000/cluster/workers | jq
```

#### 3. **Web Dashboard** (Coming Soon)

```bash
# Start web UI
legacygpu ui start --port 8080

# Open browser
open http://localhost:8080
```

### Prometheus Integration

```python
# Enable Prometheus metrics in config
monitoring:
  prometheus:
    enabled: true
    port: 9090
    metrics:
      - cluster_workers_total
      - cluster_workers_healthy
      - cluster_tasks_pending
      - cluster_tasks_running
      - cluster_tasks_completed
      - cluster_task_latency_seconds
```

```yaml
# prometheus.yml
scrape_configs:
  - job_name: 'legacygpu-cluster'
    static_configs:
      - targets: ['coordinator:9090']
```

---

## Troubleshooting

### Common Issues

#### 1. **Workers Can't Connect**

**Symptom**: Worker logs show connection errors

```
ERROR: Failed to connect to coordinator at tcp://192.168.1.100:5555
```

**Solutions**:
```bash
# Check coordinator is running
legacygpu cluster status

# Check firewall
sudo ufw status
sudo ufw allow 5555/tcp

# Check network connectivity
ping 192.168.1.100
telnet 192.168.1.100 5555

# Check coordinator bind address
# Should be 0.0.0.0, not 127.0.0.1
```

#### 2. **Tasks Stay in PENDING**

**Symptom**: Tasks submitted but never execute

**Solutions**:
```bash
# Check if workers are registered
legacygpu cluster workers

# Check worker health
curl http://coordinator:8000/cluster/health | jq '.healthy_workers'

# Check worker logs
journalctl -u legacygpu-worker -f

# Restart workers
legacygpu worker stop && legacygpu worker start
```

#### 3. **High Latency**

**Symptom**: Tasks taking longer than expected

**Solutions**:
```bash
# Check network latency
ping -c 10 worker-node

# Check GPU utilization
rocm-smi  # On worker node

# Switch to LATENCY balancing
curl -X PUT http://coordinator:8000/cluster/config/balancing \\
  -d '{"strategy": "LATENCY"}'

# Check task queue
curl http://coordinator:8000/cluster/metrics | jq '.tasks_pending'
```

#### 4. **Worker Marked Unhealthy**

**Symptom**: Worker shown as UNHEALTHY in status

**Solutions**:
```bash
# Check heartbeat settings (default: 5s interval, 15s timeout)
# Worker must send heartbeat every 5s

# Check worker process is running
ps aux | grep legacygpu

# Check system load on worker
top  # CPU usage
nvidia-smi  # Or rocm-smi for GPU

# Restart worker
kill <worker-pid>
legacygpu worker start
```

### Debug Mode

```bash
# Run with debug logging
export LEGACYGPU_LOG_LEVEL=DEBUG

legacygpu cluster start --strategy ADAPTIVE

# Or
LOG_LEVEL=DEBUG legacygpu worker start
```

### Logs Location

```
/var/log/legacygpu/coordinator.log
/var/log/legacygpu/worker-<id>.log
~/.legacygpu/logs/
```

---

## Security

### Production Security Checklist

- [ ] Use TLS/SSL for all network communication
- [ ] Enable API authentication (API keys)
- [ ] Restrict coordinator bind address to internal network
- [ ] Use firewall rules to limit access
- [ ] Run workers with minimal privileges
- [ ] Monitor for unauthorized access attempts
- [ ] Regularly update all dependencies
- [ ] Use separate network for cluster communication

### Enable Authentication

```yaml
# config.yaml
api:
  authentication:
    enabled: true
    method: "api_key"
    keys:
      - key: "your-secret-api-key"
        permissions: ["admin"]
      - key: "worker-key"
        permissions: ["worker"]
```

```bash
# Use API key in requests
curl -H "X-API-Key: your-secret-api-key" \\
  http://coordinator:8000/cluster/health
```

### Network Isolation

```bash
# Create isolated network for cluster
docker network create --driver bridge \\
  --subnet 172.20.0.0/16 \\
  --opt "com.docker.network.bridge.name"="cluster-br" \\
  cluster-net

# Run containers in isolated network
docker run --network cluster-net ...
```

---

## Best Practices

### 1. **Start Small, Scale Gradually**
- Begin with 1 coordinator + 1-2 workers
- Test thoroughly before adding more nodes
- Monitor performance at each scale

### 2. **Use Health Checks**
- Monitor coordinator and workers continuously
- Set up alerts for failures
- Implement automatic restarts

### 3. **Optimize Task Sizes**
- Avoid very small tasks (network overhead)
- Batch small inputs when possible
- Use appropriate timeout values

### 4. **Monitor Resource Usage**
- Track GPU utilization
- Monitor memory usage
- Watch network bandwidth

### 5. **Plan for Failures**
- Workers can fail - coordinator handles it
- Use retry logic for critical tasks
- Implement backup coordinator (future feature)

---

## Performance Tuning

### Coordinator Tuning

```yaml
coordinator:
  # Increase for high-throughput
  max_queue_size: 10000
  
  # Reduce for faster failure detection
  heartbeat_interval: 3.0
  heartbeat_timeout: 10.0
  
  # Best strategy for production
  balancing_strategy: "ADAPTIVE"
```

### Worker Tuning

```yaml
worker:
  # Adjust based on GPU memory
  max_concurrent_tasks: 4  # RX 580 8GB
  # max_concurrent_tasks: 6  # Vega 56 8GB
  # max_concurrent_tasks: 8  # RX 5700 8GB
  
  # Reduce for faster updates
  heartbeat_interval: 3.0
```

### Network Tuning

```bash
# Increase TCP buffer sizes (Linux)
sudo sysctl -w net.core.rmem_max=134217728
sudo sysctl -w net.core.wmem_max=134217728

# Increase socket buffer
sudo sysctl -w net.ipv4.tcp_rmem='4096 87380 134217728'
sudo sysctl -w net.ipv4.tcp_wmem='4096 65536 134217728'
```

---

## Next Steps

1. ✅ Deploy single-node cluster
2. ✅ Add 1-2 worker nodes
3. ✅ Test with real workloads
4. ✅ Monitor performance
5. ⬜ Scale to production size
6. ⬜ Implement monitoring dashboard
7. ⬜ Set up automated deployment (Kubernetes)

---

## Support & Resources

- **Documentation**: https://github.com/yourusername/legacygpu-ai/docs
- **Issues**: https://github.com/yourusername/legacygpu-ai/issues
- **Discord**: https://discord.gg/legacygpu
- **Email**: support@legacygpu.ai

---

**Last Updated**: Enero 22, 2026  
**Version**: Session 33  
**Status**: Production Ready ✅
