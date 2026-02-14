"""
Production Deployment Complete Demo

End-to-end demonstration of production deployment pipeline:
1. Train model
2. Export to ONNX
3. Deploy with async inference
4. Monitor with Prometheus/Grafana

Author: AMD GPU Computing Team
Date: January 21, 2026
"""

import asyncio
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from src.api.async_inference import AsyncInferenceQueue, JobPriority
from src.api.monitoring import InferenceMetrics
from src.inference.onnx_engine import ONNXEngine
from src.inference.onnx_export import export_to_onnx, get_model_info

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# ============================================================================
# Demo Model
# ============================================================================


class DemoModel(nn.Module):
    """Simple CNN for CIFAR-10"""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64, 10)

    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


# ============================================================================
# Step 1: Train Model
# ============================================================================


def train_demo_model():
    """Train simple demo model"""
    logger.info("=" * 80)
    logger.info("STEP 1: Training Demo Model")
    logger.info("=" * 80)

    model = DemoModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # Generate dummy training data
    logger.info("Generating dummy training data...")
    X_train = torch.randn(100, 3, 32, 32)
    y_train = torch.randint(0, 10, (100,))

    # Quick training
    logger.info("Training for 10 epochs...")
    model.train()
    for epoch in range(10):
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 5 == 0:
            logger.info(f"Epoch {epoch + 1}/10 - Loss: {loss.item():.4f}")

    logger.info("✓ Model training complete")
    return model


# ============================================================================
# Step 2: Export to ONNX
# ============================================================================


def export_model_to_onnx(model, output_path):
    """Export model to ONNX format"""
    logger.info("\n" + "=" * 80)
    logger.info("STEP 2: Exporting Model to ONNX")
    logger.info("=" * 80)

    # Export with all features
    logger.info(f"Exporting to {output_path}...")

    success = export_to_onnx(
        model=model,
        output_path=output_path,
        input_shape=(1, 3, 32, 32),
        opset_version=13,
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        optimize=True,
        verify=True,
        quantize=True,  # Enable INT8 quantization
    )

    if success:
        logger.info("✓ Export successful")

        # Get model info
        info = get_model_info(output_path)
        logger.info(f"\nModel Info:")
        logger.info(f"  - Opset Version: {info['opset_version']}")
        logger.info(f"  - Total Parameters: {info['total_parameters']:,}")
        logger.info(f"  - Number of Nodes: {info['num_nodes']}")
        logger.info(f"  - File Size: {info['file_size_mb']:.2f} MB")
        logger.info(f"  - Operators: {list(info['operators'].keys())}")
    else:
        logger.error("✗ Export failed")
        raise RuntimeError("ONNX export failed")


# ============================================================================
# Step 3: Setup Async Inference
# ============================================================================


async def setup_async_inference(model_path):
    """Setup async inference queue"""
    logger.info("\n" + "=" * 80)
    logger.info("STEP 3: Setting Up Async Inference")
    logger.info("=" * 80)

    # Load ONNX model
    logger.info(f"Loading ONNX model from {model_path}...")
    engine = ONNXEngine(
        model_path=str(model_path),
        device="cpu",  # RX 580 uses CPU inference with ROCm
        batch_size=8,
    )

    # Define inference function
    async def inference_fn(batch_data):
        """Batch inference function"""
        # Convert to numpy array
        batch_array = np.stack(batch_data)

        # Run inference
        outputs = engine.infer({"input": batch_array})

        # Return results
        return outputs["output"].tolist()

    # Create async queue
    logger.info("Creating async inference queue...")
    queue = AsyncInferenceQueue(
        inference_fn=inference_fn,
        batch_size=8,
        max_wait_time=0.1,
        num_workers=2,
        max_queue_size=1000,
        result_ttl=3600,
    )

    await queue.start()
    logger.info("✓ Async inference queue ready")

    return queue, engine


# ============================================================================
# Step 4: Run Inference Demo
# ============================================================================


async def run_inference_demo(queue):
    """Run inference demo with different priorities"""
    logger.info("\n" + "=" * 80)
    logger.info("STEP 4: Running Inference Demo")
    logger.info("=" * 80)

    # Submit test jobs with different priorities
    logger.info("Submitting test jobs...")

    test_data = []
    job_ids = []
    priorities = [
        JobPriority.LOW,
        JobPriority.NORMAL,
        JobPriority.NORMAL,
        JobPriority.HIGH,
        JobPriority.URGENT,
    ]

    for i, priority in enumerate(priorities):
        # Generate random input
        data = np.random.randn(3, 32, 32).astype(np.float32)
        test_data.append(data)

        # Submit job
        job_id = await queue.submit(data, priority=priority)
        job_ids.append(job_id)
        logger.info(f"  Job {i + 1}: ID={job_id[:8]}... Priority={priority.name}")

    # Wait for results
    logger.info("\nWaiting for results...")
    results = []
    for i, job_id in enumerate(job_ids):
        result = await queue.get_result(job_id, timeout=5.0)
        results.append(result)

        # Get predicted class
        pred_class = np.argmax(result)
        confidence = np.max(result)
        logger.info(f"  Job {i + 1}: Class={pred_class}, Confidence={confidence:.4f}")

    logger.info("✓ All inference jobs complete")

    return results


# ============================================================================
# Step 5: Show Statistics
# ============================================================================


async def show_statistics(queue):
    """Display queue statistics"""
    logger.info("\n" + "=" * 80)
    logger.info("STEP 5: Queue Statistics")
    logger.info("=" * 80)

    stats = queue.get_statistics()

    logger.info(f"Total Jobs:")
    logger.info(f"  - Submitted: {stats['total_submitted']}")
    logger.info(f"  - Completed: {stats['total_completed']}")
    logger.info(f"  - Failed: {stats['total_failed']}")

    logger.info(f"\nBatch Processing:")
    logger.info(f"  - Total Batches: {stats['total_batches']}")
    logger.info(f"  - Avg Batch Size: {stats['avg_batch_size']:.2f}")

    logger.info(f"\nPerformance:")
    logger.info(f"  - Avg Latency: {stats['avg_latency']:.4f}s")
    logger.info(f"  - Queue Size: {stats['queue_size']}")


# ============================================================================
# Step 6: Monitor with Prometheus
# ============================================================================


def setup_monitoring():
    """Setup Prometheus monitoring"""
    logger.info("\n" + "=" * 80)
    logger.info("STEP 6: Monitoring Setup")
    logger.info("=" * 80)

    logger.info("Prometheus metrics available at: http://localhost:8000/metrics")
    logger.info("Grafana dashboard at: http://localhost:3000")
    logger.info("\nMetrics include:")
    logger.info("  - inference_requests_total")
    logger.info("  - inference_latency_seconds")
    logger.info("  - gpu_utilization_percent")
    logger.info("  - gpu_memory_used_mb")
    logger.info("  - batch_size")
    logger.info("  - queue_size")

    logger.info("\n✓ Monitoring ready (start servers separately)")


# ============================================================================
# Main Demo
# ============================================================================


async def main():
    """Main demo pipeline"""
    logger.info("=" * 80)
    logger.info("PRODUCTION DEPLOYMENT COMPLETE DEMO")
    logger.info("=" * 80)
    logger.info("\nThis demo shows the complete production pipeline:")
    logger.info("  1. Train model")
    logger.info("  2. Export to ONNX (with optimization & quantization)")
    logger.info("  3. Setup async inference queue")
    logger.info("  4. Run inference with priority scheduling")
    logger.info("  5. Display statistics")
    logger.info("  6. Show monitoring setup")

    # Setup paths
    output_dir = Path("outputs/production_demo")
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "demo_model.onnx"

    try:
        # Step 1: Train model
        model = train_demo_model()

        # Step 2: Export to ONNX
        export_model_to_onnx(model, model_path)

        # Step 3: Setup async inference
        queue, engine = await setup_async_inference(model_path)

        # Step 4: Run inference demo
        results = await run_inference_demo(queue)

        # Step 5: Show statistics
        await show_statistics(queue)

        # Step 6: Monitoring setup
        setup_monitoring()

        # Cleanup
        logger.info("\n" + "=" * 80)
        logger.info("Shutting down...")
        await queue.shutdown()
        logger.info("✓ Demo complete")

        logger.info("\n" + "=" * 80)
        logger.info("NEXT STEPS:")
        logger.info("=" * 80)
        logger.info("1. Start REST API server:")
        logger.info("   python -m uvicorn src.api.server:app --reload")
        logger.info("\n2. Start Prometheus:")
        logger.info("   docker-compose up prometheus")
        logger.info("\n3. Start Grafana:")
        logger.info("   docker-compose up grafana")
        logger.info("\n4. Access Grafana dashboard:")
        logger.info("   http://localhost:3000")
        logger.info("   Username: admin, Password: admin")
        logger.info("\n5. Test API endpoints:")
        logger.info("   curl -X POST http://localhost:8000/predict \\")
        logger.info("     -H 'Content-Type: application/json' \\")
        logger.info("     -d '{\"data\": [...]}'")

    except Exception as e:
        logger.error(f"\n✗ Demo failed: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
