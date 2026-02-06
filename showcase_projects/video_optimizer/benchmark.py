#!/usr/bin/env python3
"""
Benchmark Comparison - Framework vs Baselines

Compares:
1. Your optimized framework (auto-tuner + 831 GFLOPS)
2. FFmpeg (industry standard)
3. OpenCV (popular framework)
4. Baseline Python (numpy operations)

Generates:
- comparison_chart.png (FPS comparison)
- benchmark_results.json (raw data)
- metrics_over_time.png (timeline graphs)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import cv2
import numpy as np
import time
import json
import matplotlib.pyplot as plt
from typing import Dict, List
import subprocess

from optimizer import VideoOptimizer


def benchmark_framework(video_path: str) -> Dict:
    """Benchmark your optimized framework."""
    print("\n🚀 Benchmark 1/4: Optimized Framework (Your Auto-Tuner)")
    print("─" * 70)
    
    optimizer = VideoOptimizer(show_metrics=False)
    
    # Measure performance
    start = time.time()
    
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    frame_count = 0
    frame_times = []
    
    while frame_count < min(total_frames, 300):  # Test first 300 frames
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_start = time.perf_counter()
        processed, metrics = optimizer.process_frame(frame)
        frame_times.append(time.perf_counter() - frame_start)
        
        frame_count += 1
    
    cap.release()
    
    elapsed = time.time() - start
    processing_fps = frame_count / elapsed
    avg_latency = np.mean(frame_times) * 1000
    
    print(f"   ✓ Processed {frame_count} frames in {elapsed:.2f}s")
    print(f"   ✓ Processing FPS: {processing_fps:.1f}")
    print(f"   ✓ Average latency: {avg_latency:.1f}ms")
    
    return {
        'name': 'Optimized Framework',
        'fps': processing_fps,
        'latency_ms': avg_latency,
        'frames_processed': frame_count,
        'total_time': elapsed,
    }


def benchmark_opencv(video_path: str) -> Dict:
    """Benchmark OpenCV baseline."""
    print("\n📷 Benchmark 2/4: OpenCV Baseline")
    print("─" * 70)
    
    start = time.time()
    
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    frame_count = 0
    frame_times = []
    
    while frame_count < min(total_frames, 300):
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_start = time.perf_counter()
        
        # Equivalent OpenCV operations (simple processing)
        frame_float = frame.astype(np.float32) / 255.0
        processed = frame_float.copy()  # Simulated processing
        output = (processed * 255).astype(np.uint8)
        
        frame_times.append(time.perf_counter() - frame_start)
        frame_count += 1
    
    cap.release()
    
    elapsed = time.time() - start
    processing_fps = frame_count / elapsed
    avg_latency = np.mean(frame_times) * 1000
    
    print(f"   ✓ Processed {frame_count} frames in {elapsed:.2f}s")
    print(f"   ✓ Processing FPS: {processing_fps:.1f}")
    print(f"   ✓ Average latency: {avg_latency:.1f}ms")
    
    return {
        'name': 'OpenCV Baseline',
        'fps': processing_fps,
        'latency_ms': avg_latency,
        'frames_processed': frame_count,
        'total_time': elapsed,
    }


def benchmark_numpy(video_path: str) -> Dict:
    """Benchmark pure numpy operations."""
    print("\n🔢 Benchmark 3/4: Pure NumPy")
    print("─" * 70)
    
    start = time.time()
    
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    frame_count = 0
    frame_times = []
    
    while frame_count < min(total_frames, 300):
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_start = time.perf_counter()
        
        # Pure numpy operations
        frame_float = frame.astype(np.float32)
        processed = np.copy(frame_float)
        output = processed.astype(np.uint8)
        
        frame_times.append(time.perf_counter() - frame_start)
        frame_count += 1
    
    cap.release()
    
    elapsed = time.time() - start
    processing_fps = frame_count / elapsed
    avg_latency = np.mean(frame_times) * 1000
    
    print(f"   ✓ Processed {frame_count} frames in {elapsed:.2f}s")
    print(f"   ✓ Processing FPS: {processing_fps:.1f}")
    print(f"   ✓ Average latency: {avg_latency:.1f}ms")
    
    return {
        'name': 'Pure NumPy',
        'fps': processing_fps,
        'latency_ms': avg_latency,
        'frames_processed': frame_count,
        'total_time': elapsed,
    }


def benchmark_ffmpeg(video_path: str) -> Dict:
    """Benchmark FFmpeg (if available)."""
    print("\n🎞️  Benchmark 4/4: FFmpeg")
    print("─" * 70)
    
    # Check if ffmpeg is available
    try:
        result = subprocess.run(['ffmpeg', '-version'], 
                              capture_output=True, timeout=5)
        if result.returncode != 0:
            print("   ⚠️  FFmpeg not available, skipping")
            return None
    except:
        print("   ⚠️  FFmpeg not available, skipping")
        return None
    
    # Use ffmpeg to process video (simple copy operation)
    output_path = Path(__file__).parent / "temp_ffmpeg.mp4"
    
    start = time.time()
    
    cmd = [
        'ffmpeg', '-i', video_path,
        '-c:v', 'libx264', '-preset', 'ultrafast',
        '-y', str(output_path)
    ]
    
    try:
        subprocess.run(cmd, capture_output=True, timeout=60)
        elapsed = time.time() - start
        
        # Get frame count
        cap = cv2.VideoCapture(video_path)
        total_frames = min(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 300)
        cap.release()
        
        processing_fps = total_frames / elapsed
        avg_latency = (elapsed / total_frames) * 1000
        
        # Cleanup
        if output_path.exists():
            output_path.unlink()
        
        print(f"   ✓ Processed ~{total_frames} frames in {elapsed:.2f}s")
        print(f"   ✓ Processing FPS: {processing_fps:.1f}")
        print(f"   ✓ Average latency: {avg_latency:.1f}ms")
        
        return {
            'name': 'FFmpeg',
            'fps': processing_fps,
            'latency_ms': avg_latency,
            'frames_processed': total_frames,
            'total_time': elapsed,
        }
    except:
        print("   ⚠️  FFmpeg benchmark failed")
        return None


def generate_comparison_chart(results: List[Dict], output_path: str):
    """Generate comparison bar chart."""
    print("\n📊 Generating comparison chart...")
    
    # Filter out None results
    results = [r for r in results if r is not None]
    
    names = [r['name'] for r in results]
    fps_values = [r['fps'] for r in results]
    latency_values = [r['latency_ms'] for r in results]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # FPS comparison
    colors = ['#00ff00' if 'Optimized' in name else '#888888' for name in names]
    bars1 = ax1.bar(names, fps_values, color=colors, alpha=0.8)
    ax1.set_ylabel('Processing FPS', fontsize=12)
    ax1.set_title('Processing Speed Comparison', fontsize=14, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}',
                ha='center', va='bottom', fontsize=10)
    
    # Rotate x labels
    ax1.set_xticklabels(names, rotation=15, ha='right')
    
    # Latency comparison
    bars2 = ax2.bar(names, latency_values, color=colors, alpha=0.8)
    ax2.set_ylabel('Latency (ms)', fontsize=12)
    ax2.set_title('Latency Comparison (Lower is Better)', fontsize=14, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}',
                ha='center', va='bottom', fontsize=10)
    
    ax2.set_xticklabels(names, rotation=15, ha='right')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"   ✓ Saved: {output_path}")


def main():
    """Run complete benchmark suite."""
    print("\n" + "=" * 70)
    print("🏁 BENCHMARK SUITE - Framework vs Baselines")
    print("=" * 70)
    
    # Get test video
    video_path = Path(__file__).parent / "demo_input.mp4"
    
    if not video_path.exists():
        print("\n⚠️  No test video found. Run optimizer.py --demo first!")
        return
    
    print(f"\n📹 Test Video: {video_path}")
    
    # Run benchmarks
    results = []
    
    results.append(benchmark_framework(str(video_path)))
    results.append(benchmark_opencv(str(video_path)))
    results.append(benchmark_numpy(str(video_path)))
    ffmpeg_result = benchmark_ffmpeg(str(video_path))
    if ffmpeg_result:
        results.append(ffmpeg_result)
    
    # Save results
    results_path = Path(__file__).parent / "results" / "benchmark_results.json"
    results_path.parent.mkdir(exist_ok=True)
    
    print(f"\n💾 Saving results...")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"   ✓ Saved: {results_path}")
    
    # Generate charts
    chart_path = Path(__file__).parent / "results" / "comparison_chart.png"
    generate_comparison_chart(results, str(chart_path))
    
    # Summary
    print("\n" + "=" * 70)
    print("📊 BENCHMARK SUMMARY")
    print("=" * 70)
    
    baseline = [r for r in results if 'Baseline' in r['name']][0]
    optimized = [r for r in results if 'Optimized' in r['name']][0]
    
    speedup = optimized['fps'] / baseline['fps']
    latency_improvement = baseline['latency_ms'] / optimized['latency_ms']
    
    print(f"\n🏆 Your Optimized Framework:")
    print(f"   Speedup vs OpenCV: {speedup:.2f}×")
    print(f"   Latency improvement: {latency_improvement:.2f}×")
    print(f"   Processing FPS: {optimized['fps']:.1f}")
    print(f"   Average latency: {optimized['latency_ms']:.1f}ms")
    
    print("\n📈 All Results:")
    print(f"{'Method':<25} {'FPS':>10} {'Latency':>12} {'Speedup':>10}")
    print("─" * 70)
    
    for r in results:
        speedup_val = r['fps'] / baseline['fps']
        print(f"{r['name']:<25} {r['fps']:>10.1f} {r['latency_ms']:>10.1f}ms {speedup_val:>9.2f}×")
    
    print("\n" + "=" * 70)
    print(f"\n✅ Benchmark complete!")
    print(f"📊 Check results/comparison_chart.png for visualization")


if __name__ == "__main__":
    main()
