"""Tests for Profiler module"""
import pytest
import time
from core.profiler import Profiler, ProfileEntry


def test_profiler_initialization():
    """Test profiler initialization"""
    profiler = Profiler()
    assert profiler is not None


def test_profiler_single_operation():
    """Test profiling a single operation"""
    profiler = Profiler()
    
    profiler.start("test_op")
    time.sleep(0.01)  # Simulate work
    profiler.end("test_op")
    
    summary = profiler.get_summary()
    assert "test_op" in summary
    assert summary["test_op"]["count"] == 1
    assert summary["test_op"]["avg_ms"] >= 10  # At least 10ms


def test_profiler_multiple_operations():
    """Test profiling multiple operations"""
    profiler = Profiler()
    
    for i in range(5):
        profiler.start("loop_op")
        time.sleep(0.005)
        profiler.end("loop_op")
    
    summary = profiler.get_summary()
    assert summary["loop_op"]["count"] == 5
    assert summary["loop_op"]["min_ms"] > 0
    assert summary["loop_op"]["max_ms"] > summary["loop_op"]["min_ms"]


def test_profiler_metadata():
    """Test profiler metadata storage"""
    profiler = Profiler()
    
    profiler.start("test_op", model="gpt2", batch_size=4)
    profiler.end("test_op")
    
    # Metadata is stored but not in summary
    summary = profiler.get_summary()
    assert "test_op" in summary


def test_profiler_reset():
    """Test profiler reset"""
    profiler = Profiler()
    
    profiler.start("test_op")
    profiler.end("test_op")
    
    summary = profiler.get_summary()
    assert len(summary) == 1
    
    profiler.reset()
    summary = profiler.get_summary()
    assert len(summary) == 0


def test_profiler_nested_warning():
    """Test warning for nested profiling"""
    profiler = Profiler()
    
    profiler.start("op1")
    profiler.start("op1")  # Should warn
    profiler.end("op1")
    
    # Should still work
    summary = profiler.get_summary()
    assert "op1" in summary


def test_profile_entry():
    """Test ProfileEntry dataclass"""
    entry = ProfileEntry(
        name="test",
        start_time=1.0,
        end_time=2.0,
        duration_ms=1000.0
    )
    
    assert entry.name == "test"
    assert entry.duration_ms == 1000.0
