#!/usr/bin/env python3
"""
Hybrid GEMM Kernel Compilation and Validation Script

Tasks:
1. Validate kernel compilation
2. Run unit tests
3. Generate performance baseline
4. Create validation report

Usage:
    python compile_hybrid_kernel.py [--verbose] [--benchmark] [--output report.json]
"""

import argparse
import sys
import logging
import json
from pathlib import Path
from datetime import datetime

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from src.opencl.hybrid_gemm import HybridGEMMExecutor, HybridGEMMConfig
from tests.test_gemm_hybrid import HybridGEMMTester, run_full_test_suite


def setup_logging(verbose: bool = False):
    """Configure logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)


def validate_compilation():
    """Validate that kernel compiles without errors."""
    logger = logging.getLogger(__name__)
    logger.info("=" * 80)
    logger.info("KERNEL COMPILATION VALIDATION")
    logger.info("=" * 80 + "\n")
    
    try:
        logger.info("Creating executor (compiling kernels)...")
        executor = HybridGEMMExecutor()
        logger.info("✅ Kernel compilation successful\n")
        return True, executor
    except Exception as e:
        logger.error(f"❌ Kernel compilation failed: {e}\n")
        return False, None


def run_quick_test(executor: HybridGEMMExecutor) -> bool:
    """Run quick functional test."""
    logger = logging.getLogger(__name__)
    logger.info("=" * 80)
    logger.info("QUICK FUNCTIONAL TEST")
    logger.info("=" * 80 + "\n")
    
    try:
        import numpy as np
        
        logger.info("Testing with n=512...")
        A = np.random.randn(512, 512).astype(np.float32)
        B = np.random.randn(512, 512).astype(np.float32)
        
        logger.info("Running GPU GEMM...")
        C_gpu = executor(A, B)
        
        logger.info("Computing reference (CPU)...")
        C_ref = A @ B
        
        error_rel = np.linalg.norm(C_gpu - C_ref) / np.linalg.norm(C_ref)
        
        logger.info(f"Relative error: {error_rel:.2e}")
        
        if error_rel < 1e-4:
            logger.info("✅ Functional test passed\n")
            return True
        else:
            logger.error(f"❌ Error too large: {error_rel:.2e} (threshold: 1e-4)\n")
            return False
            
    except Exception as e:
        logger.error(f"❌ Functional test failed: {e}\n")
        return False


def run_benchmark(executor: HybridGEMMExecutor, output_file: Path = None) -> dict:
    """Run performance benchmarks."""
    logger = logging.getLogger(__name__)
    
    logger.info("=" * 80)
    logger.info("PERFORMANCE BENCHMARK")
    logger.info("=" * 80 + "\n")
    
    tester = HybridGEMMTester(executor)
    results = tester.benchmark_suite(
        sizes=[512, 1024, 2048],
        iterations=10
    )
    
    # Generate plots
    plot_file = None
    if output_file:
        plot_file = output_file.parent / "plots.png"
        tester.plot_results(plot_file)
        logger.info(f"Plots saved to {plot_file}\n")
    
    return {
        'results': [r.to_dict() for r in results],
        'summary': tester._generate_summary(),
    }


def generate_validation_report(
    compilation_ok: bool,
    functional_ok: bool,
    benchmark_data: dict = None,
    output_file: Path = None
) -> dict:
    """Generate comprehensive validation report."""
    logger = logging.getLogger(__name__)
    
    report = {
        'timestamp': datetime.now().isoformat(),
        'compilation': {
            'status': 'PASS' if compilation_ok else 'FAIL',
        },
        'functional_test': {
            'status': 'PASS' if functional_ok else 'FAIL',
        },
        'overall_status': 'PASS' if (compilation_ok and functional_ok) else 'FAIL',
    }
    
    if benchmark_data:
        report['benchmark'] = benchmark_data
    
    if output_file:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
        logger.info(f"Report saved to {output_file}")
    
    return report


def print_summary(report: dict):
    """Print validation summary."""
    logger = logging.getLogger(__name__)
    
    logger.info("=" * 80)
    logger.info("VALIDATION SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Compilation:     {report['compilation']['status']}")
    logger.info(f"Functional test: {report['functional_test']['status']}")
    logger.info(f"Overall:         {report['overall_status']}")
    
    if 'benchmark' in report and 'summary' in report['benchmark']:
        summary = report['benchmark']['summary']
        logger.info(f"\nBenchmark Results:")
        logger.info(f"  Max GFLOPS:    {summary.get('max_gflops', 'N/A'):.1f}")
        logger.info(f"  Target:        {summary.get('target_gflops', 'N/A'):.1f}")
        logger.info(f"  Target Achieved: {summary.get('target_achieved', False)}")
    
    logger.info("=" * 80 + "\n")


def main():
    """Main validation pipeline."""
    parser = argparse.ArgumentParser(
        description='Hybrid GEMM Kernel Compilation and Validation'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    parser.add_argument(
        '--benchmark', '-b',
        action='store_true',
        help='Run performance benchmarks'
    )
    parser.add_argument(
        '--full-test', '-f',
        action='store_true',
        help='Run full test suite (comprehensive)'
    )
    parser.add_argument(
        '--output', '-o',
        type=Path,
        default=Path('test_results/validation_report.json'),
        help='Output file for validation report'
    )
    
    args = parser.parse_args()
    
    logger = setup_logging(args.verbose)
    
    logger.info("\n" + "=" * 80)
    logger.info("HYBRID GEMM KERNEL - VALIDATION PIPELINE")
    logger.info("=" * 80 + "\n")
    
    # Step 1: Compilation
    compilation_ok, executor = validate_compilation()
    
    if not compilation_ok:
        logger.error("Cannot proceed without successful compilation")
        sys.exit(1)
    
    # Step 2: Quick functional test
    functional_ok = run_quick_test(executor)
    
    if not functional_ok:
        logger.error("Functional test failed")
        sys.exit(1)
    
    # Step 3: Optional benchmarks
    benchmark_data = None
    if args.benchmark:
        benchmark_data = run_benchmark(executor, args.output)
    
    # Step 4: Optional full test suite
    if args.full_test:
        logger.info("Running full test suite...")
        run_full_test_suite()
    
    # Step 5: Generate report
    report = generate_validation_report(
        compilation_ok,
        functional_ok,
        benchmark_data,
        args.output
    )
    
    # Step 6: Print summary
    print_summary(report)
    
    # Return appropriate exit code
    return 0 if report['overall_status'] == 'PASS' else 1


if __name__ == '__main__':
    sys.exit(main())
