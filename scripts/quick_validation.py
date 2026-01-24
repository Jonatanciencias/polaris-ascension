#!/usr/bin/env python3
"""
Task 1.1.2.2 - Quick Functional Validation Script

Validates that the hybrid GEMM kernel produces correct results on basic tests.

Tests:
  1. Small matrix (128×128)
  2. Medium matrix (512×512)
  3. Alpha/Beta parameters
  4. Different parameter combinations
"""

import sys
import numpy as np
import logging
from pathlib import Path
import json
import time

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.opencl.hybrid_gemm import HybridGEMMExecutor


class QuickValidator:
    """Quick functional validation for hybrid GEMM kernel."""
    
    def __init__(self):
        """Initialize validator and executor."""
        logger.info("Initializing Quick Validator...")
        self.executor = HybridGEMMExecutor()
        self.results = {
            'timestamp': time.time(),
            'tests': {},
            'summary': {}
        }
        
    def test_small_matrix(self):
        """Test with small matrix 128×128."""
        logger.info("\n" + "="*80)
        logger.info("TEST 1: Small Matrix (128×128)")
        logger.info("="*80)
        
        test_name = "small_matrix_128"
        try:
            n = 128
            np.random.seed(42)
            A = np.random.randn(n, n).astype(np.float32)
            B = np.random.randn(n, n).astype(np.float32)
            
            # Execute GPU GEMM
            start = time.perf_counter()
            C_gpu = self.executor(A, B)
            gpu_time = (time.perf_counter() - start) * 1000
            
            # CPU reference
            C_ref = A @ B
            
            # Calculate error
            error_abs = np.linalg.norm(C_gpu - C_ref)
            error_rel = error_abs / np.linalg.norm(C_ref)
            
            # GFLOPS
            flops = 2 * n**3
            gflops = flops / (gpu_time / 1000) / 1e9
            
            passed = error_rel < 1e-4
            status = "✅ PASS" if passed else "❌ FAIL"
            
            logger.info(f"Matrix size: {n}×{n}")
            logger.info(f"GPU time: {gpu_time:.3f} ms")
            logger.info(f"GFLOPS: {gflops:.1f}")
            logger.info(f"Error (relative): {error_rel:.2e}")
            logger.info(f"Status: {status}")
            
            self.results['tests'][test_name] = {
                'size': n,
                'gpu_time_ms': gpu_time,
                'gflops': gflops,
                'error_rel': float(error_rel),
                'passed': passed
            }
            
            return passed
            
        except Exception as e:
            logger.error(f"Test failed with error: {e}", exc_info=True)
            self.results['tests'][test_name] = {'passed': False, 'error': str(e)}
            return False
    
    def test_medium_matrix(self):
        """Test with medium matrix 512×512."""
        logger.info("\n" + "="*80)
        logger.info("TEST 2: Medium Matrix (512×512)")
        logger.info("="*80)
        
        test_name = "medium_matrix_512"
        try:
            n = 512
            np.random.seed(43)
            A = np.random.randn(n, n).astype(np.float32)
            B = np.random.randn(n, n).astype(np.float32)
            
            # Execute GPU GEMM
            start = time.perf_counter()
            C_gpu = self.executor(A, B)
            gpu_time = (time.perf_counter() - start) * 1000
            
            # CPU reference
            C_ref = A @ B
            
            # Calculate error
            error_abs = np.linalg.norm(C_gpu - C_ref)
            error_rel = error_abs / np.linalg.norm(C_ref)
            
            # GFLOPS
            flops = 2 * n**3
            gflops = flops / (gpu_time / 1000) / 1e9
            
            passed = error_rel < 1e-4
            status = "✅ PASS" if passed else "❌ FAIL"
            
            logger.info(f"Matrix size: {n}×{n}")
            logger.info(f"GPU time: {gpu_time:.3f} ms")
            logger.info(f"GFLOPS: {gflops:.1f}")
            logger.info(f"Error (relative): {error_rel:.2e}")
            logger.info(f"Status: {status}")
            
            self.results['tests'][test_name] = {
                'size': n,
                'gpu_time_ms': gpu_time,
                'gflops': gflops,
                'error_rel': float(error_rel),
                'passed': passed
            }
            
            return passed
            
        except Exception as e:
            logger.error(f"Test failed with error: {e}", exc_info=True)
            self.results['tests'][test_name] = {'passed': False, 'error': str(e)}
            return False
    
    def test_alpha_beta(self):
        """Test alpha and beta parameters."""
        logger.info("\n" + "="*80)
        logger.info("TEST 3: Alpha/Beta Parameters")
        logger.info("="*80)
        
        test_name = "alpha_beta_params"
        try:
            n = 256
            np.random.seed(44)
            A = np.random.randn(n, n).astype(np.float32)
            B = np.random.randn(n, n).astype(np.float32)
            C = np.random.randn(n, n).astype(np.float32)
            
            test_cases = [
                (1.0, 0.0, "alpha=1.0, beta=0.0"),
                (2.5, 0.0, "alpha=2.5, beta=0.0"),
                (1.0, 1.0, "alpha=1.0, beta=1.0"),
                (2.5, 0.5, "alpha=2.5, beta=0.5"),
            ]
            
            all_passed = True
            sub_results = {}
            
            for alpha, beta, desc in test_cases:
                logger.info(f"\n  Testing: {desc}")
                
                # GPU result
                C_gpu = self.executor(A, B, C=C.copy(), alpha=alpha, beta=beta)
                
                # CPU reference
                C_ref = alpha * (A @ B) + beta * C
                
                # Error
                error_rel = np.linalg.norm(C_gpu - C_ref) / np.linalg.norm(C_ref)
                passed = error_rel < 1e-4
                
                status = "✅" if passed else "❌"
                logger.info(f"    {status} Error: {error_rel:.2e}")
                
                sub_results[desc] = {
                    'error_rel': float(error_rel),
                    'passed': passed
                }
                
                all_passed = all_passed and passed
            
            self.results['tests'][test_name] = {
                'sub_tests': sub_results,
                'passed': all_passed
            }
            
            return all_passed
            
        except Exception as e:
            logger.error(f"Test failed with error: {e}", exc_info=True)
            self.results['tests'][test_name] = {'passed': False, 'error': str(e)}
            return False
    
    def run_all_tests(self):
        """Run all validation tests."""
        logger.info("\n" + "="*80)
        logger.info("QUICK FUNCTIONAL VALIDATION - TASK 1.1.2.2")
        logger.info("="*80)
        
        results = []
        
        # Test 1: Small matrix
        try:
            result1 = self.test_small_matrix()
            results.append(result1)
        except Exception as e:
            logger.error(f"Test 1 failed: {e}")
            results.append(False)
        
        # Test 2: Medium matrix
        try:
            result2 = self.test_medium_matrix()
            results.append(result2)
        except Exception as e:
            logger.error(f"Test 2 failed: {e}")
            results.append(False)
        
        # Test 3: Alpha/Beta
        try:
            result3 = self.test_alpha_beta()
            results.append(result3)
        except Exception as e:
            logger.error(f"Test 3 failed: {e}")
            results.append(False)
        
        # Summary
        logger.info("\n" + "="*80)
        logger.info("SUMMARY")
        logger.info("="*80)
        
        total = len(results)
        passed = sum(results)
        
        logger.info(f"Tests Passed: {passed}/{total}")
        logger.info(f"Status: {'✅ ALL PASS' if all(results) else '❌ SOME FAILED'}")
        
        self.results['summary'] = {
            'total_tests': total,
            'passed': passed,
            'failed': total - passed,
            'all_passed': all(results)
        }
        
        return all(results)
    
    def save_results(self, output_file='results/quick_validation.json'):
        """Save results to JSON file."""
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        logger.info(f"\nResults saved to {output_file}")


def main():
    """Main entry point."""
    validator = QuickValidator()
    
    try:
        success = validator.run_all_tests()
        validator.save_results()
        
        if success:
            logger.info("\n✅ QUICK VALIDATION COMPLETED SUCCESSFULLY")
            return 0
        else:
            logger.error("\n❌ QUICK VALIDATION FAILED")
            return 1
            
    except Exception as e:
        logger.error(f"\nCritical error: {e}", exc_info=True)
        return 2


if __name__ == '__main__':
    sys.exit(main())
