#!/usr/bin/env python3
"""
🎯 VALIDATION: AI + BAYESIAN INTEGRATION COMPLETE
"""

import sys
from pathlib import Path


def main():
    print("🎯 VALIDATION: AI + BAYESIAN INTEGRATION COMPLETE")
    print("=" * 60)

    try:
        # Test import
        sys.path.append(str(Path(__file__).parent / "fase_7_ai_kernel_predictor" / "src"))
        from kernel_predictor import BAYESIAN_INTEGRATION_AVAILABLE, AIKernelPredictor

        print("✅ Imports successful")

        # Test initialization
        predictor = AIKernelPredictor()
        print("✅ Predictor initialized")

        # Test prediction
        result = predictor.predict_best_kernel_enhanced(512, use_bayesian=True)
        improvement = result["improvement_percent"]

        print(f"✅ Prediction successful: {result['predicted_performance']:.1f} GFLOPS")
        print(f"✅ Improvement: {improvement:.1f}%")

        if improvement >= 30:
            print("🎯 SUCCESS: AI + Bayesian Integration working perfectly!")
            print("🚀 System ready for multi-GPU scaling")
            return 0
        else:
            print("❌ FAILED: Improvement too low")
            return 1

    except Exception as e:
        print(f"❌ ERROR: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
