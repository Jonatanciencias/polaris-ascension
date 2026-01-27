#!/bin/bash
# Complete PyTorch Compilation Script for AMD Radeon RX 580 (gfx803)
# Estimated time: 4-8 hours

set -e  # Exit on error

echo "=== PyTorch Compilation for AMD Radeon RX 580 (gfx803) ==="
echo "Start time: $(date)"
echo "=========================================================="

# Activate virtual environment
source /home/jonatanciencias/Proyectos/Programacion/Radeon_RX_580/venv/bin/activate

# Navigate to build directory
cd /home/jonatanciencias/Proyectos/Programacion/Radeon_RX_580/pytorch_build

# Clean previous build artifacts
echo "Cleaning previous build artifacts..."
python3 setup.py clean 2>/dev/null || true
rm -rf build dist *.egg-info 2>/dev/null || true

# Set environment variables
echo "Configuring build environment..."
export ROCM_PATH=/opt/rocm
export HIP_PATH=/opt/rocm
export ROCM_VERSION=6.2

# CRITICAL: Target gfx803 architecture
export PYTORCH_ROCM_ARCH="gfx803"
export HIP_PLATFORM="amd"

# Enable ROCm, disable CUDA
export USE_ROCM=1
export USE_CUDA=0
export USE_CUDNN=0
export USE_NCCL=0

# Optimize compilation speed
export MAX_JOBS=8
export CMAKE_BUILD_TYPE=Release
export BUILD_TEST=0

# Disable optional features
export BUILD_CAFFE2=0
export USE_DISTRIBUTED=0
export USE_FBGEMM=0
export USE_KINETO=0
export USE_TENSORPIPE=0

# Use Intel MKL for CPU operations
export CMAKE_PREFIX_PATH="/home/jonatanciencias/Proyectos/Programacion/Radeon_RX_580/venv/lib/python3.12/site-packages/mkl:${CMAKE_PREFIX_PATH}"
export USE_MKL=1

# Version info
export PYTORCH_BUILD_VERSION=2.5.1+rocm6.2.gfx803
export PYTORCH_BUILD_NUMBER=0

echo "Configuration:"
echo "  ROCM_PATH: $ROCM_PATH"
echo "  PYTORCH_ROCM_ARCH: $PYTORCH_ROCM_ARCH"
echo "  MAX_JOBS: $MAX_JOBS"
echo "  BUILD_TYPE: $CMAKE_BUILD_TYPE"

# Start compilation
echo ""
echo "Starting compilation (this will take 4-8 hours)..."
echo "Log file: build_complete.log"
echo ""

python3 setup.py develop 2>&1 | tee build_complete.log

# Check if compilation succeeded
if [ $? -eq 0 ]; then
    echo ""
    echo "=== Compilation Successful! ==="
    echo "End time: $(date)"
    echo "================================"
    echo ""
    echo "Testing PyTorch installation..."
    python3 -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
else
    echo ""
    echo "=== Compilation Failed ==="
    echo "Check build_complete.log for errors"
    exit 1
fi
