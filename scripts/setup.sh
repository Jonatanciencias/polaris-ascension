#!/bin/bash
# 
# Setup Script for Radeon RX 580 AI Framework
# 
# This script installs necessary dependencies and configures the environment
# for running AI workloads on AMD Radeon RX 580 GPUs.
#

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "=============================================="
echo "Radeon RX 580 AI Framework Setup"
echo "=============================================="

# Check if running on Linux
if [[ "$OSTYPE" != "linux-gnu"* ]]; then
    echo -e "${RED}Error: This script is only for Linux systems${NC}"
    exit 1
fi

# Check if running as root
if [[ $EUID -eq 0 ]]; then
   echo -e "${YELLOW}Warning: Running as root. Consider using a regular user.${NC}"
fi

echo -e "\n${GREEN}[1/6] Updating system packages...${NC}"
sudo apt update

echo -e "\n${GREEN}[2/6] Installing system dependencies...${NC}"
sudo apt install -y \
    build-essential \
    cmake \
    git \
    python3 \
    python3-pip \
    python3-venv \
    ocl-icd-opencl-dev \
    opencl-headers \
    clinfo \
    mesa-opencl-icd \
    libdrm-amdgpu1 \
    || echo -e "${YELLOW}Some packages may already be installed${NC}"

echo -e "\n${GREEN}[3/6] Creating Python virtual environment...${NC}"
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "Virtual environment created"
else
    echo "Virtual environment already exists"
fi

echo -e "\n${GREEN}[4/6] Activating virtual environment...${NC}"
source venv/bin/activate

echo -e "\n${GREEN}[5/6] Installing Python dependencies...${NC}"
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt

echo -e "\n${GREEN}[6/6] Installing package in development mode...${NC}"
pip install -e .

echo -e "\n${GREEN}=============================================="
echo "Setup Complete!"
echo "==============================================${NC}"

echo -e "\n${GREEN}Running driver verification...${NC}"
python scripts/verify_drivers.py

if [ $? -eq 0 ]; then
    echo -e "\n${GREEN}✅ All drivers are properly configured!${NC}"
else
    echo -e "\n${YELLOW}⚠️  Some driver issues detected. See recommendations above.${NC}"
    echo -e "${YELLOW}For detailed setup instructions, see: docs/guides/DRIVER_SETUP_RX580.md${NC}"
fi

echo -e "\n${GREEN}=============================================="
echo "Next Steps"
echo "==============================================${NC}"
echo "  1. Activate the environment: source venv/bin/activate"
echo "  2. Verify hardware: python scripts/verify_hardware.py"
echo "  3. Run diagnostics: python scripts/diagnostics.py"
echo "  4. Download models: python scripts/download_models.py --all"
echo "  5. Test inference: python examples/demo_real_simple.py"

echo -e "\n${GREEN}📚 Documentation:${NC}"
echo "  - Quick Start: docs/guides/QUICKSTART.md"
echo "  - Driver Setup: docs/guides/DRIVER_SETUP_RX580.md"
echo "  - User Guide: docs/guides/USER_GUIDE.md"

echo -e "\n${YELLOW}Note: If you need ROCm (optional), visit:${NC}"
echo "  https://rocmdocs.amd.com/en/latest/deploy/linux/quick_start.html"

echo -e "\n${GREEN}Setup script finished successfully!${NC}"
