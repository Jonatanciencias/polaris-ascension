FROM ubuntu:22.04

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    ocl-icd-opencl-dev \
    opencl-headers \
    clinfo \
    mesa-opencl-icd \
    libdrm-amdgpu1 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Install package
RUN pip3 install -e .

# Setup environment
ENV PYTHONUNBUFFERED=1

# Default command
CMD ["python3", "scripts/verify_hardware.py"]
