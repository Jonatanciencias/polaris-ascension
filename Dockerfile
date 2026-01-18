# ==============================================================================
# Dockerfile - Radeon RX 580 AI API
# Session 17: REST API + Docker Deployment
#
# Multi-stage build para optimizar tamaño de imagen
# Base: Ubuntu 22.04 con OpenCL support
# ==============================================================================

# ------------------------------------------------------------------------------
# STAGE 1: Builder - Instala dependencias y compila código
# ------------------------------------------------------------------------------
FROM ubuntu:22.04 as builder

# Evitar prompts durante instalación
ENV DEBIAN_FRONTEND=noninteractive

# Metadata
LABEL maintainer="Radeon RX 580 AI Framework Team"
LABEL version="0.6.0-dev"
LABEL description="REST API for AMD Radeon RX 580 AI Inference"
LABEL session="17 - REST API + Docker"

# Instalar dependencias del sistema
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-dev \
    python3-pip \
    git \
    wget \
    curl \
    build-essential \
    ocl-icd-opencl-dev \
    opencl-headers \
    && rm -rf /var/lib/apt/lists/*

# Crear directorio de trabajo
WORKDIR /build

# Copiar requirements primero (mejor caching)
COPY requirements.txt .

# Instalar dependencias Python
RUN pip3 install --no-cache-dir --upgrade pip && \
    pip3 install --no-cache-dir -r requirements.txt

# Instalar dependencias adicionales para API (Session 17)
RUN pip3 install --no-cache-dir \
    fastapi==0.104.1 \
    uvicorn[standard]==0.24.0 \
    prometheus-client==0.19.0 \
    python-multipart==0.0.6 \
    pydantic==2.5.0

# ------------------------------------------------------------------------------
# STAGE 2: Runtime - Imagen final optimizada
# ------------------------------------------------------------------------------
FROM ubuntu:22.04

# Evitar prompts
ENV DEBIAN_FRONTEND=noninteractive

# Metadata
LABEL maintainer="Radeon RX 580 AI Framework Team"
LABEL version="0.6.0-dev"

# Instalar solo runtime dependencies (más ligero)
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    libgomp1 \
    curl \
    ocl-icd-opencl-dev \
    mesa-opencl-icd \
    clinfo \
    libdrm-amdgpu1 \
    && rm -rf /var/lib/apt/lists/*

# Crear usuario no-root para seguridad
RUN useradd -m -u 1000 -s /bin/bash aiuser

# Crear directorios necesarios
RUN mkdir -p /app /models /logs && \
    chown -R aiuser:aiuser /app /models /logs

# Copiar dependencias Python desde builder
COPY --from=builder /usr/local/lib/python3.10/dist-packages /usr/local/lib/python3.10/dist-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Cambiar a usuario no-root
USER aiuser
WORKDIR /app

# Copiar código del proyecto
COPY --chown=aiuser:aiuser . .

# Instalar package en modo editable
RUN pip3 install --user -e .

# Variables de entorno
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV API_HOST=0.0.0.0
ENV API_PORT=8000
ENV LOG_LEVEL=info
ENV MAX_MEMORY_MB=7000

# Health check (verifica cada 30s)
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Exponer puerto API
EXPOSE 8000

# Comando por defecto: iniciar servidor FastAPI
CMD ["uvicorn", "src.api.server:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--log-level", "info"]

# ==============================================================================
# BUILD & RUN INSTRUCTIONS
# ==============================================================================
# 
# Build:
#   docker build -t radeon-rx580-ai-api:latest .
#
# Run (CPU only):
#   docker run -d -p 8000:8000 \
#              -v $(pwd)/models:/models \
#              --name rx580-api \
#              radeon-rx580-ai-api:latest
#
# Run (with GPU - OpenCL/ROCm):
#   docker run -d -p 8000:8000 \
#              -v $(pwd)/models:/models \
#              --device=/dev/kfd \
#              --device=/dev/dri \
#              --group-add video \
#              --name rx580-api \
#              radeon-rx580-ai-api:latest
#
# View logs:
#   docker logs -f rx580-api
#
# Stop:
#   docker stop rx580-api
#
# Interactive shell:
#   docker exec -it rx580-api /bin/bash
#
# Remove:
#   docker rm -f rx580-api
#
# ==============================================================================
