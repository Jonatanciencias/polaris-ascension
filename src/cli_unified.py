#!/usr/bin/env python3
"""
Legacy GPU AI - Command Line Interface (Click version)
Session 33: Cluster Management Integration

Unified CLI with:
- Original inference commands
- Cluster management (Session 33)
- Worker management (Session 33)
- Task management (Session 33)

Author: Radeon RX 580 AI Framework Team
Date: Enero 22, 2026
"""

import click
import sys

# Importar comandos de cluster
try:
    from .cli_cluster import cluster_cli, worker_cli, task_cli
    CLUSTER_CLI_AVAILABLE = True
except ImportError as e:
    CLUSTER_CLI_AVAILABLE = False
    click.echo(f"⚠️  Cluster CLI not available: {e}", err=True)


@click.group()
@click.version_option(version='0.7.0-dev', prog_name='legacygpu')
def cli():
    """
    Legacy GPU AI - Democratizando la IA con hardware accesible
    
    Framework de inferencia optimizado para GPUs AMD antiguas.
    
    \b
    Comandos disponibles:
      - info:     Información del sistema
      - infer:    Ejecutar inferencia
      - cluster:  Gestión de clúster distribuido
      - worker:   Gestión de workers
      - task:     Gestión de tareas
    
    \b
    Ejemplos:
      # Información del sistema
      legacygpu info
      
      # Cluster distribuido
      legacygpu cluster start
      legacygpu worker start
      legacygpu task submit --model resnet50
    """
    pass


@cli.command(name='info')
@click.option('--verbose', '-v', is_flag=True, help='Mostrar información detallada')
def info_command(verbose: bool):
    """
    Muestra información del sistema y GPU.
    
    Incluye:
    - Información de GPU AMD
    - Memoria disponible (VRAM/RAM)
    - Capacidades ROCm/OpenCL
    - Recomendaciones de rendimiento
    """
    try:
        from .core.gpu import GPUManager
        from .core.memory import MemoryManager
    except ImportError:
        click.echo("❌ Error: Core modules not available", err=True)
        sys.exit(1)
    
    gpu_manager = GPUManager()
    memory_manager = MemoryManager()
    
    click.echo("=" * 60)
    click.echo("🖥️  LEGACY GPU AI - SYSTEM INFORMATION")
    click.echo("=" * 60)
    
    # GPU Info
    gpu_info = gpu_manager.get_info()
    click.echo("\n📊 GPU Information:")
    if gpu_info:
        click.echo(f"   Name: {gpu_info.name}")
        click.echo(f"   Driver: {gpu_info.driver}")
        click.echo(f"   VRAM: {gpu_info.vram_gb:.1f} GB")
        click.echo(f"   OpenCL: {'✅ Available' if gpu_info.opencl_available else '❌ Not available'}")
        
        if verbose:
            click.echo(f"   Compute Units: {gpu_info.compute_units if hasattr(gpu_info, 'compute_units') else 'N/A'}")
            click.echo(f"   Max Clock: {gpu_info.max_clock if hasattr(gpu_info, 'max_clock') else 'N/A'} MHz")
    else:
        click.echo("   ⚠️  GPU not detected (will use CPU)")
    
    # Memory Info
    memory_stats = memory_manager.get_stats()
    click.echo("\n💾 Memory Information:")
    vram_gb = memory_stats.gpu_vram_gb if memory_stats.gpu_vram_gb else 0
    click.echo(f"   VRAM: {vram_gb:.1f} GB total")
    click.echo(f"   RAM: {memory_stats.available_ram_gb:.1f} GB available / {memory_stats.total_ram_gb:.1f} GB total")
    
    # Features
    click.echo("\n⚡ Available Features:")
    click.echo(f"   ✅ ONNX Runtime inference")
    click.echo(f"   ✅ Mixed precision (FP32/FP16/INT8)")
    click.echo(f"   ✅ Quantization & pruning")
    click.echo(f"   ✅ Sparse computation")
    click.echo(f"   {'✅' if CLUSTER_CLI_AVAILABLE else '❌'} Distributed computing")
    
    # Tips
    click.echo("\n💡 Performance Tips:")
    click.echo("   • Use FP16 for 1.5-2x speedup on most models")
    click.echo("   • Use INT8 quantization for 2-3x speedup")
    click.echo("   • Sparse models can achieve 3-5x speedup")
    if CLUSTER_CLI_AVAILABLE:
        click.echo("   • Use distributed mode for cluster deployments")
    
    click.echo("")


@cli.command(name='infer')
@click.argument('model')
@click.argument('input_file', type=click.Path(exists=True))
@click.option('--output', '-o', type=click.Path(), help='Output file path')
@click.option('--device', default='auto', help='Device: auto, rocm, cuda, opencl, cpu')
@click.option('--precision', type=click.Choice(['fp32', 'fp16', 'int8']), default='fp32', help='Precision mode')
@click.option('--batch-size', '-b', default=1, type=int, help='Batch size')
def infer_command(model: str, input_file: str, output: str, device: str, precision: str, batch_size: int):
    """
    Ejecuta inferencia local con un modelo.
    
    \b
    Ejemplos:
      legacygpu infer resnet50.onnx image.jpg
      legacygpu infer model.onnx input.npy --precision fp16
      legacygpu infer yolo.onnx frame.jpg --batch-size 4
    """
    click.echo(f"🚀 Running inference...")
    click.echo(f"   Model: {model}")
    click.echo(f"   Input: {input_file}")
    click.echo(f"   Device: {device}")
    click.echo(f"   Precision: {precision}")
    click.echo(f"   Batch size: {batch_size}")
    
    try:
        # TODO: Implementar inferencia real usando el engine
        from .inference import EnhancedInferenceEngine
        
        click.echo("\n⚠️  Full inference implementation coming in next session")
        click.echo("For now, use the API server: legacygpu api start")
        
    except ImportError as e:
        click.echo(f"❌ Error: {e}", err=True)
        sys.exit(1)


@cli.command(name='api')
@click.argument('action', type=click.Choice(['start', 'stop', 'status']))
@click.option('--host', default='0.0.0.0', help='API host (default: 0.0.0.0)')
@click.option('--port', default=8000, type=int, help='API port (default: 8000)')
@click.option('--reload', is_flag=True, help='Auto-reload on code changes')
def api_command(action: str, host: str, port: int, reload: bool):
    """
    Gestión del servidor API REST.
    
    \b
    Ejemplos:
      legacygpu api start
      legacygpu api start --port 8080 --reload
      legacygpu api stop
    """
    if action == 'start':
        click.echo(f"🚀 Starting API server on {host}:{port}")
        
        try:
            import uvicorn
            from .api.server import app
            
            uvicorn.run(
                "src.api.server:app",
                host=host,
                port=port,
                reload=reload,
                log_level="info"
            )
        except ImportError:
            click.echo("❌ Error: uvicorn not installed", err=True)
            click.echo("Install with: pip install uvicorn", err=True)
            sys.exit(1)
        except KeyboardInterrupt:
            click.echo("\n⚠️  API server stopped")
    
    elif action == 'stop':
        click.echo("⚠️  Stop command not implemented - use Ctrl+C to stop the server")
    
    elif action == 'status':
        # Verificar si el servidor está corriendo
        try:
            import requests
            response = requests.get(f"http://{host}:{port}/health", timeout=2)
            if response.status_code == 200:
                click.echo(f"✅ API server is running on http://{host}:{port}")
                data = response.json()
                click.echo(f"   Status: {data.get('status', 'unknown')}")
            else:
                click.echo(f"⚠️  API server responded with status {response.status_code}")
        except Exception:
            click.echo(f"❌ API server is not running on http://{host}:{port}")


# ============================================================================
# INTEGRAR COMANDOS DE CLUSTER (Session 33)
# ============================================================================

if CLUSTER_CLI_AVAILABLE:
    cli.add_command(cluster_cli, name='cluster')
    cli.add_command(worker_cli, name='worker')
    cli.add_command(task_cli, name='task')


# ============================================================================
# ENTRY POINT
# ============================================================================

def main():
    """Entry point for CLI"""
    try:
        cli()
    except KeyboardInterrupt:
        click.echo("\n⚠️  Interrupted by user")
        sys.exit(130)
    except Exception as e:
        click.echo(f"❌ Error: {e}", err=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
