"""
API Client Demo - Ejemplo de uso de la REST API

Session 17: REST API + Docker Deployment

Demuestra cómo interactuar con la API:
- Health checks
- Cargar modelos
- Ejecutar inferencias
- Obtener métricas
- Manejo de errores

Uso:
    python examples/demo_api_client.py

Requirements:
    pip install httpx

Author: Radeon RX 580 AI Framework Team
Date: Enero 2026
"""

import time
import json
from pathlib import Path
from typing import Dict, Any, Optional

try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False
    print("Error: httpx not installed. Run: pip install httpx")
    exit(1)


# ============================================================================
# API CLIENT CLASS
# ============================================================================

class RX580APIClient:
    """
    Cliente para interactuar con Radeon RX 580 AI API.
    
    Proporciona métodos convenientes para todos los endpoints.
    
    Attributes:
        base_url: URL base de la API
        client: Cliente HTTP
        timeout: Timeout para requests
    """
    
    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        timeout: float = 30.0
    ):
        """
        Inicializa el cliente API.
        
        Args:
            base_url: URL base de la API
            timeout: Timeout en segundos
        """
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.client = httpx.Client(
            base_url=self.base_url,
            timeout=timeout
        )
    
    def __enter__(self):
        """Context manager enter"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - cierra el cliente"""
        self.client.close()
    
    # ------------------------------------------------------------------------
    # Health & Info
    # ------------------------------------------------------------------------
    
    def get_info(self) -> Dict[str, Any]:
        """
        Obtiene información del servicio.
        
        Returns:
            Dict con información básica
        """
        response = self.client.get("/")
        response.raise_for_status()
        return response.json()
    
    def health_check(self) -> Dict[str, Any]:
        """
        Ejecuta health check.
        
        Returns:
            Dict con estado de salud del servicio
        """
        response = self.client.get("/health")
        response.raise_for_status()
        return response.json()
    
    def get_metrics(self) -> str:
        """
        Obtiene métricas Prometheus.
        
        Returns:
            String con métricas en formato Prometheus
        """
        response = self.client.get("/metrics")
        response.raise_for_status()
        return response.text
    
    # ------------------------------------------------------------------------
    # Model Management
    # ------------------------------------------------------------------------
    
    def load_model(
        self,
        path: str,
        model_name: Optional[str] = None,
        compression: Optional[Dict[str, Any]] = None,
        device: str = "auto",
        optimization_level: int = 1
    ) -> Dict[str, Any]:
        """
        Carga un modelo en el servidor.
        
        Args:
            path: Ruta al archivo del modelo
            model_name: Nombre para el modelo (opcional)
            compression: Configuración de compresión (opcional)
            device: Device de inferencia (cpu/cuda/auto)
            optimization_level: Nivel de optimización ONNX (0-2)
            
        Returns:
            Dict con información del modelo cargado
        """
        request_data = {
            "path": path,
            "device": device,
            "optimization_level": optimization_level
        }
        
        if model_name:
            request_data["model_name"] = model_name
        
        if compression:
            request_data["compression"] = compression
        
        response = self.client.post("/models/load", json=request_data)
        response.raise_for_status()
        return response.json()
    
    def unload_model(self, model_name: str) -> Dict[str, Any]:
        """
        Descarga un modelo del servidor.
        
        Args:
            model_name: Nombre del modelo a descargar
            
        Returns:
            Dict con confirmación
        """
        response = self.client.delete(f"/models/{model_name}")
        response.raise_for_status()
        return response.json()
    
    def list_models(self) -> Dict[str, Any]:
        """
        Lista todos los modelos cargados.
        
        Returns:
            Dict con lista de modelos
        """
        response = self.client.get("/models")
        response.raise_for_status()
        return response.json()
    
    def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """
        Obtiene información de un modelo específico.
        
        Args:
            model_name: Nombre del modelo
            
        Returns:
            Dict con información del modelo
        """
        response = self.client.get(f"/models/{model_name}")
        response.raise_for_status()
        return response.json()
    
    # ------------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------------
    
    def predict(
        self,
        model_name: str,
        inputs: Any,
        batch_size: int = 1,
        return_metadata: bool = False
    ) -> Dict[str, Any]:
        """
        Ejecuta inferencia en un modelo.
        
        Args:
            model_name: Nombre del modelo
            inputs: Datos de entrada (dict o lista)
            batch_size: Tamaño del batch
            return_metadata: Si retornar metadata
            
        Returns:
            Dict con outputs y metadata
        """
        request_data = {
            "model_name": model_name,
            "inputs": inputs,
            "batch_size": batch_size,
            "return_metadata": return_metadata
        }
        
        response = self.client.post("/predict", json=request_data)
        response.raise_for_status()
        return response.json()


# ============================================================================
# DEMO FUNCTIONS
# ============================================================================

def print_section(title: str):
    """Imprime un título de sección"""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80 + "\n")


def demo_1_connection_test():
    """
    Demo 1: Probar conexión con la API.
    """
    print_section("Demo 1: Connection Test")
    
    try:
        with RX580APIClient() as client:
            # Obtener info del servicio
            info = client.get_info()
            print("✓ API Connection successful!")
            print(f"  Service: {info['service']}")
            print(f"  Version: {info['version']}")
            print(f"  Status: {info['status']}")
            
    except httpx.ConnectError:
        print("✗ Error: Cannot connect to API")
        print("  Make sure the server is running:")
        print("    uvicorn src.api.server:app --host 0.0.0.0 --port 8000")
        return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False
    
    return True


def demo_2_health_check():
    """
    Demo 2: Verificar salud del servicio.
    """
    print_section("Demo 2: Health Check")
    
    try:
        with RX580APIClient() as client:
            health = client.health_check()
            
            print(f"Status: {health['status']}")
            print(f"Version: {health['version']}")
            print(f"Models loaded: {health['models_loaded']}")
            print(f"Memory used: {health['memory_used_mb']:.1f} MB")
            print(f"Memory available: {health['memory_available_mb']:.1f} MB")
            print(f"Uptime: {health['uptime_seconds']:.0f} seconds")
            
            if health['status'] == 'healthy':
                print("\n✓ Service is healthy!")
            else:
                print(f"\n⚠ Service status: {health['status']}")
                
    except Exception as e:
        print(f"✗ Health check failed: {e}")
        return False
    
    return True


def demo_3_list_models():
    """
    Demo 3: Listar modelos cargados.
    """
    print_section("Demo 3: List Loaded Models")
    
    try:
        with RX580APIClient() as client:
            models_list = client.list_models()
            
            print(f"Total models: {models_list['total']}")
            
            if models_list['total'] == 0:
                print("No models currently loaded.")
            else:
                print("\nLoaded models:")
                for model in models_list['models']:
                    print(f"\n  - {model['name']}")
                    print(f"    Framework: {model['framework']}")
                    print(f"    Memory: {model['memory_mb']:.1f} MB")
                    print(f"    Device: {model['device']}")
                    print(f"    Inferences: {model['inference_count']}")
                    
    except Exception as e:
        print(f"✗ Error listing models: {e}")
        return False
    
    return True


def demo_4_model_lifecycle():
    """
    Demo 4: Ciclo de vida completo de un modelo (simulado).
    
    Nota: Requiere un archivo modelo real para funcionar completamente.
    """
    print_section("Demo 4: Model Lifecycle (Simulated)")
    
    print("This demo shows the complete model lifecycle:")
    print("1. Load model")
    print("2. Run inference")
    print("3. Get model info")
    print("4. Unload model")
    
    print("\n⚠ Note: Requires a real model file to execute")
    print("Example:")
    print("  # Load model")
    print("  response = client.load_model('/models/resnet50.onnx', 'resnet50')")
    print()
    print("  # Run inference")
    print("  result = client.predict('resnet50', {'input': [[...data...]]})")
    print()
    print("  # Get info")
    print("  info = client.get_model_info('resnet50')")
    print()
    print("  # Unload")
    print("  client.unload_model('resnet50')")
    
    return True


def demo_5_metrics():
    """
    Demo 5: Obtener métricas Prometheus.
    """
    print_section("Demo 5: Prometheus Metrics")
    
    try:
        with RX580APIClient() as client:
            metrics = client.get_metrics()
            
            # Mostrar primeras líneas
            lines = metrics.split('\n')[:20]
            
            print("Prometheus metrics (first 20 lines):")
            print("-" * 80)
            for line in lines:
                if line.strip():
                    print(line)
            print("-" * 80)
            
            print(f"\n✓ Total metrics lines: {len(metrics.split(chr(10)))}")
            
    except Exception as e:
        print(f"✗ Error getting metrics: {e}")
        return False
    
    return True


def demo_6_error_handling():
    """
    Demo 6: Manejo de errores.
    """
    print_section("Demo 6: Error Handling")
    
    try:
        with RX580APIClient() as client:
            print("Testing error scenarios...\n")
            
            # Test 1: Modelo inexistente
            print("1. Getting info for non-existent model...")
            try:
                client.get_model_info("nonexistent_model")
                print("   ✗ Should have raised error")
            except httpx.HTTPStatusError as e:
                print(f"   ✓ Correctly returned {e.response.status_code} error")
            
            # Test 2: Request inválido
            print("\n2. Sending invalid request...")
            try:
                response = client.client.post("/predict", json={})
                response.raise_for_status()
                print("   ✗ Should have raised error")
            except httpx.HTTPStatusError as e:
                print(f"   ✓ Correctly returned {e.response.status_code} error")
            
            # Test 3: Endpoint inexistente
            print("\n3. Accessing non-existent endpoint...")
            try:
                response = client.client.get("/invalid")
                response.raise_for_status()
                print("   ✗ Should have raised error")
            except httpx.HTTPStatusError as e:
                print(f"   ✓ Correctly returned {e.response.status_code} error")
            
            print("\n✓ Error handling works correctly!")
            
    except Exception as e:
        print(f"✗ Error in error handling demo: {e}")
        return False
    
    return True


def demo_7_performance_test():
    """
    Demo 7: Test simple de performance.
    """
    print_section("Demo 7: Performance Test")
    
    try:
        with RX580APIClient() as client:
            print("Running 10 health check requests...")
            
            latencies = []
            for i in range(10):
                start = time.time()
                client.health_check()
                latency = (time.time() - start) * 1000  # ms
                latencies.append(latency)
                print(f"  Request {i+1}: {latency:.2f} ms")
            
            # Estadísticas
            avg_latency = sum(latencies) / len(latencies)
            min_latency = min(latencies)
            max_latency = max(latencies)
            
            print(f"\nStatistics:")
            print(f"  Average: {avg_latency:.2f} ms")
            print(f"  Min: {min_latency:.2f} ms")
            print(f"  Max: {max_latency:.2f} ms")
            print(f"  Throughput: ~{1000/avg_latency:.0f} req/s")
            
    except Exception as e:
        print(f"✗ Performance test failed: {e}")
        return False
    
    return True


# ============================================================================
# MAIN - Ejecutar todos los demos
# ============================================================================

def main():
    """
    Ejecuta todos los demos.
    """
    print("\n" + "╔" + "=" * 78 + "╗")
    print("║" + " " * 78 + "║")
    print("║" + "      Radeon RX 580 AI API - Client Demo".center(78) + "║")
    print("║" + "      Session 17: REST API + Docker Deployment".center(78) + "║")
    print("║" + " " * 78 + "║")
    print("╚" + "=" * 78 + "╝")
    
    # Lista de demos
    demos = [
        ("Connection Test", demo_1_connection_test),
        ("Health Check", demo_2_health_check),
        ("List Models", demo_3_list_models),
        ("Model Lifecycle", demo_4_model_lifecycle),
        ("Prometheus Metrics", demo_5_metrics),
        ("Error Handling", demo_6_error_handling),
        ("Performance Test", demo_7_performance_test),
    ]
    
    results = []
    
    # Ejecutar demos
    for name, demo_func in demos:
        try:
            success = demo_func()
            results.append((name, success))
        except KeyboardInterrupt:
            print("\n\nDemo interrupted by user")
            break
        except Exception as e:
            print(f"\n✗ Demo failed with exception: {e}")
            results.append((name, False))
    
    # Resumen
    print_section("SUMMARY")
    
    total = len(results)
    passed = sum(1 for _, success in results if success)
    
    print(f"Total demos: {total}")
    print(f"Passed: {passed}")
    print(f"Failed: {total - passed}")
    print()
    
    for name, success in results:
        status = "✓" if success else "✗"
        print(f"  {status} {name}")
    
    if passed == total:
        print("\n🎉 All demos passed!")
    else:
        print(f"\n⚠ {total - passed} demo(s) failed")
    
    print("\n" + "=" * 80)
    print("\nFor more information, visit:")
    print("  - API Docs: http://localhost:8000/docs")
    print("  - Health: http://localhost:8000/health")
    print("  - Metrics: http://localhost:8000/metrics")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
