"""
Tests para REST API

Session 17: REST API + Docker Deployment

Test suite completo para validar todos los endpoints:
- Health check
- Model management (load/unload/list)
- Inference
- Metrics
- Error handling

Author: Radeon RX 580 AI Framework Team
Date: Enero 2026
"""

import pytest
from fastapi.testclient import TestClient
from datetime import datetime
from pathlib import Path

# Importar la app
from src.api.server import app, server_state


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def client():
    """
    Cliente de prueba para FastAPI.
    
    Yields:
        TestClient configurado
    """
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def mock_model_path(tmp_path):
    """
    Crea un archivo de modelo mock para testing.
    
    Args:
        tmp_path: Directorio temporal de pytest
        
    Returns:
        Path al modelo mock
    """
    model_file = tmp_path / "test_model.onnx"
    model_file.write_text("mock onnx model")
    return str(model_file)


# ============================================================================
# TESTS - Root & Health
# ============================================================================

def test_root_endpoint(client):
    """
    Test 1: Root endpoint retorna información básica.
    """
    response = client.get("/")
    
    assert response.status_code == 200
    data = response.json()
    
    assert "service" in data
    assert "version" in data
    assert data["version"] == "0.6.0-dev"
    assert data["status"] == "running"


def test_health_check(client):
    """
    Test 2: Health check endpoint funciona correctamente.
    """
    response = client.get("/health")
    
    assert response.status_code == 200
    data = response.json()
    
    # Validar campos requeridos
    assert "status" in data
    assert "version" in data
    assert "models_loaded" in data
    assert "uptime_seconds" in data
    
    # Validar tipos
    assert isinstance(data["models_loaded"], int)
    assert isinstance(data["uptime_seconds"], float)
    assert data["status"] in ["healthy", "degraded", "unhealthy"]


def test_health_check_status_format(client):
    """
    Test 3: Health check retorna campos con formato correcto.
    """
    response = client.get("/health")
    data = response.json()
    
    # Validar que memory fields son números positivos
    assert data["memory_used_mb"] >= 0
    assert data["memory_available_mb"] >= 0
    
    # Validar timestamp format
    timestamp = data["timestamp"]
    datetime.fromisoformat(timestamp)  # Should not raise


# ============================================================================
# TESTS - Metrics
# ============================================================================

def test_metrics_endpoint(client):
    """
    Test 4: Metrics endpoint retorna datos Prometheus.
    """
    response = client.get("/metrics")
    
    assert response.status_code == 200
    
    # Validar content type
    assert "text/plain" in response.headers["content-type"].lower() or \
           "text" in response.headers["content-type"].lower()


def test_metrics_format(client):
    """
    Test 5: Métricas tienen formato Prometheus válido.
    """
    response = client.get("/metrics")
    content = response.text
    
    # Métricas básicas que deben estar presentes
    # (si prometheus_client está instalado)
    if "not installed" not in content.lower():
        assert "# HELP" in content or "# TYPE" in content


# ============================================================================
# TESTS - Model Management
# ============================================================================

def test_list_models_empty(client):
    """
    Test 6: List models cuando no hay modelos cargados.
    """
    response = client.get("/models")
    
    assert response.status_code == 200
    data = response.json()
    
    assert "models" in data
    assert "total" in data
    assert isinstance(data["models"], list)
    assert data["total"] == len(data["models"])


def test_get_nonexistent_model(client):
    """
    Test 7: GET de modelo que no existe retorna 404.
    """
    response = client.get("/models/nonexistent_model")
    
    assert response.status_code == 404
    data = response.json()
    assert "detail" in data


def test_unload_nonexistent_model(client):
    """
    Test 8: DELETE de modelo que no existe retorna 404.
    """
    response = client.delete("/models/nonexistent_model")
    
    # Si engine no está disponible, puede ser 503
    assert response.status_code in [404, 503]


def test_load_model_invalid_path(client):
    """
    Test 9: Load model con path inválido retorna 404.
    """
    request_data = {
        "path": "/nonexistent/model.onnx",
        "model_name": "test_model"
    }
    
    response = client.post("/models/load", json=request_data)
    
    # Si engine no disponible: 503, si path inválido: 404
    assert response.status_code in [404, 503]


def test_load_model_invalid_extension(client):
    """
    Test 10: Load model con extensión inválida retorna error de validación.
    """
    request_data = {
        "path": "/path/to/model.txt",  # Extensión inválida
        "model_name": "test_model"
    }
    
    response = client.post("/models/load", json=request_data)
    
    # Validación Pydantic debe fallar (422)
    assert response.status_code == 422


# ============================================================================
# TESTS - Inference
# ============================================================================

def test_predict_nonexistent_model(client):
    """
    Test 11: Predict con modelo inexistente retorna 404.
    """
    request_data = {
        "model_name": "nonexistent_model",
        "inputs": {"input": [[1.0, 2.0, 3.0]]}
    }
    
    response = client.post("/predict", json=request_data)
    
    # Si engine no disponible: 503, si modelo no existe: 404
    assert response.status_code in [404, 503]


def test_predict_invalid_request(client):
    """
    Test 12: Predict con request inválido retorna 422.
    """
    # Request sin model_name (requerido)
    request_data = {
        "inputs": {"input": [[1.0, 2.0, 3.0]]}
    }
    
    response = client.post("/predict", json=request_data)
    
    # Validación Pydantic
    assert response.status_code == 422


def test_predict_empty_model_name(client):
    """
    Test 13: Predict con model_name vacío retorna 422.
    """
    request_data = {
        "model_name": "",  # Vacío (min_length=1)
        "inputs": {"input": [[1.0, 2.0, 3.0]]}
    }
    
    response = client.post("/predict", json=request_data)
    
    assert response.status_code == 422


def test_predict_with_metadata(client):
    """
    Test 14: Predict con return_metadata=True (modelo inexistente).
    """
    request_data = {
        "model_name": "test_model",
        "inputs": {"input": [[1.0]]},
        "return_metadata": True
    }
    
    response = client.post("/predict", json=request_data)
    
    # Modelo no existe, pero request es válido
    assert response.status_code in [404, 503]


# ============================================================================
# TESTS - Request Validation
# ============================================================================

def test_load_model_validation_device(client):
    """
    Test 15: Validación de campo 'device' en LoadModelRequest.
    """
    request_data = {
        "path": "/models/test.onnx",
        "device": "invalid_device"  # Debe ser cpu/cuda/auto
    }
    
    response = client.post("/models/load", json=request_data)
    
    # Validación Pydantic debe fallar
    assert response.status_code == 422


def test_load_model_validation_optimization_level(client):
    """
    Test 16: Validación de optimization_level (debe estar en 0-2).
    """
    request_data = {
        "path": "/models/test.onnx",
        "optimization_level": 5  # Fuera de rango (0-2)
    }
    
    response = client.post("/predict", json=request_data)
    
    # Validación Pydantic
    assert response.status_code == 422


def test_predict_validation_batch_size(client):
    """
    Test 17: Validación de batch_size (debe ser >= 1).
    """
    request_data = {
        "model_name": "test",
        "inputs": [[1.0]],
        "batch_size": 0  # Inválido (ge=1)
    }
    
    response = client.post("/predict", json=request_data)
    
    # Validación Pydantic
    assert response.status_code == 422


# ============================================================================
# TESTS - Error Handling
# ============================================================================

def test_invalid_endpoint(client):
    """
    Test 18: Endpoint inexistente retorna 404.
    """
    response = client.get("/invalid_endpoint")
    
    assert response.status_code == 404


def test_method_not_allowed(client):
    """
    Test 19: Método HTTP incorrecto retorna 405.
    """
    # GET en endpoint que requiere POST
    response = client.get("/predict")
    
    assert response.status_code == 405


def test_malformed_json(client):
    """
    Test 20: JSON malformado retorna 422.
    """
    response = client.post(
        "/predict",
        data="malformed json{{{",
        headers={"Content-Type": "application/json"}
    )
    
    assert response.status_code == 422


# ============================================================================
# TESTS - Server State
# ============================================================================

def test_server_state_initialization():
    """
    Test 21: ServerState se inicializa correctamente.
    """
    assert server_state is not None
    assert hasattr(server_state, 'models_info')
    assert isinstance(server_state.models_info, dict)
    assert hasattr(server_state, 'startup_time')


def test_server_state_methods():
    """
    Test 22: ServerState tiene métodos necesarios.
    """
    assert hasattr(server_state, 'add_model_info')
    assert hasattr(server_state, 'remove_model_info')
    assert hasattr(server_state, 'increment_inference_count')
    assert callable(server_state.add_model_info)


# ============================================================================
# TESTS - OpenAPI Documentation
# ============================================================================

def test_openapi_schema(client):
    """
    Test 23: OpenAPI schema está disponible.
    """
    response = client.get("/openapi.json")
    
    assert response.status_code == 200
    schema = response.json()
    
    assert "openapi" in schema
    assert "info" in schema
    assert "paths" in schema


def test_docs_endpoint(client):
    """
    Test 24: Swagger UI está disponible.
    """
    response = client.get("/docs")
    
    assert response.status_code == 200
    assert "swagger" in response.text.lower() or "openapi" in response.text.lower()


def test_redoc_endpoint(client):
    """
    Test 25: ReDoc está disponible.
    """
    response = client.get("/redoc")
    
    assert response.status_code == 200


# ============================================================================
# SUMMARY
# ============================================================================

def test_summary_statistics():
    """
    Test 26: Resumen de estadísticas de tests.
    
    Este test siempre pasa y sirve para documentar el test suite.
    """
    print("\n" + "=" * 80)
    print("TEST SUITE SUMMARY - Session 17 REST API")
    print("=" * 80)
    print(f"Total Tests: 26")
    print(f"Categories:")
    print(f"  - Root & Health: 3 tests")
    print(f"  - Metrics: 2 tests")
    print(f"  - Model Management: 5 tests")
    print(f"  - Inference: 4 tests")
    print(f"  - Request Validation: 3 tests")
    print(f"  - Error Handling: 3 tests")
    print(f"  - Server State: 2 tests")
    print(f"  - OpenAPI: 3 tests")
    print(f"  - Summary: 1 test")
    print("=" * 80)
    
    assert True  # Always pass


# ============================================================================
# PYTEST CONFIGURATION
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
