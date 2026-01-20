#!/usr/bin/env python3
"""
Security Integration Tests
Session 18 - Phase 4: Testing de Integración

Tests de integración para validar:
1. Autenticación (3 métodos)
2. RBAC (admin/user/readonly)
3. Rate limiting
4. Security headers

Requiere:
- Server corriendo en http://localhost:8000
- API keys generadas en config/api_keys.json

Usage:
    python scripts/test_security_integration.py
"""

import json
import requests
import time
from pathlib import Path
from typing import Dict, Any


# ============================================================================
# CONFIGURACIÓN
# ============================================================================

BASE_URL = "http://localhost:8000"
API_KEYS_FILE = Path(__file__).parent.parent / "config" / "api_keys.json"

# Cargar API keys
def load_api_keys() -> Dict[str, str]:
    """Carga las API keys del archivo de configuración."""
    if not API_KEYS_FILE.exists():
        print(f"❌ Error: API keys file not found: {API_KEYS_FILE}")
        print("   Run: python scripts/generate_test_keys.py")
        exit(1)
    
    with open(API_KEYS_FILE) as f:
        data = json.load(f)
    
    # Extraer primera key de cada rol
    keys = {}
    for key, info in data["keys"].items():
        role = info["role"]
        if role not in keys:
            keys[role] = key
    
    return keys


# ============================================================================
# TEST UTILITIES
# ============================================================================

class TestResult:
    """Resultado de un test."""
    def __init__(self, name: str, passed: bool, message: str = "", details: Any = None):
        self.name = name
        self.passed = passed
        self.message = message
        self.details = details
    
    def __str__(self):
        status = "✅ PASS" if self.passed else "❌ FAIL"
        msg = f": {self.message}" if self.message else ""
        return f"{status} - {self.name}{msg}"


def run_test(name: str, test_func) -> TestResult:
    """Ejecuta un test y captura el resultado."""
    try:
        result = test_func()
        if isinstance(result, TestResult):
            return result
        return TestResult(name, True, "Test passed")
    except Exception as e:
        return TestResult(name, False, str(e))


# ============================================================================
# TESTS - AUTENTICACIÓN
# ============================================================================

def test_no_auth(keys: Dict[str, str]) -> TestResult:
    """Test: Request sin autenticación debe fallar en endpoints protegidos."""
    try:
        response = requests.get(f"{BASE_URL}/models", timeout=5)
        
        if response.status_code == 401:
            return TestResult(
                "No Auth (should fail)",
                True,
                f"Correctly rejected with 401"
            )
        else:
            return TestResult(
                "No Auth (should fail)",
                False,
                f"Expected 401, got {response.status_code}"
            )
    except requests.exceptions.ConnectionError:
        return TestResult(
            "No Auth (should fail)",
            False,
            "Server not running. Start with: uvicorn src.api.server:app"
        )


def test_header_auth(keys: Dict[str, str]) -> TestResult:
    """Test: Autenticación por header (X-API-Key)."""
    headers = {"X-API-Key": keys["user"]}
    response = requests.get(f"{BASE_URL}/models", headers=headers, timeout=5)
    
    if response.status_code == 200:
        return TestResult("Header Auth", True, "Authenticated successfully")
    else:
        return TestResult("Header Auth", False, f"Status {response.status_code}")


def test_query_auth(keys: Dict[str, str]) -> TestResult:
    """Test: Autenticación por query parameter."""
    response = requests.get(
        f"{BASE_URL}/models?api_key={keys['user']}",
        timeout=5
    )
    
    if response.status_code == 200:
        return TestResult("Query Auth", True, "Authenticated successfully")
    else:
        return TestResult("Query Auth", False, f"Status {response.status_code}")


def test_bearer_auth(keys: Dict[str, str]) -> TestResult:
    """Test: Autenticación por Bearer token."""
    headers = {"Authorization": f"Bearer {keys['user']}"}
    response = requests.get(f"{BASE_URL}/models", headers=headers, timeout=5)
    
    if response.status_code == 200:
        return TestResult("Bearer Auth", True, "Authenticated successfully")
    else:
        return TestResult("Bearer Auth", False, f"Status {response.status_code}")


def test_invalid_key() -> TestResult:
    """Test: Key inválida debe fallar."""
    headers = {"X-API-Key": "invalid-key-12345"}
    response = requests.get(f"{BASE_URL}/models", headers=headers, timeout=5)
    
    if response.status_code == 401:
        return TestResult("Invalid Key", True, "Correctly rejected")
    else:
        return TestResult("Invalid Key", False, f"Expected 401, got {response.status_code}")


# ============================================================================
# TESTS - RBAC (Role-Based Access Control)
# ============================================================================

def test_readonly_access(keys: Dict[str, str]) -> TestResult:
    """Test: Readonly puede acceder a health y metrics."""
    headers = {"X-API-Key": keys["readonly"]}
    
    # Health debe ser accesible
    health_response = requests.get(f"{BASE_URL}/health", headers=headers, timeout=5)
    
    if health_response.status_code == 200:
        return TestResult("Readonly - Health", True, "Can access /health")
    else:
        return TestResult("Readonly - Health", False, f"Status {health_response.status_code}")


def test_readonly_cannot_inference(keys: Dict[str, str]) -> TestResult:
    """Test: Readonly NO puede hacer inferencia."""
    headers = {"X-API-Key": keys["readonly"]}
    data = {
        "model_name": "test",
        "inputs": {"input": [[1, 2, 3]]}
    }
    
    response = requests.post(f"{BASE_URL}/predict", headers=headers, json=data, timeout=5)
    
    # Debe fallar con 403 (Forbidden) o 503 (no engine)
    if response.status_code in [403, 503]:
        return TestResult("Readonly - Cannot Inference", True, "Correctly denied")
    else:
        return TestResult("Readonly - Cannot Inference", False, f"Expected 403/503, got {response.status_code}")


def test_user_can_list_models(keys: Dict[str, str]) -> TestResult:
    """Test: User puede listar modelos."""
    headers = {"X-API-Key": keys["user"]}
    response = requests.get(f"{BASE_URL}/models", headers=headers, timeout=5)
    
    if response.status_code == 200:
        return TestResult("User - List Models", True, "Can access /models")
    else:
        return TestResult("User - List Models", False, f"Status {response.status_code}")


def test_user_cannot_load_model(keys: Dict[str, str]) -> TestResult:
    """Test: User NO puede cargar modelos (solo admin)."""
    headers = {"X-API-Key": keys["user"]}
    data = {
        "path": "/fake/model.onnx"
    }
    
    response = requests.post(f"{BASE_URL}/models/load", headers=headers, json=data, timeout=5)
    
    # Debe fallar con 403 (Forbidden)
    if response.status_code == 403:
        return TestResult("User - Cannot Load", True, "Correctly denied (admin only)")
    elif response.status_code == 404:
        # Si llegó aquí, la autenticación pasó (malo)
        return TestResult("User - Cannot Load", False, "Should have been denied with 403")
    else:
        return TestResult("User - Cannot Load", False, f"Expected 403, got {response.status_code}")


def test_admin_can_load_model(keys: Dict[str, str]) -> TestResult:
    """Test: Admin puede cargar modelos."""
    headers = {"X-API-Key": keys["admin"]}
    data = {
        "path": "/fake/model.onnx"
    }
    
    response = requests.post(f"{BASE_URL}/models/load", headers=headers, json=data, timeout=5)
    
    # Debe pasar autenticación (aunque falle por archivo inexistente = 404 o 503)
    if response.status_code in [404, 503, 500]:
        return TestResult("Admin - Can Load", True, "Auth passed (file not found expected)")
    elif response.status_code == 403:
        return TestResult("Admin - Can Load", False, "Admin was denied (should have access)")
    else:
        return TestResult("Admin - Can Load", False, f"Unexpected status {response.status_code}")


# ============================================================================
# TESTS - RATE LIMITING
# ============================================================================

def test_rate_limit_anonymous() -> TestResult:
    """Test: Rate limiting para requests anónimos."""
    # Hacer muchos requests rápidos
    count = 0
    rate_limited = False
    
    for i in range(110):  # Más del límite (100/min)
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        if response.status_code == 429:
            rate_limited = True
            count = i + 1
            break
    
    if rate_limited:
        return TestResult("Rate Limit - Anonymous", True, f"Rate limited after {count} requests")
    else:
        return TestResult("Rate Limit - Anonymous", False, "No rate limit enforced")


def test_rate_limit_headers() -> TestResult:
    """Test: Verificar headers de rate limiting."""
    response = requests.get(f"{BASE_URL}/health", timeout=5)
    
    # Buscar headers relevantes
    headers = response.headers
    rate_headers = [h for h in headers if 'rate' in h.lower() or 'ratelimit' in h.lower()]
    
    if rate_headers:
        return TestResult("Rate Limit - Headers", True, f"Found: {rate_headers}")
    else:
        return TestResult("Rate Limit - Headers", False, "No rate limit headers found")


# ============================================================================
# TESTS - SECURITY HEADERS
# ============================================================================

def test_security_headers() -> TestResult:
    """Test: Verificar presencia de security headers."""
    response = requests.get(f"{BASE_URL}/health", timeout=5)
    headers = response.headers
    
    required_headers = [
        "x-content-type-options",
        "x-frame-options",
        "x-xss-protection",
    ]
    
    found = []
    missing = []
    
    for header in required_headers:
        if header in headers:
            found.append(f"{header}: {headers[header]}")
        else:
            missing.append(header)
    
    if not missing:
        return TestResult("Security Headers", True, f"All present: {', '.join(found)}")
    else:
        return TestResult("Security Headers", False, f"Missing: {', '.join(missing)}")


def test_cors_headers() -> TestResult:
    """Test: Verificar CORS headers."""
    response = requests.get(f"{BASE_URL}/health", timeout=5)
    headers = response.headers
    
    if "access-control-allow-origin" in headers:
        return TestResult("CORS Headers", True, f"Origin: {headers['access-control-allow-origin']}")
    else:
        return TestResult("CORS Headers", False, "CORS headers not found")


# ============================================================================
# MAIN - EJECUTAR TODOS LOS TESTS
# ============================================================================

def main():
    """Ejecuta todos los tests de integración."""
    
    print("=" * 80)
    print("🔐 Security Integration Tests - Session 18 Phase 4")
    print("=" * 80)
    print()
    
    # Cargar API keys
    print("📦 Loading API keys...")
    try:
        keys = load_api_keys()
        print(f"   ✅ Loaded keys for roles: {', '.join(keys.keys())}")
        print()
    except Exception as e:
        print(f"   ❌ Error loading keys: {e}")
        return
    
    # Verificar servidor
    print("🌐 Checking server...")
    try:
        response = requests.get(f"{BASE_URL}/", timeout=5)
        print(f"   ✅ Server running at {BASE_URL}")
        server_info = response.json()
        print(f"   Version: {server_info.get('version', 'unknown')}")
        security_info = server_info.get('security', {})
        if isinstance(security_info, dict):
            print(f"   Security: {security_info.get('enabled', False)}")
        else:
            print(f"   Security: {security_info}")
        print()
    except requests.exceptions.ConnectionError:
        print(f"   ❌ Server not running at {BASE_URL}")
        print("      Start with: uvicorn src.api.server:app")
        return
    
    # Ejecutar tests
    results = []
    
    print("=" * 80)
    print("🧪 Running Tests")
    print("=" * 80)
    print()
    
    # Autenticación
    print("1️⃣  Authentication Tests")
    results.append(run_test("No Auth", lambda: test_no_auth(keys)))
    results.append(run_test("Header Auth", lambda: test_header_auth(keys)))
    results.append(run_test("Query Auth", lambda: test_query_auth(keys)))
    results.append(run_test("Bearer Auth", lambda: test_bearer_auth(keys)))
    results.append(run_test("Invalid Key", test_invalid_key))
    print()
    
    # RBAC
    print("2️⃣  RBAC Tests")
    results.append(run_test("Readonly - Health", lambda: test_readonly_access(keys)))
    results.append(run_test("Readonly - Cannot Inference", lambda: test_readonly_cannot_inference(keys)))
    results.append(run_test("User - List Models", lambda: test_user_can_list_models(keys)))
    results.append(run_test("User - Cannot Load", lambda: test_user_cannot_load_model(keys)))
    results.append(run_test("Admin - Can Load", lambda: test_admin_can_load_model(keys)))
    print()
    
    # Rate Limiting
    print("3️⃣  Rate Limiting Tests")
    print("   (This may take a moment...)")
    results.append(run_test("Rate Limit - Anonymous", test_rate_limit_anonymous))
    results.append(run_test("Rate Limit - Headers", test_rate_limit_headers))
    print()
    
    # Security Headers
    print("4️⃣  Security Headers Tests")
    results.append(run_test("Security Headers", test_security_headers))
    results.append(run_test("CORS Headers", test_cors_headers))
    print()
    
    # Resultados
    print("=" * 80)
    print("📊 Test Results")
    print("=" * 80)
    print()
    
    for result in results:
        print(f"   {result}")
    
    # Resumen
    passed = sum(1 for r in results if r.passed)
    failed = sum(1 for r in results if not r.passed)
    total = len(results)
    percentage = (passed / total * 100) if total > 0 else 0
    
    print()
    print("=" * 80)
    print(f"✅ Passed: {passed}/{total} ({percentage:.1f}%)")
    if failed > 0:
        print(f"❌ Failed: {failed}/{total}")
    print("=" * 80)
    
    # Exit code
    exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
