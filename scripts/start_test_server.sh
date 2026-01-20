#!/bin/bash
# Start API Server for Testing
# Session 18 - Phase 4: Security Integration Testing

echo "=========================================="
echo "🚀 Starting API Server (Testing Mode)"
echo "=========================================="
echo ""

# Configuration
export API_KEY_AUTH_ENABLED=true
export RATE_LIMIT_ENABLED=true
export SECURITY_HEADERS_ENABLED=true
export CORS_ORIGINS="*"

echo "✅ Security Configuration:"
echo "   - Authentication: ENABLED"
echo "   - Rate Limiting: ENABLED"
echo "   - Security Headers: ENABLED"
echo "   - CORS: * (all origins)"
echo ""

# Check if api_keys.json exists
if [ ! -f "config/api_keys.json" ]; then
    echo "⚠️  API keys not found. Generating..."
    python3 scripts/generate_test_keys.py
    echo ""
fi

echo "🌐 Starting server on http://localhost:8000"
echo "   Docs: http://localhost:8000/docs"
echo "   Health: http://localhost:8000/health"
echo ""
echo "Press Ctrl+C to stop"
echo "=========================================="
echo ""

# Start server (will fail if dependencies not installed, which is expected)
cd "$(dirname "$0")/.." && python3 -m uvicorn src.api.server:app --host 0.0.0.0 --port 8000 --reload
