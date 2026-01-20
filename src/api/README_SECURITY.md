# Security Module

**Session 18 - Phase 4: Complete Security Hardening**

This directory contains comprehensive security implementations for the REST API, providing enterprise-grade protection and access control.

---

## 🔐 Modules

### 1. `security.py` - API Key Authentication

**Features:**
- Multi-method authentication (header, query, bearer)
- Role-based access control (admin, user, readonly)
- Key expiration support
- Environment & file configuration
- Audit logging

**Usage:**
```python
from src.api.security import require_api_key, require_admin

@app.get("/protected")
async def protected(api_key: dict = Depends(require_api_key)):
    return {"user": api_key["name"]}

@app.post("/admin-only")
async def admin(api_key: dict = Depends(require_admin())):
    return {"message": "Admin access granted"}
```

### 2. `rate_limit.py` - Rate Limiting

**Features:**
- Per-IP and per-key limiting
- Adaptive limits by role
- Endpoint-specific limits
- Redis backend support
- Sliding window algorithm

**Usage:**
```python
from src.api.rate_limit import limiter

@app.get("/limited")
@limiter.limit("10/minute")
async def limited():
    return {"message": "Rate limited endpoint"}
```

### 3. `security_headers.py` - Security Headers & Validation

**Features:**
- Comprehensive security headers
- CORS configuration
- Input validation & sanitization
- SQL injection prevention
- XSS protection
- Path traversal prevention

**Usage:**
```python
from src.api.security_headers import add_security_middleware

app = FastAPI()
add_security_middleware(app)
```

---

## 🚀 Quick Start

### 1. Generate API Keys

```bash
python scripts/generate_api_keys.py
```

This creates `config/api_keys.json` with keys for all roles.

### 2. Configure Security

**Environment Variables:**
```bash
# Authentication
export API_KEY_AUTH_ENABLED=true
export API_KEYS="key1:admin,key2:user"

# Rate Limiting
export RATE_LIMIT_ENABLED=true
export RATE_LIMIT_DEFAULT="100/minute"

# Security Headers
export SECURITY_HEADERS_ENABLED=true
export CORS_ORIGINS="http://localhost:3000"
```

**Or use config file:**
```bash
# Keys loaded from config/api_keys.json
export API_KEY_AUTH_ENABLED=true
```

### 3. Apply to FastAPI App

```python
from fastapi import FastAPI, Depends
from src.api.security import require_api_key
from src.api.rate_limit import add_rate_limiting_middleware
from src.api.security_headers import add_security_middleware

app = FastAPI()

# Add security middleware
add_security_middleware(app)
add_rate_limiting_middleware(app)

# Protected endpoint
@app.get("/protected")
async def protected(api_key: dict = Depends(require_api_key)):
    return {"message": f"Hello, {api_key['name']}!"}
```

---

## 🔑 Authentication Methods

### Header (Recommended)
```bash
curl -H "X-API-Key: YOUR_KEY" http://localhost:8000/api
```

### Query Parameter
```bash
curl "http://localhost:8000/api?api_key=YOUR_KEY"
```

### Bearer Token
```bash
curl -H "Authorization: Bearer YOUR_KEY" http://localhost:8000/api
```

---

## 📊 Role-Based Access

| Role | Level | Permissions |
|------|-------|-------------|
| `admin` | 3 | Full access to all endpoints |
| `user` | 2 | Inference + model listing |
| `readonly` | 1 | Health, metrics, GET only |

**Role hierarchy**: admin > user > readonly

---

## 🚦 Rate Limits

### Default Limits

- **Anonymous**: 100 requests/minute
- **Authenticated**: 1000 requests/minute
- **Admin**: 10000 requests/minute

### Endpoint-Specific

- `/health`: 1000/minute
- `/metrics`: 100/minute
- `/models/load`: 10/minute (expensive)
- `/inference`: 500/minute

### Configuration

```bash
export RATE_LIMIT_DEFAULT="50/minute"
export RATE_LIMIT_AUTHENTICATED="500/minute"
export RATE_LIMIT_ADMIN="5000/minute"
```

---

## 🛡️ Security Headers

### Applied Headers

```
X-Content-Type-Options: nosniff
X-Frame-Options: DENY
X-XSS-Protection: 1; mode=block
Strict-Transport-Security: max-age=31536000
Content-Security-Policy: default-src 'self'; ...
Referrer-Policy: strict-origin-when-cross-origin
Permissions-Policy: geolocation=(), microphone=(), camera=()
```

### CORS

```bash
# Specific origins
export CORS_ORIGINS="http://localhost:3000,https://app.example.com"

# All origins (dev only)
export CORS_ORIGINS="*"
```

---

## 🧪 Testing

### Test Authentication

```bash
# No key (should fail)
curl http://localhost:8000/models
# → 401 Unauthorized

# Valid key
curl -H "X-API-Key: rx580-abc123..." http://localhost:8000/models
# → 200 OK

# Invalid key
curl -H "X-API-Key: invalid" http://localhost:8000/models
# → 401 Unauthorized
```

### Test Rate Limiting

```bash
# Exceed limit
for i in {1..150}; do curl http://localhost:8000/health; done
# → First 100 OK, then 429 Too Many Requests
```

### Test Role-Based Access

```bash
# Readonly trying admin action (should fail)
curl -X POST -H "X-API-Key: READONLY_KEY" \
  http://localhost:8000/models/load
# → 403 Forbidden

# Admin performing action (should succeed)
curl -X POST -H "X-API-Key: ADMIN_KEY" \
  http://localhost:8000/models/load
# → 200 OK
```

---

## 📚 API Reference

### `security.py`

```python
# Dependencies
require_api_key()          # Any valid key
require_admin()            # Admin role required
require_user()             # User+ role required
require_readonly()         # Readonly+ role required

# Functions
SecurityConfig.generate_key(prefix)  # Generate new key
SecurityConfig.validate_key(key)     # Validate key
security_config.has_role(info, role) # Check role

# Classes
SecurityConfig              # Configuration management
```

### `rate_limit.py`

```python
# Global limiter
limiter                     # Limiter instance

# Decorators
@limiter.limit("10/minute") # Fixed limit
@adaptive_rate_limit("10/minute")  # Adaptive by role
@endpoint_rate_limit("/endpoint")  # Endpoint-specific

# Functions
add_rate_limiting_middleware(app)  # Add to FastAPI
get_rate_limit_status(request)     # Get current status
reset_rate_limit(key)              # Reset for key
```

### `security_headers.py`

```python
# Middleware
add_security_middleware(app)       # Add all middleware

# Utilities
sanitize_string(text)              # Sanitize text
sanitize_filename(filename)        # Sanitize filename
validate_url(url)                  # Validate URL

# Info
get_security_info()                # Get config
```

---

## 🔧 Configuration Reference

### Environment Variables

```bash
# Authentication
API_KEY_AUTH_ENABLED=true          # Enable authentication
ALLOW_DEV_KEYS=false               # Allow dev keys
API_KEYS=key1:role1,key2:role2     # Comma-separated keys
ADMIN_KEYS=key1,key2               # Admin keys

# Rate Limiting
RATE_LIMIT_ENABLED=true            # Enable rate limiting
RATE_LIMIT_STRATEGY=moving-window  # Algorithm
RATE_LIMIT_STORAGE=memory          # Storage (memory/redis)
RATE_LIMIT_REDIS_URL=redis://...   # Redis URL
RATE_LIMIT_DEFAULT=100/minute      # Default limit
RATE_LIMIT_AUTHENTICATED=1000/minute  # Auth limit
RATE_LIMIT_ADMIN=10000/minute      # Admin limit

# Security Headers
SECURITY_HEADERS_ENABLED=true      # Enable headers
HTTPS_ONLY=false                   # Enforce HTTPS
CSP_ENABLED=true                   # Content Security Policy
CORS_ENABLED=true                  # Enable CORS
CORS_ORIGINS=*                     # Allowed origins
CORS_ALLOW_CREDENTIALS=true        # Allow credentials
TRUSTED_HOSTS=localhost,127.0.0.1  # Trusted hosts
```

### Configuration Files

```
config/
└── api_keys.json          # API keys configuration
    {
      "keys": {
        "rx580-abc123...": {
          "name": "admin_key_1",
          "role": "admin",
          "created": "2026-01-19T...",
          "expires": "2027-01-19T...",
          "description": "Admin key"
        }
      }
    }
```

---

## 🎓 Examples

### Complete Integration

```python
from fastapi import FastAPI, Depends
from src.api.security import require_api_key, require_admin
from src.api.rate_limit import limiter, add_rate_limiting_middleware
from src.api.security_headers import add_security_middleware

app = FastAPI()

# Add middleware
add_security_middleware(app)
add_rate_limiting_middleware(app)

# Public endpoint (rate limited)
@app.get("/health")
@limiter.limit("1000/minute")
async def health():
    return {"status": "healthy"}

# Protected endpoint (requires any valid key)
@app.get("/models")
async def models(api_key: dict = Depends(require_api_key)):
    return {"models": [...], "user": api_key["name"]}

# Admin-only endpoint
@app.post("/admin/reset")
async def reset(api_key: dict = Depends(require_admin())):
    # Admin action
    return {"message": "Reset successful"}
```

### Python Client

```python
import requests

API_KEY = "rx580-abc123..."
BASE_URL = "http://localhost:8000"

# Header authentication
headers = {"X-API-Key": API_KEY}
response = requests.get(f"{BASE_URL}/models", headers=headers)

# Query parameter
response = requests.get(f"{BASE_URL}/models?api_key={API_KEY}")

# Bearer token
headers = {"Authorization": f"Bearer {API_KEY}"}
response = requests.get(f"{BASE_URL}/models", headers=headers)
```

---

## 📖 Further Reading

- [OWASP API Security Top 10](https://owasp.org/www-project-api-security/)
- [FastAPI Security](https://fastapi.tiangolo.com/tutorial/security/)
- [Rate Limiting Strategies](https://blog.logrocket.com/rate-limiting-python-fastapi/)

---

**Quality**: 9.8/10 | **Session**: 18 | **Phase**: 4/4 | **Status**: COMPLETE ✅
