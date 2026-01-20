#!/usr/bin/env python3
"""
Quick API Key Generator (Standalone)
Generates API keys without external dependencies for testing.
"""

import secrets
import json
from datetime import datetime, timedelta
from pathlib import Path


def generate_key(prefix: str = "rx580") -> str:
    """Generate a secure API key."""
    random_part = secrets.token_urlsafe(32)
    return f"{prefix}-{random_part}"


def main():
    """Generate test API keys."""
    
    # Configuration
    expires_days = 365
    expires_date = (datetime.now() + timedelta(days=expires_days)).isoformat()
    created = datetime.now().isoformat()
    
    # Generate keys
    admin_key_1 = generate_key("rx580-admin")
    admin_key_2 = generate_key("rx580-admin")
    user_key_1 = generate_key("rx580-user")
    user_key_2 = generate_key("rx580-user")
    readonly_key_1 = generate_key("rx580-readonly")
    readonly_key_2 = generate_key("rx580-readonly")
    
    # Create config structure
    keys_data = {
        "generated_at": created,
        "keys": {
            admin_key_1: {
                "name": "admin_key_1",
                "role": "admin",
                "created": created,
                "expires": expires_date,
                "description": "Admin key 1 for testing"
            },
            admin_key_2: {
                "name": "admin_key_2",
                "role": "admin",
                "created": created,
                "expires": expires_date,
                "description": "Admin key 2 for testing"
            },
            user_key_1: {
                "name": "user_key_1",
                "role": "user",
                "created": created,
                "expires": expires_date,
                "description": "User key 1 for testing"
            },
            user_key_2: {
                "name": "user_key_2",
                "role": "user",
                "created": created,
                "expires": expires_date,
                "description": "User key 2 for testing"
            },
            readonly_key_1: {
                "name": "readonly_key_1",
                "role": "readonly",
                "created": created,
                "expires": expires_date,
                "description": "Readonly key 1 for testing"
            },
            readonly_key_2: {
                "name": "readonly_key_2",
                "role": "readonly",
                "created": created,
                "expires": expires_date,
                "description": "Readonly key 2 for testing"
            }
        },
        "roles": {
            "admin": ["full access", "model management", "configuration", "user management"],
            "user": ["inference", "model listing", "health checks", "metrics"],
            "readonly": ["health checks", "metrics", "GET requests only"]
        }
    }
    
    # Create config directory if needed
    config_dir = Path(__file__).parent.parent / "config"
    config_dir.mkdir(exist_ok=True)
    
    # Save to file
    output_file = config_dir / "api_keys.json"
    with open(output_file, "w") as f:
        json.dump(keys_data, f, indent=2)
    
    print("=" * 80)
    print("✅ API Keys Generated Successfully!")
    print("=" * 80)
    print(f"\n📁 Output: {output_file}\n")
    
    print("📦 Generated Keys:\n")
    
    print("🔑 ADMIN Keys (Full Access):")
    print(f"   1. {admin_key_1}")
    print(f"   2. {admin_key_2}\n")
    
    print("🔑 USER Keys (Inference + Listing):")
    print(f"   1. {user_key_1}")
    print(f"   2. {user_key_2}\n")
    
    print("🔑 READONLY Keys (Health + Metrics):")
    print(f"   1. {readonly_key_1}")
    print(f"   2. {readonly_key_2}\n")
    
    print("=" * 80)
    print("📚 Usage Examples:\n")
    
    print("1️⃣  Header Authentication:")
    print(f'   curl -H "X-API-Key: {admin_key_1}" http://localhost:8000/models\n')
    
    print("2️⃣  Query Parameter:")
    print(f'   curl "http://localhost:8000/models?api_key={user_key_1}"\n')
    
    print("3️⃣  Bearer Token:")
    print(f'   curl -H "Authorization: Bearer {readonly_key_1}" http://localhost:8000/health\n')
    
    print("4️⃣  Environment Variable:")
    print(f'   export API_KEY_AUTH_ENABLED=true')
    print(f'   # Server will load from {output_file}\n')
    
    print("=" * 80)
    print("⚠️  Security Notes:")
    print("   - Store keys securely")
    print("   - Don't commit api_keys.json to git")
    print("   - Use environment variables in production")
    print("   - Rotate keys regularly")
    print("=" * 80)


if __name__ == "__main__":
    main()
