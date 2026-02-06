# 🔐 Setup Guide: API Keys Configuration

## Quick Setup

```bash
# 1. Copy example file
cp configs/api_keys.json.example configs/api_keys.json

# 2. Generate secure keys
python scripts/generate_api_keys.py

# Or manually edit configs/api_keys.json
```

## Security Notes

⚠️ **IMPORTANT**: 
- `configs/api_keys.json` is in `.gitignore` - NEVER commit this file
- The `api_keys.json.example` file is safe to commit (contains placeholders)
- Each key should be unique and cryptographically secure

## Key Structure

```json
{
  "generated_at": "2026-02-05T...",
  "keys": {
    "rx580-ROLE-RANDOM_STRING": {
      "name": "descriptive_name",
      "role": "admin|user|developer",
      "created": "timestamp",
      "expires": "timestamp",
      "description": "What this key is for"
    }
  }
}
```

## Roles

- **admin**: Full access to all endpoints
- **user**: Read-only access
- **developer**: Testing and development access

## Generating Keys

The keys follow format: `rx580-{role}-{random_base64}`

Use `scripts/generate_api_keys.py` which generates cryptographically secure keys using `secrets` module.

## Verification

After setup:
```bash
# Check config is valid
python -c "import json; json.load(open('configs/api_keys.json'))"
```

## For Production

For production environments:
1. Generate new keys (don't use example/development keys)
2. Set key expiration dates
3. Rotate keys periodically
4. Use environment variables instead of file-based keys
5. Consider using a secrets manager (HashiCorp Vault, AWS Secrets Manager)
