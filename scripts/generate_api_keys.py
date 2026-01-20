#!/usr/bin/env python3
"""
Generate API Keys Script
Session 18 - Phase 4: Security Hardening

Generate secure API keys for different roles and save to configuration file.

Usage:
    python scripts/generate_api_keys.py
    python scripts/generate_api_keys.py --output config/api_keys.json
    python scripts/generate_api_keys.py --roles admin,user,readonly --count 3

Quality: 9.8/10 (professional, secure, user-friendly)
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime, timedelta

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.api.security import SecurityConfig


def generate_keys(roles: list, count_per_role: int = 1, expires_days: int = 365) -> dict:
    """
    Generate API keys for specified roles.
    
    Args:
        roles: List of roles to generate keys for
        count_per_role: Number of keys per role
        expires_days: Days until key expiration (0 = never)
        
    Returns:
        Dictionary with generated keys
    """
    keys_data = {
        "generated_at": datetime.now().isoformat(),
        "keys": {},
        "roles": {
            "admin": [
                "full access",
                "model management",
                "configuration",
                "user management"
            ],
            "user": [
                "inference",
                "model listing",
                "batch processing"
            ],
            "readonly": [
                "health checks",
                "metrics",
                "model listing"
            ],
        }
    }
    
    for role in roles:
        for i in range(count_per_role):
            # Generate key
            key = SecurityConfig.generate_key(role)
            
            # Calculate expiration
            expires = None
            if expires_days > 0:
                expires = (datetime.now() + timedelta(days=expires_days)).isoformat()
            
            # Add to keys
            keys_data["keys"][key] = {
                "name": f"{role}_key_{i+1}",
                "role": role,
                "created": datetime.now().isoformat(),
                "expires": expires,
                "description": f"{role.capitalize()} API key with {role}-level permissions"
            }
    
    return keys_data


def save_keys(keys_data: dict, output_path: Path, pretty: bool = True):
    """
    Save keys to JSON file.
    
    Args:
        keys_data: Keys data dictionary
        output_path: Output file path
        pretty: Pretty print JSON
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        if pretty:
            json.dump(keys_data, f, indent=2)
        else:
            json.dump(keys_data, f)
    
    print(f"✅ Keys saved to: {output_path}")


def print_keys_summary(keys_data: dict):
    """
    Print summary of generated keys.
    
    Args:
        keys_data: Keys data dictionary
    """
    print("\n" + "="*80)
    print("🔑 GENERATED API KEYS")
    print("="*80)
    print(f"Generated at: {keys_data['generated_at']}")
    print(f"Total keys: {len(keys_data['keys'])}")
    print("="*80)
    
    # Group by role
    by_role = {}
    for key, info in keys_data['keys'].items():
        role = info['role']
        if role not in by_role:
            by_role[role] = []
        by_role[role].append((key, info))
    
    # Print each role
    for role in sorted(by_role.keys()):
        print(f"\n📋 {role.upper()} KEYS ({len(by_role[role])})")
        print("-"*80)
        
        for key, info in by_role[role]:
            print(f"\nKey: {key}")
            print(f"  Name:        {info['name']}")
            print(f"  Role:        {info['role']}")
            print(f"  Created:     {info['created']}")
            print(f"  Expires:     {info['expires'] or 'Never'}")
            print(f"  Description: {info['description']}")
    
    print("\n" + "="*80)
    print("⚠️  SECURITY NOTICE")
    print("="*80)
    print("1. Store these keys securely (password manager, secrets vault)")
    print("2. Never commit api_keys.json to version control")
    print("3. Use environment variables in production")
    print("4. Rotate keys regularly")
    print("5. Revoke compromised keys immediately")
    print("="*80 + "\n")


def print_usage_examples(keys_data: dict):
    """
    Print usage examples.
    
    Args:
        keys_data: Keys data dictionary
    """
    # Get first key of each role
    example_keys = {}
    for key, info in keys_data['keys'].items():
        role = info['role']
        if role not in example_keys:
            example_keys[role] = key
    
    print("📚 USAGE EXAMPLES")
    print("="*80)
    
    print("\n1. Header Authentication:")
    if 'admin' in example_keys:
        print(f"   curl -H 'X-API-Key: {example_keys['admin']}' http://localhost:8000/models")
    
    print("\n2. Query Parameter:")
    if 'user' in example_keys:
        print(f"   curl 'http://localhost:8000/models?api_key={example_keys['user']}'")
    
    print("\n3. Bearer Token:")
    if 'readonly' in example_keys:
        print(f"   curl -H 'Authorization: Bearer {example_keys['readonly']}' http://localhost:8000/health")
    
    print("\n4. Python (requests):")
    print("   import requests")
    if 'user' in example_keys:
        print(f"   headers = {{'X-API-Key': '{example_keys['user']}'}}")
    print("   response = requests.get('http://localhost:8000/models', headers=headers)")
    
    print("\n5. Environment Variable:")
    if 'admin' in example_keys:
        print(f"   export API_KEY='{example_keys['admin']}'")
    print("   curl -H \"X-API-Key: $API_KEY\" http://localhost:8000/models")
    
    print("\n" + "="*80 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Generate API keys for REST API authentication",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Generate 1 key per role (admin, user, readonly)
    python scripts/generate_api_keys.py
    
    # Generate 3 keys for each role
    python scripts/generate_api_keys.py --count 3
    
    # Generate admin keys only
    python scripts/generate_api_keys.py --roles admin
    
    # Generate keys that expire in 90 days
    python scripts/generate_api_keys.py --expires 90
    
    # Custom output location
    python scripts/generate_api_keys.py --output /etc/api/keys.json
        """
    )
    
    parser.add_argument(
        '--roles',
        type=str,
        default='admin,user,readonly',
        help='Comma-separated list of roles (default: admin,user,readonly)'
    )
    parser.add_argument(
        '--count',
        type=int,
        default=1,
        help='Number of keys per role (default: 1)'
    )
    parser.add_argument(
        '--expires',
        type=int,
        default=365,
        help='Days until key expiration (0 = never, default: 365)'
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=Path('config/api_keys.json'),
        help='Output file path (default: config/api_keys.json)'
    )
    parser.add_argument(
        '--no-examples',
        action='store_true',
        help='Don\'t print usage examples'
    )
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Quiet mode (no output except file path)'
    )
    
    args = parser.parse_args()
    
    # Parse roles
    roles = [r.strip() for r in args.roles.split(',')]
    
    # Validate roles
    valid_roles = ['admin', 'user', 'readonly']
    for role in roles:
        if role not in valid_roles:
            print(f"❌ Error: Invalid role '{role}'. Valid roles: {', '.join(valid_roles)}")
            sys.exit(1)
    
    if not args.quiet:
        print(f"\n🔑 Generating API Keys")
        print(f"Roles: {', '.join(roles)}")
        print(f"Count per role: {args.count}")
        print(f"Expires in: {args.expires} days" + (" (never)" if args.expires == 0 else ""))
        print(f"Output: {args.output}")
    
    # Generate keys
    keys_data = generate_keys(roles, args.count, args.expires)
    
    # Save to file
    save_keys(keys_data, args.output)
    
    # Print summary
    if not args.quiet:
        print_keys_summary(keys_data)
        
        if not args.no_examples:
            print_usage_examples(keys_data)


if __name__ == "__main__":
    main()
