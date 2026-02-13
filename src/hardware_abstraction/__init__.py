# Hardware Abstraction Package
# Low-level hardware interfaces and monitoring
try:
    from .check_integration_status import (
        check_hybrid_system_integration,
        check_technique_integration,
        generate_integration_report,
    )
except (ImportError, AttributeError):
    check_technique_integration = None
    check_hybrid_system_integration = None
    generate_integration_report = None

try:
    from .debug_test import main as debug_test_main
except (ImportError, AttributeError):
    debug_test_main = None

__all__ = [
    "check_technique_integration",
    "check_hybrid_system_integration",
    "generate_integration_report",
    "debug_test_main",
]
