# Hardware Abstraction Package
# Low-level hardware interfaces and monitoring
from typing import Any, cast

try:
    from .check_integration_status import (
        check_hybrid_system_integration,
        check_technique_integration,
        generate_integration_report,
    )
except (ImportError, AttributeError):
    check_technique_integration = cast(Any, None)
    check_hybrid_system_integration = cast(Any, None)
    generate_integration_report = cast(Any, None)

try:
    from .debug_test import main as debug_test_main
except (ImportError, AttributeError):
    debug_test_main = cast(Any, None)

__all__ = [
    "check_technique_integration",
    "check_hybrid_system_integration",
    "generate_integration_report",
    "debug_test_main",
]
