"""
Logging setup helper used by legacy examples.
"""

from __future__ import annotations

import logging
from typing import Union


def setup_logging(level: Union[str, int] = "INFO") -> logging.Logger:
    """Configure root logging with a concise format."""
    if isinstance(level, str):
        numeric = getattr(logging, level.upper(), logging.INFO)
    else:
        numeric = int(level)

    logging.basicConfig(
        level=numeric,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    return logging.getLogger(__name__)
