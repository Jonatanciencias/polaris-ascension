"""
Shared helpers to persist benchmark reports in JSON/Markdown formats.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Sequence


def timestamp_tag() -> str:
    """Return a compact timestamp used in benchmark report filenames."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def ensure_output_dir(output_dir: str | Path) -> Path:
    """Create output directory (if needed) and return it as Path."""
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path


def report_paths(
    prefix: str,
    output_dir: str | Path = "results/benchmark_reports",
    stamp: str | None = None,
) -> tuple[Path, Path]:
    """
    Build deterministic JSON/Markdown paths for a benchmark report.
    """
    stamp = stamp or timestamp_tag()
    base = ensure_output_dir(output_dir)
    return (
        base / f"{prefix}_{stamp}.json",
        base / f"{prefix}_{stamp}.md",
    )


def save_json_report(path: Path, payload: dict[str, Any]) -> None:
    """Persist benchmark payload as prettified JSON."""
    path.write_text(json.dumps(payload, indent=2, sort_keys=False))


def save_markdown_report(path: Path, content: str) -> None:
    """Persist benchmark report as Markdown text."""
    path.write_text(content)


def markdown_table(headers: Sequence[str], rows: Iterable[Sequence[Any]]) -> str:
    """Render a simple GitHub-compatible Markdown table."""
    head = "| " + " | ".join(headers) + " |"
    sep = "| " + " | ".join(["---"] * len(headers)) + " |"
    body = ["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows]
    return "\n".join([head, sep, *body])
