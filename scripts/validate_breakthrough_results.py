#!/usr/bin/env python3
"""Validate breakthrough lab results.json files against the canonical schema.

Usage:
  python scripts/validate_breakthrough_results.py
  python scripts/validate_breakthrough_results.py --file research/breakthrough_lab/t3_online_control/results.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable

from jsonschema import Draft202012Validator


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _iter_result_files(repo_root: Path, files: list[str] | None) -> list[Path]:
    if files:
        out = [(repo_root / rel).resolve() for rel in files]
        return sorted(out)
    return sorted((repo_root / "research/breakthrough_lab").glob("t*/results.json"))


def _format_error(err, base: Path) -> str:
    path = ".".join(str(p) for p in err.absolute_path)
    json_path = path if path else "<root>"
    rel = err.path if err.path else []
    return f"{base}: {json_path}: {err.message} (path={list(rel)})"


def _validate_track_name(data: dict, results_path: Path, repo_root: Path) -> str | None:
    expected_track = results_path.parent.name
    actual_track = data.get("track")
    if actual_track != expected_track:
        rel = results_path.relative_to(repo_root)
        return (
            f"{rel}: track mismatch (expected '{expected_track}', got '{actual_track}')"
        )
    return None


def _validate_schema_version(data: dict, results_path: Path, repo_root: Path) -> str | None:
    if data.get("schema_version") != "1.0.0":
        rel = results_path.relative_to(repo_root)
        return f"{rel}: schema_version must be '1.0.0' (got '{data.get('schema_version')}')"
    return None


def validate_files(
    *,
    repo_root: Path,
    schema_path: Path,
    result_files: Iterable[Path],
) -> tuple[list[Path], list[str]]:
    passed: list[Path] = []
    failures: list[str] = []

    schema_data = _load_json(schema_path)
    validator = Draft202012Validator(schema_data)

    for path in result_files:
        try:
            data = _load_json(path)
        except json.JSONDecodeError as exc:
            failures.append(f"{path.relative_to(repo_root)}: invalid JSON ({exc})")
            continue

        errors = sorted(validator.iter_errors(data), key=lambda e: (list(e.absolute_path), e.message))
        if errors:
            for err in errors:
                failures.append(_format_error(err, path.relative_to(repo_root)))
            continue

        track_error = _validate_track_name(data, path, repo_root)
        if track_error:
            failures.append(track_error)
            continue

        version_error = _validate_schema_version(data, path, repo_root)
        if version_error:
            failures.append(version_error)
            continue

        passed.append(path)

    return passed, failures


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate breakthrough results.json files.")
    parser.add_argument(
        "--schema",
        default="research/breakthrough_lab/results.schema.json",
        help="Schema path relative to repository root.",
    )
    parser.add_argument(
        "--file",
        action="append",
        help="Specific results.json path relative to repository root. Repeatable.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = _repo_root()
    schema_path = (repo_root / args.schema).resolve()

    if not schema_path.exists():
        print(f"Schema not found: {schema_path}")
        return 2

    result_files = _iter_result_files(repo_root, args.file)
    if not result_files:
        print("No results.json files found.")
        return 2

    passed, failures = validate_files(
        repo_root=repo_root,
        schema_path=schema_path,
        result_files=result_files,
    )

    print(f"Schema: {schema_path.relative_to(repo_root)}")
    print(f"Validated: {len(result_files)} files")
    print(f"Passed: {len(passed)}")
    print(f"Failed: {len(failures)}")

    if failures:
        print("\nFailures:")
        for failure in failures:
            print(f"- {failure}")
        return 1

    print("\nAll breakthrough results contracts are valid.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
