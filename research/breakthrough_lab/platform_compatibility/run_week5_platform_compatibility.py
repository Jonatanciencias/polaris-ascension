#!/usr/bin/env python3
"""Week 5 Block 4 runner: Rusticl/ROCm platform compatibility feasibility."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_DIR_DEFAULT = REPO_ROOT / "research" / "breakthrough_lab" / "platform_compatibility"


INVENTORY_SNIPPET = r"""
import json
import pyopencl as cl

rows = []
for i, platform in enumerate(cl.get_platforms()):
    try:
        devices = platform.get_devices(device_type=cl.device_type.GPU)
    except Exception as exc:
        rows.append({
            "platform_index": i,
            "platform_name": platform.name,
            "vendor": platform.vendor,
            "version": platform.version,
            "error": str(exc),
        })
        continue

    rows.append(
        {
            "platform_index": i,
            "platform_name": platform.name,
            "vendor": platform.vendor,
            "version": platform.version,
            "gpu_device_count": len(devices),
            "devices": [
                {
                    "name": dev.name,
                    "vendor": dev.vendor,
                    "version": dev.version,
                    "driver": dev.driver_version,
                    "global_mem_gb": round(dev.global_mem_size / (1024 ** 3), 3),
                    "max_work_group_size": int(dev.max_work_group_size),
                }
                for dev in devices
            ],
        }
    )

print(json.dumps({"platforms": rows}))
"""


MICROBENCH_SNIPPET = r"""
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pyopencl as cl

warnings.filterwarnings("ignore", message=".*PyOpenCL compiler caching failed.*")

size = 1024
iterations = 5
warmup = 2
rng = np.random.default_rng(123)
kernel_source = Path("src/kernels/gemm_tile24_production.cl").read_text()
kernel_name = "gemm_tile24_vectorized"

results = []

for platform in cl.get_platforms():
    try:
        gpu_devices = platform.get_devices(device_type=cl.device_type.GPU)
    except Exception as exc:
        results.append(
            {
                "platform": platform.name,
                "status": "error",
                "error": f"get_devices_failed: {exc}",
            }
        )
        continue

    amd_devices = [
        dev
        for dev in gpu_devices
        if "amd" in (dev.vendor or "").lower() or "radeon" in (dev.name or "").lower()
    ]
    if not amd_devices:
        results.append({"platform": platform.name, "status": "skip_no_amd_gpu"})
        continue

    dev = amd_devices[0]
    try:
        ctx = cl.Context([dev])
        queue = cl.CommandQueue(ctx)
        program = cl.Program(ctx, kernel_source).build(options=["-cl-fast-relaxed-math"])
        kernel = getattr(program, kernel_name)

        a = rng.standard_normal((size, size), dtype=np.float32)
        b = rng.standard_normal((size, size), dtype=np.float32)
        c = np.zeros((size, size), dtype=np.float32)

        mf = cl.mem_flags
        a_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a)
        b_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=b)
        c_buf = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=c)

        local_size = (12, 12)
        tile = 24
        global_size = (
            ((size + tile - 1) // tile) * local_size[0],
            ((size + tile - 1) // tile) * local_size[1],
        )

        kernel.set_args(
            np.int32(size),
            np.int32(size),
            np.int32(size),
            np.float32(1.0),
            a_buf,
            b_buf,
            np.float32(0.0),
            c_buf,
        )

        for _ in range(warmup):
            cl.enqueue_nd_range_kernel(queue, kernel, global_size, local_size)
            queue.finish()

        elapsed = []
        for _ in range(iterations):
            t0 = time.perf_counter()
            cl.enqueue_nd_range_kernel(queue, kernel, global_size, local_size)
            queue.finish()
            elapsed.append(time.perf_counter() - t0)

        out = np.empty_like(c)
        cl.enqueue_copy(queue, out, c_buf).wait()

        flops = 2 * size * size * size
        peak_gflops = float(flops / min(elapsed) / 1e9)
        avg_gflops = float(flops / (sum(elapsed) / len(elapsed)) / 1e9)
        max_error = float(np.max(np.abs(out - (a @ b))))

        results.append(
            {
                "platform": platform.name,
                "platform_version": platform.version,
                "device": dev.name,
                "driver": dev.driver_version,
                "status": "ok",
                "size": size,
                "peak_gflops": peak_gflops,
                "avg_gflops": avg_gflops,
                "max_error": max_error,
            }
        )
    except Exception as exc:
        results.append(
            {
                "platform": platform.name,
                "device": dev.name,
                "status": "error",
                "error": str(exc),
            }
        )

print(json.dumps({"microbench": results}))
"""


def _run_command(
    command: list[str],
    *,
    env_patch: dict[str, str] | None = None,
    timeout: int = 120,
) -> dict[str, Any]:
    env = os.environ.copy()
    if env_patch:
        env.update(env_patch)
    try:
        proc = subprocess.run(
            command,
            cwd=str(REPO_ROOT),
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return {
            "command": command,
            "returncode": int(proc.returncode),
            "stdout": proc.stdout,
            "stderr": proc.stderr,
        }
    except FileNotFoundError as exc:
        return {
            "command": command,
            "returncode": 127,
            "stdout": "",
            "stderr": str(exc),
        }


def _run_python_snippet(
    snippet: str,
    *,
    env_patch: dict[str, str] | None = None,
    timeout: int = 120,
) -> dict[str, Any]:
    result = _run_command(
        [sys.executable, "-c", snippet],
        env_patch=env_patch,
        timeout=timeout,
    )
    parsed: Any | None = None
    if result["stdout"].strip():
        try:
            parsed = json.loads(result["stdout"])
        except json.JSONDecodeError:
            parsed = None
    result["json"] = parsed
    return result


def _platform_has_gpu(platform_payload: dict[str, Any]) -> bool:
    return int(platform_payload.get("gpu_device_count", 0)) > 0


def _find_platform(
    inventory: dict[str, Any],
    name: str,
) -> dict[str, Any] | None:
    for entry in inventory.get("platforms", []):
        if str(entry.get("platform_name", "")).lower() == name.lower():
            return entry
    return None


def _microbench_map(microbench: dict[str, Any]) -> dict[str, dict[str, Any]]:
    mapping: dict[str, dict[str, Any]] = {}
    for row in microbench.get("microbench", []):
        platform = str(row.get("platform", "")).lower()
        if platform:
            mapping[platform] = row
    return mapping


def _evaluate(report: dict[str, Any]) -> dict[str, Any]:
    base_inventory = report["inventory_default"]["json"] or {"platforms": []}
    rusticl_inventory = report["inventory_rusticl_enabled"]["json"] or {"platforms": []}
    base_microbench = report["microbench_default"]["json"] or {"microbench": []}
    rusticl_microbench = report["microbench_rusticl_enabled"]["json"] or {"microbench": []}

    base_clover = _find_platform(base_inventory, "Clover")
    base_rusticl = _find_platform(base_inventory, "rusticl")
    rusticl_enabled = _find_platform(rusticl_inventory, "rusticl")

    clover_present = bool(base_clover and _platform_has_gpu(base_clover))
    rusticl_default_visible = bool(base_rusticl and _platform_has_gpu(base_rusticl))
    rusticl_activatable = bool(rusticl_enabled and _platform_has_gpu(rusticl_enabled))

    default_map = _microbench_map(base_microbench)
    enabled_map = _microbench_map(rusticl_microbench)
    clover_perf = enabled_map.get("clover", default_map.get("clover", {}))
    rusticl_perf = enabled_map.get("rusticl", {})

    clover_avg = float(clover_perf.get("avg_gflops", 0.0) or 0.0)
    rusticl_avg = float(rusticl_perf.get("avg_gflops", 0.0) or 0.0)
    rusticl_vs_clover_ratio = float(rusticl_avg / clover_avg) if clover_avg > 0 else 0.0

    rocminfo_cmd = report["rocminfo"]
    rocmsmi_cmd = report["rocm_smi"]
    rocm_tools_available = bool(
        rocminfo_cmd["returncode"] == 0 or rocmsmi_cmd["returncode"] == 0
    )

    production_platform_lock = "cl.get_platforms()[0]" in (
        REPO_ROOT / "src/benchmarking/production_kernel_benchmark.py"
    ).read_text()

    checks = {
        "clover_default_gpu_available": {
            "observed": clover_present,
            "required": True,
            "pass": clover_present,
        },
        "rusticl_gpu_visible_without_env": {
            "observed": rusticl_default_visible,
            "required": True,
            "pass": rusticl_default_visible,
        },
        "rusticl_gpu_activatable_with_env": {
            "observed": rusticl_activatable,
            "required": True,
            "pass": rusticl_activatable,
        },
        "rusticl_perf_ratio_vs_clover": {
            "observed": rusticl_vs_clover_ratio,
            "required_min": 0.9,
            "pass": rusticl_vs_clover_ratio >= 0.9,
        },
        "rocm_tools_present": {
            "observed": rocm_tools_available,
            "required": False,
            "pass": True,
        },
        "production_platform_selection_explicit": {
            "observed": not production_platform_lock,
            "required": True,
            "pass": not production_platform_lock,
        },
    }

    if not clover_present:
        decision = "stop"
        rationale = "No stable OpenCL production platform detected (Clover missing)."
    elif not rusticl_activatable:
        decision = "iterate"
        rationale = "Rusticl not activatable in current host; keep Clover only and re-evaluate platform stack."
    elif rusticl_vs_clover_ratio < 0.9:
        decision = "iterate"
        rationale = "Rusticl can run but performance gap vs Clover is above threshold."
    elif production_platform_lock:
        decision = "refine"
        rationale = (
            "Rusticl is technically viable in shadow mode, but production path is pinned to platform index 0; "
            "selector hardening is required before any canary promotion."
        )
    else:
        decision = "promote"
        rationale = "Rusticl/ROCm compatibility gate passed for controlled canary progression."

    return {
        "checks": checks,
        "decision": decision,
        "rationale": rationale,
        "summary": {
            "clover_default_gpu_available": clover_present,
            "rusticl_gpu_visible_without_env": rusticl_default_visible,
            "rusticl_gpu_activatable_with_env": rusticl_activatable,
            "rusticl_avg_gflops": rusticl_avg,
            "clover_avg_gflops": clover_avg,
            "rusticl_vs_clover_ratio": rusticl_vs_clover_ratio,
            "rocm_tools_available": rocm_tools_available,
            "production_platform_selection_hardcoded": production_platform_lock,
        },
    }


def _markdown(report: dict[str, Any]) -> str:
    evaluation = report["evaluation"]
    lines: list[str] = []
    lines.append("# Week 5 Block 4 - Rusticl/ROCm Compatibility Report")
    lines.append("")
    lines.append(f"- Date: {report['metadata']['timestamp_utc']}")
    lines.append(f"- Host: `{report['metadata']['hostname']}`")
    lines.append(f"- Python: `{report['metadata']['python']}`")
    lines.append("")
    lines.append("## Compatibility Summary")
    lines.append("")
    summary = evaluation["summary"]
    lines.append(f"- Clover GPU available (default): {summary['clover_default_gpu_available']}")
    lines.append(
        f"- Rusticl GPU visible (default env): {summary['rusticl_gpu_visible_without_env']}"
    )
    lines.append(
        f"- Rusticl GPU activatable (`RUSTICL_ENABLE=radeonsi`): {summary['rusticl_gpu_activatable_with_env']}"
    )
    lines.append(f"- Clover avg GFLOPS (microbench): {summary['clover_avg_gflops']:.3f}")
    lines.append(f"- Rusticl avg GFLOPS (microbench): {summary['rusticl_avg_gflops']:.3f}")
    lines.append(f"- Rusticl/Clover ratio: {summary['rusticl_vs_clover_ratio']:.3f}")
    lines.append(f"- ROCm tools present: {summary['rocm_tools_available']}")
    lines.append(
        f"- Production platform selection hardcoded index-0: {summary['production_platform_selection_hardcoded']}"
    )
    lines.append("")
    lines.append("## Guardrail Checks")
    lines.append("")
    lines.append("| Check | Observed | Requirement | Pass |")
    lines.append("| --- | --- | --- | --- |")
    for name, payload in evaluation["checks"].items():
        observed = payload["observed"]
        if "required_min" in payload:
            requirement = f">= {payload['required_min']}"
        elif "required" in payload:
            requirement = str(payload["required"])
        else:
            requirement = "n/a"
        lines.append(f"| {name} | {observed} | {requirement} | {payload['pass']} |")
    lines.append("")
    lines.append("## Formal Decision")
    lines.append("")
    lines.append(f"- Decision: `{evaluation['decision']}`")
    lines.append(f"- Rationale: {evaluation['rationale']}")
    lines.append("")
    lines.append("## Raw Command Exit Codes")
    lines.append("")
    lines.append(f"- `verify_hardware.py`: {report['verify_hardware']['returncode']}")
    lines.append(f"- `verify_drivers.py --json`: {report['verify_drivers']['returncode']}")
    lines.append(f"- `clinfo --list`: {report['clinfo']['returncode']}")
    lines.append(f"- `rocminfo`: {report['rocminfo']['returncode']}")
    lines.append(f"- `rocm-smi`: {report['rocm_smi']['returncode']}")
    lines.append("")
    return "\n".join(lines) + "\n"


def run() -> dict[str, Any]:
    report = {
        "metadata": {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "hostname": os.uname().nodename,
            "python": sys.version.split()[0],
        },
        "verify_hardware": _run_command([sys.executable, "scripts/verify_hardware.py"]),
        "verify_drivers": _run_command([sys.executable, "scripts/verify_drivers.py", "--json"]),
        "clinfo": _run_command(["clinfo", "--list"]),
        "rocminfo": _run_command(["rocminfo"]),
        "rocm_smi": _run_command(["rocm-smi", "--showproductname"]),
        "inventory_default": _run_python_snippet(INVENTORY_SNIPPET),
        "inventory_rusticl_enabled": _run_python_snippet(
            INVENTORY_SNIPPET, env_patch={"RUSTICL_ENABLE": "radeonsi"}
        ),
        "microbench_default": _run_python_snippet(MICROBENCH_SNIPPET),
        "microbench_rusticl_enabled": _run_python_snippet(
            MICROBENCH_SNIPPET, env_patch={"RUSTICL_ENABLE": "radeonsi"}
        ),
    }
    report["evaluation"] = _evaluate(report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run Week 5 Block 4 compatibility feasibility checks."
    )
    parser.add_argument(
        "--output-dir",
        default=str(OUTPUT_DIR_DEFAULT.relative_to(REPO_ROOT)),
        help="Output directory relative to repository root.",
    )
    args = parser.parse_args()

    output_dir = (REPO_ROOT / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    report = run()
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    json_path = output_dir / f"week5_platform_compatibility_{ts}.json"
    md_path = output_dir / f"week5_platform_compatibility_{ts}.md"
    json_path.write_text(json.dumps(report, indent=2))
    md_path.write_text(_markdown(report))

    print(f"Platform compatibility JSON: {json_path}")
    print(f"Platform compatibility MD:   {md_path}")
    print(f"Decision: {report['evaluation']['decision']}")
    print(f"Rationale: {report['evaluation']['rationale']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
