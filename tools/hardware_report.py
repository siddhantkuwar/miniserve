"""Collect the Phase 0 machine and environment baseline.

This intentionally uses only the Python standard library. The report must be
available before MLX or a model has been installed.
"""

from __future__ import annotations

import importlib.metadata
import json
import platform
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results" / "hardware.json"


def command(*args: str) -> str | None:
    try:
        result = subprocess.run(args, capture_output=True, text=True, check=False)
    except OSError:
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None


def sysctl(name: str) -> str | None:
    return command("sysctl", "-n", name)


def numeric(value: str | None) -> int | float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except ValueError:
        return None
    return int(number) if number.is_integer() else number


def memory_gb() -> float | None:
    bytes_value = numeric(sysctl("hw.memsize"))
    return round(bytes_value / (1024**3), 2) if bytes_value else None


def hardware_field(name: str) -> str | None:
    output = command("system_profiler", "SPHardwareDataType")
    if not output:
        return None
    match = re.search(rf"^\s+{re.escape(name)}:\s+(.+)$", output, re.MULTILINE)
    return match.group(1).strip() if match else None


def memory_from_system_profiler() -> float | None:
    value = hardware_field("Memory")
    if not value:
        return None
    match = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*GB", value, re.IGNORECASE)
    return float(match.group(1)) if match else None


def swap_usage() -> dict[str, int | None]:
    output = command("sysctl", "vm.swapusage") or ""
    usage: dict[str, int | None] = {}
    for label, value, unit in re.findall(r"(total|used|free)\s*=\s*([0-9.]+)\s*([MG])B?", output):
        multiplier = 1024**3 if unit == "G" else 1024**2
        usage[f"{label}_bytes"] = int(float(value) * multiplier)
    return {key: usage.get(key) for key in ("total_bytes", "used_bytes", "free_bytes")}


def power_status() -> str:
    output = command("pmset", "-g", "batt") or ""
    if "AC Power" in output:
        return "plugged_in"
    if "Battery Power" in output:
        return "battery"
    return "unknown"


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def available_disk_gb() -> float | None:
    return round(shutil.disk_usage(ROOT).free / (1024**3), 2)


def copy_bandwidth_gb_s() -> float | None:
    """Measure a rough host-memory copy rate without requiring NumPy.

    This is a Python-level baseline, not a GPU roofline measurement. It is
    labeled accordingly so it cannot be mistaken for MLX/Metal bandwidth.
    """
    size = 64 * 1024 * 1024
    source = bytearray(size)
    target = bytearray(size)
    start = time.perf_counter()
    target[:] = source
    elapsed = time.perf_counter() - start
    if elapsed <= 0:
        return None
    return round(size / elapsed / (1024**3), 3)


def build_report() -> dict[str, Any]:
    os_version = platform.mac_ver()[0] or platform.platform()
    chip = sysctl("machdep.cpu.brand_string") or hardware_field("Chip")
    report: dict[str, Any] = {
        "report_version": 1,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "machine": {
            "chip": chip,
            "unified_memory_gb": memory_gb() or memory_from_system_profiler(),
            "macos": os_version,
            "python": platform.python_version(),
            "python_arch": platform.machine(),
            "processor": platform.processor(),
        },
        "storage": {"available_disk_gb": available_disk_gb()},
        "power": {"status": power_status()},
        "memory": {
            **swap_usage(),
            "note": "Use macOS memory_pressure for detailed pressure during benchmarks.",
        },
        "software": {
            "uv": command("uv", "--version"),
            "mlx": package_version("mlx"),
            "mlx_lm": package_version("mlx-lm"),
        },
        "bandwidth": {
            "python_host_copy_gb_s": copy_bandwidth_gb_s(),
            "mlx_device_bandwidth_gb_s": None,
            "note": "Device bandwidth will be measured after MLX is installed.",
        },
    }
    return report


def main() -> None:
    report = build_report()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    print(f"\nWrote {OUTPUT}", file=sys.stderr)


if __name__ == "__main__":
    main()
