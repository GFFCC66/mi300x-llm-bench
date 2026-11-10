"""Collect GPU identity and VRAM via ``rocm-smi``.

The collector is safe to run on a machine without ROCm: a missing binary
returns ``available: false`` and does not raise. Parsing is best-effort.
Raw command output is kept so an unfamiliar ``rocm-smi`` layout can still
be read by hand.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from collections.abc import Callable
from typing import Protocol

_PRODUCT_SERIES = re.compile(r"GPU\[(\d+)\].*Card series:\s*(.+)", re.IGNORECASE)
_VRAM_BYTES = re.compile(
    r"GPU\[(\d+)\].*VRAM Total(?: Used)? Memory \(B\):\s*(\d+)",
    re.IGNORECASE,
)
_VRAM_USED = re.compile(r"VRAM Total Used Memory \(B\)", re.IGNORECASE)
_VRAM_MIB = re.compile(
    r"GPU\[(\d+)\].*VRAM Total Memory \(MiB\):\s*([0-9.]+)",
    re.IGNORECASE,
)


def parse_rocm_smi(product_text: str, mem_text: str) -> list[dict[str, object]]:
    gpus: dict[int, dict[str, object]] = {}
    for line in product_text.splitlines():
        match = _PRODUCT_SERIES.search(line)
        if match:
            gpus.setdefault(int(match.group(1)), {})["name"] = match.group(2).strip()
    for line in mem_text.splitlines():
        match = _VRAM_BYTES.search(line)
        if match:
            index = int(match.group(1))
            entry = gpus.setdefault(index, {})
            amount = int(match.group(2))
            if _VRAM_USED.search(line):
                entry["vram_used_bytes"] = amount
            else:
                entry["vram_total_bytes"] = amount
            continue
        match = _VRAM_MIB.search(line)
        if match:
            gpus.setdefault(int(match.group(1)), {})["vram_total_bytes"] = int(
                float(match.group(2)) * 1024 * 1024
            )
    rows: list[dict[str, object]] = []
    for index in sorted(gpus):
        row: dict[str, object] = {"index": index}
        row.update(gpus[index])
        rows.append(row)
    return rows


def _unavailable(message: str) -> dict[str, object]:
    return {
        "available": False,
        "tool": None,
        "message": message,
        "gpus": [],
        "commands": [],
    }


class _Command(Protocol):
    returncode: int
    stdout: str
    stderr: str


def _default_run(argv: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(argv, capture_output=True, text=True, check=False)


def collect_gpu_info(
    runner: Callable[[list[str]], _Command] | None = None,
    which: Callable[[str], str | None] | None = None,
) -> dict[str, object]:
    """Run ``rocm-smi`` if it is on ``PATH``. Never raises ``FileNotFoundError``."""
    finder = which or shutil.which
    if finder("rocm-smi") is None:
        return _unavailable(
            "rocm-smi not found on PATH. GPU info was not collected. "
            "Newer ROCm releases also ship amd-smi; this collector calls rocm-smi only."
        )
    run = runner or _default_run
    argv_list = (
        ["rocm-smi", "--showproductname"],
        ["rocm-smi", "--showmeminfo", "vram"],
    )
    commands: list[dict[str, object]] = []
    outputs: list[str] = []
    try:
        for argv in argv_list:
            completed = run(argv)
            commands.append(
                {
                    "argv": argv,
                    "returncode": completed.returncode,
                    "stdout": completed.stdout,
                    "stderr": completed.stderr,
                }
            )
            outputs.append(completed.stdout or "")
    except FileNotFoundError:
        return _unavailable(
            "rocm-smi was listed on PATH but could not be executed. GPU info was not collected."
        )
    product_text, mem_text = outputs
    return {
        "available": True,
        "tool": "rocm-smi",
        "message": "collected",
        "gpus": parse_rocm_smi(product_text, mem_text),
        "commands": commands,
    }


def format_gpu_info(info: dict[str, object]) -> str:
    if not info.get("available"):
        return str(info.get("message", "rocm-smi unavailable"))
    gpus = info.get("gpus")
    if not isinstance(gpus, list) or not gpus:
        return "rocm-smi ran, but no GPU rows were parsed. Re-run with --json for the raw output."
    lines = []
    for gpu in gpus:
        if not isinstance(gpu, dict):
            continue
        name = gpu.get("name", "unknown")
        total = gpu.get("vram_total_bytes")
        used = gpu.get("vram_used_bytes")
        total_text = _gib(total)
        used_text = _gib(used)
        lines.append(f"GPU {gpu.get('index')}: {name}  VRAM {used_text} used / {total_text} total")
    return "\n".join(lines)


def _gib(value: object) -> str:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return "n/a"
    return f"{value / (1024**3):.2f} GiB"
