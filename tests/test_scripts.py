from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def test_launch_scripts_are_valid_bash() -> None:
    vllm = (REPO / "scripts" / "launch_vllm_rocm.sh").read_text(encoding="utf-8")
    sglang = (REPO / "scripts" / "launch_sglang_rocm.sh").read_text(encoding="utf-8")
    assert "vllm serve" in vllm
    assert "bfloat16" in vllm
    assert "--tensor-parallel-size" in vllm
    assert "HIP_VISIBLE_DEVICES" in vllm
    assert "sglang.launch_server" in sglang
    assert "--context-length" in sglang
    assert "bfloat16" in sglang
    for script in (
        REPO / "scripts" / "launch_vllm_rocm.sh",
        REPO / "scripts" / "launch_sglang_rocm.sh",
    ):
        completed = subprocess.run(["bash", "-n", str(script)], check=False)
        assert completed.returncode == 0


def test_gpu_script_runs() -> None:
    completed = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "collect_gpu_info.py")],
        check=False,
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": str(REPO / "src")},
    )
    assert completed.returncode == 0
    assert "available" in completed.stdout
