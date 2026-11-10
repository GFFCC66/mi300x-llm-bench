from __future__ import annotations

import subprocess

from mi300x_bench.gpu_info import collect_gpu_info, format_gpu_info, parse_rocm_smi


class _Proc:
    def __init__(self, stdout: str, returncode: int = 0, stderr: str = "") -> None:
        self.stdout = stdout
        self.returncode = returncode
        self.stderr = stderr


_PRODUCT = """
====================================== Product Info ======================================
GPU[0]		: Card series: 		AMD INSTINCT MI300X
GPU[1]		: Card series: 		AMD INSTINCT MI300X
================================== End of ROCm SMI Log ===================================
"""

_MEM = """
GPU[0]		: VRAM Total Memory (B): 206158430208
GPU[0]		: VRAM Total Used Memory (B): 1073741824
GPU[1]		: VRAM Total Memory (MiB): 196608
"""


def test_parse_product_and_vram() -> None:
    gpus = parse_rocm_smi(_PRODUCT, _MEM)
    assert gpus[0]["index"] == 0
    assert gpus[0]["name"] == "AMD INSTINCT MI300X"
    assert gpus[0]["vram_total_bytes"] == 206158430208
    assert gpus[0]["vram_used_bytes"] == 1073741824
    assert gpus[1]["index"] == 1
    assert gpus[1]["vram_total_bytes"] == 196608 * 1024 * 1024


def test_missing_rocm_smi_is_not_an_error() -> None:
    info = collect_gpu_info(which=lambda _name: None)
    assert info["available"] is False
    assert info["gpus"] == []
    assert "rocm-smi" in str(info["message"])
    assert "amd-smi" in str(info["message"])
    text = format_gpu_info(info)
    assert "rocm-smi" in text


def test_collect_uses_runner_output() -> None:
    calls: list[list[str]] = []

    def runner(argv: list[str]) -> _Proc:
        calls.append(argv)
        if "--showproductname" in argv:
            return _Proc(_PRODUCT)
        return _Proc(_MEM)

    info = collect_gpu_info(runner=runner, which=lambda _name: "/usr/bin/rocm-smi")
    assert info["available"] is True
    assert info["tool"] == "rocm-smi"
    assert calls == [
        ["rocm-smi", "--showproductname"],
        ["rocm-smi", "--showmeminfo", "vram"],
    ]
    gpus = info["gpus"]
    assert isinstance(gpus, list)
    assert gpus[0]["name"] == "AMD INSTINCT MI300X"
    formatted = format_gpu_info(info)
    assert "GPU 0" in formatted
    assert "GiB" in formatted


def test_live_collector_does_not_raise() -> None:
    info = collect_gpu_info()
    assert isinstance(info["available"], bool)
    assert isinstance(info["gpus"], list)
    if not info["available"]:
        assert "rocm-smi" in str(info["message"])
    which = subprocess.run(["bash", "-lc", "command -v rocm-smi"], capture_output=True, text=True)
    if which.returncode != 0:
        assert info["available"] is False
