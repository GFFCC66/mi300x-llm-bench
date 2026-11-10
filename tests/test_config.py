from __future__ import annotations

from pathlib import Path

import pytest

from mi300x_bench.config import load_sweep_config, specs_from_mapping


def test_sweep_expands_cartesian_product(tmp_path: Path) -> None:
    path = tmp_path / "sweep.yaml"
    path.write_text(
        """
base_url: http://127.0.0.1:9/v1
model: demo
concurrency: [1, 4]
input_len: [8, 32, 128]
output_len: 16
num_requests: 2
ignore_eos: false
""",
        encoding="utf-8",
    )
    config = load_sweep_config(path)
    assert len(config.specs) == 6
    assert [spec.input_len for spec in config.specs] == [8, 8, 32, 32, 128, 128]
    assert [spec.concurrency for spec in config.specs] == [1, 4, 1, 4, 1, 4]
    assert config.specs[0].output_len == 16
    assert config.specs[0].ignore_eos is False
    assert config.specs[0].api_key is None


def test_dataset_and_input_len_are_mutually_exclusive() -> None:
    with pytest.raises(ValueError, match="dataset or input_len"):
        specs_from_mapping(
            {
                "base_url": "http://127.0.0.1:9/v1",
                "model": "demo",
                "dataset": "prompts.jsonl",
                "input_len": [128],
            }
        )


def test_unknown_key_and_bad_numbers() -> None:
    with pytest.raises(ValueError, match="unknown config keys"):
        specs_from_mapping({"base_url": "http://x", "model": "m", "conc": 1})
    with pytest.raises(ValueError, match="concurrency"):
        specs_from_mapping({"base_url": "http://x", "model": "m", "concurrency": 0})
    with pytest.raises(ValueError, match="boolean"):
        specs_from_mapping({"base_url": "http://x", "model": "m", "num_requests": True})
    with pytest.raises(ValueError, match="base_url and model"):
        specs_from_mapping({})


def test_empty_api_key_is_dropped_and_checked_in_configs_load() -> None:
    config = specs_from_mapping(
        {"base_url": "http://127.0.0.1:9/v1/", "model": " m ", "api_key": "EMPTY"}
    )
    assert config.specs[0].api_key is None
    assert config.specs[0].base_url == "http://127.0.0.1:9/v1"
    assert config.specs[0].model == "m"

    repo = Path(__file__).resolve().parents[1]
    mock = load_sweep_config(repo / "configs" / "mock_sweep.yaml")
    assert len(mock.specs) == 4
    assert mock.specs[0].input_len == 64
    hardware = load_sweep_config(repo / "configs" / "mi300x_sweep.yaml")
    assert [spec.input_len for spec in hardware.specs] == [
        8192,
        8192,
        8192,
        32768,
        32768,
        32768,
        131072,
        131072,
        131072,
    ]
    assert {spec.concurrency for spec in hardware.specs} == {1, 4, 16}
    assert all(spec.output_len == 128 for spec in hardware.specs)
    assert hardware.specs[0].model == "meta-llama/Llama-3.1-70B-Instruct"
    assert hardware.specs[0].ignore_eos is True
