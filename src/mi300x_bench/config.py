"""YAML sweep config and the resolved spec for a single measured point."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

_KNOWN_KEYS = {
    "base_url",
    "model",
    "api_key",
    "num_requests",
    "warmup",
    "concurrency",
    "input_len",
    "output_len",
    "temperature",
    "ignore_eos",
    "seed",
    "timeout_s",
    "dataset",
    "output_dir",
    "save_requests",
    "skip_health_check",
}


@dataclass(frozen=True)
class RunSpec:
    base_url: str
    model: str
    api_key: str | None
    num_requests: int
    warmup: int
    concurrency: int
    input_len: int | None
    output_len: int
    temperature: float
    ignore_eos: bool
    seed: int
    timeout_s: float
    dataset: str | None
    save_requests: bool
    skip_health_check: bool

    @property
    def prompt_source(self) -> str:
        return "dataset" if self.dataset else "synthetic"


@dataclass(frozen=True)
class SweepConfig:
    output_dir: Path
    specs: tuple[RunSpec, ...]
    source: Path | None = None


def _reject_bool(value: object, name: str) -> None:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a number, not a boolean")


def _positive_int(value: object, name: str) -> int:
    _reject_bool(value, name)
    if not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _nonnegative_int(value: object, name: str) -> int:
    _reject_bool(value, name)
    if not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be an integer >= 0")
    return value


def _nonnegative_float(value: object, name: str) -> float:
    _reject_bool(value, name)
    if not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number >= 0")
    if value < 0:
        raise ValueError(f"{name} must be a number >= 0")
    return float(value)


def _positive_float(value: object, name: str) -> float:
    _reject_bool(value, name)
    if not isinstance(value, (int, float)) or value <= 0:
        raise ValueError(f"{name} must be a number > 0")
    return float(value)


def _as_list(value: object, name: str) -> list[object]:
    if isinstance(value, list):
        if not value:
            raise ValueError(f"{name} must not be empty")
        return list(value)
    return [value]


def _api_key(value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError("api_key must be a string or null")
    stripped = value.strip()
    if not stripped or stripped.upper() == "EMPTY":
        return None
    return stripped


def _optional_path(value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError("dataset must be a file path or null")
    return value.strip()


def specs_from_mapping(data: dict[str, object], *, source: Path | None = None) -> SweepConfig:
    unknown = sorted(set(data) - _KNOWN_KEYS)
    if unknown:
        raise ValueError(f"unknown config keys: {', '.join(unknown)}")
    if "base_url" not in data or "model" not in data:
        raise ValueError("base_url and model are required")
    base_url = data["base_url"]
    model = data["model"]
    if not isinstance(base_url, str) or not base_url.strip():
        raise ValueError("base_url must be a non-empty string")
    if not isinstance(model, str) or not model.strip():
        raise ValueError("model must be a non-empty string")

    dataset = _optional_path(data.get("dataset"))
    if dataset is not None and "input_len" in data:
        raise ValueError(
            "set dataset or input_len, not both; a dataset is not re-tokenized to input_len"
        )

    input_values: list[int | None]
    if dataset is not None:
        input_values = [None]
    else:
        raw_input_lens = _as_list(data.get("input_len", 256), "input_len")
        input_values = [_positive_int(item, "input_len") for item in raw_input_lens]

    raw_output_lens = _as_list(data.get("output_len", 64), "output_len")
    output_lens = [_positive_int(item, "output_len") for item in raw_output_lens]
    concurrencies = [
        _positive_int(item, "concurrency")
        for item in _as_list(data.get("concurrency", 1), "concurrency")
    ]

    output_dir_value = data.get("output_dir", "results")
    if not isinstance(output_dir_value, str) or not output_dir_value.strip():
        raise ValueError("output_dir must be a non-empty string")

    ignore_eos = data.get("ignore_eos", False)
    save_requests = data.get("save_requests", False)
    skip_health_check = data.get("skip_health_check", False)
    for name, flag in (
        ("ignore_eos", ignore_eos),
        ("save_requests", save_requests),
        ("skip_health_check", skip_health_check),
    ):
        if not isinstance(flag, bool):
            raise ValueError(f"{name} must be true or false")

    common = dict(
        base_url=base_url.strip().rstrip("/"),
        model=model.strip(),
        api_key=_api_key(data.get("api_key")),
        num_requests=_positive_int(data.get("num_requests", 8), "num_requests"),
        warmup=_nonnegative_int(data.get("warmup", 0), "warmup"),
        temperature=_nonnegative_float(data.get("temperature", 0), "temperature"),
        ignore_eos=ignore_eos,
        seed=_nonnegative_int(data.get("seed", 0), "seed"),
        timeout_s=_positive_float(data.get("timeout_s", 600), "timeout_s"),
        dataset=dataset,
        save_requests=save_requests,
        skip_health_check=skip_health_check,
    )
    specs: list[RunSpec] = []
    for input_len in input_values:
        for output_len in output_lens:
            for concurrency in concurrencies:
                specs.append(
                    RunSpec(
                        input_len=input_len,
                        output_len=output_len,
                        concurrency=concurrency,
                        **common,
                    )
                )
    return SweepConfig(output_dir=Path(output_dir_value), specs=tuple(specs), source=source)


def load_sweep_config(path: Path) -> SweepConfig:
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError("config file must be a YAML mapping")
    return specs_from_mapping(loaded, source=path)
