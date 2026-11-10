from __future__ import annotations

import json
from pathlib import Path
from urllib import request as urlrequest

import pytest

from mi300x_bench.cli import main
from mi300x_bench.config import RunSpec, specs_from_mapping
from mi300x_bench.mock_server import MockServer
from mi300x_bench.report import SUMMARY_COLUMNS, load_result_rows
from mi300x_bench.runner import run_spec


def _spec(base_url: str, **overrides: object) -> RunSpec:
    data: dict[str, object] = {
        "base_url": base_url,
        "model": "mock",
        "num_requests": 4,
        "concurrency": 2,
        "input_len": 12,
        "output_len": 5,
        "warmup": 1,
        "timeout_s": 10,
        "save_requests": True,
    }
    data.update(overrides)
    return specs_from_mapping(data).specs[0]


def test_runner_against_mock_measures_streamed_tokens() -> None:
    with MockServer(ttft_s=0.05, itl_s=0.01) as server:
        result = run_spec(_spec(server.base_url))
    summary = result["summary"]
    meta = result["meta"]
    assert isinstance(summary, dict)
    assert isinstance(meta, dict)
    assert summary["completed"] == 4
    assert summary["failed"] == 0
    assert summary["completion_tokens_total"] == 20
    assert summary["prompt_tokens_mean"] == pytest.approx(12)
    assert summary["output_tokens_per_s"] > 0
    ttft = summary["ttft_ms"]
    itl = summary["itl_ms"]
    e2e = summary["e2e_ms"]
    assert isinstance(ttft, dict) and isinstance(itl, dict) and isinstance(e2e, dict)
    assert ttft["p50"] >= 40
    assert itl["p50"] >= 5
    assert e2e["p50"] <= e2e["p90"] <= e2e["p99"]
    requests = result["requests"]
    assert isinstance(requests, list)
    assert len(requests) == 4
    assert all(item["chunk_count"] == 5 and item["streamed"] is True for item in requests)
    assert all(len(item["itl_s"]) == 4 for item in requests)
    assert "closed-loop" in meta["methodology"]


def test_failed_server_is_counted() -> None:
    with MockServer(fail_status=500) as server:
        result = run_spec(_spec(server.base_url, warmup=0, num_requests=3, concurrency=3))
    summary = result["summary"]
    assert isinstance(summary, dict)
    assert summary["completed"] == 0
    assert summary["failed"] == 3
    assert not summary["output_tokens_per_s"]
    assert summary["errors"]


def test_mock_non_streaming_usage() -> None:
    with MockServer() as server:
        payload = {
            "model": "mock",
            "messages": [{"role": "user", "content": "a b"}],
            "max_tokens": 3,
            "stream": False,
        }
        req = urlrequest.Request(
            server.base_url + "/chat/completions",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urlrequest.urlopen(req, timeout=5) as response:
            body = json.load(response)
    assert body["usage"]["prompt_tokens"] == 2
    assert body["usage"]["completion_tokens"] == 3
    assert "t0 t1 t2" == body["choices"][0]["message"]["content"]


def test_cli_run_writes_json_and_csv(tmp_path: Path) -> None:
    with MockServer() as server:
        output = tmp_path / "run.json"
        code = main(
            [
                "run",
                "--base-url",
                server.base_url,
                "--model",
                "mock",
                "--concurrency",
                "4",
                "--input-len",
                "8",
                "--output-len",
                "4",
                "--num-requests",
                "4",
                "--output",
                str(output),
            ]
        )
    assert code == 0
    saved = json.loads(output.read_text(encoding="utf-8"))
    assert saved["summary"]["completed"] == 4
    rows = load_result_rows(output.with_suffix(".csv"))
    assert rows[0]["completed"] == 4
    assert rows[0]["target_input_len"] == 8
    assert isinstance(rows[0]["requests_per_s"], float)
    assert isinstance(rows[0]["ttft_ms_p50"], float)
    assert set(SUMMARY_COLUMNS).issuperset({"output_tokens_per_s", "ttft_ms_p50", "e2e_ms_p99"})


def test_cli_sweep_and_unreachable_server(tmp_path: Path) -> None:
    with MockServer() as server:
        config = tmp_path / "sweep.yaml"
        out = tmp_path / "out"
        config.write_text(
            f"""
base_url: {server.base_url}
model: mock
num_requests: 2
warmup: 0
concurrency: [1, 2]
input_len: [4]
output_len: [3]
timeout_s: 10
output_dir: {out}
""",
            encoding="utf-8",
        )
        assert main(["sweep", "--config", str(config)]) == 0
    rows = load_result_rows(out / "sweep.json")
    assert len(rows) == 2
    assert (out / "sweep.csv").is_file()
    assert list((out / "points").glob("*.json"))

    code = main(
        [
            "run",
            "--base-url",
            "http://127.0.0.1:1/v1",
            "--model",
            "missing",
            "--num-requests",
            "1",
            "--timeout",
            "1",
            "--input-len",
            "4",
            "--output-len",
            "2",
        ]
    )
    assert code == 2


def test_warmup_failure_aborts() -> None:
    with MockServer(fail_status=503) as server:
        with pytest.raises(ConnectionError, match="warmup"):
            run_spec(_spec(server.base_url, warmup=1, num_requests=1, concurrency=1))


def test_help_and_gpu_info_exit_zero() -> None:
    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0
    assert main(["gpu-info"]) == 0
    assert main(["gpu-info", "--json"]) == 0
