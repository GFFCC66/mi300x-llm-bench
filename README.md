# mi300x-bench

Closed-loop benchmark for OpenAI-compatible LLM servers. It is aimed at a single **AMD Instinct MI300X** (192 GB HBM3, ROCm) serving a 70B-class model with vLLM, or optionally SGLang, and it also runs against any other compatible endpoint — including the mock server in this repo, so the client can be tested on a machine with no GPU.

The client sends streaming `POST /v1/chat/completions` requests, times the chunks, and reports:

- output, total, and request throughput
- time to first token (TTFT)
- inter-chunk latency (ITL)
- time per output token (TPOT)
- end-to-end latency (E2E)
- p50 / p90 / p99 for each of those latencies

A YAML file sweeps concurrency and context length (the checked-in MI300X example uses 8k / 32k / 128k input words). Results are JSON and CSV. `plot` draws them with matplotlib.

No MI300X measurements are included. The results table below is empty on purpose.

## Motivation

A 70B model in bf16 is about 140 GB of weights (`70e9` parameters × 2 bytes). That fits in one MI300X's 192 GB, so a 70B model can be served at tensor-parallel size 1 instead of being sharded across several smaller GPUs. Sharding adds all-reduce traffic on every layer. Single-GPU serving avoids that traffic, and the HBM that remains after the weights is the budget for KV cache.

That budget is the reason to measure long context explicitly. For Llama 3.1 70B (80 layers, 8 KV heads, head dim 128) a bf16 KV cache is about 320 KiB per token:

```text
2 (K and V) * 80 layers * 8 heads * 128 dims * 2 bytes = 327680 bytes
```

One 128k sequence is on the order of 40 GiB of KV cache before allocator overhead. One such sequence can be plausible next to a 140 GB bf16 weight load; several concurrent 128k sequences are a different question. Those figures are a capacity sketch from the public model shape, not a number measured by this repo. The sweep exists so you can see, on hardware, where latency moves and where the server runs out of cache.

vLLM and SGLang both expose an OpenAI-compatible server. This toolkit speaks only that protocol, so the same command compares the two stacks, or a multi-GPU deployment, without importing either framework.

## Install

Python 3.10 or newer.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

`dev` installs pytest, ruff, and matplotlib. The runtime dependencies are `httpx` and `PyYAML`. Plotting alone is `pip install -e ".[plot]"`.

You do not install vLLM or ROCm to run the tests. Install a ROCm build of vLLM (or SGLang) only on the machine that will serve the model.

## Quickstart (no GPU)

Terminal A:

```bash
python -m mi300x_bench mock --port 8000
```

Terminal B:

```bash
python -m mi300x_bench run \
  --base-url http://127.0.0.1:8000/v1 \
  --model mock \
  --concurrency 4 \
  --input-len 128 \
  --output-len 32 \
  --num-requests 8 \
  --output results/mock.json

python -m mi300x_bench sweep --config configs/mock_sweep.yaml
python -m mi300x_bench plot --input results/mock_sweep/sweep.json --output results/mock_sweep.png
```

`input-len` is a word count for synthetic prompts, not a tokenizer length. The mock counts those words and reports them as `prompt_tokens`. Its delays and token strings are not a hardware result.

`mi300x-bench` is the same CLI after installation.

## Quickstart (MI300X)

Collect GPU identity first. If `rocm-smi` is not on `PATH`, the command still exits 0 and says so. Newer ROCm images also ship `amd-smi`; this collector does not call it.

```bash
python -m mi300x_bench gpu-info
python scripts/collect_gpu_info.py
```

Start vLLM on one GPU. The script only execs `vllm serve`. It assumes a ROCm build of vLLM is already installed. `MAX_MODEL_LEN` defaults to 8192 because KV cache, not the weight load, is what runs out first. Raise it toward 131072 only after the server stays up.

```bash
chmod +x scripts/launch_vllm_rocm.sh
MODEL=meta-llama/Llama-3.1-70B-Instruct \
MAX_MODEL_LEN=8192 \
scripts/launch_vllm_rocm.sh
```

The process listens on `http://0.0.0.0:8000/v1`. Then, from another shell:

```bash
python -m mi300x_bench sweep --config configs/mi300x_sweep.yaml
python -m mi300x_bench plot \
  --input results/mi300x_sweep/sweep.json \
  --output results/mi300x_sweep.png
```

An optional SGLang launcher is `scripts/launch_sglang_rocm.sh` (port 30000). Point `--base-url` at `http://127.0.0.1:30000/v1`. The client does not care which server it is.

Flags worth knowing on the vLLM script, all overridable by environment variables: `MODEL`, `HOST`, `PORT`, `TP` (default 1), `MAX_MODEL_LEN`, `GPU_MEMORY_UTILIZATION` (default 0.90), `MAX_NUM_SEQS`, `HIP_VISIBLE_DEVICES`.

## Example config

`configs/mi300x_sweep.yaml` is the single-GPU sweep. It is a plan, not a log.

```yaml
base_url: http://127.0.0.1:8000/v1
model: meta-llama/Llama-3.1-70B-Instruct
num_requests: 20
warmup: 2
concurrency: [1, 4, 16]
input_len: [8192, 32768, 131072]
output_len: [128]
temperature: 0
ignore_eos: true
timeout_s: 3600
output_dir: results/mi300x_sweep
```

The sweep is the cartesian product of `concurrency`, `input_len`, and `output_len`. Order is input length, then output length, then concurrency. `configs/mock_sweep.yaml` is the same schema at a size the mock server can finish quickly.

`ignore_eos: true` is sent as an extra JSON field so vLLM and SGLang generate the full `output_len` instead of stopping on an end-of-sequence token. Leave it false for a strict OpenAI endpoint, which may reject unknown fields. The example hardware config turns it on; the CLI leaves it off unless you pass `--ignore-eos`.

A local file replaces synthetic prompts. JSONL accepts `{"prompt": "..."}`, OpenAI `messages`, or ShareGPT `conversations`. A `.txt` file is one prompt per line. Do not set `input_len` in that mode.

```bash
python -m mi300x_bench run \
  --base-url http://127.0.0.1:8000/v1 \
  --model meta-llama/Llama-3.1-70B-Instruct \
  --dataset prompts.jsonl \
  --output-len 128 \
  --concurrency 4 \
  --num-requests 20 \
  --ignore-eos \
  --output results/dataset.json
```

Other flags: `--api-key` (sent as `Authorization: Bearer`), `--temperature`, `--seed`, `--warmup`, `--timeout`, `--save-requests`, `--skip-health-check`. The client ignores `HTTP_PROXY` / `HTTPS_PROXY` so the URL you pass is the one it calls. Each run starts with `GET {base_url}/models` unless you skip the health check. A failed warmup aborts the run. In a sweep that also stops later points; points already finished stay on disk. Exit code 0 means every requested point finished and at least one request succeeded. Exit code 2 means the server was unreachable, the sweep stopped early, or every request failed.

## Metrics

Each point is closed-loop. The client keeps at most `concurrency` requests in flight and has no separate arrival-rate limiter (this matches an infinite request rate in serving benchmarks). There is no open-loop Poisson mode.

The wall clock used for throughput starts when the first timed request is issued and stops when the last timed request finishes, including failed ones. Warmup requests run first, one after another, and are dropped.

Per-request latencies start when that request is issued, which is after it takes a concurrency slot. Time spent waiting for a slot is not part of TTFT or E2E. It still affects throughput, because the wall clock covers the whole overlap.

| Field | Definition |
| --- | --- |
| `output_tokens_per_s` | Successful completion tokens ÷ wall clock. This is generated tokens over the whole benchmark, not a pure decode-phase rate. |
| `total_tokens_per_s` | Successful prompt tokens + completion tokens, ÷ wall clock. Prompt tokens are prefill work spread across the same wall clock, so this is not a prefill-only rate either. |
| `requests_per_s` | Successful requests ÷ wall clock. |
| `ttft_ms` | Issue time → first streamed chunk whose content is non-empty. An opening role-only chunk does not count. |
| `itl_ms` | Gaps between consecutive non-empty content chunks, pooled across successful requests. vLLM often sends one token per chunk; a server may also coalesce tokens, so ITL is a chunk gap. |
| `tpot_ms` | Per successful request, `(e2e - ttft) / (completion_tokens - 1)` when at least two completion tokens were counted and a first chunk was observed. The distribution is then taken over requests. |
| `e2e_ms` | Issue time → end of the response. |
| `prompt_tokens_total` | Sum of `usage.prompt_tokens` from successful responses. |
| `completion_tokens_total` | Sum of `usage.completion_tokens`. If the server sends no usage object, the client falls back to the number of non-empty content chunks. |

`mean`, `p50`, `p90`, and `p99` use linear interpolation, the same rule as the default `numpy.percentile`. With a short run, p99 sits next to the slowest sample and is not a stable tail. The example sweep uses 20 requests per point; raise `num_requests` before quoting a tail.

Failed requests are excluded from the latency distributions and from the token totals. They are counted in `failed`, and their error strings (up to 10) are stored. If any successful response is missing a prompt-token count, the prompt total is null and `total_tokens_per_s` is null rather than a guessed rate.

Synthetic `input_len` is exact in words and only approximate in tokenizer tokens. Compare `prompt_tokens_mean` with `target_input_len` to see what the server actually billed. The mock's word count matches `input_len`. A real model's tokenizer will not match it exactly.

The request body is `max_tokens=output_len`, `temperature`, `stream=true`, and `stream_options.include_usage=true` (supported by current vLLM, SGLang, and the OpenAI API). The model may still stop early unless `ignore_eos` is set, so `completion_tokens_mean` can be below `output_len`.

`--save-requests` adds each call's TTFT, ITL list, token counts, and error to the JSON. The default JSON is the meta block plus the summary, so a long sweep stays small. A sweep writes `sweep.json`, `sweep.csv`, and one JSON/CSV pair per point under `points/`.

## Results

Nothing here was measured on an MI300X, or on any other GPU. Fill the table from `results/mi300x_sweep/sweep.csv` after a real run. Do not copy numbers from the mock server into this table.

| Model | Precision | GPUs | Input tokens | Output tokens | Concurrency | Output tok/s | TTFT p50 (ms) | E2E p99 (ms) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| TBD — to be filled after running on MI300X | TBD — to be filled after running on MI300X | TBD — to be filled after running on MI300X | TBD — to be filled after running on MI300X | TBD — to be filled after running on MI300X | TBD — to be filled after running on MI300X | TBD — to be filled after running on MI300X | TBD — to be filled after running on MI300X | TBD — to be filled after running on MI300X |

## Roadmap

- Sample `rocm-smi` power and VRAM while a point is running, still no-op when the tool is absent.
- Repeat each point and report mean and spread across repeats.
- An open-loop request-rate mode next to the closed loop.
- A side-by-side of prefix caching on and off.
- Read `vllm bench serve` JSON into the same CSV so one plot can show both clients.

## Development

```bash
pip install -e ".[dev]"
ruff check .
ruff format --check .
pytest
```

GitHub Actions runs those three commands on Python 3.10, 3.11, and 3.12. The launch scripts are syntax-checked (`bash -n`) and are not executed. CI does not install ROCm, vLLM, or SGLang.

## 中文简介

这是一个面向 AMD Instinct MI300X（192GB HBM3，ROCm）的 LLM 推理压测小工具。Llama-3.1-70B-Instruct 的 bf16 权重约 140GB，可以放进单张 MI300X，从而不必为了放下模型而做多卡张量并行。权重之外剩下的显存要留给 KV cache，所以长上下文（示例扫描为 8k / 32k / 128k 输入）和高并发是否站得住，需要在机器上测，而不是靠估算。

工具通过 vLLM 或 SGLang 的 OpenAI 兼容接口发送流式请求，记录吞吐（tokens/s）、首 token 延迟、token 间隔、TPOT，以及 p50/p90/p99 端到端延迟。它不绑定 GPU：对着任何兼容服务都能跑，仓库自带一个 mock server，CPU 上的 pytest 会打这个 mock。启动脚本在 `scripts/launch_vllm_rocm.sh`（可选 SGLang：`scripts/launch_sglang_rocm.sh`），`rocm-smi` 不存在时 GPU 信息采集会正常退出并说明原因。

仓库里没有实测数字。上面的结果表是空的，标记为 TBD，需要在 MI300X 上跑完 `configs/mi300x_sweep.yaml` 之后再填。

## License

MIT. See [LICENSE](LICENSE).
