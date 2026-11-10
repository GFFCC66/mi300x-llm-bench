#!/usr/bin/env bash
# Optional SGLang server with an OpenAI-compatible API on one MI300X.
# The benchmark client uses the same /v1/chat/completions path as vLLM.
# Point --base-url at http://HOST:PORT/v1.
#
# The memory note in scripts/launch_vllm_rocm.sh applies here too: bf16
# Llama-3.1-70B weights fit in 192 GB, and context length is limited by the
# remaining HBM. Requires a ROCm build of SGLang. Not run in CI.

set -euo pipefail

MODEL="${MODEL:-meta-llama/Llama-3.1-70B-Instruct}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-30000}"
TP="${TP:-1}"
CONTEXT_LENGTH="${CONTEXT_LENGTH:-8192}"
MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.90}"

export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0}"

exec python -m sglang.launch_server \
  --model-path "${MODEL}" \
  --host "${HOST}" \
  --port "${PORT}" \
  --dtype bfloat16 \
  --tp "${TP}" \
  --context-length "${CONTEXT_LENGTH}" \
  --mem-fraction-static "${MEM_FRACTION_STATIC}"
