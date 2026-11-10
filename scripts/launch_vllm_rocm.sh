#!/usr/bin/env bash
# Start a vLLM OpenAI-compatible server on ROCm for one AMD Instinct MI300X.
#
# Llama-3.1-70B-Instruct in bf16 is about 140 GB of weights (70e9 parameters
# times 2 bytes). One MI300X has 192 GB of HBM3, so those weights fit on a
# single GPU and tensor-parallel size stays 1. The memory left over is the
# budget for KV cache, activations, and the runtime. Long context and high
# concurrency share that budget. If vLLM exits because the KV cache does not
# fit, lower MAX_MODEL_LEN or MAX_NUM_SEQS. This script does not measure
# throughput and it is not run in CI.
#
# Install a ROCm build of vLLM first so the `vllm` command exists
# (pip wheel or the ROCm vLLM container for your ROCm version).

set -euo pipefail

MODEL="${MODEL:-meta-llama/Llama-3.1-70B-Instruct}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
TP="${TP:-1}"
# Llama 3.1 was trained at 131072 tokens. 8192 is a conservative default:
# the weight load is not what runs out first.
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-64}"

export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0}"

exec vllm serve "${MODEL}" \
  --host "${HOST}" \
  --port "${PORT}" \
  --dtype bfloat16 \
  --tensor-parallel-size "${TP}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
  --max-num-seqs "${MAX_NUM_SEQS}" \
  --disable-log-requests
