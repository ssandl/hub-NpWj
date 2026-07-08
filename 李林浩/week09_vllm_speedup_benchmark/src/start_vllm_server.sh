#!/usr/bin/env bash
set -euo pipefail

# Week09 assignment: start an OpenAI-compatible vLLM service.
# You can override these variables before running this script, for example:
#   MODEL_NAME=Qwen/Qwen2.5-1.5B-Instruct PORT=8001 bash start_vllm_server.sh

MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-0.5B-Instruct}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-week09-qwen}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-2048}"

python -m vllm.entrypoints.openai.api_server \
  --model "${MODEL_NAME}" \
  --served-model-name "${SERVED_MODEL_NAME}" \
  --host "${HOST}" \
  --port "${PORT}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
  --max-model-len "${MAX_MODEL_LEN}"
