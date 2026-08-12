#!/usr/bin/env bash
set -euo pipefail

# Start an 8-GPU local vLLM OpenAI-compatible server for Qwen, run Harbor with
# the uncertainty-sampling agent, and stop the model server when Harbor exits.
#
# Required:
#   MODEL_PATH=/path/to/Qwen3.5-27B
#
# Common optional settings:
#   VLLM_PYTHON=/path/to/python-with-vllm
#   SERVED_MODEL_NAME=Qwen3.5-27B
#   CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
#   TENSOR_PARALLEL_SIZE=2
#   DATA_PARALLEL_SIZE=4
#   VLLM_DISABLE_NCCL_FOR_DP_SYNC=1
#   VLLM_PORT=8000
#   MODEL_MAX_INPUT_TOKENS=32768
#   MODEL_MAX_OUTPUT_TOKENS=8192
#   UNCERTAINTY_NUM_SAMPLES=5
#   UNCERTAINTY_METHOD=trajectory_tau
#   HARBOR_N_CONCURRENT=8
#   HARBOR_AGENT_MAX_TURNS=80
#   MAX_CONSECUTIVE_REPEATED_ACTIONS=8
#
# Example:
#   MODEL_PATH=/data/models/Qwen3.5-27B \
#   TENSOR_PARALLEL_SIZE=2 \
#   DATA_PARALLEL_SIZE=4 \
#   VLLM_PYTHON=/opt/conda/envs/vllm/bin/python \
#   ./run_local_qwen_uncertainty_harbor.sh \
#     --dataset terminal-bench/terminal-bench-2 \
#     --force-build \
#     --yes

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
HARBOR_SCRIPT="${HARBOR_SCRIPT:-${PROJECT_ROOT}/scripts/harbor/run_harbor_uncertainty_agent.sh}"

MODEL_PATH="${MODEL_PATH:-}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-Qwen3.5-27B}"
HOST="${VLLM_HOST:-127.0.0.1}"
PORT="${VLLM_PORT:-8000}"
API_BASE="http://${HOST}:${PORT}/v1"
VLLM_LOG="${VLLM_LOG:-${PROJECT_ROOT}/vllm_qwen_${PORT}.log}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-2}"
DATA_PARALLEL_SIZE="${DATA_PARALLEL_SIZE:-4}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
MODEL_MAX_INPUT_TOKENS="${MODEL_MAX_INPUT_TOKENS:-32768}"
MODEL_MAX_OUTPUT_TOKENS="${MODEL_MAX_OUTPUT_TOKENS:-8192}"
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.90}"
VLLM_DTYPE="${VLLM_DTYPE:-bfloat16}"
VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-8}"
VLLM_DATA_PARALLEL_BACKEND="${VLLM_DATA_PARALLEL_BACKEND:-mp}"
VLLM_DISABLE_NCCL_FOR_DP_SYNC="${VLLM_DISABLE_NCCL_FOR_DP_SYNC:-1}"
VLLM_STARTUP_TIMEOUT_SEC="${VLLM_STARTUP_TIMEOUT_SEC:-1800}"
HARBOR_N_CONCURRENT="${HARBOR_N_CONCURRENT:-8}"

if [[ -z "${MODEL_PATH}" ]]; then
  echo "ERROR: MODEL_PATH is required, e.g. MODEL_PATH=/data/models/Qwen3.5-27B" >&2
  exit 2
fi

if [[ ! -e "${MODEL_PATH}" ]]; then
  echo "ERROR: MODEL_PATH does not exist: ${MODEL_PATH}" >&2
  exit 2
fi

if [[ ! -f "${HARBOR_SCRIPT}" ]]; then
  echo "ERROR: Harbor uncertainty runner not found: ${HARBOR_SCRIPT}" >&2
  exit 2
fi

if [[ -n "${VLLM_PYTHON:-}" ]]; then
  VLLM_CMD=("${VLLM_PYTHON}" -m vllm.entrypoints.openai.api_server)
else
  VLLM_CMD=(python -m vllm.entrypoints.openai.api_server)
fi

cleanup() {
  local code=$?
  if [[ -n "${VLLM_PID:-}" ]] && kill -0 "${VLLM_PID}" >/dev/null 2>&1; then
    echo "Stopping vLLM server pid=${VLLM_PID}"
    kill "${VLLM_PID}" >/dev/null 2>&1 || true
    wait "${VLLM_PID}" >/dev/null 2>&1 || true
  fi
  exit "${code}"
}
trap cleanup EXIT INT TERM

echo "Starting local vLLM server"
echo "  model path: ${MODEL_PATH}"
echo "  served name: ${SERVED_MODEL_NAME}"
echo "  api base: ${API_BASE}"
echo "  cuda devices: ${CUDA_VISIBLE_DEVICES}"
echo "  tensor parallel size: ${TENSOR_PARALLEL_SIZE}"
echo "  data parallel size: ${DATA_PARALLEL_SIZE}"
echo "  data parallel backend: ${VLLM_DATA_PARALLEL_BACKEND}"
echo "  disable nccl for dp sync: ${VLLM_DISABLE_NCCL_FOR_DP_SYNC}"
echo "  log: ${VLLM_LOG}"

mkdir -p "$(dirname "${VLLM_LOG}")"

VLLM_ARGS=(
  --model "${MODEL_PATH}" \
  --served-model-name "${SERVED_MODEL_NAME}" \
  --host "${HOST}" \
  --port "${PORT}" \
  --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}" \
  --data-parallel-size "${DATA_PARALLEL_SIZE}" \
  --data-parallel-backend "${VLLM_DATA_PARALLEL_BACKEND}" \
  --dtype "${VLLM_DTYPE}" \
  --max-model-len "${MODEL_MAX_INPUT_TOKENS}" \
  --gpu-memory-utilization "${VLLM_GPU_MEMORY_UTILIZATION}" \
  --max-num-seqs "${VLLM_MAX_NUM_SEQS}"
)

if [[ "${VLLM_DISABLE_NCCL_FOR_DP_SYNC}" == "1" || "${VLLM_DISABLE_NCCL_FOR_DP_SYNC}" == "true" ]]; then
  VLLM_ARGS+=(--disable-nccl-for-dp-synchronization)
fi

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" "${VLLM_CMD[@]}" \
  "${VLLM_ARGS[@]}" \
  ${VLLM_EXTRA_ARGS:-} \
  >"${VLLM_LOG}" 2>&1 &

VLLM_PID=$!

echo "Waiting for vLLM server to become ready"
deadline=$((SECONDS + VLLM_STARTUP_TIMEOUT_SEC))
while (( SECONDS < deadline )); do
  if ! kill -0 "${VLLM_PID}" >/dev/null 2>&1; then
    echo "ERROR: vLLM server exited before becoming ready. Last log lines:" >&2
    echo "---- likely root-cause lines ----" >&2
    grep -Ein "traceback|error|exception|failed|cuda|nccl|out of memory|oom|no space|shared_memory" "${VLLM_LOG}" | tail -200 >&2 || true
    echo "---- final log tail ----" >&2
    tail -300 "${VLLM_LOG}" >&2 || true
    exit 1
  fi

  if python - "${API_BASE}" >/dev/null 2>&1 <<'PY'
import json
import sys
import urllib.request

api_base = sys.argv[1].rstrip("/")
try:
    with urllib.request.urlopen(api_base + "/models", timeout=5) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    if data.get("data") is not None:
        sys.exit(0)
except Exception:
    pass
sys.exit(1)
PY
  then
    break
  fi

  sleep 5
done

if (( SECONDS >= deadline )); then
  echo "ERROR: vLLM server did not become ready within ${VLLM_STARTUP_TIMEOUT_SEC}s" >&2
  echo "---- likely root-cause lines ----" >&2
  grep -Ein "traceback|error|exception|failed|cuda|nccl|out of memory|oom|no space|shared_memory" "${VLLM_LOG}" | tail -200 >&2 || true
  echo "---- final log tail ----" >&2
  tail -300 "${VLLM_LOG}" >&2 || true
  exit 1
fi

echo "vLLM server is ready"

export OPENAI_API_KEY="${OPENAI_API_KEY:-dummy}"
export OPENAI_BASE_URL="${API_BASE}"
export MODEL_MAX_INPUT_TOKENS
export MODEL_MAX_OUTPUT_TOKENS
export AGENT_TIMEOUT_MULTIPLIER="${AGENT_TIMEOUT_MULTIPLIER:-6}"
export UNCERTAINTY_NUM_SAMPLES="${UNCERTAINTY_NUM_SAMPLES:-5}"
export UNCERTAINTY_METHOD="${UNCERTAINTY_METHOD:-trajectory_tau}"
export UNCERTAINTY_TEMPERATURE="${UNCERTAINTY_TEMPERATURE:-0.7}"
export HARBOR_AGENT_MAX_TURNS="${HARBOR_AGENT_MAX_TURNS:-80}"
export MAX_CONSECUTIVE_REPEATED_ACTIONS="${MAX_CONSECUTIVE_REPEATED_ACTIONS:-8}"

echo "Starting Harbor job"
exec_args=(
  --model "openai/${SERVED_MODEL_NAME}"
)

has_harbor_concurrency_arg=false
for arg in "$@"; do
  if [[ "${arg}" == "--n-concurrent" || "${arg}" == --n-concurrent=* || "${arg}" == "-n" ]]; then
    has_harbor_concurrency_arg=true
    break
  fi
done

if [[ "${has_harbor_concurrency_arg}" == false ]]; then
  exec_args+=(--n-concurrent "${HARBOR_N_CONCURRENT}")
fi

exec_args+=("$@")

if [[ -x "${HARBOR_SCRIPT}" ]]; then
  "${HARBOR_SCRIPT}" "${exec_args[@]}"
else
  bash "${HARBOR_SCRIPT}" "${exec_args[@]}"
fi
