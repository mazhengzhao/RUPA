#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-${SCRIPT_DIR}}"
HARBOR_BIN="${HARBOR_BIN:-harbor}"

if [[ ! -x "${HARBOR_BIN}" ]]; then
  if command -v harbor >/dev/null 2>&1; then
    HARBOR_BIN="$(command -v harbor)"
  else
    echo "ERROR: harbor executable not found. Set HARBOR_BIN=/path/to/harbor" >&2
    exit 2
  fi
fi

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

: "${OPENAI_API_KEY:?Set OPENAI_API_KEY before running this script.}"
export OPENAI_API_KEY
if [[ -n "${OPENAI_BASE_URL:-}" ]]; then
  export OPENAI_BASE_URL
fi

CMD=(
  "${HARBOR_BIN}" run
  --agent-import-path harbor_uncertainty_agent:UncertaintySamplingTerminus2 \
  --agent-timeout-multiplier "${AGENT_TIMEOUT_MULTIPLIER:-4}" \
  --agent-kwarg num_samples="${UNCERTAINTY_NUM_SAMPLES:-5}" \
  --agent-kwarg uncertainty_method="${UNCERTAINTY_METHOD:-trajectory_tau}" \
  --agent-kwarg temperature="${UNCERTAINTY_TEMPERATURE:-0.7}" \
  --agent-kwarg max_turns="${HARBOR_AGENT_MAX_TURNS:-80}" \
  --agent-kwarg max_consecutive_repeated_actions="${MAX_CONSECUTIVE_REPEATED_ACTIONS:-8}"
)

if [[ -n "${MODEL_MAX_INPUT_TOKENS:-}" || -n "${MODEL_MAX_OUTPUT_TOKENS:-}" ]]; then
  MODEL_INFO="{\"max_input_tokens\":${MODEL_MAX_INPUT_TOKENS:-32768},\"max_output_tokens\":${MODEL_MAX_OUTPUT_TOKENS:-8192}}"
  CMD+=(--agent-kwarg "model_info=${MODEL_INFO}")
fi

CMD+=("$@")

exec "${CMD[@]}"
