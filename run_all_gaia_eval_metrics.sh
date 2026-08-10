#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "Usage: $0 <harbor_job_root> [python_bin]" >&2
  echo "Example: $0 PATH_TO_JOB_ROOT PATH_TO_PYTHON" >&2
  exit 1
fi

JOB_ROOT="$1"
PYTHON_BIN="${2:-${PYTHON_BIN:-python}}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-agent-confidence}"
mkdir -p "${MPLCONFIGDIR}"

echo "Using Python: ${PYTHON_BIN}"
echo "Harbor job root: ${JOB_ROOT}"
echo "Matplotlib config dir: ${MPLCONFIGDIR}"
echo

echo "=== Entropy ==="
"${PYTHON_BIN}" "${SCRIPT_DIR}/evaluate_gaia_entropy_metrics.py" "${JOB_ROOT}"
echo

echo "=== Sequence Probability ==="
"${PYTHON_BIN}" "${SCRIPT_DIR}/evaluate_gaia_sequence_prob_metrics.py" "${JOB_ROOT}"
echo

echo "=== UProp ==="
"${PYTHON_BIN}" "${SCRIPT_DIR}/evaluate_gaia_uprop_metrics.py" "${JOB_ROOT}"
echo

echo "=== TRACER ==="
"${PYTHON_BIN}" "${SCRIPT_DIR}/evaluate_gaia_tracer_metrics.py" "${JOB_ROOT}"
echo

echo "=== SAUP ==="
"${PYTHON_BIN}" "${SCRIPT_DIR}/evaluate_gaia_saup_metrics.py" "${JOB_ROOT}"
echo

echo "=== TAU ==="
"${PYTHON_BIN}" "${SCRIPT_DIR}/evaluate_gaia_tau_metrics.py" "${JOB_ROOT}"

echo "=== Trajectory TAU ==="
"${PYTHON_BIN}" "${SCRIPT_DIR}/evaluate_gaia_trajectory_tau_metrics.py" "${JOB_ROOT}"
