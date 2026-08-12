OPENAI_API_KEY="${OPENAI_API_KEY:?Set OPENAI_API_KEY}" \
OPENAI_BASE_URL="${OPENAI_BASE_URL:?Set OPENAI_BASE_URL}" \
harbor run \
  --dataset terminal-bench/terminal-bench-2 \
  --agent-import-path scripts.harbor.harbor_uncertainty_agent:UncertaintySamplingTerminus2 \
  --model openai/gemma4-31B \
  --agent-timeout-multiplier 3 \
  --agent-kwarg num_samples=3 \
  --agent-kwarg uncertainty_method=tracer \
  --agent-kwarg temperature=0.7 \
  --force-build \
  --yes
