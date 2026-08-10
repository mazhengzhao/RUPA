# RUPA

This repository is the official implementation for **From Sequence to Structure: Relational uncertainty Propagation for LLM Agents**

RUPA (Relational Uncertainty Propagation for Agents) is a graph-based uncertainty quantification framework for long-horizon LLM agents.


## 1. Environment Setup

### 1.1 Create a Python environment

```bash
python -m venv .venv
source .venv/bin/activate
```

### 1.2 Install dependencies

Install everything from the repository-level requirements file:

```bash
pip install -r requirements.txt
```

## 2. Harbor Task Execution

The agent benchmark runner lives in `agent-tracer/`. After configuring your model API credentials in the environment or `.env` file, you can launch a small test run with:

```bash
cd agent-tracer
tau2 run \
  --domain airline \
  --agent-llm gpt-4.1 \
  --user-llm gpt-4.1 \
  --num-trials 1 \
  --num-tasks 5 \
  --calculate-uncertainty
```

For full benchmark runs, adjust the domain, model names, task IDs, and concurrency settings as needed. Generated trajectories and metrics should be kept outside version control.

If you want to run the Harbor sampling agent used in this repository, use the wrapper script:

```bash
export OPENAI_API_KEY=YOUR_KEY
export OPENAI_BASE_URL=YOUR_BASE_URL
bash run_harbor_uncertainty_agent.sh --dataset PATH_TO_HARBOR_DATASET --model MODEL_NAME --yes
```

You can also control the sampling behavior with environment variables such as `UNCERTAINTY_METHOD=trajectory_tau`, `UNCERTAINTY_NUM_SAMPLES=5`, and `UNCERTAINTY_TEMPERATURE=0.7`.

## 3. Confidence Analysis

All analysis scripts expect a Harbor job root containing `result.json` and per-trial subdirectories.

### 3.1 Run all metrics

```bash
bash run_all_eval_metrics.sh PATH_TO_JOB_ROOT [python_bin]
```

This evaluates the supported uncertainty methods, including entropy, sequence probability, UProp, TRACER, SAUP, TAU, and RUPA.

### 3.2 Run individual evaluations

```bash
python evaluate_gaia_entropy_metrics.py PATH_TO_JOB_ROOT
python evaluate_gaia_sequence_prob_metrics.py PATH_TO_JOB_ROOT
python evaluate_gaia_uprop_metrics.py PATH_TO_JOB_ROOT
python evaluate_gaia_tracer_metrics.py PATH_TO_JOB_ROOT
python evaluate_gaia_saup_metrics.py PATH_TO_JOB_ROOT
python evaluate_gaia_tau_metrics.py PATH_TO_JOB_ROOT
python evaluate_gaia_trajectory_tau_metrics.py PATH_TO_JOB_ROOT
```

### 3.3 Prefix confidence curves

```bash
python evaluate_prefix_uq_curves.py PATH_TO_JOB_ROOT --mode percent --prefix-percents 0.1,0.2,0.3,0.5,0.7,1.0
python evaluate_prefix_uq_curves.py PATH_TO_JOB_ROOT --mode steps --prefix-steps 1,2,4,8,12,16
python plot_prefix_uq_curves.py PATH_TO_JOB_ROOT/prefix_uq_curves_percent.json --metric auroc
python plot_prefix_uq_curves.py PATH_TO_JOB_ROOT/prefix_uq_curves_steps.json --metric auprc
```
