# RUPA

RUPA (Relational Uncertainty Propagation for Agents) is a graph-based uncertainty quantification framework for long-horizon LLM agents.

This repository contains:
- the core RUPA scoring implementation
- evaluation scripts for baseline methods
- prefix-based confidence analysis scripts
- graph-structure and ablation studies
- plotting scripts for publication-quality figures
- the underlying `tau2` agent simulation code used by the evaluation pipeline

## 1. Environment Setup

### 1.1 Create a Python environment

```bash
python -m venv .venv
source .venv/bin/activate
```

### 1.2 Install dependencies

Install the `tau2` package in editable mode so that the agent simulator and domain code are available:

```bash
pip install -e agent-tracer
```

Install the analysis stack used by the confidence scripts if it is not already present:

```bash
pip install numpy pandas scikit-learn matplotlib tqdm
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

### 3.4 Graph and ablation studies

```bash
python evaluate_trajectory_tau_graph_experiments.py PATH_TO_JOB_ROOT
python evaluate_trajectory_tau_ablation.py PATH_TO_JOB_ROOT
python evaluate_trajectory_tau_extended_ablation.py PATH_TO_JOB_ROOT
```

### 3.5 Publication figures

```bash
python plot_entropy_matched_bins.py PATH_TO_JOB_ROOT/trajectory_tau_graph_experiments_summary.json
python plot_trajectory_tau_parameter_ablation.py PATH_TO_JOB_ROOT/trajectory_tau_extended_ablation_summary.json
```

## 4. Repository Hygiene

This repository ignores:
- `jobs/` and other run outputs
- logs and temporary files
- generated figures and intermediate artifacts
- cache directories such as `__pycache__`, `.pytest_cache`, `.mypy_cache`, and `.ruff_cache`

## 5. Notes

- All uncertainty metrics are reported as failure-detection scores: higher values indicate higher failure risk.
- Use placeholder paths such as `PATH_TO_JOB_ROOT` when sharing commands publicly.
