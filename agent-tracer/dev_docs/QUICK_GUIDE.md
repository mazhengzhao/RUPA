# Quick Reference: Uncertainty Analysis

## 🚀 Common Commands

### Run Simulations

* **Basic simulation** (log probs only)

```bash
# Using your Gemini/Vertex AI models
tau2 run \
  --domain airline \
  --num-tasks 50 \
  --num-trials 1 \
  --max-steps 50 \
  --max-errors 5 \
  --agent-llm vertex_ai/gemini-2.5-flash \
  --user-llm vertex_ai/gemini-2.5-flash \
```

* **Advanced simulation** (with TRACER metrics: U_i, Da, Do)

```bash
tau2 run \
  --domain airline \
  --num-tasks 50 \
  --num-trials 1 \
  --max-steps 50 \
  --max-errors 5 \
  --max-concurrency 2 \
  --agent-llm vertex_ai/gemini-2.5-flash \
  --user-llm vertex_ai/gemini-2.5-flash \
  --calculate-uncertainty
```

**Note**: The `--calculate-uncertainty` flag now enables:
- **U_i**: Single-step uncertainty (token-level confidence)
- **Da**: Inquiry Drift (semantic distance from goal)
- **Do**: Inference Gap (action-observation coherence)
- **TRACER Score**: Trajectory-level aggregation of all metrics (weighted RMS)

### Run separate uncertainty tests
```bash
# Simple test runner (no dependencies needed)
python tests/run_uncertainty_tests.py

# With pytest (if installed)
pytest tests/test_uncertainty.py -v
```

### Analyze Simulations
```bash
# Basic analysis (auto-saves to data/uncertainty/)
python -m tau2.scripts.analyze_uncertainty data/simulations/your_file.json

# Detailed turn-by-turn view with Da/Do scores (auto-saves)
python -m tau2.scripts.analyze_uncertainty data/simulations/your_file.json --detailed

# Include verbose statistics in output
python -m tau2.scripts.analyze_uncertainty data/simulations/your_file.json --verbose

# Custom output location
python -m tau2.scripts.analyze_uncertainty data/simulations/your_file.json \
  --output results/custom_analysis.json

# Don't save, just display results
python -m tau2.scripts.analyze_uncertainty data/simulations/your_file.json --no-save

# Custom TRACER configuration (adjust α, β, γ weights, top-k, and ensemble)
python -m tau2.scripts.analyze_uncertainty data/simulations/your_file.json \
  --tracer-config '{"alpha": 4.0, "beta": 4.0, "gamma": 5.0, "top_k_percentile": 0.26, "ensemble_weight_max": 0.2}'

# Skip AUROC calculation (if ground truth not available)
python -m tau2.scripts.analyze_uncertainty data/simulations/your_file.json --no-auroc
```

**Note**: By default, results are automatically saved to `data/uncertainty/` with the same filename as your simulation file for easy cross-referencing.

**AUROC Calculation**: The script automatically calculates AUROC (Area Under ROC Curve) to evaluate whether TRACER scores predict task failure. Requires both TRACER scores and ground truth (reward) data.

**TRACER Improvements**: The default configuration uses optimized ensemble method combining top-k aggregation (80%) with max risk (20%), improving AUROC from 0.6488 to 0.7007 (+8.0%). This captures both sustained problems and critical failure moments, achieving "good" predictive power.

### Batch Process All Simulations
```bash
# Analyze all simulations at once (skips already analyzed files)
./scripts/batch_analyze_uncertainty.sh

# With verbose statistics
./scripts/batch_analyze_uncertainty.sh --verbose

# With detailed view for each
./scripts/batch_analyze_uncertainty.sh --detailed
```

**Note on AUROC**: The uncertainty analysis script automatically calculates AUROC (Area Under ROC Curve) to evaluate whether TRACER scores can predict task failure. This is included in the standard analysis output when both TRACER scores and ground truth data are available.

### Optimize TRACER Parameters
Find the best TRACER parameters (α, β, γ, top-k, ensemble) for your dataset.

**Performance Note:** The script uses a two-phase approach for 100-1000× speed improvement:
- **Phase 1:** Pre-computes step-level metrics (U_i, Da, Do) once (runs automatically)
- **Phase 2:** Fast parameter search using cached metrics (only arithmetic operations)

This means fine-grained searches with 300+ configurations now complete in minutes instead of hours!

```bash
# AUTO MODE (default): Run all 3 stages in one command
python -m tau2.scripts.optimize_tracer_parameters \
  data/simulations/my_simulation_results.json

# RANDOM SEARCH (recommended for finding global optima)
# Tests 1000 random configurations across wide parameter ranges
python -m tau2.scripts.optimize_tracer_parameters \
  data/simulations/my_simulation_results.json \
  --mode random \
  --n-random 1000

# WIDE SEARCH (most thorough - grid + random)
# Tests ~2800 configurations with extended ranges
python -m tau2.scripts.optimize_tracer_parameters \
  data/simulations/my_simulation_results.json \
  --mode wide \
  --n-random 2000

# Don't save, just display results
python -m tau2.scripts.optimize_tracer_parameters \
  data/simulations/my_simulation_results.json \
  --no-save

# Stage 1: Coarse grid search only (explores wide parameter ranges)
# Tests ~1500 configurations (10×5×6×5 grid)
python -m tau2.scripts.optimize_tracer_parameters \
  data/simulations/my_simulation_results.json \
  --mode coarse

# Stage 2: Fine-grained search (refines around best coarse params)
# Tests ~286 configurations with smaller step sizes
python -m tau2.scripts.optimize_tracer_parameters \
  data/simulations/my_simulation_results.json \
  --mode fine \
  --alpha-center 6.0 --alpha-range 3.0 --alpha-step 0.5 \
  --topk-center 0.25 --topk-range 0.1 --topk-step 0.02

# Stage 3: Ensemble optimization (tunes ensemble weight)
# Tests 9 ensemble weights while keeping other params fixed
python -m tau2.scripts.optimize_tracer_parameters \
  data/simulations/my_simulation_results.json \
  --mode ensemble \
  --alpha 4.0 --beta 4.0 --gamma 5.0 --top-k 0.26 \
  --ensemble-weights 0.05 0.1 0.15 0.2 0.25 0.3

# Custom grid search (specify your own parameter values)
python -m tau2.scripts.optimize_tracer_parameters \
  data/simulations/my_simulation_results.json \
  --mode custom \
  --alpha-values 1.25 1.5 1.75 2.0 2.25 2.5 \
  --beta-values 7.0 7.5 8.0 8.5 9.0 \
  --gamma-values 2.5 3.0 3.5 4.0 \
  --topk-values 0.20 0.21 0.22 0.23 0.24 \
  --ensemble-values 0.0 0.15 0.2 0.25
```

**How it works**: The optimizer loads simulation(s), and systematically tests different parameter combinations to maximize AUROC (failure prediction accuracy). The script outputs the best configuration and shows stage-by-stage progress.

**Optimization Modes**:
- **auto** (default): Runs all 3 stages sequentially (coarse → fine → ensemble)
- **random** (recommended for global optima): Tests N random configurations across wide ranges - often finds better solutions than grid search
- **wide** (most thorough): Combines broad grid (~1800 configs) + random search (1000+ configs) for maximum coverage
- **coarse**: Tests 1,500 configurations across wide ranges (good for initial exploration)
- **fine**: Tests ~286 configurations with finer granularity around a center point
- **ensemble**: Tests different ensemble weights (mean top-k vs. max risk balance)
- **custom**: Full control - specify exact values for all parameters

**Recommendation:** If auto mode gives poor AUROC (<0.65), try **random** or **wide** mode to escape local optima.

**Note**: Requires simulation file(s) with ground truth labels (reward info). Random search takes 10-15 minutes for 1000 configs, wide search takes 25-30 minutes.

## 📝 Code Examples

### Calculate Single-Step Uncertainty (U_i)

```python
from tau2.metrics.uncertainty import calculate_normalized_entropy

# Your message with logprobs
message = {
    "role": "assistant",
    "content": "I can help you with that.",
    "logprobs": {
        "content": [
            {"token": "I", "logprob": -0.074},
            {"token": " can", "logprob": -0.0008},
            # ... more tokens
        ]
    }
}

# Calculate uncertainty
ui = calculate_normalized_entropy(message['logprobs'])
print(f"Uncertainty (U_i): {ui:.4f}")
```


### Diagnose and optimize (ablation study)
```bash
# Full ablation study (all variants)
python -m tau2.scripts.diagnose_and_optimize_tracer \
  data/uncertainty/my_simulation_results.json

# Diagnostic only (fast check)
python -m tau2.scripts.diagnose_and_optimize_tracer \
  data/uncertainty/my_simulation_results.json \
  --diagnostic-only

# Test specific variant
python -m tau2.scripts.diagnose_and_optimize_tracer \
  data/uncertainty/my_simulation_results.json \
  --variant max
```

### Calculate Semantic Distance Metrics

```python
from tau2.metrics.uncertainty import (
    calculate_inquiry_drift,
    calculate_inference_gap,
    calculate_semantic_distance
)

# Calculate Inquiry Drift (Da)
initial_goal = "Cancel my flight reservation"
conversation = [
    "Hi, I need to cancel a reservation",
    "Sure, can you provide your booking ID?",
    "It's ABC123"
]
da_score = calculate_inquiry_drift(initial_goal, conversation)
print(f"Inquiry Drift (Da): {da_score:.4f}")

# Calculate Inference Gap (Do) - Agent Coherence
agent_action = "Tool: get_customer_info(customer_id='C1001')"
observation = "Customer: John Smith, Phone: 555-1234"
do_score = calculate_inference_gap(agent_action, observation)
print(f"Inference Gap (Do): {do_score:.4f}")

# Calculate Inference Gap (Do) - User Coherence
agent_request = "Please provide your customer ID"
user_response = "My ID is C1001"
do_score = calculate_inference_gap(agent_request, user_response)
print(f"User Coherence (Do): {do_score:.4f}")
```

### Calculate TRACER Trajectory Score

```python
from tau2.metrics.uncertainty import TRACERConfig, calculate_tracer_score

# Step data with all metrics
steps = [
    {'ui': 0.1, 'da': 0.2, 'do_agent': 0.3, 'do_user': None},
    {'ui': 0.15, 'da': 0.25, 'do_agent': None, 'do_user': 0.35},
    {'ui': 0.2, 'da': 0.3, 'do_agent': 0.4, 'do_user': None}
]

# Calculate with default config (α=4.0, β=4.0, γ=5.0)
result = calculate_tracer_score(steps)
print(f"TRACER Score: {result['tracer_score']:.4f}")
print(f"Mean Penalty: {result['mean_penalty']:.4f}")
print(f"Steps: {result['num_steps']}")

# Calculate with custom weights
config = TRACERConfig(alpha=2.0, beta=1.0, gamma=0.5)
result = calculate_tracer_score(steps, config)
print(f"TRACER Score (custom): {result['tracer_score']:.4f}")
```

### Get Detailed Uncertainty Statistics

```python
from tau2.metrics.uncertainty import get_uncertainty_stats

stats = get_uncertainty_stats(message['logprobs'])
print(f"Normalized Entropy: {stats.normalized_entropy:.4f}")
print(f"Token Count: {stats.token_count}")
print(f"Mean Probability: {stats.mean_probability:.4f}")
print(f"Max Uncertainty: {stats.max_uncertainty:.4f}")
```

### Access All Metrics from Real-Time Simulation

If you ran simulation with `--calculate-uncertainty`, all TRACER metrics are embedded:

```python
from tau2.data_model.simulation import Results

# Load simulation (run with --calculate-uncertainty)
results = Results.load("data/simulations/your_file.json")

# Access TRACER trajectory score
for sim in results.simulations:
    if sim.tracer_metrics:
        print(f"Task {sim.task_id}:")
        print(f"  TRACER Score: {sim.tracer_metrics['tracer_score']:.4f}")
        print(f"  Mean Penalty: {sim.tracer_metrics['mean_penalty']:.4f}")
        print(f"  Steps: {sim.tracer_metrics['num_steps']}")
        
        # Ground truth correlation
        if sim.reward_info:
            status = "✅ Pass" if sim.reward_info.reward == 1.0 else "❌ Fail"
            print(f"  Result: {status}")
    
    # Access step-level metrics
    for msg in sim.messages:
        # Single-step uncertainty (U_i)
        if msg.uncertainty is not None:
            u = msg.uncertainty
            print(f"{msg.role}:")
            print(f"  U_i: {u['normalized_entropy']:.4f}")
            print(f"  Tokens: {u['token_count']}")
            print(f"  Mean probability: {u['mean_probability']:.4f}")
        
        # Semantic distance metrics (Da, Do)
        if hasattr(msg, 'da_score') and msg.da_score is not None:
            print(f"  Da (Inquiry Drift): {msg.da_score:.4f}")
        
        if hasattr(msg, 'do_score') and msg.do_score is not None:
            print(f"  Do (Inference Gap): {msg.do_score:.4f}")
            print(f"  Do Type: {msg.do_type}")
```

### Analyze a Simulation File (Post-Processing)

For simulations run WITHOUT `--calculate-uncertainty` (only U_i, no Da/Do):

```python
from pathlib import Path
from tau2.data_model.simulation import Results
from tau2.scripts.analyze_uncertainty import analyze_results

# Load simulation
sim_path = Path("data/simulations/your_file.json")
results = Results.load(sim_path)

# Analyze (extracts U_i from logprobs, but Da/Do require real-time calculation)
analysis = analyze_results(results, verbose=True)

# Access results
for sim in analysis.results:
    print(f"Task: {sim.task_id}")
    print(f"  Agent uncertainty (U_i): {sim.summary['mean_uncertainty_agent']:.4f}")
    print(f"  User uncertainty (U_i): {sim.summary['mean_uncertainty_user']:.4f}")
    
    # Semantic distance metrics (only if run with --calculate-uncertainty)
    if sim.summary.get('mean_da_score'):
        print(f"  Inquiry Drift (Da): {sim.summary['mean_da_score']:.4f}")
    if sim.summary.get('mean_do_score'):
        print(f"  Inference Gap (Do): {sim.summary['mean_do_score']:.4f}")
```

### Process Multiple Files

```python
from pathlib import Path
from tau2.data_model.simulation import Results
from tau2.scripts.analyze_uncertainty import analyze_results
import json

sim_dir = Path("data/simulations")
results_dir = Path("results/uncertainty")
results_dir.mkdir(exist_ok=True)

for sim_file in sim_dir.glob("*.json"):
    print(f"Processing {sim_file.name}...")
    
    # Load and analyze
    results = Results.load(sim_file)
    analysis = analyze_results(results, verbose=True)
    
    # Save
    output_file = results_dir / f"uncertainty_{sim_file.name}"
    with open(output_file, 'w') as f:
        json.dump(analysis.model_dump(), f, indent=2)
    
    print(f"  ✅ Saved to {output_file}")
```

## 📁 File Locations

| What | Where |
|------|-------|
| Core metrics (U_i, Da, Do) | `src/tau2/metrics/uncertainty.py` |
| TRACER aggregation | `src/tau2/metrics/uncertainty.py` |
| AUROC evaluation | `src/tau2/scripts/analyze_uncertainty.py` |
| Real-time calculation | `src/tau2/orchestrator/orchestrator.py` |
| Data models (messages) | `src/tau2/data_model/message.py` |
| Data models (simulations) | `src/tau2/data_model/simulation.py` |
| CLI analysis tool | `src/tau2/scripts/analyze_uncertainty.py` |
| Tests (pytest) | `tests/test_uncertainty.py` |
| Tests (simple) | `tests/run_uncertainty_tests.py` |

## 🔧 Import Patterns

```python
# Import uncertainty functions
from tau2.metrics.uncertainty import (
    calculate_normalized_entropy,
    get_uncertainty_stats,
    calculate_token_uncertainties,
)

# Import semantic distance functions
from tau2.metrics.uncertainty import (
    calculate_inquiry_drift,
    calculate_inference_gap,
    calculate_semantic_distance,
    EmbeddingService,
)

# Import TRACER aggregation functions
from tau2.metrics.uncertainty import (
    TRACERConfig,
    calculate_situational_weight,
    calculate_tracer_score,
    calculate_tracer_from_trajectory,
)

# Import data models
from tau2.metrics.uncertainty import TokenUncertainty, UncertaintyStats
```

## 📊 Understanding Metric Values

### U_i (Single-Step Uncertainty)

| Range | Interpretation | Typical Examples |
|-------|----------------|------------------|
| 0.00 - 0.10 | Very confident | IDs, confirmations, facts |
| 0.10 - 0.30 | Normal | Standard conversation |
| 0.30 - 0.60 | Moderate uncertainty | Complex reasoning |
| 0.60 - 1.00 | High uncertainty | Confusion, ambiguity |
| > 1.00 | Very high uncertainty | Model struggling |

### Da (Inquiry Drift)

| Range | Interpretation | Action |
|-------|----------------|--------|
| 0.00 - 0.30 | Focused on goal | ✅ On track |
| 0.30 - 0.60 | Moderate drift | ⚠️ Monitor |
| > 0.60 | Significant drift | 🚨 Refocus conversation |

### Do (Inference Gap)

| Range | Interpretation | Meaning |
|-------|----------------|---------|
| 0.00 - 0.30 | High coherence | Action matched expectation |
| 0.30 - 0.60 | Moderate gap | Some mismatch |
| > 0.60 | Poor coherence | Action failed or misaligned |

**Do Types**:
- **agent_coherence**: Distance between agent's tool call and observation
- **user_coherence**: Distance between agent's request and user's response

### TRACER Score (Trajectory-Level)

The TRACER score aggregates U_i, Da, and Do into a single trajectory score using top-k aggregation with ensemble:

**Formula**: `TRACER = (1-w) × mean(top_k%(U_i + Penalty_i)) + w × max(all_risks)`

Where `Penalty_i = α·Da_i + β·Do_agent_i + γ·Do_user_i` (default: α=4.0, β=4.0, γ=5.0)

| Range | Interpretation | Typical Scenarios |
|-------|----------------|-------------------|
| 0.00 - 0.10 | Very low uncertainty | Successful, confident trajectories |
| 0.10 - 0.30 | Moderate uncertainty | Normal operation, some complexity |
| 0.30 - 0.50 | High uncertainty | Challenging tasks, confusion |
| > 0.50 | Very high uncertainty | Critical issues, likely failure |

**Key Insights**:
- Lower TRACER → More confident agent, better situational awareness
- Higher TRACER → Less confident agent, poor situational awareness
- Compare passed vs failed tasks to find predictive thresholds
- Adjust α, β, γ weights based on domain importance

### AUROC (Predictive Power)

AUROC measures how well TRACER scores predict task failure (1.0 = perfect, 0.5 = random):

| AUROC Range | Interpretation | Recommendation |
|-------------|----------------|----------------|
| 0.90 - 1.00 | Excellent | Deploy TRACER for failure prediction |
| 0.80 - 0.90 | Good | Use with confidence monitoring |
| 0.70 - 0.80 | Fair | Combine with other signals |
| 0.60 - 0.70 | Poor | Tune α,β,γ or collect more data |
| < 0.60 | Very poor | Hypothesis may not hold |

**Interpretation Tips**:
- AUROC > 0.7 → TRACER can reliably predict failures
- Use optimal threshold from evaluation to flag risky tasks
- Compare mean TRACER scores for passed vs failed to validate
- Need 50+ samples for statistically robust AUROC
- **Automatically calculated** when analyzing sims with ground truth data

## 🐛 Troubleshooting

### Import Error: "No module named 'tau2'"

**Solution**: Make sure you're in the project directory and the package is installed:
```bash
cd /path/to/agent-uncertainty
pip install -e .
# or
pdm install
```

### Tests Fail

**Solution**: Use the simple test runner that doesn't require pytest:
```bash
python tests/run_uncertainty_tests.py
```

### CLI Script Doesn't Run

**Solution**: Run with PYTHONPATH:
```bash
PYTHONPATH=src python -m tau2.scripts.analyze_uncertainty simulation.json
```

## 💡 Tips

1. **Batch Processing**: Use a loop to process all simulations in a directory
2. **Filtering**: Focus on high-uncertainty turns (U_i > 0.5) to find problems
3. **Comparison**: Compare agent vs user uncertainty ratios
4. **Visualization**: Export to JSON and visualize with matplotlib/seaborn
5. **Thresholds**: Set domain-specific thresholds for intervention
6. **Semantic Distance**: Use `--calculate-uncertainty` for Da/Do metrics (requires Vertex AI)
7. **Cost Management**: Da/Do metrics require embedding API calls (~$0.0001 per call)
8. **Combined Analysis**: Look at U_i + Da + Do together for complete picture
9. **TRACER Scores**: Use trajectory-level TRACER score to compare models and predict task outcomes
10. **Custom Weights**: Adjust α, β, γ based on what matters in your domain (e.g., α=2.0 to emphasize goal-tracking)
11. **Failure Prediction**: AUROC is automatically calculated when you analyze simulations with ground truth
12. **Threshold Tuning**: Check the optimal_threshold in AUROC metrics to find your failure cutoff
13. **Statistical Power**: Need 50+ samples for robust AUROC; combine multiple benchmark runs
14. **Interpret AUROC**: AUROC > 0.7 = good predictor, < 0.6 = poor (may need different weights)

## 📈 Common Analysis Patterns

### Find High-Uncertainty Moments
```python
for sim in analysis.results:
    for turn in sim.uncertainty_scores:
        if turn.ui_score > 0.5:
            print(f"⚠️  High uncertainty at turn {turn.turn}")
            print(f"   Actor: {turn.actor}, U_i: {turn.ui_score:.4f}")
```

### Find Goal Drift (High Da)
```python
for sim in analysis.results:
    for turn in sim.uncertainty_scores:
        if turn.da_score and turn.da_score > 0.6:
            print(f"🎯 Goal drift detected at turn {turn.turn}")
            print(f"   Da: {turn.da_score:.4f}")
            print(f"   Content: {turn.content_preview}")
```

### Find Coordination Failures (High Do)
```python
for sim in analysis.results:
    for turn in sim.uncertainty_scores:
        if turn.do_score and turn.do_score > 0.6:
            print(f"💥 Coherence issue at turn {turn.turn}")
            print(f"   Do: {turn.do_score:.4f} ({turn.do_type})")
            print(f"   Content: {turn.content_preview}")
```

### Combined Metric Analysis
```python
for sim in analysis.results:
    for turn in sim.uncertainty_scores:
        # Critical situation: uncertain AND drifting AND incoherent
        if (turn.ui_score > 0.5 and 
            turn.da_score and turn.da_score > 0.6 and
            turn.do_score and turn.do_score > 0.6):
            print(f"🚨 CRITICAL at turn {turn.turn}")
            print(f"   U_i: {turn.ui_score:.4f}, Da: {turn.da_score:.4f}, Do: {turn.do_score:.4f}")
```

### Compare Models
```python
# After analyzing simulations with different models
gpt4_analysis = analyze_results(gpt4_results)
gemini_analysis = analyze_results(gemini_results)

print(f"GPT-4 agent uncertainty: {gpt4_analysis.results[0].summary['mean_uncertainty_agent']:.4f}")
print(f"Gemini agent uncertainty: {gemini_analysis.results[0].summary['mean_uncertainty_agent']:.4f}")

# Compare semantic metrics if available
if gpt4_analysis.results[0].summary.get('mean_da_score'):
    print(f"GPT-4 inquiry drift: {gpt4_analysis.results[0].summary['mean_da_score']:.4f}")
    print(f"Gemini inquiry drift: {gemini_analysis.results[0].summary['mean_da_score']:.4f}")
```

### Track All Metrics Over Trajectory
```python
sim = analysis.results[0]
print("Turn | Actor | U_i  | Da   | Do   | Do Type")
print("-" * 60)
for turn in sim.uncertainty_scores:
    da = f"{turn.da_score:.3f}" if turn.da_score else "  —  "
    do = f"{turn.do_score:.3f}" if turn.do_score else "  —  "
    do_type = turn.do_type[:12] if turn.do_type else "—"
    print(f"{turn.turn:4d} | {turn.actor:5s} | {turn.ui_score:.3f} | {da} | {do} | {do_type}")
```

### Analyze TRACER Scores Across Tasks
```python
from tau2.data_model.simulation import Results

results = Results.load("data/simulations/your_file.json")

# Collect TRACER scores by outcome
passed_tracer = []
failed_tracer = []

for sim in results.simulations:
    if sim.tracer_metrics and sim.reward_info:
        tracer = sim.tracer_metrics['tracer_score']
        if sim.reward_info.reward == 1.0:
            passed_tracer.append(tracer)
        else:
            failed_tracer.append(tracer)

# Compare
import numpy as np
print(f"Passed tasks - Mean TRACER: {np.mean(passed_tracer):.4f}")
print(f"Failed tasks - Mean TRACER: {np.mean(failed_tracer):.4f}")
print(f"Difference: {np.mean(failed_tracer) - np.mean(passed_tracer):.4f}")
```

### Custom TRACER Weighting
```python
from tau2.metrics.uncertainty import TRACERConfig, calculate_tracer_from_trajectory

# Emphasize inquiry drift (higher alpha weight)
config_high_da = TRACERConfig(alpha=6.0, beta=4.0, gamma=5.0)

# Ignore user coherence (focus on agent)
config_agent_only = TRACERConfig(alpha=4.0, beta=4.0, gamma=0.0)

# Calculate with different configs
for sim in results.simulations:
    tracer_default = calculate_tracer_from_trajectory(sim.messages)
    tracer_high_da = calculate_tracer_from_trajectory(sim.messages, config_high_da)
    
    print(f"Task {sim.task_id}:")
    print(f"  Default:  {tracer_default['tracer_score']:.4f}")
    print(f"  High Da:  {tracer_high_da['tracer_score']:.4f}")
```

### Access AUROC Metrics (Failure Prediction)
```python
import json
from pathlib import Path

# Load analysis results (includes AUROC automatically)
with open('data/uncertainty/your_analysis.json', 'r') as f:
    analysis = json.load(f)

# Access AUROC metrics
if analysis.get('auroc_metrics'):
    auroc = analysis['auroc_metrics']
    print(f"AUROC: {auroc['auroc']:.4f}")
    print(f"Accuracy: {auroc['accuracy']:.4f}")
    print(f"Optimal threshold: {auroc['optimal_threshold']:.4f}")
    print(f"Precision: {auroc['precision']:.4f}")
    print(f"Recall: {auroc['recall']:.4f}")
    print(f"F1 Score: {auroc['f1_score']:.4f}")
    
    # Check if TRACER is predictive
    if auroc['auroc'] > 0.7:
        print(f"✅ TRACER has good predictive power!")
        print(f"Flag tasks with TRACER > {auroc['optimal_threshold']:.4f}")
    else:
        print(f"⚠️  TRACER needs more data or weight tuning")
    
    # Score distributions
    print(f"\nFailed tasks: TRACER = {auroc['mean_tracer_failures']:.4f}")
    print(f"Passed tasks: TRACER = {auroc['mean_tracer_successes']:.4f}")
else:
    print("No AUROC metrics (need TRACER scores + ground truth)")
```

---

## 🎓 Understanding the Metrics

### The Four-Layer TRACER System

1. **U_i (Token-Level)**: How confident is the model when generating text?
2. **Da (Goal-Level)**: Is the conversation drifting from the original goal?
3. **Do (Action-Level)**: Do actions match their expected outcomes?
4. **TRACER (Trajectory-Level)**: Overall uncertainty considering all factors

### When to Use Each Metric

- **U_i alone**: Fast, real-time confidence monitoring
- **Da + U_i**: Detect when uncertain model loses focus
- **Do + U_i**: Detect when actions don't match expectations
- **TRACER**: Single score to compare trajectories and predict outcomes
- **All together**: Complete situational awareness picture

### Practical Interpretation

**Scenario 1**: Low U_i, Low Da, Low Do → Low TRACER = ✅ Perfect (score < 2.0)
**Scenario 2**: High U_i, Low Da, Low Do → Moderate TRACER = ⚠️ Uncertain but on track (score 2.0-4.0)
**Scenario 3**: Low U_i, High Da, Low Do → Moderate TRACER = ⚠️ Confident but lost (score 2.0-4.0)
**Scenario 4**: Any High Do → High TRACER = 🚨 Coordination problem (score > 4.0)
**Scenario 5**: All High → Very High TRACER = 🚨 Critical failure (score > 6.0)

**Using TRACER for Decision Making**:
- Set threshold based on your domain (e.g., 4.0)
- If TRACER > threshold → Trigger intervention, human handoff, or retry
- Track TRACER over time to detect degradation
- Compare TRACER between different agent models or prompts

---

**For detailed documentation, see**: `STEP2_UNCERTAINTY_CALCULATION.md`  
**For TRACER framework details, see**: Research paper on Task Risk Assessment via Contextual Entropy and Reasoning

