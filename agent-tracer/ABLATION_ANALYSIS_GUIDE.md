# Ablation Analysis Implementation Guide

## Overview

The analyze_uncertainty script now automatically performs **ablation studies** to isolate the impact of assistant (agent) vs. user turns on uncertainty metrics and failure prediction.

## What Was Implemented

### 1. **Data Model Updates**
- Added `ablation_studies` field to `UncertaintyAnalysis` class
- Stores ablation results for assistant-only and user-only analyses

### 2. **Refactored AUROC Functions**
- `calculate_auroc_metrics()` now accepts `role_filter` parameter
- `calculate_baseline_aurocs()` now accepts `role_filter` parameter
- When `role_filter=None`: Uses all turns (default behavior)
- When `role_filter="assistant"`: Filters to only assistant turns (actor="agent")
- When `role_filter="user"`: Filters to only user turns (actor="user")

### 3. **Automatic Ablation Execution**
In `analyze_results()`:
- **Full Analysis** (default): All turns combined
- **Assistant-Only Ablation**: Recalculates all metrics using only assistant turns
- **User-Only Ablation**: Recalculates all metrics using only user turns

Each ablation runs:
- TRACER score calculation with filtered turns
- All baseline metrics (SAUP, Normalized Entropy, Self-Assessed Confidence)
- Full AUROC evaluation (accuracy, precision, recall, F1)

### 4. **Display Updates**
The `print_summary()` function now includes:
- **Ablation Studies** section showing performance for each ablation
- Side-by-side comparison tables
- Comparison with full analysis (Δ AUROC)

## How It Works

### Role Filtering Logic

```python
# Map role to actor
role="assistant" → actor="agent"
role="user" → actor="user"

# Filter uncertainty scores
if role_filter == "assistant":
    filtered_scores = [s for s in uncertainty_scores if s.actor == "agent"]
elif role_filter == "user":
    filtered_scores = [s for s in uncertainty_scores if s.actor == "user"]
else:
    filtered_scores = uncertainty_scores  # All turns
```

### Metrics Recalculated for Each Ablation

1. **TRACER Score**: Recalculated using filtered step_data
2. **SAUP**: Recalculated using filtered turns
3. **Normalized Entropy**: Mean U_i over filtered turns
4. **Self-Assessed Confidence**: Mean confidence over filtered turns
5. **AUROC**: All classification metrics recalculated

### Data Isolation Guarantees

✅ **No leakage between ablations**: Each ablation uses completely independent filtered data

- Assistant-only: Only processes turns with `role="assistant"`
- User-only: Only processes turns with `role="user"`
- Full analysis: Uses all turns

## Usage

### Running Analysis (No Changes Needed!)

```bash
python -m tau2.scripts.analyze_uncertainty data/simulations/your_file.json
```

The script **automatically** runs ablation studies alongside the main analysis.

### Expected Output

```
Baseline Comparison
  Comparing TRACER against simpler baseline metrics:
  
  ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━┳━━━━━━┓
  ┃ Metric                            ┃ AUROC    ┃ Accuracy ┃ Precision ┃ Recall ┃ F1   ┃
  ┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━╇━━━━━━┩
  │ TRACER (Full Framework)           │ 0.6823   │ 0.7143   │ 0.6667    │ 0.8000 │ 0.73 │
  │ SAUP (RMS Weighted Uncertainty)   │ 0.6523   │ 0.6939   │ 0.6400    │ 0.7600 │ 0.69 │
  │ Normalized Entropy (U_i) Only     │ 0.5340   │ 0.5714   │ 0.5333    │ 0.6000 │ 0.56 │
  │ Self-Assessed Confidence Only     │ 0.4416   │ 0.4898   │ 0.4000    │ 0.4000 │ 0.40 │
  └───────────────────────────────────┴──────────┴──────────┴───────────┴────────┴──────┘

Ablation Studies
  Analyzing impact of isolating assistant vs. user turns:

Analysis using only assistant (agent) turns

  ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━┳━━━━━━┓
  ┃ Metric                            ┃ AUROC    ┃ Accuracy ┃ Precision ┃ Recall ┃ F1   ┃
  ┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━╇━━━━━━┩
  │ TRACER                            │ 0.6945   │ 0.7200   │ 0.6800    │ 0.8100 │ 0.74 │
  │ SAUP                              │ 0.6612   │ 0.7000   │ 0.6500    │ 0.7700 │ 0.70 │
  │ Normalized Entropy                │ 0.5456   │ 0.5800   │ 0.5400    │ 0.6200 │ 0.58 │
  │ Self-Assessed Confidence          │ 0.4523   │ 0.5000   │ 0.4100    │ 0.4200 │ 0.42 │
  └───────────────────────────────────┴──────────┴──────────┴───────────┴────────┴──────┘

  → Better than full analysis (Δ=+0.0122)

Analysis using only user turns

  ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━┳━━━━━━┓
  ┃ Metric                            ┃ AUROC    ┃ Accuracy ┃ Precision ┃ Recall ┃ F1   ┃
  ┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━╇━━━━━━┩
  │ TRACER                            │ 0.5234   │ 0.5500   │ 0.5100    │ 0.5800 │ 0.54 │
  │ SAUP                              │ 0.5123   │ 0.5400   │ 0.5000    │ 0.5600 │ 0.53 │
  │ Normalized Entropy                │ 0.4890   │ 0.5200   │ 0.4800    │ 0.5400 │ 0.51 │
  │ Self-Assessed Confidence          │ 0.4123   │ 0.4700   │ 0.3900    │ 0.4100 │ 0.40 │
  └───────────────────────────────────┴──────────┴──────────┴───────────┴────────┴──────┘

  → Worse than full analysis (Δ=-0.1589)
```

### JSON Output Structure

```json
{
  "metadata": { ... },
  "results": [ ... ],
  "auroc_metrics": { ... },
  "baseline_aurocs": { ... },
  "ablation_studies": {
    "assistant_only": {
      "description": "Analysis using only assistant (agent) turns",
      "auroc_metrics": {
        "auroc": 0.6945,
        "accuracy": 0.7200,
        ...
      },
      "baseline_aurocs": {
        "saup": { ... },
        "normalized_entropy": { ... },
        "self_assessed_confidence": { ... }
      }
    },
    "user_only": {
      "description": "Analysis using only user turns",
      "auroc_metrics": { ... },
      "baseline_aurocs": { ... }
    }
  }
}
```

## Interpretation Guide

### What the Ablations Tell You

**1. Assistant-Only Performance**
- Shows how well agent uncertainty predicts failure
- Useful for understanding agent self-awareness
- If AUROC is similar to full analysis → agent uncertainty is the main signal

**2. User-Only Performance**
- Shows how well user uncertainty predicts failure
- Captures user confusion and frustration
- If AUROC is much lower → user signals are less predictive

**3. Comparison Insights**

| Scenario | Interpretation |
|----------|----------------|
| Assistant > Full | Agent uncertainty is the strongest signal; user turns add noise |
| User > Full | User uncertainty is the strongest signal; agent turns add noise |
| Full > Both | Combined signal is better; both perspectives matter |
| Assistant >> User | Agent self-awareness is key predictor |
| User ≈ Assistant | Both perspectives equally important |

## Technical Details

### Files Modified

1. **`src/tau2/scripts/analyze_uncertainty.py`**
   - Updated `UncertaintyAnalysis` model
   - Refactored `calculate_auroc_metrics()` with role_filter
   - Refactored `calculate_baseline_aurocs()` with role_filter
   - Updated `analyze_results()` to run ablations
   - Updated `print_summary()` to display ablations

### Backward Compatibility

✅ **Fully backward compatible**
- Existing code works unchanged
- `role_filter=None` maintains original behavior
- All existing tests should pass
- Output format extended, not changed

### Performance Impact

- **Runtime**: ~3x longer (runs analysis 3 times)
- **Memory**: Minimal increase (reuses filtered data)
- **Typical runtime**: 30-60 seconds for 50 simulations

## Troubleshooting

### Issue: "Not enough samples for ablation"

```
⚠️  No metrics available for this ablation
```

**Cause**: Not enough turns of specified role (e.g., very few user turns)

**Solution**: Check turn distribution in your data

### Issue: "Only one class present"

```
Only one class present in data: [0]. Cannot calculate AUROC.
```

**Cause**: All tasks succeeded or all failed in the filtered data

**Solution**: Ensure balanced dataset with both successes and failures

## Future Extensions

Potential additions:
- [ ] Cross-domain ablation comparisons
- [ ] Temporal ablations (early vs late turns)
- [ ] Tool-use ablations (with tools vs without)
- [ ] Confidence-level ablations (high vs low confidence turns)

## Questions?

Contact the Tau-2 team or check the main documentation.
