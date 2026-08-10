#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Run ablation studies for Trajectory_TAU.

All variants are evaluated as failure-detection scores:
    higher score => more likely failure.

Usage:
    python evaluate_trajectory_tau_ablation.py jobs/2026-07-01__10-00-09
    python evaluate_trajectory_tau_ablation.py --root jobs/2026-07-01__10-00-09 --variants full,no_graph,ui_only

Output:
    <root>/trajectory_tau_ablation_summary.json
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from evaluate_trajectory_tau_graph_experiments import (
    DEFAULT_ROOT,
    TrajectoryTAUConfig,
    evaluate_failure_scores,
    finite_float,
    load_dataset,
    trajectory_tau_module,
)


calculate_trajectory_tau_score = trajectory_tau_module.calculate_trajectory_tau_score


ABLATION_OVERRIDES: dict[str, dict[str, Any]] = {
    "full": {},
    "ui_only": {
        "alpha": 0.0,
        "beta": 0.0,
        "graph_weight": 0.0,
        "graph_uncertainty_weight": 0.0,
    },
    "no_graph": {
        "graph_weight": 0.0,
        "graph_uncertainty_weight": 0.0,
    },
    "no_graph_propagation": {
        "graph_weight": 0.0,
    },
    "no_graph_effective_ui": {
        "graph_uncertainty_weight": 0.0,
    },
    "no_trajectory_propagation": {
        "alpha": 0.0,
        "graph_weight": 0.0,
    },
    "no_structured_history": {
        "momentum_weight": 0.0,
        "repetition_weight": 0.0,
        "observation_weight": 0.0,
        "stagnation_weight": 0.0,
    },
    "no_momentum": {
        "momentum_weight": 0.0,
    },
    "no_repetition": {
        "repetition_weight": 0.0,
    },
    "no_observation": {
        "observation_weight": 0.0,
    },
    "no_stagnation": {
        "stagnation_weight": 0.0,
    },
    "no_interaction_gap": {
        "beta": 0.0,
    },
    "no_goal_interaction": {
        "interaction_goal_weight": 0.0,
    },
    "no_user_interaction": {
        "interaction_user_weight": 0.0,
    },
}


DEFAULT_VARIANTS = [
    "full",
    "ui_only",
    "no_graph",
    "no_graph_propagation",
    "no_graph_effective_ui",
    "no_trajectory_propagation",
    "no_structured_history",
    "no_momentum",
    "no_repetition",
    "no_observation",
    "no_stagnation",
    "no_interaction_gap",
    "no_goal_interaction",
    "no_user_interaction",
]


def save_json(path: Path, data: Any) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [json_safe(v) for v in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def parse_csv_list(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def build_config(base_config: TrajectoryTAUConfig, overrides: dict[str, Any]) -> TrajectoryTAUConfig:
    values = asdict(trajectory_tau_module._normalize_config(base_config))
    values.update(overrides)
    return TrajectoryTAUConfig(**values)


def summarize_tau_info(tau_info: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "tau_score",
        "tau_confidence",
        "num_steps",
        "mean_risk",
        "max_risk",
        "mean_propagation",
        "mean_combined_propagation",
        "mean_graph_uncertainty",
        "mean_effective_ui",
        "mean_interaction_gap",
    ]
    return {key: finite_float(tau_info.get(key)) for key in keys}


def run_ablation(
    dataset: list[dict[str, Any]],
    base_config: TrajectoryTAUConfig,
    variants: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    variant_configs = {
        variant: build_config(base_config, ABLATION_OVERRIDES[variant])
        for variant in variants
    }
    rows: list[dict[str, Any]] = []

    for item in dataset:
        row: dict[str, Any] = {
            "trial": item["trial"],
            "reward": item["reward"],
            "success": item["success"],
            "failure": item["failure"],
            "exception_type": item["exception_type"],
        }
        for variant, config in variant_configs.items():
            tau_info = calculate_trajectory_tau_score(item["steps"], item["goal_text"], config)
            summary = summarize_tau_info(tau_info)
            row[f"{variant}_score"] = summary["tau_score"]
            row[f"{variant}_num_steps"] = summary["num_steps"]
            row[f"{variant}_mean_propagation"] = summary["mean_propagation"]
            row[f"{variant}_mean_combined_propagation"] = summary["mean_combined_propagation"]
            row[f"{variant}_mean_graph_uncertainty"] = summary["mean_graph_uncertainty"]
            row[f"{variant}_mean_effective_ui"] = summary["mean_effective_ui"]
            row[f"{variant}_mean_interaction_gap"] = summary["mean_interaction_gap"]
        rows.append(row)

    trial_df = pd.DataFrame(rows)
    metric_rows: list[dict[str, Any]] = []
    for variant in variants:
        score_col = f"{variant}_score"
        metrics = evaluate_failure_scores(trial_df, score_col)
        metrics.update(
            {
                "variant": variant,
                "score_column": score_col,
                "description": describe_variant(variant),
                "overrides": ABLATION_OVERRIDES[variant],
            }
        )
        for key in (
            "mean_propagation",
            "mean_combined_propagation",
            "mean_graph_uncertainty",
            "mean_effective_ui",
            "mean_interaction_gap",
        ):
            col = f"{variant}_{key}"
            metrics[f"avg_{key}"] = finite_float(trial_df[col].mean()) if col in trial_df else None
        metric_rows.append(metrics)

    metrics_df = pd.DataFrame(metric_rows)
    if "full" in variants:
        full_row = metrics_df[metrics_df["variant"] == "full"].iloc[0]
        metrics_df["delta_auroc_vs_full"] = metrics_df["auroc"] - float(full_row["auroc"])
        metrics_df["delta_auprc_vs_full"] = metrics_df["auprc"] - float(full_row["auprc"])
        metrics_df["delta_f1_vs_full"] = metrics_df["f1_at_threshold"] - float(full_row["f1_at_threshold"])
    else:
        metrics_df["delta_auroc_vs_full"] = None
        metrics_df["delta_auprc_vs_full"] = None
        metrics_df["delta_f1_vs_full"] = None

    config_payload = {
        variant: {
            "overrides": ABLATION_OVERRIDES[variant],
            "resolved_config": asdict(variant_configs[variant]),
        }
        for variant in variants
    }
    return trial_df, metrics_df, config_payload


def describe_variant(variant: str) -> str:
    descriptions = {
        "full": "Complete Trajectory_TAU with graph uncertainty propagation and interaction modeling.",
        "ui_only": "Only token/message uncertainty is used; trajectory propagation, graph propagation, and interaction gap are disabled.",
        "no_graph": "Graph propagation is fully disabled in both propagation and effective uncertainty.",
        "no_graph_propagation": "Graph uncertainty is removed from the propagation term but still contributes to effective uncertainty.",
        "no_graph_effective_ui": "Graph uncertainty is removed from effective uncertainty but still contributes to propagation.",
        "no_trajectory_propagation": "The alpha-weighted trajectory propagation term is disabled.",
        "no_structured_history": "Momentum, repetition, observation, and stagnation history components are disabled.",
        "no_momentum": "The temporal uncertainty momentum component is disabled.",
        "no_repetition": "The tool/action repetition component is disabled.",
        "no_observation": "The environment observation instability component is disabled.",
        "no_stagnation": "The action stagnation component is disabled.",
        "no_interaction_gap": "The beta-weighted goal/user interaction gap term is disabled.",
        "no_goal_interaction": "Goal coverage contribution inside the interaction gap is disabled.",
        "no_user_interaction": "Latest-user feedback contribution inside the interaction gap is disabled.",
    }
    return descriptions.get(variant, variant)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Trajectory_TAU ablation experiments.")
    parser.add_argument("root", nargs="?", type=Path, default=None)
    parser.add_argument("--root", dest="root_flag", type=Path, default=None)
    parser.add_argument("--eval-key", default=None)
    parser.add_argument("--include-exceptions", action="store_true")
    parser.add_argument(
        "--variants",
        default=",".join(DEFAULT_VARIANTS),
        help=f"Comma-separated variants. Available: {','.join(ABLATION_OVERRIDES)}",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Output summary JSON path. Default: <root>/trajectory_tau_ablation_summary.json",
    )
    parser.add_argument(
        "--include-trial-scores",
        action="store_true",
        help="Include per-trial scores in the summary JSON.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = (args.root_flag or args.root or DEFAULT_ROOT).expanduser().resolve()
    if not (root / "result.json").exists():
        raise FileNotFoundError(f"Missing result.json: {root / 'result.json'}")

    variants = parse_csv_list(args.variants)
    unknown = [variant for variant in variants if variant not in ABLATION_OVERRIDES]
    if unknown:
        raise ValueError(f"Unknown variants: {unknown}. Available: {sorted(ABLATION_OVERRIDES)}")
    if not variants:
        raise ValueError("At least one variant is required")

    base_config = TrajectoryTAUConfig()
    dataset, dataset_info = load_dataset(root, args.eval_key, args.include_exceptions)
    if not dataset:
        raise RuntimeError("No usable trials found")

    trial_df, metrics_df, variant_configs = run_ablation(dataset, base_config, variants)
    output_json = (args.output_json or (root / "trajectory_tau_ablation_summary.json")).expanduser().resolve()
    output_json.parent.mkdir(parents=True, exist_ok=True)

    summary: dict[str, Any] = {
        "root": str(root),
        "eval_key": dataset_info["eval_key"],
        "include_exceptions": bool(args.include_exceptions),
        "positive_label": "failure",
        "score_direction": "higher_score_more_likely_failure",
        "trajectory_tau_base_config": asdict(base_config),
        "variant_configs": variant_configs,
        "counters": dataset_info["counters"],
        "variant_metrics": metrics_df.to_dict(orient="records"),
        "output_json": str(output_json),
    }
    if args.include_trial_scores:
        summary["trial_scores"] = trial_df.to_dict(orient="records")

    save_json(output_json, json_safe(summary))

    print("=" * 80)
    print("Trajectory_TAU ablation")
    print("=" * 80)
    print(f"Root: {root}")
    print(f"Eval key: {dataset_info['eval_key']}")
    print(f"Used trials: {dataset_info['counters']['used_trials']}")
    print("Score direction: higher score => more likely failure")
    print()
    display_cols = [
        "variant",
        "n",
        "n_success",
        "n_failure",
        "auroc",
        "auprc",
        "delta_auroc_vs_full",
        "delta_auprc_vs_full",
        "f1_at_threshold",
    ]
    print(metrics_df[display_cols].to_string(index=False))
    print()
    print(f"Saved summary JSON: {output_json}")


if __name__ == "__main__":
    main()
