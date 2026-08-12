#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Extended ablation suite for Trajectory_TAU.

This script is designed to support the paper's ablation section with three
complementary analyses:

1. Component ablation: remove each major method component.
2. Graph-structure ablation: delete or perturb graph edges to test whether
   the learned/constructed structure matters.
3. Hyperparameter ablation: sweep edge-family weights, edge-type weights, and
   history-propagation weights to measure sensitivity.

All variants are evaluated as failure-detection scores:
    higher score => more likely failure.

Outputs:
    <root>/trajectory_tau_extended_ablation_summary.json
"""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import asdict
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from evaluate_trajectory_tau_graph_experiments import (
    DEFAULT_ROOT,
    TrajectoryTAUConfig,
    edge_payloads_to_uncertainty,
    evaluate_failure_scores,
    finite_float,
    load_dataset,
    stable_seed,
    trajectory_tau_module,
)


calculate_trajectory_tau_score = trajectory_tau_module.calculate_trajectory_tau_score
build_dependency_edges = trajectory_tau_module.build_dependency_edges
calculate_trajectory_propagation = trajectory_tau_module.calculate_trajectory_propagation
calculate_goal_coverage_gap = trajectory_tau_module.calculate_goal_coverage_gap


COMPONENT_ABLATIONS: dict[str, dict[str, Any]] = {
    "full": {},
    "no_graph": {
        "graph_weight": 0.0,
        "graph_uncertainty_weight": 0.0,
    },
    "no_history_propagation": {
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
}


GRAPH_VARIANTS: dict[str, dict[str, Any]] = {
    "full_graph": {
        "kind": "full",
        "description": "Original trajectory graph with all dependency edges.",
    },
    "no_graph": {
        "kind": "none",
        "description": "Remove all graph edges.",
    },
    "sequential_only": {
        "kind": "keep_types",
        "keep_types": {"sequential", "latest_user"},
        "description": "Keep only sequential and latest-user dependencies.",
    },
    "drop_repetition_edges": {
        "kind": "drop_types",
        "drop_types": {"repetition", "tool_repetition"},
        "description": "Remove repetition-related edges.",
    },
    "drop_feedback_edges": {
        "kind": "drop_types",
        "drop_types": {"feedback_instability", "feedback_response"},
        "description": "Remove feedback-related edges.",
    },
    "drop_progression_edges": {
        "kind": "drop_types",
        "drop_types": {"progression"},
        "description": "Remove progression edges.",
    },
    "drop_parallel_edges": {
        "kind": "drop_types",
        "drop_types": {"parallel"},
        "description": "Remove parallel-branch edges.",
    },
    "random_dropout_50": {
        "kind": "dropout",
        "drop_rate": 0.5,
        "description": "Randomly delete 50% of edges while keeping the remaining structure.",
    },
    "random_graph": {
        "kind": "random_graph",
        "description": "Rewire edges to random historical sources while preserving edge types and weights.",
    },
}


EDGE_FAMILY_MAP = {
    "sequential": "sequential",
    "latest_user": "latest_user",
    "tool_repetition": "repetition",
    "repetition": "repetition",
    "feedback_instability": "feedback",
    "feedback_response": "feedback",
    "progression": "progression",
    "parallel": "parallel",
    "token_overlap": "token_overlap",
}


EDGE_FAMILY_SCALES = [0.0, 0.5, 1.0, 1.25, 1.5]
EDGE_TYPE_SCALES = [0.0, 0.5, 1.0, 1.25, 1.5]
HISTORY_WEIGHT_SCALES = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25]
TEMPORAL_WEIGHT_SCALES = [0.5, 0.65, 0.8, 0.9, 0.97]


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


def describe_component_variant(variant: str) -> str:
    descriptions = {
        "full": "Full Trajectory_TAU.",
        "no_graph": "Remove graph propagation terms.",
        "no_history_propagation": "Remove structured-history terms.",
        "no_momentum": "Remove uncertainty momentum.",
        "no_repetition": "Remove repetition pressure.",
        "no_observation": "Remove observation instability.",
        "no_stagnation": "Remove action stagnation.",
        "no_interaction_gap": "Remove interaction-gap term.",
    }
    return descriptions.get(variant, variant)


def describe_graph_variant(variant: str) -> str:
    return GRAPH_VARIANTS[variant]["description"]


def scale_edges(
    edges: list[Any],
    family_scale: dict[str, float],
    edge_type_scale: dict[str, float],
) -> list[Any]:
    if not family_scale and not edge_type_scale:
        return edges

    scaled_edges = []
    for edge in edges:
        family = EDGE_FAMILY_MAP.get(edge.edge_type, edge.edge_type)
        scale = float(family_scale.get(family, 1.0)) * float(edge_type_scale.get(edge.edge_type, 1.0))
        if scale <= 0.0:
            continue
        scaled_edges.append(
            trajectory_tau_module.TrajectoryGraphEdge(
                source_index=edge.source_index,
                target_index=edge.target_index,
                weight=float(edge.weight * scale),
                edge_type=edge.edge_type,
            )
        )
    return scaled_edges


def transform_edges(
    edges: list[Any],
    graph_nodes: list[Any],
    spec: dict[str, Any],
    rng: random.Random,
) -> list[Any]:
    kind = spec["kind"]
    if kind == "full":
        return edges
    if kind == "none":
        return []
    if kind == "keep_types":
        keep_types = set(spec.get("keep_types", set()))
        return [edge for edge in edges if edge.edge_type in keep_types]
    if kind == "drop_types":
        drop_types = set(spec.get("drop_types", set()))
        return [edge for edge in edges if edge.edge_type not in drop_types]
    if kind == "dropout":
        drop_rate = float(spec.get("drop_rate", 0.5))
        kept = [edge for edge in edges if rng.random() >= drop_rate]
        return kept
    if kind == "random_graph":
        if not graph_nodes:
            return []
        source_indices = [node.index for node in graph_nodes]
        randomized = []
        for edge in edges:
            randomized.append(
                trajectory_tau_module.TrajectoryGraphEdge(
                    source_index=rng.choice(source_indices),
                    target_index=edge.target_index,
                    weight=edge.weight,
                    edge_type=edge.edge_type,
                )
            )
        return randomized
    raise ValueError(f"Unknown graph variant kind: {kind}")


def calculate_graph_uncertainty_variant(
    current: Any,
    graph_nodes: list[Any],
    latest_user_index: Optional[int],
    config: Any,
    graph_variant: str,
    rng: random.Random,
    family_scale: Optional[dict[str, float]] = None,
    edge_type_scale: Optional[dict[str, float]] = None,
) -> dict[str, Any]:
    if graph_variant == "no_graph":
        return edge_payloads_to_uncertainty([], graph_nodes)

    edges = build_dependency_edges(current, graph_nodes, latest_user_index, config)
    edges = transform_edges(edges, graph_nodes, GRAPH_VARIANTS[graph_variant], rng)
    edges = scale_edges(edges, family_scale or {}, edge_type_scale or {})
    return edge_payloads_to_uncertainty(edges, graph_nodes)


def calculate_score_with_graph_variant(
    messages: list[Any],
    goal_text: str,
    config: Any,
    graph_variant: str,
    rng: random.Random,
    family_scale: Optional[dict[str, float]] = None,
    edge_type_scale: Optional[dict[str, float]] = None,
) -> dict[str, Any]:
    config = trajectory_tau_module._normalize_config(config)

    history: list[Any] = []
    graph_nodes: list[Any] = []
    latest_user = None
    latest_user_index: Optional[int] = None
    risks: list[float] = []
    per_step: list[dict[str, Any]] = []

    for msg in messages:
        state = trajectory_tau_module._build_state(msg)
        if state.role not in {"assistant", "user", "environment"}:
            continue

        if state.role == "assistant":
            propagation = calculate_trajectory_propagation(state, history, config)
            graph = calculate_graph_uncertainty_variant(
                state,
                graph_nodes,
                latest_user_index,
                config,
                graph_variant,
                rng,
                family_scale=family_scale,
                edge_type_scale=edge_type_scale,
            )
            interaction_gap = calculate_goal_coverage_gap(state, latest_user, goal_text, config)
            effective_ui = state.ui + config.graph_uncertainty_weight * graph["graph_uncertainty"]
            combined_propagation = min(
                1.5,
                propagation["propagation"] + config.graph_weight * graph["graph_uncertainty"],
            )
            risk = effective_ui * (1.0 + config.alpha * combined_propagation + config.beta * interaction_gap)

            risks.append(float(risk))
            per_step.append(
                {
                    "ui": state.ui,
                    "effective_ui": float(effective_ui),
                    "risk": float(risk),
                    "propagation": propagation["propagation"],
                    "combined_propagation": float(combined_propagation),
                    "graph_uncertainty": graph["graph_uncertainty"],
                    "graph_edge_count": graph["graph_edge_count"],
                    "graph_total_weight": graph["graph_total_weight"],
                    "graph_max_edge_weight": graph["graph_max_edge_weight"],
                    "interaction_gap": interaction_gap,
                    "momentum": propagation["momentum"],
                    "repetition": propagation["repetition"],
                    "observation": propagation["observation"],
                    "stagnation": propagation["stagnation"],
                }
            )
            node_uncertainty = float(max(state.ui, effective_ui, risk))
        else:
            node_uncertainty = state.observation_quality

        graph_nodes.append(
            trajectory_tau_module.TrajectoryGraphNode(
                index=len(graph_nodes),
                state=state,
                propagated_uncertainty=float(max(0.0, min(1.5, node_uncertainty))),
            )
        )
        history.append(state)
        if state.role == "user":
            latest_user = state
            latest_user_index = graph_nodes[-1].index

    if not risks:
        return {"tau_score": 0.0, "tau_confidence": 1.0, "num_steps": 0, "per_step": []}

    tau = float(np.mean(risks))
    return {
        "tau_score": tau,
        "tau_confidence": float(1.0 / (1.0 + tau)),
        "num_steps": len(risks),
        "mean_risk": float(np.mean(risks)),
        "max_risk": float(np.max(risks)),
        "mean_ui": float(np.mean([step["ui"] for step in per_step])),
        "max_ui": float(np.max([step["ui"] for step in per_step])),
        "mean_propagation": float(np.mean([step["propagation"] for step in per_step])),
        "mean_combined_propagation": float(np.mean([step["combined_propagation"] for step in per_step])),
        "mean_graph_uncertainty": float(np.mean([step["graph_uncertainty"] for step in per_step])),
        "max_graph_uncertainty": float(np.max([step["graph_uncertainty"] for step in per_step])),
        "mean_effective_ui": float(np.mean([step["effective_ui"] for step in per_step])),
        "mean_interaction_gap": float(np.mean([step["interaction_gap"] for step in per_step])),
        "mean_graph_edge_count": float(np.mean([step["graph_edge_count"] for step in per_step])),
        "per_step": per_step,
    }


def summarize_score_metrics(df: pd.DataFrame, score_col: str) -> dict[str, Any]:
    metrics = evaluate_failure_scores(df, score_col)
    metrics["score_column"] = score_col
    return metrics


def add_delta_columns(metrics_df: pd.DataFrame, full_variant: str) -> pd.DataFrame:
    metrics_df = metrics_df.copy()
    full_rows = metrics_df[metrics_df["variant"] == full_variant]
    if full_rows.empty:
        metrics_df["delta_auroc_vs_full"] = None
        metrics_df["delta_auprc_vs_full"] = None
        metrics_df["delta_f1_vs_full"] = None
        return metrics_df

    full_row = full_rows.iloc[0]
    metrics_df["delta_auroc_vs_full"] = metrics_df["auroc"] - float(full_row["auroc"])
    metrics_df["delta_auprc_vs_full"] = metrics_df["auprc"] - float(full_row["auprc"])
    metrics_df["delta_f1_vs_full"] = metrics_df["f1_at_threshold"] - float(full_row["f1_at_threshold"])
    return metrics_df


def run_component_ablation(
    dataset: list[dict[str, Any]],
    base_config: TrajectoryTAUConfig,
    variants: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    variant_configs = {
        variant: build_config(base_config, COMPONENT_ABLATIONS[variant])
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
            row[f"{variant}_score"] = finite_float(tau_info.get("tau_score"))
        rows.append(row)

    trial_df = pd.DataFrame(rows)
    metric_rows: list[dict[str, Any]] = []
    for variant in variants:
        metrics = summarize_score_metrics(trial_df, f"{variant}_score")
        metrics.update(
            {
                "variant": variant,
                "description": describe_component_variant(variant),
                "overrides": COMPONENT_ABLATIONS[variant],
            }
        )
        metric_rows.append(metrics)

    metrics_df = pd.DataFrame(metric_rows)
    metrics_df = add_delta_columns(metrics_df, "full")
    return trial_df, metrics_df, {
        variant: {
            "overrides": COMPONENT_ABLATIONS[variant],
            "resolved_config": asdict(variant_configs[variant]),
        }
        for variant in variants
    }


def run_graph_structure_ablation(
    dataset: list[dict[str, Any]],
    base_config: TrajectoryTAUConfig,
    variants: list[str],
    random_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []

    for item in dataset:
        row: dict[str, Any] = {
            "trial": item["trial"],
            "reward": item["reward"],
            "success": item["success"],
            "failure": item["failure"],
            "exception_type": item["exception_type"],
        }
        for variant in variants:
            rng = random.Random(stable_seed(random_seed, f"{item['trial']}:{variant}"))
            result = calculate_score_with_graph_variant(
                item["steps"],
                item["goal_text"],
                base_config,
                graph_variant=variant,
                rng=rng,
            )
            row[f"{variant}_score"] = finite_float(result.get("tau_score"))
            row[f"{variant}_mean_graph_uncertainty"] = finite_float(result.get("mean_graph_uncertainty"))
            row[f"{variant}_mean_graph_edge_count"] = finite_float(result.get("mean_graph_edge_count"))
            row[f"{variant}_mean_effective_ui"] = finite_float(result.get("mean_effective_ui"))
        rows.append(row)

    trial_df = pd.DataFrame(rows)
    metric_rows: list[dict[str, Any]] = []
    for variant in variants:
        metrics = summarize_score_metrics(trial_df, f"{variant}_score")
        metrics.update(
            {
                "variant": variant,
                "description": describe_graph_variant(variant),
                "graph_kind": GRAPH_VARIANTS[variant]["kind"],
            }
        )
        metric_rows.append(metrics)

    metrics_df = pd.DataFrame(metric_rows)
    metrics_df = add_delta_columns(metrics_df, "full_graph")
    return trial_df, metrics_df


def build_param_variants(base_config: TrajectoryTAUConfig) -> list[dict[str, Any]]:
    variants: list[dict[str, Any]] = [
        {
            "name": "full",
            "kind": "reference",
            "description": "Full Trajectory_TAU with default parameters.",
            "config": base_config,
            "param_name": "full",
            "param_value": None,
        }
    ]

    edge_family_targets = [
        "sequential",
        "latest_user",
        "repetition",
        "feedback",
        "progression",
        "parallel",
        "token_overlap",
    ]
    for family in edge_family_targets:
        for value in EDGE_FAMILY_SCALES:
            variants.append(
                {
                    "name": f"edge_{family}_{str(value).replace('.', 'p')}",
                    "kind": "edge_family_scale",
                    "description": f"Scale {family} edge weights by {value}.",
                    "config": base_config,
                    "family_scale": {family: value},
                    "param_name": f"edge_family::{family}",
                    "param_value": value,
                }
            )

    edge_type_targets = [
        "sequential",
        "latest_user",
        "tool_repetition",
        "repetition",
        "feedback_instability",
        "feedback_response",
        "progression",
        "parallel",
        "token_overlap",
    ]
    for edge_type in edge_type_targets:
        for value in EDGE_TYPE_SCALES:
            variants.append(
                {
                    "name": f"edge_weight_{edge_type}_{str(value).replace('.', 'p')}",
                    "kind": "edge_type_scale",
                    "description": f"Scale {edge_type} edge weights by {value}.",
                    "config": base_config,
                    "edge_type_scale": {edge_type: value},
                    "param_name": f"edge_weight::{edge_type}",
                    "param_value": value,
                }
            )

    history_param_values = {
        "alpha": [0.0, 0.5, 1.0, 1.5, 2.0],
        "beta": [0.0, 0.3, 0.7, 1.0, 1.3],
        "momentum_weight": HISTORY_WEIGHT_SCALES,
        "repetition_weight": HISTORY_WEIGHT_SCALES,
        "observation_weight": [0.0, 0.10, 0.20, 0.30, 0.40],
        "stagnation_weight": [0.0, 0.05, 0.10, 0.15, 0.25],
        "graph_weight": [0.0, 0.10, 0.20, 0.30, 0.45, 0.60],
        "graph_uncertainty_weight": [0.0, 0.15, 0.25, 0.35, 0.50, 0.65],
        "history_decay": [0.40, 0.60, 0.80, 0.90, 0.97],
        "uncertainty_decay": [0.50, 0.65, 0.75, 0.85, 0.95],
        "graph_edge_decay": TEMPORAL_WEIGHT_SCALES,
        "graph_max_neighbors": [2, 4, 6, 8, 10, 12],
        "recent_window": [1, 3, 5, 7, 9],
        "novelty_window": [1, 2, 4, 6, 8],
    }
    for param, values in history_param_values.items():
        for value in values:
            overrides = {param: value}
            variants.append(
                {
                    "name": f"{param}_{str(value).replace('.', 'p')}",
                    "kind": "history_weight",
                    "description": f"Set {param} to {value}.",
                    "config": build_config(base_config, overrides),
                    "overrides": overrides,
                    "param_name": param,
                    "param_value": value,
                }
            )

    return variants


def run_parameter_ablation(
    dataset: list[dict[str, Any]],
    base_config: TrajectoryTAUConfig,
    random_seed: int,
) -> pd.DataFrame:
    variants = build_param_variants(base_config)
    rows: list[dict[str, Any]] = []

    for item in dataset:
        row: dict[str, Any] = {
            "trial": item["trial"],
            "reward": item["reward"],
            "success": item["success"],
            "failure": item["failure"],
            "exception_type": item["exception_type"],
        }
        for variant in variants:
            rng = random.Random(stable_seed(random_seed, f"{item['trial']}:{variant['name']}"))
            if variant["kind"] == "edge_family_scale":
                result = calculate_score_with_graph_variant(
                    item["steps"],
                    item["goal_text"],
                    base_config,
                    graph_variant="full_graph",
                    rng=rng,
                    family_scale=variant["family_scale"],
                )
            elif variant["kind"] == "edge_type_scale":
                result = calculate_score_with_graph_variant(
                    item["steps"],
                    item["goal_text"],
                    base_config,
                    graph_variant="full_graph",
                    rng=rng,
                    edge_type_scale=variant["edge_type_scale"],
                )
            elif variant["kind"] == "history_weight":
                result = calculate_trajectory_tau_score(item["steps"], item["goal_text"], variant["config"])
            elif variant["kind"] == "reference":
                result = calculate_trajectory_tau_score(item["steps"], item["goal_text"], variant["config"])
            else:
                raise ValueError(f"Unknown parameter variant kind: {variant['kind']}")
            row[f"{variant['name']}_score"] = finite_float(result.get("tau_score"))
        rows.append(row)

    trial_df = pd.DataFrame(rows)
    metric_rows: list[dict[str, Any]] = []
    for variant in variants:
        metrics = summarize_score_metrics(trial_df, f"{variant['name']}_score")
        metrics.update(
            {
                "variant": variant["name"],
                "kind": variant["kind"],
                "description": variant["description"],
                "overrides": variant.get("overrides", {}),
                "family_scale": variant.get("family_scale", {}),
                "edge_type_scale": variant.get("edge_type_scale", {}),
                "param_name": variant.get("param_name"),
                "param_value": variant.get("param_value"),
            }
        )
        metric_rows.append(metrics)

    metrics_df = add_delta_columns(pd.DataFrame(metric_rows), "full")
    metrics_df = metrics_df.sort_values(
        by=["param_name", "param_value", "variant"],
        na_position="first",
        kind="mergesort",
    ).reset_index(drop=True)
    return metrics_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run extended Trajectory_TAU ablation experiments.")
    parser.add_argument("root", nargs="?", type=Path, default=None)
    parser.add_argument("--root", dest="root_flag", type=Path, default=None)
    parser.add_argument("--eval-key", default=None)
    parser.add_argument("--include-exceptions", action="store_true")
    parser.add_argument(
        "--component-variants",
        default="full,no_graph,no_history_propagation,no_momentum,no_repetition,no_observation,no_stagnation,no_interaction_gap",
        help=f"Comma-separated component ablations. Available: {','.join(COMPONENT_ABLATIONS)}",
    )
    parser.add_argument(
        "--graph-variants",
        default="full_graph,no_graph,sequential_only,drop_repetition_edges,drop_feedback_edges,drop_progression_edges,drop_parallel_edges,random_dropout_50,random_graph",
        help=f"Comma-separated graph variants. Available: {','.join(GRAPH_VARIANTS)}",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Output summary JSON path. Default: <root>/trajectory_tau_extended_ablation_summary.json",
    )
    parser.add_argument(
        "--include-trial-scores",
        action="store_true",
        help="Include per-trial scores in the summary JSON.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=0,
        help="Random seed used for stochastic graph perturbations.",
    )
    parser.add_argument(
        "--max-trials",
        type=int,
        default=0,
        help="Optional limit for quick debugging. Default 0 uses all trials.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = (args.root_flag or args.root or DEFAULT_ROOT).expanduser().resolve()
    if not (root / "result.json").exists():
        raise FileNotFoundError(f"Missing result.json: {root / 'result.json'}")

    component_variants = parse_csv_list(args.component_variants)
    graph_variants = parse_csv_list(args.graph_variants)

    unknown_components = [variant for variant in component_variants if variant not in COMPONENT_ABLATIONS]
    if unknown_components:
        raise ValueError(f"Unknown component variants: {unknown_components}")
    unknown_graphs = [variant for variant in graph_variants if variant not in GRAPH_VARIANTS]
    if unknown_graphs:
        raise ValueError(f"Unknown graph variants: {unknown_graphs}")

    base_config = TrajectoryTAUConfig()
    dataset, dataset_info = load_dataset(root, args.eval_key, args.include_exceptions)
    if not dataset:
        raise RuntimeError("No usable trials found")
    if args.max_trials and args.max_trials > 0:
        dataset = dataset[: args.max_trials]
        dataset_info = {
            **dataset_info,
            "counters": {
                **dataset_info["counters"],
                "used_trials": len(dataset),
            },
        }

    component_trial_df, component_metrics_df, component_configs = run_component_ablation(
        dataset,
        base_config,
        component_variants,
    )
    graph_trial_df, graph_metrics_df = run_graph_structure_ablation(
        dataset,
        base_config,
        graph_variants,
        args.random_seed,
    )
    param_metrics_df = run_parameter_ablation(dataset, base_config, args.random_seed)

    output_json = (args.output_json or (root / "trajectory_tau_extended_ablation_summary.json")).expanduser().resolve()
    output_json.parent.mkdir(parents=True, exist_ok=True)

    summary: dict[str, Any] = {
        "root": str(root),
        "eval_key": dataset_info["eval_key"],
        "include_exceptions": bool(args.include_exceptions),
        "positive_label": "failure",
        "score_direction": "higher_score_more_likely_failure",
        "trajectory_tau_base_config": asdict(base_config),
        "counters": dataset_info["counters"],
        "component_ablation_metrics": component_metrics_df.to_dict(orient="records"),
        "graph_structure_ablation_metrics": graph_metrics_df.to_dict(orient="records"),
        "parameter_ablation_metrics": param_metrics_df.to_dict(orient="records"),
        "component_variant_configs": component_configs,
        "graph_variants": {
            variant: {
                "kind": GRAPH_VARIANTS[variant]["kind"],
                "description": GRAPH_VARIANTS[variant]["description"],
            }
            for variant in graph_variants
        },
        "output_json": str(output_json),
    }
    if args.include_trial_scores:
        summary["component_trial_scores"] = component_trial_df.to_dict(orient="records")
        summary["graph_trial_scores"] = graph_trial_df.to_dict(orient="records")

    save_json(output_json, json_safe(summary))

    print("=" * 80)
    print("Trajectory_TAU extended ablation")
    print("=" * 80)
    print(f"Root: {root}")
    print(f"Eval key: {dataset_info['eval_key']}")
    print(f"Used trials: {dataset_info['counters']['used_trials']}")
    print()
    print("Component ablation")
    print(component_metrics_df[["variant", "n", "auroc", "auprc", "f1_at_threshold", "delta_auroc_vs_full", "delta_auprc_vs_full"]].to_string(index=False))
    print()
    print("Graph-structure ablation")
    print(graph_metrics_df[["variant", "n", "auroc", "auprc", "f1_at_threshold", "delta_auroc_vs_full", "delta_auprc_vs_full"]].to_string(index=False))
    print()
    print("Parameter ablation")
    print(param_metrics_df[["variant", "kind", "n", "auroc", "auprc", "f1_at_threshold", "delta_auroc_vs_full", "delta_auprc_vs_full"]].head(20).to_string(index=False))
    print()
    print(f"Saved summary JSON: {output_json}")


if __name__ == "__main__":
    main()
