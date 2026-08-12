#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Analysis experiments for Trajectory_TAU graph modeling.

This script is intentionally standalone: it does not modify the existing
evaluation scripts or the trajectory_tau implementation. It runs two analyses:

1. Full Graph vs No-Graph vs Sequential-Only vs Random Graph.
2. Entropy-matched bins: within similar token-uncertainty bins, compare whether
   graph signals still predict failure.

All scores are evaluated as failure-detection scores:
    higher score => more likely failure.

Usage:
    python evaluate_trajectory_tau_graph_experiments.py jobs/2026-06-26__09-42-41
    python evaluate_trajectory_tau_graph_experiments.py --root jobs/2026-06-26__09-42-41 --n-bins 5 --random-seed 0

Outputs:
    <root>/trajectory_tau_graph_experiments_variants.csv
    <root>/trajectory_tau_graph_experiments_entropy_bins.csv
    <root>/trajectory_tau_graph_experiments_trials.csv
    <root>/trajectory_tau_graph_experiments_summary.json
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import random
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

import evaluate_gaia_prefix_confidence_metrics as prefix_metrics


DEFAULT_ROOT = Path("~/jobs/2026-05-16__19-11-56")
PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRAJECTORY_TAU_PATH = PROJECT_ROOT / "agent-tracer" / "src" / "tau2" / "metrics" / "trajectory_tau.py"


def load_trajectory_tau_module():
    spec = importlib.util.spec_from_file_location("trajectory_tau_module_graph_experiments", TRAJECTORY_TAU_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load module from {TRAJECTORY_TAU_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


trajectory_tau_module = load_trajectory_tau_module()
TrajectoryTAUConfig = trajectory_tau_module.TrajectoryTAUConfig


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data: Any) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def finite_float(value: Any) -> Optional[float]:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def get_eval_payload(result_data: dict[str, Any], eval_key: Optional[str]) -> tuple[str, dict[str, Any]]:
    evals = result_data.get("stats", {}).get("evals", {})
    if not evals:
        raise ValueError("No stats.evals found in result.json")

    if eval_key is not None:
        if eval_key not in evals:
            available = ", ".join(evals.keys())
            raise ValueError(f"Eval key {eval_key!r} not found. Available: {available}")
        return eval_key, evals[eval_key]

    first_key = next(iter(evals.keys()))
    return first_key, evals[first_key]


def extract_reward_mapping(eval_payload: dict[str, Any]) -> dict[str, float]:
    reward_stats = eval_payload.get("reward_stats", {}).get("reward", {})
    mapping: dict[str, float] = {}
    for reward_str, trial_names in reward_stats.items():
        reward = finite_float(reward_str)
        if reward is None:
            continue
        for trial_name in trial_names:
            mapping[str(trial_name)] = reward
    return mapping


def extract_exception_mapping(eval_payload: dict[str, Any]) -> dict[str, str]:
    exception_stats = eval_payload.get("exception_stats", {})
    mapping: dict[str, str] = {}
    for exception_type, trial_names in exception_stats.items():
        for trial_name in trial_names:
            mapping[str(trial_name)] = str(exception_type)
    return mapping


def trial_exception_type(trial_dir: Path, exception_mapping: dict[str, str]) -> Optional[str]:
    if trial_dir.name in exception_mapping:
        return exception_mapping[trial_dir.name]

    trial_result_path = trial_dir / "result.json"
    if not trial_result_path.exists():
        return None

    try:
        trial_result = load_json(trial_result_path)
    except Exception:
        return None

    exception_info = trial_result.get("exception_info")
    if isinstance(exception_info, dict):
        return exception_info.get("exception_type")
    return None


def extract_goal_text(traj_data: dict[str, Any], trial_dir: Path) -> str:
    for step in traj_data.get("steps", []):
        if step.get("source") == "user":
            message = step.get("message")
            if isinstance(message, str) and message.strip():
                return message

    config_path = trial_dir / "config.json"
    if config_path.exists():
        try:
            config = load_json(config_path)
            task_cfg = config.get("task", {})
            for key in ("name", "source", "ref"):
                value = task_cfg.get(key)
                if isinstance(value, str) and value.strip():
                    return value
        except Exception:
            pass

    return ""


def stable_seed(base_seed: int, trial_name: str) -> int:
    digest = hashlib.sha256(f"{base_seed}:{trial_name}".encode("utf-8")).hexdigest()
    return int(digest[:16], 16) % (2**32)


def clone_config(config: Any, **overrides: Any) -> Any:
    normalized = trajectory_tau_module._normalize_config(config)
    values = asdict(normalized)
    values.update(overrides)
    return TrajectoryTAUConfig(**values)


def edge_payloads_to_uncertainty(
    edges: list[Any],
    graph_nodes: list[Any],
) -> dict[str, Any]:
    if not edges:
        return {
            "graph_uncertainty": 0.0,
            "graph_edge_count": 0,
            "graph_total_weight": 0.0,
            "graph_max_edge_weight": 0.0,
            "graph_edges": [],
        }

    node_by_index = {node.index: node for node in graph_nodes}
    weighted_uncertainty = 0.0
    total_weight = 0.0
    payloads: list[dict[str, Any]] = []

    for edge in edges:
        source = node_by_index.get(edge.source_index)
        if source is None:
            continue
        weighted_uncertainty += edge.weight * source.propagated_uncertainty
        total_weight += edge.weight
        payloads.append(
            {
                "source_index": edge.source_index,
                "target_index": edge.target_index,
                "weight": edge.weight,
                "edge_type": edge.edge_type,
                "source_role": source.state.role,
                "source_uncertainty": source.propagated_uncertainty,
            }
        )

    graph_uncertainty = 0.0 if total_weight == 0.0 else weighted_uncertainty / total_weight
    return {
        "graph_uncertainty": float(max(0.0, min(1.5, graph_uncertainty))),
        "graph_edge_count": len(payloads),
        "graph_total_weight": float(total_weight),
        "graph_max_edge_weight": float(max((edge["weight"] for edge in payloads), default=0.0)),
        "graph_edges": payloads,
    }


def sequential_only_graph_uncertainty(
    graph_nodes: list[Any],
    latest_user_index: Optional[int],
    config: Any,
) -> dict[str, Any]:
    if not graph_nodes:
        return edge_payloads_to_uncertainty([], graph_nodes)

    target_index = len(graph_nodes)
    edges = [
        trajectory_tau_module.TrajectoryGraphEdge(
            source_index=graph_nodes[-1].index,
            target_index=target_index,
            weight=trajectory_tau_module._edge_strength("sequential", 0.35, 1, config),
            edge_type="sequential",
        )
    ]
    if latest_user_index is not None:
        user_age = target_index - latest_user_index
        edges.append(
            trajectory_tau_module.TrajectoryGraphEdge(
                source_index=latest_user_index,
                target_index=target_index,
                weight=trajectory_tau_module._edge_strength("latest_user", 0.75, user_age, config),
                edge_type="latest_user",
            )
        )
    return edge_payloads_to_uncertainty(edges, graph_nodes)


def random_graph_uncertainty(
    current: Any,
    graph_nodes: list[Any],
    latest_user_index: Optional[int],
    config: Any,
    rng: random.Random,
) -> dict[str, Any]:
    if not graph_nodes:
        return edge_payloads_to_uncertainty([], graph_nodes)

    real_edges = trajectory_tau_module.build_dependency_edges(current, graph_nodes, latest_user_index, config)
    source_indices = [node.index for node in graph_nodes]
    randomized_edges = []
    for edge in real_edges:
        randomized_edges.append(
            trajectory_tau_module.TrajectoryGraphEdge(
                source_index=rng.choice(source_indices),
                target_index=edge.target_index,
                weight=edge.weight,
                edge_type=edge.edge_type,
            )
        )
    return edge_payloads_to_uncertainty(randomized_edges, graph_nodes)


def graph_uncertainty_for_mode(
    mode: str,
    current: Any,
    graph_nodes: list[Any],
    latest_user_index: Optional[int],
    config: Any,
    rng: random.Random,
) -> dict[str, Any]:
    if mode == "full":
        return trajectory_tau_module.calculate_graph_uncertainty(current, graph_nodes, latest_user_index, config)
    if mode == "no_graph":
        return edge_payloads_to_uncertainty([], graph_nodes)
    if mode == "sequential_only":
        return sequential_only_graph_uncertainty(graph_nodes, latest_user_index, config)
    if mode == "random_graph":
        return random_graph_uncertainty(current, graph_nodes, latest_user_index, config, rng)
    raise ValueError(f"Unknown graph mode: {mode}")


def calculate_score_with_graph_mode(
    messages: list[Any],
    goal_text: str,
    config: Any,
    graph_mode: str,
    rng: random.Random,
) -> dict[str, Any]:
    config = trajectory_tau_module._normalize_config(config)
    if graph_mode == "no_graph":
        config = clone_config(config, graph_weight=0.0, graph_uncertainty_weight=0.0)

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
            propagation = trajectory_tau_module.calculate_trajectory_propagation(state, history, config)
            graph = graph_uncertainty_for_mode(graph_mode, state, graph_nodes, latest_user_index, config, rng)
            interaction_gap = trajectory_tau_module.calculate_goal_coverage_gap(state, latest_user, goal_text, config)
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
                    "graph_edge_count": float(graph["graph_edge_count"]),
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


def load_dataset(root: Path, eval_key: Optional[str], include_exceptions: bool) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    result_data = load_json(root / "result.json")
    resolved_eval_key, eval_payload = get_eval_payload(result_data, eval_key)
    reward_mapping = extract_reward_mapping(eval_payload)
    exception_mapping = extract_exception_mapping(eval_payload)

    dataset: list[dict[str, Any]] = []
    counters = {
        "rewarded_trials": len(reward_mapping),
        "skipped_exception": 0,
        "missing_trajectory": 0,
        "used_trials": 0,
    }

    for trial_dir in sorted(root.iterdir()):
        if not trial_dir.is_dir():
            continue
        trial_name = trial_dir.name
        if trial_name not in reward_mapping:
            continue

        exception_type = trial_exception_type(trial_dir, exception_mapping)
        if exception_type and not include_exceptions:
            counters["skipped_exception"] += 1
            continue

        traj_path = trial_dir / "agent" / "trajectory.json"
        if not traj_path.exists():
            counters["missing_trajectory"] += 1
            continue

        try:
            traj_data = load_json(traj_path)
        except Exception:
            counters["missing_trajectory"] += 1
            continue

        reward = reward_mapping[trial_name]
        dataset.append(
            {
                "trial": trial_name,
                "reward": reward,
                "success": int(reward > 0.5),
                "failure": int(reward <= 0.5),
                "exception_type": exception_type,
                "steps": traj_data.get("steps", []),
                "goal_text": extract_goal_text(traj_data, trial_dir),
            }
        )

    counters["used_trials"] = len(dataset)
    return dataset, {"eval_key": resolved_eval_key, "counters": counters}


def best_youden_threshold(y_true: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    fpr, tpr, thresholds = roc_curve(y_true, scores)
    j_scores = tpr - fpr
    best_idx = int(np.argmax(j_scores))
    return float(thresholds[best_idx]), float(j_scores[best_idx])


def evaluate_failure_scores(df: pd.DataFrame, score_col: str) -> dict[str, Any]:
    eval_df = df[df[score_col].notna()].copy()
    if eval_df.empty:
        raise ValueError(f"No valid scores for {score_col}")

    y_true = eval_df["failure"].to_numpy(dtype=int)
    scores = eval_df[score_col].to_numpy(dtype=float)
    if len(np.unique(y_true)) < 2:
        raise ValueError(f"Need both classes to compute metrics for {score_col}")

    threshold, youden_j = best_youden_threshold(y_true, scores)
    y_pred = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "score_column": score_col,
        "n": int(len(eval_df)),
        "n_success": int((eval_df["success"] == 1).sum()),
        "n_failure": int((eval_df["failure"] == 1).sum()),
        "auroc": float(roc_auc_score(y_true, scores)),
        "auprc": float(average_precision_score(y_true, scores)),
        "best_threshold_youden": threshold,
        "youden_j": youden_j,
        "accuracy_at_threshold": float(accuracy_score(y_true, y_pred)),
        "precision_at_threshold": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall_at_threshold": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_at_threshold": float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def step_mean_logprob(step: dict[str, Any]) -> Optional[float]:
    logprobs = prefix_metrics.collect_logprobs_from_step(step)
    if not logprobs:
        return None
    return float(np.mean(np.asarray(logprobs, dtype=np.float64)))


def calculate_sequence_prob_score(steps: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Step-level sequence NLL baseline.

    The score keeps sequence accumulation while reducing token-count effects:
        sequence_step_nll = -sum(step_mean_logprob)
    """

    step_means: list[float] = []
    n_agent_steps = 0
    for step in steps:
        if step.get("source") != "agent":
            continue
        n_agent_steps += 1
        mean_logprob = step_mean_logprob(step)
        if mean_logprob is not None:
            step_means.append(mean_logprob)

    if not step_means:
        return {
            "sequence_prob_score": None,
            "sequence_prob_num_steps": n_agent_steps,
        }

    score = float(-np.sum(np.asarray(step_means, dtype=np.float64)))
    return {
        "sequence_prob_score": score,
        "sequence_prob_num_steps": n_agent_steps,
        "sequence_prob_mean_step_logprob": float(np.mean(step_means)),
    }


def calculate_baseline_scores(
    steps: list[dict[str, Any]],
    tracer_config: Any,
    saup_config: Any,
    uprop_config: Any,
) -> dict[str, Any]:
    entropy_info = prefix_metrics.calculate_entropy_prefix(steps)
    tracer_info = prefix_metrics.calculate_tracer_prefix(steps, tracer_config)
    saup_info = prefix_metrics.calculate_saup_prefix(steps, saup_config)
    uprop_info = prefix_metrics.calculate_uprop_prefix(steps, uprop_config)
    sequence_prob_info = calculate_sequence_prob_score(steps)

    return {
        "entropy_mean_ui": finite_float(entropy_info.get("entropy_risk")),
        "entropy_num_steps": entropy_info.get("entropy_n_agent_steps"),
        "tracer_score": finite_float(tracer_info.get("tracer_risk")),
        "tracer_num_steps": tracer_info.get("tracer_num_steps"),
        "saup_score": finite_float(saup_info.get("saup_risk")),
        "saup_num_steps": saup_info.get("saup_num_steps"),
        "uprop_score": finite_float(uprop_info.get("uprop_risk")),
        "uprop_num_steps": uprop_info.get("uprop_num_steps"),
        "sequence_prob_score": finite_float(sequence_prob_info.get("sequence_prob_score")),
        "sequence_prob_num_steps": sequence_prob_info.get("sequence_prob_num_steps"),
    }


def run_variant_experiment(dataset: list[dict[str, Any]], config: Any, random_seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    trial_rows: list[dict[str, Any]] = []
    variant_rows: list[dict[str, Any]] = []
    modes = ["full", "no_graph", "sequential_only", "random_graph"]
    tracer_config = prefix_metrics.TRACERConfig()
    saup_config = prefix_metrics.SAUPConfig()
    uprop_config = prefix_metrics.UPropApproxConfig()

    for item in dataset:
        row = {
            "trial": item["trial"],
            "reward": item["reward"],
            "success": item["success"],
            "failure": item["failure"],
            "exception_type": item["exception_type"],
        }
        row.update(calculate_baseline_scores(item["steps"], tracer_config, saup_config, uprop_config))
        for mode in modes:
            rng = random.Random(stable_seed(random_seed, f"{item['trial']}:{mode}"))
            result = calculate_score_with_graph_mode(
                item["steps"],
                item["goal_text"],
                config,
                graph_mode=mode,
                rng=rng,
            )
            prefix = mode
            row[f"{prefix}_tau_score"] = finite_float(result.get("tau_score"))
            row[f"{prefix}_mean_ui"] = finite_float(result.get("mean_ui"))
            row[f"{prefix}_mean_graph_uncertainty"] = finite_float(result.get("mean_graph_uncertainty"))
            row[f"{prefix}_max_graph_uncertainty"] = finite_float(result.get("max_graph_uncertainty"))
            row[f"{prefix}_mean_effective_ui"] = finite_float(result.get("mean_effective_ui"))
            row[f"{prefix}_mean_graph_edge_count"] = finite_float(result.get("mean_graph_edge_count"))
            row[f"{prefix}_num_steps"] = int(result.get("num_steps", 0) or 0)
        trial_rows.append(row)

    trial_df = pd.DataFrame(trial_rows)
    for mode in modes:
        score_col = f"{mode}_tau_score"
        metrics = evaluate_failure_scores(trial_df, score_col)
        metrics.update({"variant": mode, "score_column": score_col})
        variant_rows.append(metrics)

    variant_df = pd.DataFrame(variant_rows)
    full_auroc = float(variant_df.loc[variant_df["variant"] == "full", "auroc"].iloc[0])
    full_auprc = float(variant_df.loc[variant_df["variant"] == "full", "auprc"].iloc[0])
    variant_df["delta_auroc_vs_full"] = variant_df["auroc"] - full_auroc
    variant_df["delta_auprc_vs_full"] = variant_df["auprc"] - full_auprc
    return trial_df, variant_df


def run_entropy_matched_bins(trial_df: pd.DataFrame, n_bins: int) -> pd.DataFrame:
    df = trial_df.copy()
    df = df[df["full_mean_ui"].notna()].copy()
    unique_entropy = df["full_mean_ui"].nunique()
    actual_bins = min(n_bins, int(unique_entropy))
    if actual_bins < 2:
        raise ValueError("Need at least two distinct entropy values for entropy-matched bins")

    df["entropy_bin"] = pd.qcut(df["full_mean_ui"], q=actual_bins, labels=False, duplicates="drop")
    rows = []
    score_cols = {
        "entropy_mean_ui": "entropy_mean_ui",
        "tracer_score": "tracer_score",
        "saup_score": "saup_score",
        "uprop_score": "uprop_score",
        "sequence_prob_score": "sequence_prob_score",
        "graph_mean_uncertainty": "full_mean_graph_uncertainty",
        "graph_max_uncertainty": "full_max_graph_uncertainty",
        "trajectory_tau_score": "full_tau_score",
        "no_graph_tau_score": "no_graph_tau_score",
    }

    for bin_id, bin_df in df.groupby("entropy_bin", sort=True):
        row: dict[str, Any] = {
            "entropy_bin": int(bin_id),
            "n": int(len(bin_df)),
            "n_success": int((bin_df["success"] == 1).sum()),
            "n_failure": int((bin_df["failure"] == 1).sum()),
            "entropy_min": float(bin_df["full_mean_ui"].min()),
            "entropy_max": float(bin_df["full_mean_ui"].max()),
            "entropy_mean": float(bin_df["full_mean_ui"].mean()),
            "failure_rate": float(bin_df["failure"].mean()),
        }
        for label, score_col in score_cols.items():
            if score_col not in bin_df:
                row[f"{label}_auroc"] = None
                row[f"{label}_auprc"] = None
                continue
            valid_df = bin_df[bin_df[score_col].notna()].copy()
            if valid_df.empty:
                row[f"{label}_auroc"] = None
                row[f"{label}_auprc"] = None
                continue
            if len(np.unique(valid_df["failure"].to_numpy(dtype=int))) < 2:
                row[f"{label}_auroc"] = None
                row[f"{label}_auprc"] = None
                continue
            scores = valid_df[score_col].to_numpy(dtype=float)
            y_true = valid_df["failure"].to_numpy(dtype=int)
            row[f"{label}_auroc"] = float(roc_auc_score(y_true, scores))
            row[f"{label}_auprc"] = float(average_precision_score(y_true, scores))
        rows.append(row)

    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run graph-ablation and entropy-matched-bin analyses for Trajectory_TAU."
    )
    parser.add_argument("root", nargs="?", type=Path, default=None)
    parser.add_argument("--root", dest="root_flag", type=Path, default=None)
    parser.add_argument("--eval-key", default=None)
    parser.add_argument("--include-exceptions", action="store_true")
    parser.add_argument("--n-bins", type=int, default=5)
    parser.add_argument("--random-seed", type=int, default=0)
    parser.add_argument("--output-prefix", default="trajectory_tau_graph_experiments")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = (args.root_flag or args.root or DEFAULT_ROOT).expanduser().resolve()
    if not (root / "result.json").exists():
        raise FileNotFoundError(f"Missing result.json: {root / 'result.json'}")

    config = TrajectoryTAUConfig()
    dataset, dataset_info = load_dataset(root, args.eval_key, args.include_exceptions)
    if not dataset:
        raise RuntimeError("No usable trials found")

    trial_df, variant_df = run_variant_experiment(dataset, config, args.random_seed)
    bins_df = run_entropy_matched_bins(trial_df, args.n_bins)

    trial_csv = root / f"{args.output_prefix}_trials.csv"
    variant_csv = root / f"{args.output_prefix}_variants.csv"
    bins_csv = root / f"{args.output_prefix}_entropy_bins.csv"
    summary_json = root / f"{args.output_prefix}_summary.json"

    trial_df.to_csv(trial_csv, index=False)
    variant_df.to_csv(variant_csv, index=False)
    bins_df.to_csv(bins_csv, index=False)

    summary = {
        "root": str(root),
        "eval_key": dataset_info["eval_key"],
        "include_exceptions": bool(args.include_exceptions),
        "random_seed": args.random_seed,
        "n_bins": args.n_bins,
        "trajectory_tau_config": asdict(config),
        "counters": dataset_info["counters"],
        "variant_metrics": variant_df.to_dict(orient="records"),
        "entropy_bin_metrics": bins_df.to_dict(orient="records"),
        "outputs": {
            "trial_csv": str(trial_csv),
            "variant_csv": str(variant_csv),
            "entropy_bins_csv": str(bins_csv),
            "summary_json": str(summary_json),
        },
    }
    save_json(summary_json, summary)

    print("=" * 80)
    print("Trajectory_TAU graph experiments")
    print("=" * 80)
    print(f"Root: {root}")
    print(f"Eval key: {dataset_info['eval_key']}")
    print(f"Used trials: {dataset_info['counters']['used_trials']}")
    print()
    print("Variant experiment: higher score => more likely failure")
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
        "accuracy_at_threshold",
    ]
    print(variant_df[display_cols].to_string(index=False))
    print()
    print("Entropy-matched bins")
    print(bins_df.to_string(index=False))
    print()
    print("Saved:")
    print(f"  Trials: {trial_csv}")
    print(f"  Variants: {variant_csv}")
    print(f"  Entropy bins: {bins_csv}")
    print(f"  Summary: {summary_json}")


if __name__ == "__main__":
    main()
