#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Evaluate trajectory-aware TAU metrics on Harbor GAIA runs.

This script evaluates the stronger TAU variant implemented in:
    agent-tracer/src/tau2/metrics/trajectory_tau.py

Compared to the baseline TAU script, this version measures uncertainty
propagation through trajectory structure rather than plain text similarity.

Usage:
    python evaluate_gaia_trajectory_tau_metrics.py /path/to/harbor_job
    python evaluate_gaia_trajectory_tau_metrics.py --root /path/to/harbor_job
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
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
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None


DEFAULT_ROOT = Path("~/jobs/2026-05-16__19-11-56")
PROJECT_ROOT = Path(__file__).resolve().parents[2]
AGENT_TRACER_SRC = PROJECT_ROOT / "agent-tracer" / "src"
TRAJECTORY_TAU_PATH = AGENT_TRACER_SRC / "tau2" / "metrics" / "trajectory_tau.py"


def load_trajectory_tau_module():
    spec = importlib.util.spec_from_file_location("trajectory_tau_module", TRAJECTORY_TAU_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load module from {TRAJECTORY_TAU_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


trajectory_tau_module = load_trajectory_tau_module()
TrajectoryTAUConfig = trajectory_tau_module.TrajectoryTAUConfig
calculate_trajectory_tau_score = trajectory_tau_module.calculate_trajectory_tau_score


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
    steps = traj_data.get("steps", [])
    for step in steps:
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


def collect_rows(
    root: Path,
    reward_mapping: dict[str, float],
    exception_mapping: dict[str, str],
    config: TrajectoryTAUConfig,
    include_exceptions: bool,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    rows: list[dict[str, Any]] = []
    counters = {
        "rewarded_trials": len(reward_mapping),
        "missing_trajectory": 0,
        "skipped_exception": 0,
        "missing_tau": 0,
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
        except Exception as exc:
            print(f"[WARN] failed to load {traj_path}: {exc}")
            counters["missing_tau"] += 1
            continue

        goal_text = extract_goal_text(traj_data, trial_dir)
        tau_info = calculate_trajectory_tau_score(traj_data.get("steps", []), goal_text, config)
        tau_score = tau_info.get("tau_score")
        tau_confidence = tau_info.get("tau_confidence")
        if tau_score is None or tau_confidence is None:
            counters["missing_tau"] += 1
            continue

        reward = reward_mapping[trial_name]
        success = int(reward > 0.5)
        rows.append(
            {
                "trial": trial_name,
                "reward": reward,
                "success": success,
                "failure": 1 - success,
                "exception_type": exception_type,
                "tau_score": tau_score,
                "tau_confidence": tau_confidence,
                "num_steps": tau_info.get("num_steps"),
                "mean_risk": tau_info.get("mean_risk"),
                "max_risk": tau_info.get("max_risk"),
                "mean_propagation": tau_info.get("mean_propagation"),
                "mean_combined_propagation": tau_info.get("mean_combined_propagation"),
                "mean_graph_uncertainty": tau_info.get("mean_graph_uncertainty"),
                "mean_effective_ui": tau_info.get("mean_effective_ui"),
                "mean_interaction_gap": tau_info.get("mean_interaction_gap"),
            }
        )

    counters["used_trials"] = len(rows)
    return rows, counters


def best_youden_threshold(y_true: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    fpr, tpr, thresholds = roc_curve(y_true, scores)
    j_scores = tpr - fpr
    best_idx = int(np.argmax(j_scores))
    return float(thresholds[best_idx]), float(j_scores[best_idx])


def evaluate_scores(df: pd.DataFrame, score_col: str, positive_label: str) -> dict[str, Any]:
    if positive_label == "failure":
        y_true = df["failure"].to_numpy(dtype=int)
        title_positive = "higher"
    elif positive_label == "success":
        y_true = df["success"].to_numpy(dtype=int)
        title_positive = "higher"
    else:
        raise ValueError(f"Unsupported positive_label: {positive_label}")

    scores = df[score_col].to_numpy(dtype=float)
    if len(np.unique(y_true)) < 2:
        raise ValueError("Need both classes to compute AUROC/AUPRC")

    threshold, youden_j = best_youden_threshold(y_true, scores)
    y_pred = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    return {
        "score_column": score_col,
        "positive_label": positive_label,
        "threshold_positive_direction": title_positive,
        "n": int(len(df)),
        "n_success": int((df["success"] == 1).sum()),
        "n_failure": int((df["failure"] == 1).sum()),
        "auroc": float(roc_auc_score(y_true, scores)),
        "auprc": float(average_precision_score(y_true, scores)),
        "best_threshold_youden": threshold,
        "youden_j": youden_j,
        "accuracy_at_threshold": float(accuracy_score(y_true, y_pred)),
        "precision_at_threshold": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall_at_threshold": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_at_threshold": float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },
        "score_stats_by_success": (
            df.groupby("success")[score_col]
            .describe()
            .reset_index()
            .to_dict(orient="records")
        ),
    }


def maybe_plot_curves(
    df: pd.DataFrame,
    score_col: str,
    positive_label: str,
    root: Path,
    prefix: str,
) -> dict[str, str]:
    if plt is None:
        print("[WARN] matplotlib not installed; skip plots")
        return {}

    if positive_label == "failure":
        y_true = df["failure"].to_numpy(dtype=int)
        title_label = "Failure"
    else:
        y_true = df["success"].to_numpy(dtype=int)
        title_label = "Success"

    scores = df[score_col].to_numpy(dtype=float)
    auroc = roc_auc_score(y_true, scores)
    auprc = average_precision_score(y_true, scores)

    paths = {
        "roc_plot": str(root / f"{prefix}_roc.png"),
        "pr_plot": str(root / f"{prefix}_pr.png"),
    }

    fpr, tpr, _ = roc_curve(y_true, scores)
    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, label=f"AUROC = {auroc:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"GAIA Trajectory TAU {title_label} Detection")
    plt.legend()
    plt.tight_layout()
    plt.savefig(paths["roc_plot"], dpi=200)
    plt.close()

    precision, recall, _ = precision_recall_curve(y_true, scores)
    plt.figure(figsize=(6, 6))
    plt.plot(recall, precision, label=f"AUPRC = {auprc:.4f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"GAIA Trajectory TAU {title_label} Detection")
    plt.legend()
    plt.tight_layout()
    plt.savefig(paths["pr_plot"], dpi=200)
    plt.close()

    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate Harbor GAIA trajectory-aware TAU scores against rewards."
    )
    parser.add_argument(
        "root",
        nargs="?",
        type=Path,
        default=None,
        help="Harbor job root containing result.json and trial directories.",
    )
    parser.add_argument(
        "--root",
        dest="root_flag",
        type=Path,
        default=None,
        help="Harbor job root containing result.json and trial directories.",
    )
    parser.add_argument(
        "--eval-key",
        default=None,
        help="stats.evals key to use. Defaults to the first evaluator.",
    )
    parser.add_argument("--alpha", type=float, default=1.0, help="Outer alpha weight for trajectory propagation.")
    parser.add_argument("--beta", type=float, default=0.7, help="Outer beta weight for interaction gap.")
    parser.add_argument("--momentum-weight", type=float, default=0.30)
    parser.add_argument("--repetition-weight", type=float, default=0.30)
    parser.add_argument("--observation-weight", type=float, default=0.25)
    parser.add_argument("--stagnation-weight", type=float, default=0.15)
    parser.add_argument("--interaction-goal-weight", type=float, default=0.55)
    parser.add_argument("--interaction-user-weight", type=float, default=0.45)
    parser.add_argument("--history-decay", type=float, default=0.80)
    parser.add_argument("--uncertainty-decay", type=float, default=0.75)
    parser.add_argument("--recent-window", type=int, default=5)
    parser.add_argument("--novelty-window", type=int, default=4)
    parser.add_argument("--graph-weight", type=float, default=0.30)
    parser.add_argument("--graph-uncertainty-weight", type=float, default=0.35)
    parser.add_argument("--graph-edge-decay", type=float, default=0.85)
    parser.add_argument("--graph-max-neighbors", type=int, default=8)
    parser.add_argument(
        "--score-col",
        default="tau_score",
        choices=[
            "tau_confidence",
            "tau_score",
            "mean_risk",
            "max_risk",
            "mean_propagation",
            "mean_combined_propagation",
            "mean_graph_uncertainty",
            "mean_effective_ui",
            "mean_interaction_gap",
        ],
        help="Score column used for evaluation.",
    )
    parser.add_argument(
        "--positive-label",
        default="failure",
        choices=["success", "failure"],
        help="Which label the score should predict.",
    )
    parser.add_argument(
        "--include-exceptions",
        action="store_true",
        help="Include trials with exception_info/exception_stats if they have rewards and trajectories.",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Do not save ROC/PR plots.",
    )
    parser.add_argument(
        "--output-prefix",
        default="gaia_trajectory_tau_eval",
        help="Prefix for CSV, summary JSON, and plots.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = (args.root_flag or args.root or DEFAULT_ROOT).expanduser().resolve()
    result_path = root / "result.json"
    if not result_path.exists():
        raise FileNotFoundError(f"Missing result.json: {result_path}")

    config = TrajectoryTAUConfig(
        alpha=args.alpha,
        beta=args.beta,
        momentum_weight=args.momentum_weight,
        repetition_weight=args.repetition_weight,
        observation_weight=args.observation_weight,
        stagnation_weight=args.stagnation_weight,
        interaction_goal_weight=args.interaction_goal_weight,
        interaction_user_weight=args.interaction_user_weight,
        history_decay=args.history_decay,
        uncertainty_decay=args.uncertainty_decay,
        recent_window=args.recent_window,
        novelty_window=args.novelty_window,
        graph_weight=args.graph_weight,
        graph_uncertainty_weight=args.graph_uncertainty_weight,
        graph_edge_decay=args.graph_edge_decay,
        graph_max_neighbors=args.graph_max_neighbors,
    )

    result_data = load_json(result_path)
    eval_key, eval_payload = get_eval_payload(result_data, args.eval_key)
    reward_mapping = extract_reward_mapping(eval_payload)
    exception_mapping = extract_exception_mapping(eval_payload)

    rows, counters = collect_rows(
        root=root,
        reward_mapping=reward_mapping,
        exception_mapping=exception_mapping,
        config=config,
        include_exceptions=args.include_exceptions,
    )

    if not rows:
        raise RuntimeError("No valid trials found with reward and trajectory-aware TAU metrics.")

    df = pd.DataFrame(rows)
    df = df[df[args.score_col].notna()].copy()
    if df.empty:
        raise RuntimeError(f"No non-null scores found for {args.score_col}")

    metrics = evaluate_scores(df, args.score_col, args.positive_label)

    csv_path = root / f"{args.output_prefix}.csv"
    summary_path = root / f"{args.output_prefix}_summary.json"
    df.sort_values(args.score_col, ascending=False).to_csv(csv_path, index=False)

    plot_paths = {}
    if not args.no_plot:
        plot_paths = maybe_plot_curves(df, args.score_col, args.positive_label, root, args.output_prefix)

    summary = {
        "root": str(root),
        "eval_key": eval_key,
        "trajectory_tau_config": asdict(config),
        "score_column": args.score_col,
        "positive_label": args.positive_label,
        "include_exceptions": bool(args.include_exceptions),
        "counters": counters,
        "metrics": metrics,
        "outputs": {
            "csv": str(csv_path),
            "summary_json": str(summary_path),
            **plot_paths,
        },
    }
    save_json(summary_path, summary)

    print("=" * 80)
    print("GAIA Trajectory TAU evaluation")
    print("=" * 80)
    print(f"Root: {root}")
    print(f"Eval key: {eval_key}")
    print(f"Used trials: {metrics['n']} | success: {metrics['n_success']} | failure: {metrics['n_failure']}")
    print(f"Skipped exception trials: {counters['skipped_exception']}")
    print()
    print(f"Score: {args.score_col}")
    print(f"Positive label: {args.positive_label}")
    print(f"AUROC: {metrics['auroc']:.6f}")
    print(f"AUPRC: {metrics['auprc']:.6f}")
    print(f"Best threshold (Youden): {metrics['best_threshold_youden']:.6f}")
    print(f"Accuracy/F1 at threshold: {metrics['accuracy_at_threshold']:.6f} / {metrics['f1_at_threshold']:.6f}")
    print()
    print("Saved:")
    print(f"  CSV: {csv_path}")
    print(f"  Summary: {summary_path}")
    for label, path in plot_paths.items():
        print(f"  {label}: {path}")
    print()
    print("Top-20 highest score trials:")
    cols = [
        "trial",
        "reward",
        "success",
        args.score_col,
        "tau_score",
        "mean_risk",
        "mean_propagation",
        "mean_interaction_gap",
        "num_steps",
    ]
    print(df.sort_values(args.score_col, ascending=False).head(20)[cols].to_string(index=False))


if __name__ == "__main__":
    main()
