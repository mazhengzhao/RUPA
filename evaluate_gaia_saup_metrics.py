#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Evaluate SAUP uncertainty metrics on Harbor GAIA runs.

This script follows two local patterns:
1. Harbor result loading from evaluate_gaia_entropy_metrics.py:
   - read <job_root>/result.json
   - map trial_name -> reward from stats.evals[*].reward_stats.reward
   - skip timeout/cancelled/error trials by default
   - read each trial's agent/trajectory.json
2. Agent Tracer SAUP aggregation from
   agent-tracer/src/tau2/metrics/uncertainty.py:
   - compute per-step uncertainty U_i
   - compute SAUP with RMS aggregation over W_i * U_i
   - use the final SAUP score as a failure-detection score

Harbor GAIA trajectories usually store token logprobs as:
    steps[*].metrics.logprobs: list[float]

Some Harbor traces do not contain Tau-2 semantic fields (Da/Do). In that case:
    W_i = 1
    SAUP = sqrt(mean(U_i^2))

Usage:
    python evaluate_gaia_saup_metrics.py ~/jobs/2026-05-16__19-11-56
    python evaluate_gaia_saup_metrics.py --root ~/jobs/2026-05-16__19-11-56

Outputs:
    <root>/gaia_saup_eval.csv
    <root>/gaia_saup_eval_summary.json
    <root>/gaia_saup_eval_roc.png      (unless --no-plot)
    <root>/gaia_saup_eval_pr.png       (unless --no-plot)
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
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


@dataclass
class SAUPConfig:
    """Minimal copy of the SAUP-related weights from Agent Tracer."""

    alpha: float = 1.0
    beta: float = 1.0
    gamma: float = 1.0


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


def safe_mean(values: list[float]) -> Optional[float]:
    values = [x for x in values if math.isfinite(x)]
    if not values:
        return None
    return float(np.mean(values))


def safe_std(values: list[float]) -> Optional[float]:
    values = [x for x in values if math.isfinite(x)]
    if not values:
        return None
    return float(np.std(values))


def safe_max(values: list[float]) -> Optional[float]:
    values = [x for x in values if math.isfinite(x)]
    if not values:
        return None
    return float(np.max(values))


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


def collect_logprobs_from_step(step: dict[str, Any]) -> list[float]:
    metrics = step.get("metrics", {})
    raw_logprobs = metrics.get("logprobs", [])
    if not isinstance(raw_logprobs, list):
        return []

    logprobs = []
    for item in raw_logprobs:
        value = finite_float(item)
        if value is not None:
            logprobs.append(value)
    return logprobs


def entropy_from_logprobs(logprobs: list[float]) -> Optional[float]:
    if not logprobs:
        return None
    return float(-np.mean(np.asarray(logprobs, dtype=np.float64)))


def extract_step_metric(step: dict[str, Any], key: str) -> Optional[float]:
    value = finite_float(step.get(key))
    if value is not None:
        return value

    metrics = step.get("metrics", {})
    if isinstance(metrics, dict):
        value = finite_float(metrics.get(key))
        if value is not None:
            return value

    aliases = {
        "do_agent": ["do_agent", "do_score_agent", "agent_coherence", "agent_do"],
        "do_user": ["do_user", "do_score_user", "user_coherence", "user_do"],
        "da": ["da", "da_score", "inquiry_drift", "repetition_score"],
    }

    for alias in aliases.get(key, []):
        value = finite_float(step.get(alias))
        if value is not None:
            return value
        if isinstance(metrics, dict):
            value = finite_float(metrics.get(alias))
            if value is not None:
                return value

    return None


def calculate_saup_score(step_data: list[dict[str, Any]], config: SAUPConfig) -> dict[str, Any]:
    """
    Entropy/semantic SAUP aggregation.

    Mirrors Agent Tracer's formula:
        W_i = 1 + alpha * Da_i + beta * Do_agent_i + gamma * Do_user_i
        SAUP = sqrt(mean((W_i * U_i)^2))
    """

    if not step_data:
        return {
            "saup_score": None,
            "num_steps": 0,
            "mean_weight": None,
            "std_weight": None,
            "mean_ui": None,
            "std_ui": None,
            "max_ui": None,
            "max_weighted_uncertainty": None,
        }

    weights: list[float] = []
    weighted_uncertainties: list[float] = []
    ui_values: list[float] = []

    for step in step_data:
        ui = finite_float(step.get("ui"))
        if ui is None:
            continue

        da = step.get("da")
        do_agent = step.get("do_agent")
        do_user = step.get("do_user")

        da_val = da if da is not None else 0.0
        do_agent_val = do_agent if do_agent is not None else 0.0
        do_user_val = do_user if do_user is not None else 0.0

        weight = 1.0 + (
            config.alpha * da_val +
            config.beta * do_agent_val +
            config.gamma * do_user_val
        )
        weighted_uncertainty = weight * ui

        weights.append(float(weight))
        weighted_uncertainties.append(float(weighted_uncertainty))
        ui_values.append(float(ui))

    if not weighted_uncertainties:
        return {
            "saup_score": None,
            "num_steps": 0,
            "mean_weight": None,
            "std_weight": None,
            "mean_ui": None,
            "std_ui": None,
            "max_ui": None,
            "max_weighted_uncertainty": None,
        }

    weighted_uncertainties_array = np.asarray(weighted_uncertainties, dtype=np.float64)
    saup_score = float(np.sqrt(np.mean(weighted_uncertainties_array ** 2)))

    return {
        "saup_score": saup_score,
        "num_steps": len(weighted_uncertainties),
        "mean_weight": safe_mean(weights),
        "std_weight": safe_std(weights),
        "mean_ui": safe_mean(ui_values),
        "std_ui": safe_std(ui_values),
        "max_ui": safe_max(ui_values),
        "max_weighted_uncertainty": safe_max(weighted_uncertainties),
    }


def extract_saup_from_trajectory(traj_path: Path, config: SAUPConfig) -> Optional[dict[str, Any]]:
    try:
        trajectory = load_json(traj_path)
    except Exception as exc:
        print(f"[WARN] failed to load {traj_path}: {exc}")
        return None

    step_data: list[dict[str, Any]] = []
    all_logprobs: list[float] = []
    n_agent_steps = 0
    n_steps_with_semantic_signals = 0

    for step in trajectory.get("steps", []):
        if step.get("source") != "agent":
            continue

        n_agent_steps += 1
        logprobs = collect_logprobs_from_step(step)
        if not logprobs:
            continue

        ui = entropy_from_logprobs(logprobs)
        if ui is None:
            continue

        da = extract_step_metric(step, "da")
        do_agent = extract_step_metric(step, "do_agent")
        do_user = extract_step_metric(step, "do_user")
        if da is not None or do_agent is not None or do_user is not None:
            n_steps_with_semantic_signals += 1

        step_data.append(
            {
                "ui": ui,
                "da": da,
                "do_agent": do_agent,
                "do_user": do_user,
            }
        )
        all_logprobs.extend(logprobs)

    if not step_data:
        return None

    saup = calculate_saup_score(step_data, config)
    saup.update(
        {
            "token_entropy": entropy_from_logprobs(all_logprobs),
            "n_logprobs": len(all_logprobs),
            "n_agent_steps": n_agent_steps,
            "n_steps_with_semantic_signals": n_steps_with_semantic_signals,
        }
    )
    return saup


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


def collect_rows(
    root: Path,
    reward_mapping: dict[str, float],
    exception_mapping: dict[str, str],
    config: SAUPConfig,
    include_exceptions: bool,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    rows: list[dict[str, Any]] = []
    counters = {
        "rewarded_trials": len(reward_mapping),
        "missing_trajectory": 0,
        "skipped_exception": 0,
        "missing_saup": 0,
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

        saup_info = extract_saup_from_trajectory(traj_path, config)
        if saup_info is None:
            counters["missing_saup"] += 1
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
                **saup_info,
            }
        )

    counters["used_trials"] = len(rows)
    return rows, counters


def best_youden_threshold(y_true: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    fpr, tpr, thresholds = roc_curve(y_true, scores)
    j_scores = tpr - fpr
    best_idx = int(np.argmax(j_scores))
    return float(thresholds[best_idx]), float(j_scores[best_idx])


def evaluate_scores(df: pd.DataFrame, score_col: str) -> dict[str, Any]:
    y_true = df["failure"].to_numpy(dtype=int)
    scores = df[score_col].to_numpy(dtype=float)

    if len(np.unique(y_true)) < 2:
        raise ValueError("Need both success and failure samples to compute AUROC/AUPRC")

    threshold, youden_j = best_youden_threshold(y_true, scores)
    y_pred = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    return {
        "score_column": score_col,
        "n": int(len(df)),
        "n_success": int((df["success"] == 1).sum()),
        "n_failure": int((df["failure"] == 1).sum()),
        "positive_label": "failure",
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


def maybe_plot_curves(df: pd.DataFrame, score_col: str, root: Path, prefix: str) -> dict[str, str]:
    if plt is None:
        print("[WARN] matplotlib not installed; skip plots")
        return {}

    y_true = df["failure"].to_numpy(dtype=int)
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
    plt.title("GAIA SAUP Failure Detection")
    plt.legend()
    plt.tight_layout()
    plt.savefig(paths["roc_plot"], dpi=200)
    plt.close()

    precision, recall, _ = precision_recall_curve(y_true, scores)
    plt.figure(figsize=(6, 6))
    plt.plot(recall, precision, label=f"AUPRC = {auprc:.4f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("GAIA SAUP Failure Detection")
    plt.legend()
    plt.tight_layout()
    plt.savefig(paths["pr_plot"], dpi=200)
    plt.close()

    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate Harbor GAIA trajectory SAUP scores against rewards."
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
    parser.add_argument(
        "--alpha",
        type=float,
        default=1.0,
        help="SAUP alpha weight for Da.",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=1.0,
        help="SAUP beta weight for Do_agent.",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=1.0,
        help="SAUP gamma weight for Do_user.",
    )
    parser.add_argument(
        "--score-col",
        default="saup_score",
        choices=[
            "saup_score",
            "token_entropy",
            "mean_ui",
            "max_ui",
            "mean_weight",
            "max_weighted_uncertainty",
        ],
        help="Score column used for failure detection.",
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
        default="gaia_saup_eval",
        help="Prefix for CSV, summary JSON, and plots.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = (args.root_flag or args.root or DEFAULT_ROOT).expanduser().resolve()
    result_path = root / "result.json"
    if not result_path.exists():
        raise FileNotFoundError(f"Missing result.json: {result_path}")

    config = SAUPConfig(alpha=args.alpha, beta=args.beta, gamma=args.gamma)

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
        raise RuntimeError("No valid trials found with reward and trajectory uncertainty.")

    df = pd.DataFrame(rows)
    df = df[df[args.score_col].notna()].copy()
    if df.empty:
        raise RuntimeError(f"No non-null scores found for {args.score_col}")

    metrics = evaluate_scores(df, args.score_col)

    csv_path = root / f"{args.output_prefix}.csv"
    summary_path = root / f"{args.output_prefix}_summary.json"
    df.sort_values(args.score_col, ascending=False).to_csv(csv_path, index=False)

    plot_paths = {} if args.no_plot else maybe_plot_curves(df, args.score_col, root, args.output_prefix)

    summary = {
        "root": str(root),
        "eval_key": eval_key,
        "saup_config": asdict(config),
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
    print("GAIA SAUP evaluation")
    print("=" * 80)
    print(f"Root: {root}")
    print(f"Eval key: {eval_key}")
    print(f"Used trials: {metrics['n']} | success: {metrics['n_success']} | failure: {metrics['n_failure']}")
    print(f"Skipped exception trials: {counters['skipped_exception']}")
    print()
    print(f"Score: {args.score_col} (higher means more likely failure)")
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
    print("Top-20 highest risk trials:")
    cols = [
        "trial",
        "reward",
        "success",
        args.score_col,
        "mean_ui",
        "mean_weight",
        "max_weighted_uncertainty",
        "n_agent_steps",
        "n_steps_with_semantic_signals",
    ]
    print(df.sort_values(args.score_col, ascending=False).head(20)[cols].to_string(index=False))


if __name__ == "__main__":
    main()
