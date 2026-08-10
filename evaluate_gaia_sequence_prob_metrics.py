#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Evaluate a sequence-generation-probability baseline on Harbor GAIA runs.

The raw sequence probability is higher for more confident/success-like
generations. For consistency with the other Agent UQ scripts, the default
failure-detection score is the step-level cumulative negative log-likelihood:

    sequence_step_nll = - sum_t mean_token_logprob(step_t)

This preserves sequence accumulation while reducing direct token-count bias.
Higher score => more likely failure.

Outputs:
    <root>/gaia_sequence_prob_eval.csv
    <root>/gaia_sequence_prob_eval_summary.json
    <root>/gaia_sequence_prob_roc.png       (unless --no-plot)
    <root>/gaia_sequence_prob_pr.png        (unless --no-plot)
"""

from __future__ import annotations

import argparse
import json
import math
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
SKIPPED_EXCEPTION_TYPES = {"AgentTimeoutError", "TimeoutError", "CancelledError"}


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
            raise ValueError(f"Eval key {eval_key!r} not found. Available: {', '.join(evals)}")
        return eval_key, evals[eval_key]
    first_key = next(iter(evals.keys()))
    return first_key, evals[first_key]


def extract_reward_mapping(eval_payload: dict[str, Any]) -> dict[str, float]:
    mapping: dict[str, float] = {}
    for reward_str, trial_names in eval_payload.get("reward_stats", {}).get("reward", {}).items():
        reward = finite_float(reward_str)
        if reward is None:
            continue
        for trial_name in trial_names:
            mapping[str(trial_name)] = reward
    return mapping


def extract_exception_mapping(eval_payload: dict[str, Any]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for exception_type, trial_names in eval_payload.get("exception_stats", {}).items():
        for trial_name in trial_names:
            mapping[str(trial_name)] = str(exception_type)
    return mapping


def collect_logprobs_from_step(step: dict[str, Any]) -> list[float]:
    raw_logprobs = step.get("metrics", {}).get("logprobs", [])
    if not isinstance(raw_logprobs, list):
        return []
    values: list[float] = []
    for item in raw_logprobs:
        value = finite_float(item)
        if value is not None:
            values.append(value)
    return values


def extract_sequence_prob_from_trajectory(traj_path: Path) -> Optional[dict[str, Any]]:
    try:
        trajectory = load_json(traj_path)
    except Exception as exc:
        print(f"[WARN] failed to load {traj_path}: {exc}")
        return None

    all_logprobs: list[float] = []
    step_mean_logprobs: list[float] = []
    n_agent_steps = 0

    for step in trajectory.get("steps", []):
        if step.get("source") != "agent":
            continue
        n_agent_steps += 1
        logprobs = collect_logprobs_from_step(step)
        if not logprobs:
            continue
        all_logprobs.extend(logprobs)
        step_mean_logprobs.append(float(np.mean(np.asarray(logprobs, dtype=np.float64))))

    if not all_logprobs:
        return None

    arr = np.asarray(all_logprobs, dtype=np.float64)
    sequence_logprob = float(np.sum(arr))
    mean_token_logprob = float(np.mean(arr))
    mean_step_logprob = float(np.mean(step_mean_logprobs)) if step_mean_logprobs else None
    step_logprob_sum = float(np.sum(np.asarray(step_mean_logprobs, dtype=np.float64)))
    return {
        "sequence_step_nll": float(-step_logprob_sum),
        "sequence_nll": float(-mean_token_logprob),
        "sequence_logprob": sequence_logprob,
        "step_logprob_sum": step_logprob_sum,
        "mean_token_logprob": mean_token_logprob,
        "mean_step_logprob": mean_step_logprob,
        "geometric_mean_probability": float(math.exp(max(-745.0, min(0.0, mean_token_logprob)))),
        "sequence_probability_clipped": float(math.exp(max(-745.0, min(0.0, sequence_logprob)))),
        "n_logprobs": len(all_logprobs),
        "n_agent_steps": n_agent_steps,
    }


def best_youden_threshold(y_true: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    fpr, tpr, thresholds = roc_curve(y_true, scores)
    j_scores = tpr - fpr
    best_idx = int(np.argmax(j_scores))
    return float(thresholds[best_idx]), float(j_scores[best_idx])


def evaluate_scores(df: pd.DataFrame, score_col: str) -> dict[str, Any]:
    y_true = df["failure"].to_numpy(dtype=int)
    scores = df[score_col].to_numpy(dtype=float)
    if len(np.unique(y_true)) < 2:
        raise ValueError("Need both success and failure classes to compute metrics")
    threshold, youden_j = best_youden_threshold(y_true, scores)
    y_pred = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "score_column": score_col,
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
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "score_stats_by_success": df.groupby("success")[score_col].describe().to_dict(),
    }


def maybe_plot_curves(df: pd.DataFrame, score_col: str, root: Path, prefix: str) -> dict[str, str]:
    if plt is None:
        return {}
    y_true = df["failure"].to_numpy(dtype=int)
    scores = df[score_col].to_numpy(dtype=float)
    auroc = roc_auc_score(y_true, scores)
    auprc = average_precision_score(y_true, scores)
    paths: dict[str, str] = {}

    roc_path = root / f"{prefix}_roc.png"
    fpr, tpr, _ = roc_curve(y_true, scores)
    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, label=f"AUROC = {auroc:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Sequence Probability Failure Detection")
    plt.legend()
    plt.tight_layout()
    plt.savefig(roc_path, dpi=200)
    plt.close()
    paths["roc"] = str(roc_path)

    pr_path = root / f"{prefix}_pr.png"
    precision, recall, _ = precision_recall_curve(y_true, scores)
    plt.figure(figsize=(6, 6))
    plt.plot(recall, precision, label=f"AUPRC = {auprc:.4f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Sequence Probability Failure Detection")
    plt.legend()
    plt.tight_layout()
    plt.savefig(pr_path, dpi=200)
    plt.close()
    paths["pr"] = str(pr_path)
    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate sequence generation probability on Harbor GAIA runs.")
    parser.add_argument("root", nargs="?", type=Path, default=None)
    parser.add_argument("--root", dest="root_flag", type=Path, default=None)
    parser.add_argument("--eval-key", default=None)
    parser.add_argument("--include-exceptions", action="store_true")
    parser.add_argument(
        "--score-col",
        default="sequence_step_nll",
        choices=["sequence_step_nll", "sequence_nll"],
        help=(
            "Default sequence_step_nll is -sum of per-step mean logprob. "
            "sequence_nll is -mean token logprob and matches the entropy proxy."
        ),
    )
    parser.add_argument("--output-prefix", default="gaia_sequence_prob")
    parser.add_argument("--no-plot", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = (args.root_flag or args.root or DEFAULT_ROOT).expanduser().resolve()
    if not (root / "result.json").exists():
        raise FileNotFoundError(f"Missing result.json: {root / 'result.json'}")

    result_data = load_json(root / "result.json")
    eval_key, eval_payload = get_eval_payload(result_data, args.eval_key)
    reward_mapping = extract_reward_mapping(eval_payload)
    exception_mapping = extract_exception_mapping(eval_payload)

    rows: list[dict[str, Any]] = []
    counters = {"rewarded_trials": len(reward_mapping), "skipped_exception": 0, "missing_trajectory": 0, "missing_score": 0}
    for trial_dir in sorted(root.iterdir()):
        if not trial_dir.is_dir() or trial_dir.name not in reward_mapping:
            continue
        exception_type = exception_mapping.get(trial_dir.name)
        if exception_type in SKIPPED_EXCEPTION_TYPES and not args.include_exceptions:
            counters["skipped_exception"] += 1
            continue
        traj_path = trial_dir / "agent" / "trajectory.json"
        if not traj_path.exists():
            counters["missing_trajectory"] += 1
            continue
        info = extract_sequence_prob_from_trajectory(traj_path)
        if info is None:
            counters["missing_score"] += 1
            continue
        reward = reward_mapping[trial_dir.name]
        success = int(reward > 0.5)
        rows.append({"trial": trial_dir.name, "reward": reward, "success": success, "failure": 1 - success, "exception_type": exception_type, **info})

    df = pd.DataFrame(rows)
    counters["used_trials"] = int(len(df))
    if df.empty:
        raise RuntimeError("No valid trials found with sequence probability scores")

    metrics = evaluate_scores(df, args.score_col)
    csv_path = root / f"{args.output_prefix}_eval.csv"
    summary_path = root / f"{args.output_prefix}_eval_summary.json"
    df.sort_values(args.score_col, ascending=False).to_csv(csv_path, index=False)
    plot_paths = {} if args.no_plot else maybe_plot_curves(df, args.score_col, root, args.output_prefix)
    save_json(
        summary_path,
        {
            "root": str(root),
            "eval_key": eval_key,
            "positive_label": "failure",
            "score_direction": "higher_score_more_likely_failure",
            "baseline": "sequence_generation_probability",
            "default_score_definition": "sequence_step_nll = -sum_t mean_token_logprob(step_t)",
            "counters": counters,
            "metrics": metrics,
            "outputs": {"csv": str(csv_path), "summary_json": str(summary_path), **plot_paths},
        },
    )

    print("=" * 80)
    print("GAIA Sequence Probability evaluation")
    print("=" * 80)
    print(f"Root: {root}")
    print(f"Eval key: {eval_key}")
    print(f"Score: {args.score_col} (higher means more likely failure)")
    print(f"Used trials: {len(df)}")
    print(f"AUROC: {metrics['auroc']:.6f}")
    print(f"AUPRC: {metrics['auprc']:.6f}")
    print(f"F1 @ threshold: {metrics['f1_at_threshold']:.6f}")
    print(f"Saved CSV: {csv_path}")
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()
