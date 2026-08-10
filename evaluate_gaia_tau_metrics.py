#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Evaluate TAU-style confidence/risk metrics on Harbor GAIA runs.

Reference idea:
    Agent-Confidence/agent-tracer/src/tau2/metrics/uncertainty_prop.py

Core formula adapted from calculate_tau_score:
    risk_i = U_i * (1 + alpha * H_i + beta * C_i)
    tau_score = mean(risk_i)

Where:
    - U_i: step uncertainty from token logprobs
    - H_i: temporal influence from previous trajectory history
    - C_i: interaction consistency with previous user turn and task goal

Practical note:
The original file uses an external embedding API for semantic distance. This
script uses a local lexical distance fallback so it can run directly on Harbor
logs without network dependencies.

Usage:
    python evaluate_gaia_tau_metrics.py /path/to/harbor_job
    python evaluate_gaia_tau_metrics.py --root /path/to/harbor_job

Outputs:
    <root>/gaia_tau_eval.csv
    <root>/gaia_tau_eval_summary.json
    <root>/gaia_tau_eval_roc.png
    <root>/gaia_tau_eval_pr.png
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

STOP_WORDS = {
    "the", "is", "are", "was", "were", "be", "been", "being",
    "a", "an", "to", "of", "in", "on", "at", "for", "with",
    "from", "by", "as", "or", "and", "but", "if", "then",
    "this", "that", "these", "those", "it", "its", "i", "you",
    "he", "she", "we", "they", "my", "your", "his", "her",
    "am", "can", "will", "would", "could", "should", "may",
    "have", "has", "had", "do", "does", "did",
    "please", "sorry", "wait", "help", "need", "want",
}


@dataclass
class TAUConfig:
    alpha: float = 0.8
    beta: float = 0.8
    decay: float = 0.75
    lambda_goal: float = 0.4


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


def tokenize(text: str) -> set[str]:
    if not text:
        return set()

    tokens = set()
    for raw in text.lower().split():
        cleaned = raw.strip('.,!?;:()[]{}"\'-')
        if not cleaned or cleaned in STOP_WORDS:
            continue
        if cleaned.replace(".", "").replace("-", "").isdigit():
            continue
        tokens.add(cleaned)
    return tokens


def calculate_semantic_distance(text_a: str, text_b: str) -> float:
    """
    Local lexical fallback for the semantic-distance role in TAU.
    Uses Jaccard distance over filtered tokens.
    """

    tokens_a = tokenize(text_a)
    tokens_b = tokenize(text_b)
    if not tokens_a or not tokens_b:
        return 0.0

    union = tokens_a | tokens_b
    if not union:
        return 0.0

    intersection = tokens_a & tokens_b
    similarity = float(len(intersection)) / float(len(union))
    return 1.0 - similarity


def build_step_representation(step: dict[str, Any]) -> str:
    parts: list[str] = []

    message = step.get("message")
    if isinstance(message, str) and message.strip():
        parts.append(message)

    tool_calls = step.get("tool_calls")
    if tool_calls:
        parts.append(json.dumps(tool_calls, ensure_ascii=False, sort_keys=True))

    observation = step.get("observation")
    if observation:
        parts.append(json.dumps(observation, ensure_ascii=False, sort_keys=True))

    return "\n".join(parts)


def collect_logprobs_from_step(step: dict[str, Any]) -> list[float]:
    metrics = step.get("metrics", {})
    raw_logprobs = metrics.get("logprobs", [])
    if not isinstance(raw_logprobs, list):
        return []

    values = []
    for item in raw_logprobs:
        value = finite_float(item)
        if value is not None:
            values.append(value)
    return values


def entropy_from_logprobs(logprobs: list[float]) -> Optional[float]:
    if not logprobs:
        return None
    return float(-np.mean(np.asarray(logprobs, dtype=np.float64)))


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


def calculate_temporal_influence(current_text: str, history_texts: list[str], decay: float) -> float:
    if not current_text.strip() or not history_texts:
        return 0.0

    influence = 0.0
    weight_sum = 0.0

    for i, past_text in enumerate(reversed(history_texts)):
        if not past_text.strip():
            continue
        weight = decay ** i
        distance = calculate_semantic_distance(current_text, past_text)
        influence += weight * distance
        weight_sum += weight

    if weight_sum == 0.0:
        return 0.0
    return influence / weight_sum


def calculate_interaction_consistency(
    agent_text: str,
    prev_user_text: Optional[str],
    goal_text: Optional[str],
    lambda_goal: float,
) -> float:
    score = 0.0
    if prev_user_text:
        score += calculate_semantic_distance(agent_text, prev_user_text)
    if goal_text:
        score += lambda_goal * calculate_semantic_distance(agent_text, goal_text)
    return score


def calculate_tau_from_trajectory(trial_dir: Path, config: TAUConfig) -> Optional[dict[str, Any]]:
    traj_path = trial_dir / "agent" / "trajectory.json"
    try:
        traj_data = load_json(traj_path)
    except Exception as exc:
        print(f"[WARN] failed to load {traj_path}: {exc}")
        return None

    goal_text = extract_goal_text(traj_data, trial_dir)
    history_texts: list[str] = []
    prev_user: Optional[str] = None
    risks: list[float] = []
    ui_values: list[float] = []
    temporal_values: list[float] = []
    consistency_values: list[float] = []
    all_logprobs: list[float] = []
    n_agent_steps = 0

    for step in traj_data.get("steps", []):
        source = step.get("source")
        if source not in {"agent", "user"}:
            continue

        text = build_step_representation(step)

        if source == "agent":
            n_agent_steps += 1
            logprobs = collect_logprobs_from_step(step)
            if logprobs:
                ui = entropy_from_logprobs(logprobs)
                all_logprobs.extend(logprobs)
            else:
                ui = None

            if ui is not None:
                h_val = calculate_temporal_influence(text, history_texts, config.decay)
                c_val = calculate_interaction_consistency(text, prev_user, goal_text, config.lambda_goal)
                risk = ui * (1.0 + config.alpha * h_val + config.beta * c_val)

                risks.append(float(risk))
                ui_values.append(float(ui))
                temporal_values.append(float(h_val))
                consistency_values.append(float(c_val))

        history_texts.append(text)
        if source == "user":
            prev_user = text

    if not risks:
        return None

    tau_score = float(np.mean(risks))
    tau_confidence = float(1.0 / (1.0 + tau_score))

    return {
        "tau_score": tau_score,
        "tau_confidence": tau_confidence,
        "num_steps": len(risks),
        "mean_risk": safe_mean(risks),
        "max_risk": safe_max(risks),
        "mean_ui": safe_mean(ui_values),
        "mean_temporal_influence": safe_mean(temporal_values),
        "mean_interaction_consistency": safe_mean(consistency_values),
        "token_entropy": entropy_from_logprobs(all_logprobs),
        "n_logprobs": len(all_logprobs),
        "n_agent_steps": n_agent_steps,
    }


def collect_rows(
    root: Path,
    reward_mapping: dict[str, float],
    exception_mapping: dict[str, str],
    config: TAUConfig,
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

        tau_info = calculate_tau_from_trajectory(trial_dir, config)
        if tau_info is None:
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
                **tau_info,
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
        threshold_positive = "higher"
    elif positive_label == "success":
        y_true = df["success"].to_numpy(dtype=int)
        threshold_positive = "higher"
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
        "threshold_positive_direction": threshold_positive,
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
    plt.title(f"GAIA TAU {title_label} Detection")
    plt.legend()
    plt.tight_layout()
    plt.savefig(paths["roc_plot"], dpi=200)
    plt.close()

    precision, recall, _ = precision_recall_curve(y_true, scores)
    plt.figure(figsize=(6, 6))
    plt.plot(recall, precision, label=f"AUPRC = {auprc:.4f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"GAIA TAU {title_label} Detection")
    plt.legend()
    plt.tight_layout()
    plt.savefig(paths["pr_plot"], dpi=200)
    plt.close()

    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate Harbor GAIA trajectory TAU scores against rewards."
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
        default=0.8,
        help="TAU alpha weight for temporal influence.",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=0.8,
        help="TAU beta weight for interaction consistency.",
    )
    parser.add_argument(
        "--decay",
        type=float,
        default=0.75,
        help="Temporal decay for history influence.",
    )
    parser.add_argument(
        "--lambda-goal",
        type=float,
        default=0.4,
        help="Goal consistency weight inside interaction consistency.",
    )
    parser.add_argument(
        "--score-col",
        default="tau_score",
        choices=[
            "tau_confidence",
            "tau_score",
            "token_entropy",
            "mean_ui",
            "mean_risk",
            "max_risk",
            "mean_temporal_influence",
            "mean_interaction_consistency",
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
        default="gaia_tau_eval",
        help="Prefix for CSV, summary JSON, and plots.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = (args.root_flag or args.root or DEFAULT_ROOT).expanduser().resolve()
    result_path = root / "result.json"
    if not result_path.exists():
        raise FileNotFoundError(f"Missing result.json: {result_path}")

    config = TAUConfig(
        alpha=args.alpha,
        beta=args.beta,
        decay=args.decay,
        lambda_goal=args.lambda_goal,
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
        raise RuntimeError("No valid trials found with reward and trajectory uncertainty.")

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
        "tau_config": asdict(config),
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
    print("GAIA TAU evaluation")
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
        "mean_ui",
        "mean_temporal_influence",
        "mean_interaction_consistency",
        "n_agent_steps",
    ]
    print(df.sort_values(args.score_col, ascending=False).head(20)[cols].to_string(index=False))


if __name__ == "__main__":
    main()
