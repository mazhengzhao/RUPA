#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Evaluate a UProp-inspired baseline on Harbor GAIA runs.

Reference:
    UProp: Investigating the Uncertainty Propagation of LLMs in Multi-Step
    Agentic Decision-Making (arXiv:2506.17419v1).

The full UProp method samples multiple Trajectory-Dependent Decision Processes
(TDPs) and estimates pointwise mutual information (PMI). Harbor GAIA result
folders usually contain only one realized trajectory per trial, so this script
implements a deterministic offline approximation:

    1. Intrinsic uncertainty IU_t is estimated as predictive entropy proxy
       from stored token logprobs: IU_t = -mean(logprob_t).
    2. Extrinsic propagation is approximated from the preceding-decision
       semantic variance. For step t, the previous agent decision y_{t-1}
       is compared with earlier agent decisions by string similarity, and
       PMI_t is approximated as -log(mean GaussianKernel(distance)).
    3. The per-step UProp score is IU_t + pmi_weight * cumulative_PMI_t, with
       a length-normalized aggregate trajectory score.

All scores follow the project convention:
    higher score => more likely failure.

Outputs:
    <root>/gaia_uprop_eval.csv
    <root>/gaia_uprop_eval_summary.json
    <root>/gaia_uprop_roc.png       (unless --no-plot)
    <root>/gaia_uprop_pr.png        (unless --no-plot)
"""

from __future__ import annotations

import argparse
import difflib
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
SKIPPED_EXCEPTION_TYPES = {"AgentTimeoutError", "TimeoutError", "CancelledError"}


@dataclass
class UPropApproxConfig:
    pmi_weight: float = 1.0
    kernel_tau: float = 1.0
    max_history: int = 8
    min_kernel: float = 1e-8


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


def entropy_from_logprobs(logprobs: list[float]) -> Optional[float]:
    if not logprobs:
        return None
    return float(-np.mean(np.asarray(logprobs, dtype=np.float64)))


def text_of_step(step: dict[str, Any]) -> str:
    chunks = []
    message = step.get("message")
    if isinstance(message, str) and message.strip():
        chunks.append(message.strip())
    for call in step.get("tool_calls") or []:
        if isinstance(call, dict):
            fn = call.get("function_name")
            args = call.get("arguments")
            chunks.append(json.dumps({"function": fn, "arguments": args}, ensure_ascii=False, sort_keys=True))
    return "\n".join(chunks)


def string_distance(a: str, b: str) -> float:
    if not a or not b:
        return 1.0
    ratio = difflib.SequenceMatcher(None, a[:4000], b[:4000]).ratio()
    return float(max(0.0, min(1.0, 1.0 - ratio)))


def gaussian_kernel(distance: float, tau: float) -> float:
    tau = max(1e-6, float(tau))
    return float(math.exp(-0.5 * (distance / tau) ** 2))


def pmi_proxy_from_preceding_variance(
    previous_text: str,
    historical_texts: list[str],
    config: UPropApproxConfig,
) -> float:
    if not previous_text or not historical_texts:
        return 0.0
    history = historical_texts[-max(1, config.max_history):]
    kernels = [
        gaussian_kernel(string_distance(previous_text, other), config.kernel_tau)
        for other in history
        if other
    ]
    if not kernels:
        return 0.0
    neighborhood_mass = max(config.min_kernel, float(np.mean(kernels)))
    return float(-math.log(neighborhood_mass))


def calculate_uprop_approx(
    step_infos: list[dict[str, Any]],
    config: UPropApproxConfig,
) -> dict[str, Any]:
    if not step_infos:
        return {
            "uprop_score": None,
            "intrinsic_uncertainty": None,
            "extrinsic_uncertainty": None,
            "num_steps": 0,
        }

    texts: list[str] = []
    intrinsic_values: list[float] = []
    pmi_values: list[float] = []
    step_scores: list[float] = []
    cumulative_pmi = 0.0

    for idx, info in enumerate(step_infos):
        iu = float(info["intrinsic_uncertainty"])
        previous_text = texts[-1] if texts else ""
        pmi = pmi_proxy_from_preceding_variance(previous_text, texts[:-1], config)
        cumulative_pmi += pmi
        step_score = iu + config.pmi_weight * cumulative_pmi
        intrinsic_values.append(iu)
        pmi_values.append(pmi)
        step_scores.append(float(step_score))
        texts.append(info["text"])

    # Mirrors UProp's length-normalization idea by normalizing with the average
    # uncertainty inflation ratio induced by extrinsic propagation.
    sigmas = [
        1.0 + (config.pmi_weight * pmi / max(iu, 1e-8))
        for iu, pmi in zip(intrinsic_values, pmi_values)
    ]
    lambda_z = float(np.sum(sigmas)) if sigmas else 1.0
    uprop_score = float(np.sum(step_scores) / max(lambda_z, 1e-8))

    return {
        "uprop_score": uprop_score,
        "intrinsic_uncertainty": safe_mean(intrinsic_values),
        "extrinsic_uncertainty": safe_mean(pmi_values),
        "cumulative_pmi": float(cumulative_pmi),
        "mean_step_uprop": safe_mean(step_scores),
        "lambda_z": lambda_z,
        "num_steps": len(step_infos),
        "max_intrinsic_uncertainty": float(np.max(intrinsic_values)),
        "max_pmi_proxy": float(np.max(pmi_values)) if pmi_values else 0.0,
    }


def extract_uprop_from_trajectory(traj_path: Path, config: UPropApproxConfig) -> Optional[dict[str, Any]]:
    try:
        trajectory = load_json(traj_path)
    except Exception as exc:
        print(f"[WARN] failed to load {traj_path}: {exc}")
        return None

    step_infos: list[dict[str, Any]] = []
    all_logprobs: list[float] = []
    n_agent_steps = 0
    for step in trajectory.get("steps", []):
        if step.get("source") != "agent":
            continue
        n_agent_steps += 1
        logprobs = collect_logprobs_from_step(step)
        if not logprobs:
            continue
        iu = entropy_from_logprobs(logprobs)
        if iu is None:
            continue
        all_logprobs.extend(logprobs)
        step_infos.append({"intrinsic_uncertainty": iu, "text": text_of_step(step)})

    if not step_infos:
        return None
    result = calculate_uprop_approx(step_infos, config)
    result.update(
        {
            "token_entropy": entropy_from_logprobs(all_logprobs),
            "n_logprobs": len(all_logprobs),
            "n_agent_steps": n_agent_steps,
        }
    )
    return result


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
    plt.title("UProp-inspired Failure Detection")
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
    plt.title("UProp-inspired Failure Detection")
    plt.legend()
    plt.tight_layout()
    plt.savefig(pr_path, dpi=200)
    plt.close()
    paths["pr"] = str(pr_path)
    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a UProp-inspired baseline on Harbor GAIA runs.")
    parser.add_argument("root", nargs="?", type=Path, default=None)
    parser.add_argument("--root", dest="root_flag", type=Path, default=None)
    parser.add_argument("--eval-key", default=None)
    parser.add_argument("--include-exceptions", action="store_true")
    parser.add_argument("--score-col", default="uprop_score", choices=["uprop_score", "intrinsic_uncertainty", "extrinsic_uncertainty", "mean_step_uprop"])
    parser.add_argument("--pmi-weight", type=float, default=1.0)
    parser.add_argument("--kernel-tau", type=float, default=1.0)
    parser.add_argument("--max-history", type=int, default=8)
    parser.add_argument("--output-prefix", default="gaia_uprop")
    parser.add_argument("--no-plot", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = (args.root_flag or args.root or DEFAULT_ROOT).expanduser().resolve()
    if not (root / "result.json").exists():
        raise FileNotFoundError(f"Missing result.json: {root / 'result.json'}")

    config = UPropApproxConfig(
        pmi_weight=args.pmi_weight,
        kernel_tau=args.kernel_tau,
        max_history=args.max_history,
    )
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
        info = extract_uprop_from_trajectory(traj_path, config)
        if info is None:
            counters["missing_score"] += 1
            continue
        reward = reward_mapping[trial_dir.name]
        success = int(reward > 0.5)
        rows.append({"trial": trial_dir.name, "reward": reward, "success": success, "failure": 1 - success, "exception_type": exception_type, **info})

    df = pd.DataFrame(rows)
    counters["used_trials"] = int(len(df))
    if df.empty:
        raise RuntimeError("No valid trials found with UProp-inspired scores")

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
            "baseline": "uprop_inspired_single_trajectory_approximation",
            "note": "The original UProp requires multiple sampled TDPs. This is an offline approximation from one logged Harbor trajectory.",
            "config": asdict(config),
            "counters": counters,
            "metrics": metrics,
            "outputs": {"csv": str(csv_path), "summary_json": str(summary_path), **plot_paths},
        },
    )

    print("=" * 80)
    print("GAIA UProp-inspired evaluation")
    print("=" * 80)
    print(f"Root: {root}")
    print(f"Eval key: {eval_key}")
    print("Baseline note: single-trajectory approximation of UProp; not full TDP sampling.")
    print(f"Score: {args.score_col} (higher means more likely failure)")
    print(f"Used trials: {len(df)}")
    print(f"AUROC: {metrics['auroc']:.6f}")
    print(f"AUPRC: {metrics['auprc']:.6f}")
    print(f"F1 @ threshold: {metrics['f1_at_threshold']:.6f}")
    print(f"Saved CSV: {csv_path}")
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()
