#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Save prefix-curve UQ evaluation metrics to JSON.

This standalone script evaluates how different uncertainty scores predict final
task failure as more of the agent trajectory prefix is revealed. It does not
modify existing evaluation scripts.

All methods are evaluated as failure-detection scores:
    higher score => more likely failure.

Examples:
    python evaluate_prefix_uq_curves.py jobs/2026-07-01__10-00-09
    python evaluate_prefix_uq_curves.py jobs/2026-07-01__10-00-09 --mode percent --prefix-percents 0.1,0.2,0.3,0.5,0.7,1.0
    python evaluate_prefix_uq_curves.py jobs/2026-07-01__10-00-09 --mode steps --prefix-steps 1,2,4,8,16,32,64

Output:
    <root>/prefix_uq_curves_percent.json
    or
    <root>/prefix_uq_curves_steps.json
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
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PREFIX_MODULE_PATH = Path(__file__).resolve().parent / "evaluate_gaia_prefix_confidence_metrics.py"


def load_prefix_module():
    spec = importlib.util.spec_from_file_location("prefix_metrics_module", PREFIX_MODULE_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load module from {PREFIX_MODULE_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


pm = load_prefix_module()


METHODS = {
    "entropy": {
        "risk_key": "entropy_risk",
        "num_steps_key": "entropy_n_agent_steps",
        "label": "Entropy",
    },
    "tracer": {
        "risk_key": "tracer_risk",
        "num_steps_key": "tracer_num_steps",
        "label": "TRACER",
    },
    "saup": {
        "risk_key": "saup_risk",
        "num_steps_key": "saup_num_steps",
        "label": "SAUP",
    },
    "uprop": {
        "risk_key": "uprop_risk",
        "num_steps_key": "uprop_num_steps",
        "label": "UProp",
    },
    "tau": {
        "risk_key": "tau_risk",
        "num_steps_key": "tau_num_steps",
        "label": "TAU",
    },
    "trajectory_tau": {
        "risk_key": "trajectory_tau_risk",
        "num_steps_key": "trajectory_tau_num_steps",
        "label": "Trajectory TAU",
    },
}


def save_json(path: Path, data: Any) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def finite_float(value: Any) -> Optional[float]:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def parse_float_list(raw: str) -> list[float]:
    values = []
    for part in raw.split(","):
        part = part.strip()
        if part:
            values.append(float(part))
    if not values:
        raise ValueError("Expected at least one numeric value")
    return values


def parse_int_list(raw: str) -> list[int]:
    values = []
    for part in raw.split(","):
        part = part.strip()
        if part:
            values.append(int(part))
    if not values:
        raise ValueError("Expected at least one integer value")
    return values


def truncate_to_agent_steps(steps: list[dict[str, Any]], prefix_agent_count: int) -> tuple[list[dict[str, Any]], int, int]:
    agent_total = sum(1 for step in steps if step.get("source") == "agent")
    if agent_total == 0:
        return [], 0, 0

    target = max(1, min(int(prefix_agent_count), agent_total))
    agent_seen = 0
    prefix_steps: list[dict[str, Any]] = []
    for step in steps:
        prefix_steps.append(step)
        if step.get("source") == "agent":
            agent_seen += 1
            if agent_seen >= target:
                break
    return prefix_steps, agent_total, target


def load_dataset(root: Path, eval_key: Optional[str], include_exceptions: bool) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    result_data = pm.load_json(root / "result.json")
    resolved_eval_key, eval_payload = pm.get_eval_payload(result_data, eval_key)
    reward_mapping = pm.extract_reward_mapping(eval_payload)
    exception_mapping = pm.extract_exception_mapping(eval_payload)

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

        exception_type = pm.trial_exception_type(trial_dir, exception_mapping)
        if exception_type and not include_exceptions:
            counters["skipped_exception"] += 1
            continue

        traj_path = trial_dir / "agent" / "trajectory.json"
        if not traj_path.exists():
            counters["missing_trajectory"] += 1
            continue

        try:
            trajectory = pm.load_json(traj_path)
        except Exception:
            counters["missing_trajectory"] += 1
            continue

        reward = reward_mapping[trial_name]
        steps = trajectory.get("steps", [])
        dataset.append(
            {
                "trial": trial_name,
                "trial_dir": trial_dir,
                "reward": reward,
                "success": int(reward > 0.5),
                "failure": int(reward <= 0.5),
                "exception_type": exception_type,
                "steps": steps,
                "n_agent_steps_total": sum(1 for step in steps if step.get("source") == "agent"),
            }
        )

    counters["used_trials"] = len(dataset)
    return dataset, {"eval_key": resolved_eval_key, "counters": counters}


def compute_prefix_scores(
    prefix_steps: list[dict[str, Any]],
    trial_dir: Path,
    configs: dict[str, Any],
) -> dict[str, Any]:
    scores: dict[str, Any] = {}
    scores.update(pm.calculate_entropy_prefix(prefix_steps))
    scores.update(pm.calculate_tracer_prefix(prefix_steps, configs["tracer"]))
    scores.update(pm.calculate_saup_prefix(prefix_steps, configs["saup"]))
    scores.update(pm.calculate_uprop_prefix(prefix_steps, configs["uprop"]))
    scores.update(pm.calculate_tau_prefix(prefix_steps, trial_dir, configs["tau"]))
    scores.update(pm.calculate_trajectory_tau_prefix(prefix_steps, trial_dir, configs["trajectory_tau"]))
    return scores


def best_youden_threshold(y_true: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    fpr, tpr, thresholds = roc_curve(y_true, scores)
    j_scores = tpr - fpr
    best_idx = int(np.argmax(j_scores))
    return float(thresholds[best_idx]), float(j_scores[best_idx])


def evaluate_failure_scores(rows: list[dict[str, Any]], score_key: str) -> dict[str, Any]:
    df = pd.DataFrame(rows)
    df = df[df[score_key].notna()].copy()
    if df.empty:
        return {"score_key": score_key, "n": 0, "error": "no_valid_scores"}

    y_true = df["failure"].to_numpy(dtype=int)
    scores = df[score_key].to_numpy(dtype=float)
    if len(np.unique(y_true)) < 2:
        return {"score_key": score_key, "n": int(len(df)), "error": "single_class"}

    threshold, youden_j = best_youden_threshold(y_true, scores)
    y_pred = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "score_key": score_key,
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
    }


def prefix_points_for_item(
    mode: str,
    point_value: float | int,
    steps: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], int, int]:
    if mode == "percent":
        return pm.truncate_to_prefix(steps, float(point_value))
    if mode == "steps":
        return truncate_to_agent_steps(steps, int(point_value))
    raise ValueError(f"Unsupported mode: {mode}")


def run_prefix_curve_experiment(
    dataset: list[dict[str, Any]],
    mode: str,
    points: list[float | int],
    configs: dict[str, Any],
) -> dict[str, Any]:
    point_results = []

    for point in points:
        rows: list[dict[str, Any]] = []
        for item in dataset:
            prefix_steps, agent_total, prefix_agent_count = prefix_points_for_item(mode, point, item["steps"])
            if not prefix_steps:
                continue

            scores = compute_prefix_scores(prefix_steps, item["trial_dir"], configs)
            row = {
                "trial": item["trial"],
                "reward": item["reward"],
                "success": item["success"],
                "failure": item["failure"],
                "exception_type": item["exception_type"],
                "n_agent_steps_total": agent_total,
                "prefix_agent_steps": prefix_agent_count,
                "prefix_percent_actual": None if agent_total == 0 else prefix_agent_count / agent_total,
            }
            for method_name, spec in METHODS.items():
                row[f"{method_name}_score"] = finite_float(scores.get(spec["risk_key"]))
                row[f"{method_name}_num_steps"] = scores.get(spec["num_steps_key"])
            rows.append(row)

        method_metrics = {}
        for method_name in METHODS:
            method_metrics[method_name] = evaluate_failure_scores(rows, f"{method_name}_score")

        point_results.append(
            {
                "mode": mode,
                "point": point,
                "point_label": f"{point:.2f}" if isinstance(point, float) else str(point),
                "n_trials_with_prefix": len(rows),
                "mean_prefix_agent_steps": float(np.mean([r["prefix_agent_steps"] for r in rows])) if rows else None,
                "mean_prefix_percent_actual": float(np.mean([r["prefix_percent_actual"] for r in rows])) if rows else None,
                "method_metrics": method_metrics,
            }
        )

    return {"points": point_results}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate prefix UQ curves and save metrics to JSON."
    )
    parser.add_argument("root", nargs="?", type=Path, default=None)
    parser.add_argument("--root", dest="root_flag", type=Path, default=None)
    parser.add_argument("--eval-key", default=None)
    parser.add_argument("--include-exceptions", action="store_true")
    parser.add_argument(
        "--mode",
        choices=["percent", "steps"],
        default="percent",
        help="Use prefix percentage or absolute agent-step count on the x-axis.",
    )
    parser.add_argument(
        "--prefix-percents",
        default="0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0",
        help="Comma-separated prefix percentages for --mode percent.",
    )
    parser.add_argument(
        "--prefix-steps",
        default="1,2,4,8,16,32,64",
        help="Comma-separated absolute agent step counts for --mode steps.",
    )
    parser.add_argument("--output-json", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = (args.root_flag or args.root).expanduser().resolve()
    if not (root / "result.json").exists():
        raise FileNotFoundError(f"Missing result.json: {root / 'result.json'}")

    if args.mode == "percent":
        points: list[float | int] = parse_float_list(args.prefix_percents)
        for value in points:
            if value <= 0.0 or value > 1.0:
                raise ValueError(f"Prefix percent must be in (0, 1], got {value}")
    else:
        points = parse_int_list(args.prefix_steps)
        for value in points:
            if value <= 0:
                raise ValueError(f"Prefix steps must be positive, got {value}")

    configs = {
        "tracer": pm.TRACERConfig(),
        "saup": pm.SAUPConfig(),
        "uprop": pm.UPropApproxConfig(),
        "tau": pm.TAUConfig(),
        "trajectory_tau": pm.TrajectoryTAUConfig(),
    }
    dataset, dataset_info = load_dataset(root, args.eval_key, args.include_exceptions)
    if not dataset:
        raise RuntimeError("No usable trials found")

    curve = run_prefix_curve_experiment(dataset, args.mode, points, configs)
    output_json = args.output_json or (root / f"prefix_uq_curves_{args.mode}.json")
    result = {
        "root": str(root),
        "eval_key": dataset_info["eval_key"],
        "mode": args.mode,
        "points": points,
        "positive_label": "failure",
        "score_direction": "higher_score_more_likely_failure",
        "methods": METHODS,
        "configs": {
            "tracer": asdict(configs["tracer"]),
            "saup": asdict(configs["saup"]),
            "uprop": asdict(configs["uprop"]),
            "tau": asdict(configs["tau"]),
            "trajectory_tau": asdict(configs["trajectory_tau"]),
        },
        "counters": dataset_info["counters"],
        **curve,
    }
    save_json(output_json, result)

    print("=" * 80)
    print("Prefix UQ curve evaluation")
    print("=" * 80)
    print(f"Root: {root}")
    print(f"Mode: {args.mode}")
    print(f"Points: {points}")
    print(f"Used trials: {dataset_info['counters']['used_trials']}")
    print(f"Saved: {output_json}")
    print()
    for point_result in result["points"]:
        print(f"Point {point_result['point_label']} | n={point_result['n_trials_with_prefix']}")
        for method_name, metrics in point_result["method_metrics"].items():
            if "auroc" in metrics:
                print(f"  {method_name:15s} AUROC={metrics['auroc']:.4f} AUPRC={metrics['auprc']:.4f}")
            else:
                print(f"  {method_name:15s} {metrics.get('error', 'no_metric')}")


if __name__ == "__main__":
    main()
