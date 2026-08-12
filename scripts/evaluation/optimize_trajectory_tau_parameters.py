#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Random-search optimizer for trajectory-aware TAU on Harbor GAIA runs.

This script searches over the parameters used by:
    evaluate_gaia_trajectory_tau_metrics.py
    agent-tracer/src/tau2/metrics/trajectory_tau.py

Objective:
    1. Maximize AUROC
    2. Break ties with AUPRC

Usage:
    python optimize_trajectory_tau_parameters.py /path/to/harbor_job
    python optimize_trajectory_tau_parameters.py /path/to/harbor_job --n-random 300
"""

from __future__ import annotations

import argparse
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
from sklearn.metrics import average_precision_score, roc_auc_score


DEFAULT_ROOT = Path("~/jobs/2026-05-16__19-11-56")
PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRAJECTORY_TAU_PATH = PROJECT_ROOT / "agent-tracer" / "src" / "tau2" / "metrics" / "trajectory_tau.py"


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


def load_dataset(
    root: Path,
    eval_key: Optional[str],
    include_exceptions: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
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
            continue

        dataset.append(
            {
                "trial": trial_name,
                "success": int(reward_mapping[trial_name] > 0.5),
                "goal_text": extract_goal_text(traj_data, trial_dir),
                "steps": traj_data.get("steps", []),
            }
        )

    counters["used_trials"] = len(dataset)
    return dataset, {"eval_key": resolved_eval_key, "counters": counters}


def evaluate_config(dataset: list[dict[str, Any]], config: Any) -> dict[str, Any]:
    rows = []
    for item in dataset:
        result = calculate_trajectory_tau_score(item["steps"], item["goal_text"], config)
        tau_confidence = finite_float(result.get("tau_confidence"))
        tau_score = finite_float(result.get("tau_score"))
        if tau_confidence is None or tau_score is None:
            continue
        rows.append(
            {
                "trial": item["trial"],
                "success": item["success"],
                "tau_confidence": tau_confidence,
                "tau_score": tau_score,
            }
        )

    if not rows:
        raise ValueError("No usable scores produced")

    df = pd.DataFrame(rows)
    y_true = df["success"].to_numpy(dtype=int)
    scores = df["tau_confidence"].to_numpy(dtype=float)

    if len(np.unique(y_true)) < 2:
        raise ValueError("Need both success and failure classes")

    return {
        "n": int(len(df)),
        "n_success": int((df["success"] == 1).sum()),
        "n_failure": int((df["success"] == 0).sum()),
        "auroc": float(roc_auc_score(y_true, scores)),
        "auprc": float(average_precision_score(y_true, scores)),
    }


def sample_config(rng: random.Random) -> Any:
    alpha = rng.uniform(0.2, 2.5)
    beta = rng.uniform(0.2, 2.5)

    raw_weights = [
        rng.uniform(0.05, 1.0),
        rng.uniform(0.05, 1.0),
        rng.uniform(0.05, 1.0),
        rng.uniform(0.05, 1.0),
    ]
    weight_sum = sum(raw_weights)
    momentum_weight, repetition_weight, observation_weight, stagnation_weight = [
        value / weight_sum for value in raw_weights
    ]

    interaction_goal_weight = rng.uniform(0.2, 0.9)
    interaction_user_weight = 1.0 - interaction_goal_weight

    history_decay = rng.uniform(0.55, 0.98)
    uncertainty_decay = rng.uniform(0.55, 0.98)
    recent_window = rng.randint(3, 8)
    novelty_window = rng.randint(2, 6)
    graph_weight = rng.uniform(0.05, 0.60)
    graph_uncertainty_weight = rng.uniform(0.10, 0.70)
    graph_edge_decay = rng.uniform(0.60, 0.98)
    graph_max_neighbors = rng.randint(4, 12)

    return TrajectoryTAUConfig(
        alpha=alpha,
        beta=beta,
        momentum_weight=momentum_weight,
        repetition_weight=repetition_weight,
        observation_weight=observation_weight,
        stagnation_weight=stagnation_weight,
        interaction_goal_weight=interaction_goal_weight,
        interaction_user_weight=interaction_user_weight,
        history_decay=history_decay,
        uncertainty_decay=uncertainty_decay,
        recent_window=recent_window,
        novelty_window=novelty_window,
        graph_weight=graph_weight,
        graph_uncertainty_weight=graph_uncertainty_weight,
        graph_edge_decay=graph_edge_decay,
        graph_max_neighbors=graph_max_neighbors,
    )


def rank_key(result: dict[str, Any]) -> tuple[float, float]:
    return (result["auroc"], result["auprc"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Optimize trajectory-aware TAU parameters on a Harbor GAIA job."
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
        "--include-exceptions",
        action="store_true",
        help="Include trials with exception_info/exception_stats if they have rewards and trajectories.",
    )
    parser.add_argument(
        "--n-random",
        type=int,
        default=200,
        help="Number of random configurations to test.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=20,
        help="How many top configurations to save.",
    )
    parser.add_argument(
        "--output-prefix",
        default="trajectory_tau_optimization",
        help="Prefix for output CSV and JSON files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = (args.root_flag or args.root or DEFAULT_ROOT).expanduser().resolve()
    result_path = root / "result.json"
    if not result_path.exists():
        raise FileNotFoundError(f"Missing result.json: {result_path}")

    rng = random.Random(args.seed)
    dataset, meta = load_dataset(root, args.eval_key, args.include_exceptions)
    if not dataset:
        raise RuntimeError("No usable trials found for optimization.")

    results = []
    best_result: Optional[dict[str, Any]] = None

    for idx in range(args.n_random):
        config = sample_config(rng)
        try:
            metrics = evaluate_config(dataset, config)
        except Exception as exc:
            results.append(
                {
                    "iteration": idx,
                    "status": "error",
                    "error": str(exc),
                    **asdict(config),
                }
            )
            continue

        row = {
            "iteration": idx,
            "status": "ok",
            **metrics,
            **asdict(config),
        }
        results.append(row)

        if best_result is None or rank_key(row) > rank_key(best_result):
            best_result = row

    if best_result is None:
        raise RuntimeError("No successful configuration found during search.")

    results_df = pd.DataFrame(results)
    ok_df = results_df[results_df["status"] == "ok"].copy()
    ok_df = ok_df.sort_values(["auroc", "auprc"], ascending=[False, False])

    output_csv = root / f"{args.output_prefix}.csv"
    output_json = root / f"{args.output_prefix}_best.json"

    results_df.to_csv(output_csv, index=False)
    save_json(
        output_json,
        {
            "root": str(root),
            "eval_key": meta["eval_key"],
            "seed": args.seed,
            "n_random": args.n_random,
            "include_exceptions": bool(args.include_exceptions),
            "dataset_counters": meta["counters"],
            "best_result": best_result,
            "top_results": ok_df.head(args.top_k).to_dict(orient="records"),
            "outputs": {
                "csv": str(output_csv),
                "best_json": str(output_json),
            },
        },
    )

    print("=" * 80)
    print("Trajectory TAU optimization")
    print("=" * 80)
    print(f"Root: {root}")
    print(f"Eval key: {meta['eval_key']}")
    print(f"Usable trials: {meta['counters']['used_trials']}")
    print(f"Random configs tested: {args.n_random}")
    print()
    print(f"Best AUROC: {best_result['auroc']:.6f}")
    print(f"Best AUPRC: {best_result['auprc']:.6f}")
    print("Best config:")
    for key, value in best_result.items():
        if key in {"iteration", "status", "n", "n_success", "n_failure", "auroc", "auprc"}:
            continue
        print(f"  {key}: {value}")
    print()
    print("Top-10 configs:")
    display_cols = [
        "iteration",
        "auroc",
        "auprc",
        "alpha",
        "beta",
        "momentum_weight",
        "repetition_weight",
        "observation_weight",
        "stagnation_weight",
        "interaction_goal_weight",
        "history_decay",
        "uncertainty_decay",
        "recent_window",
        "novelty_window",
        "graph_weight",
        "graph_uncertainty_weight",
        "graph_edge_decay",
        "graph_max_neighbors",
    ]
    print(ok_df.head(10)[display_cols].to_string(index=False))
    print()
    print("Saved:")
    print(f"  CSV: {output_csv}")
    print(f"  Best JSON: {output_json}")


if __name__ == "__main__":
    main()
