#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Evaluate whether early-trajectory confidence can predict final GAIA success.

This script truncates each Harbor trajectory to the first p% of agent steps and
computes confidence scores from four methods on that prefix:
    - Entropy confidence
    - TRACER confidence
    - SAUP confidence
    - TAU confidence

For risk-style methods, confidence is converted as:
    confidence = 1 / (1 + risk)

The script then evaluates how well each prefix confidence predicts final task
success using AUROC, AUPRC, Accuracy, Precision, Recall, and F1.

Usage:
    python evaluate_gaia_prefix_confidence_metrics.py /path/to/harbor_job
    python evaluate_gaia_prefix_confidence_metrics.py /path/to/harbor_job --prefix-percents 0.25,0.5,0.75
"""

from __future__ import annotations

import argparse
import difflib
import importlib.util
import json
import math
import sys
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
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


DEFAULT_ROOT = Path("~/jobs/2026-05-16__19-11-56")
PROJECT_ROOT = Path(__file__).resolve().parent
AGENT_TRACER_SRC = PROJECT_ROOT / "agent-tracer" / "src"
TRAJECTORY_TAU_PATH = AGENT_TRACER_SRC / "tau2" / "metrics" / "trajectory_tau.py"

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
class TRACERConfig:
    top_k_percentile: float = 0.1
    ensemble_weight_max: float = 0.1


@dataclass
class SAUPConfig:
    alpha: float = 1.0
    beta: float = 1.0
    gamma: float = 1.0


@dataclass
class TAUConfig:
    alpha: float = 0.8
    beta: float = 0.8
    decay: float = 0.75
    lambda_goal: float = 0.4


@dataclass
class UPropApproxConfig:
    pmi_weight: float = 1.0
    kernel_tau: float = 1.0
    max_history: int = 8
    min_kernel: float = 1e-8


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


def safe_mean(values: list[float]) -> Optional[float]:
    values = [x for x in values if math.isfinite(x)]
    if not values:
        return None
    return float(np.mean(values))


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


def string_distance(text_a: str, text_b: str) -> float:
    if not text_a or not text_b:
        return 1.0
    ratio = difflib.SequenceMatcher(None, text_a[:4000], text_b[:4000]).ratio()
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


def extract_goal_text(steps: list[dict[str, Any]], trial_dir: Path) -> str:
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


def calculate_interaction_consistency(agent_text: str, prev_user_text: Optional[str], goal_text: Optional[str], lambda_goal: float) -> float:
    score = 0.0
    if prev_user_text:
        score += calculate_semantic_distance(agent_text, prev_user_text)
    if goal_text:
        score += lambda_goal * calculate_semantic_distance(agent_text, goal_text)
    return score


def risk_to_confidence(risk: Optional[float]) -> Optional[float]:
    if risk is None or not math.isfinite(risk):
        return None
    return float(1.0 / (1.0 + risk))


def truncate_to_prefix(steps: list[dict[str, Any]], prefix_percent: float) -> tuple[list[dict[str, Any]], int, int]:
    agent_total = sum(1 for step in steps if step.get("source") == "agent")
    if agent_total == 0:
        return [], 0, 0

    prefix_agent_count = max(1, int(math.ceil(agent_total * prefix_percent)))
    agent_seen = 0
    prefix_steps: list[dict[str, Any]] = []

    for step in steps:
        prefix_steps.append(step)
        if step.get("source") == "agent":
            agent_seen += 1
            if agent_seen >= prefix_agent_count:
                break

    return prefix_steps, agent_total, prefix_agent_count


def calculate_entropy_prefix(prefix_steps: list[dict[str, Any]]) -> dict[str, Any]:
    all_logprobs: list[float] = []
    n_agent_steps = 0

    for step in prefix_steps:
        if step.get("source") != "agent":
            continue
        n_agent_steps += 1
        all_logprobs.extend(collect_logprobs_from_step(step))

    entropy = entropy_from_logprobs(all_logprobs)
    return {
        "entropy_risk": entropy,
        "entropy_confidence": risk_to_confidence(entropy),
        "entropy_n_logprobs": len(all_logprobs),
        "entropy_n_agent_steps": n_agent_steps,
    }


def calculate_tracer_prefix(prefix_steps: list[dict[str, Any]], config: TRACERConfig) -> dict[str, Any]:
    ui_values: list[float] = []

    for step in prefix_steps:
        if step.get("source") != "agent":
            continue
        logprobs = collect_logprobs_from_step(step)
        ui = entropy_from_logprobs(logprobs)
        if ui is not None:
            ui_values.append(ui)

    if not ui_values:
        return {
            "tracer_risk": None,
            "tracer_confidence": None,
            "tracer_num_steps": 0,
        }

    if config.top_k_percentile >= 1.0:
        top_k = ui_values
    else:
        sorted_risks = sorted(ui_values, reverse=True)
        top_k_count = max(1, int(config.top_k_percentile * len(sorted_risks)))
        top_k = sorted_risks[:top_k_count]

    mean_top_k = float(np.mean(top_k))
    max_risk = float(np.max(ui_values))
    tracer_risk = (1.0 - config.ensemble_weight_max) * mean_top_k + config.ensemble_weight_max * max_risk

    return {
        "tracer_risk": float(tracer_risk),
        "tracer_confidence": risk_to_confidence(tracer_risk),
        "tracer_num_steps": len(ui_values),
    }


def calculate_saup_prefix(prefix_steps: list[dict[str, Any]], config: SAUPConfig) -> dict[str, Any]:
    weighted_uncertainties: list[float] = []
    n_steps = 0

    for step in prefix_steps:
        if step.get("source") != "agent":
            continue

        logprobs = collect_logprobs_from_step(step)
        ui = entropy_from_logprobs(logprobs)
        if ui is None:
            continue

        da = extract_step_metric(step, "da") or 0.0
        do_agent = extract_step_metric(step, "do_agent") or 0.0
        do_user = extract_step_metric(step, "do_user") or 0.0
        weight = 1.0 + config.alpha * da + config.beta * do_agent + config.gamma * do_user
        weighted_uncertainties.append(float(weight * ui))
        n_steps += 1

    if not weighted_uncertainties:
        return {
            "saup_risk": None,
            "saup_confidence": None,
            "saup_num_steps": 0,
        }

    arr = np.asarray(weighted_uncertainties, dtype=np.float64)
    saup_risk = float(np.sqrt(np.mean(arr ** 2)))

    return {
        "saup_risk": saup_risk,
        "saup_confidence": risk_to_confidence(saup_risk),
        "saup_num_steps": n_steps,
    }


def calculate_uprop_prefix(prefix_steps: list[dict[str, Any]], config: UPropApproxConfig) -> dict[str, Any]:
    texts: list[str] = []
    intrinsic_values: list[float] = []
    pmi_values: list[float] = []
    step_scores: list[float] = []
    all_logprobs: list[float] = []
    n_agent_steps = 0
    cumulative_pmi = 0.0

    for step in prefix_steps:
        if step.get("source") != "agent":
            continue
        n_agent_steps += 1
        logprobs = collect_logprobs_from_step(step)
        if not logprobs:
            continue
        iu = entropy_from_logprobs(logprobs)
        if iu is None:
            continue

        text = build_step_representation(step)
        previous_text = texts[-1] if texts else ""
        pmi = pmi_proxy_from_preceding_variance(previous_text, texts[:-1], config)
        cumulative_pmi += pmi

        step_score = float(iu + config.pmi_weight * cumulative_pmi)
        intrinsic_values.append(float(iu))
        pmi_values.append(float(pmi))
        step_scores.append(step_score)
        all_logprobs.extend(logprobs)
        texts.append(text)

    if not step_scores:
        return {
            "uprop_risk": None,
            "uprop_confidence": None,
            "uprop_num_steps": n_agent_steps,
            "uprop_intrinsic_uncertainty": None,
            "uprop_extrinsic_uncertainty": None,
        }

    sigmas = [
        1.0 + (config.pmi_weight * pmi / max(iu, 1e-8))
        for iu, pmi in zip(intrinsic_values, pmi_values)
    ]
    lambda_z = float(np.sum(sigmas)) if sigmas else 1.0
    uprop_risk = float(np.sum(step_scores) / max(lambda_z, 1e-8))

    return {
        "uprop_risk": uprop_risk,
        "uprop_confidence": risk_to_confidence(uprop_risk),
        "uprop_num_steps": n_agent_steps,
        "uprop_intrinsic_uncertainty": safe_mean(intrinsic_values),
        "uprop_extrinsic_uncertainty": safe_mean(pmi_values),
        "uprop_cumulative_pmi": float(cumulative_pmi),
        "uprop_mean_step_score": safe_mean(step_scores),
        "uprop_lambda_z": lambda_z,
        "uprop_token_entropy": entropy_from_logprobs(all_logprobs),
    }


def calculate_tau_prefix(prefix_steps: list[dict[str, Any]], trial_dir: Path, config: TAUConfig) -> dict[str, Any]:
    goal_text = extract_goal_text(prefix_steps, trial_dir)
    history_texts: list[str] = []
    prev_user: Optional[str] = None
    risks: list[float] = []
    n_agent_steps = 0

    for step in prefix_steps:
        source = step.get("source")
        if source not in {"agent", "user"}:
            continue

        text = build_step_representation(step)

        if source == "agent":
            n_agent_steps += 1
            ui = entropy_from_logprobs(collect_logprobs_from_step(step))
            if ui is not None:
                h_val = calculate_temporal_influence(text, history_texts, config.decay)
                c_val = calculate_interaction_consistency(text, prev_user, goal_text, config.lambda_goal)
                risk = ui * (1.0 + config.alpha * h_val + config.beta * c_val)
                risks.append(float(risk))

        history_texts.append(text)
        if source == "user":
            prev_user = text

    if not risks:
        return {
            "tau_risk": None,
            "tau_confidence": None,
            "tau_num_steps": 0,
        }

    tau_risk = float(np.mean(risks))
    tau_confidence = risk_to_confidence(tau_risk)
    return {
        "tau_risk": tau_risk,
        "tau_confidence": tau_confidence,
        "tau_num_steps": n_agent_steps,
    }


def calculate_trajectory_tau_prefix(
    prefix_steps: list[dict[str, Any]],
    trial_dir: Path,
    config: Any,
) -> dict[str, Any]:
    goal_text = extract_goal_text(prefix_steps, trial_dir)
    tau_info = calculate_trajectory_tau_score(prefix_steps, goal_text, config)
    tau_risk = finite_float(tau_info.get("tau_score"))
    tau_confidence = finite_float(tau_info.get("tau_confidence"))

    return {
        "trajectory_tau_risk": tau_risk,
        "trajectory_tau_confidence": tau_confidence,
        "trajectory_tau_num_steps": int(tau_info.get("num_steps", 0) or 0),
        "trajectory_tau_mean_propagation": finite_float(tau_info.get("mean_propagation")),
        "trajectory_tau_mean_combined_propagation": finite_float(tau_info.get("mean_combined_propagation")),
        "trajectory_tau_mean_graph_uncertainty": finite_float(tau_info.get("mean_graph_uncertainty")),
        "trajectory_tau_mean_effective_ui": finite_float(tau_info.get("mean_effective_ui")),
        "trajectory_tau_mean_interaction_gap": finite_float(tau_info.get("mean_interaction_gap")),
    }


def evaluate_confidence_scores(df: pd.DataFrame, score_col: str) -> dict[str, Any]:
    eval_df = df[df[score_col].notna()].copy()
    if eval_df.empty:
        raise ValueError(f"No valid scores found for {score_col}")

    y_true = eval_df["success"].to_numpy(dtype=int)
    scores = eval_df[score_col].to_numpy(dtype=float)

    if len(np.unique(y_true)) < 2:
        raise ValueError(f"Need both success/failure classes for {score_col}")

    fpr, tpr, thresholds = roc_curve(y_true, scores)
    j_scores = tpr - fpr
    best_idx = int(np.argmax(j_scores))
    threshold = float(thresholds[best_idx])
    y_pred = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    return {
        "score_column": score_col,
        "n": int(len(eval_df)),
        "n_success": int((eval_df["success"] == 1).sum()),
        "n_failure": int((eval_df["success"] == 0).sum()),
        "positive_label": "success",
        "auroc": float(roc_auc_score(y_true, scores)),
        "auprc": float(average_precision_score(y_true, scores)),
        "best_threshold_youden": threshold,
        "youden_j": float(j_scores[best_idx]),
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
    }


def parse_prefix_percents(raw: str) -> list[float]:
    values: list[float] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        value = float(part)
        if value <= 0.0 or value > 1.0:
            raise ValueError(f"Prefix percent must be in (0, 1], got {value}")
        values.append(value)
    if not values:
        raise ValueError("No valid prefix percents provided")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate early prefix confidence for Entropy/TRACER/SAUP/UProp/TAU on Harbor GAIA runs."
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
        "--prefix-percents",
        default="0.5",
        help="Comma-separated prefix percentages of agent steps, e.g. 0.25,0.5,0.75",
    )
    parser.add_argument(
        "--include-exceptions",
        action="store_true",
        help="Include trials with exception_info/exception_stats if they have rewards and trajectories.",
    )
    parser.add_argument(
        "--tracer-top-k-percentile",
        type=float,
        default=0.1,
        help="TRACER top-k percentile.",
    )
    parser.add_argument(
        "--tracer-ensemble-weight-max",
        type=float,
        default=0.1,
        help="TRACER ensemble weight for max risk.",
    )
    parser.add_argument("--saup-alpha", type=float, default=1.0)
    parser.add_argument("--saup-beta", type=float, default=1.0)
    parser.add_argument("--saup-gamma", type=float, default=1.0)
    parser.add_argument("--uprop-pmi-weight", type=float, default=1.0)
    parser.add_argument("--uprop-kernel-tau", type=float, default=1.0)
    parser.add_argument("--uprop-max-history", type=int, default=8)
    parser.add_argument("--uprop-min-kernel", type=float, default=1e-8)
    parser.add_argument("--tau-alpha", type=float, default=0.8)
    parser.add_argument("--tau-beta", type=float, default=0.8)
    parser.add_argument("--tau-decay", type=float, default=0.75)
    parser.add_argument("--tau-lambda-goal", type=float, default=0.4)
    parser.add_argument(
        "--output-prefix",
        default="gaia_prefix_confidence_eval",
        help="Prefix for output CSV and summary files.",
    )
    parser.add_argument("--trajectory-tau-alpha", type=float, default=1.0)
    parser.add_argument("--trajectory-tau-beta", type=float, default=0.7)
    parser.add_argument("--trajectory-tau-momentum-weight", type=float, default=0.30)
    parser.add_argument("--trajectory-tau-repetition-weight", type=float, default=0.30)
    parser.add_argument("--trajectory-tau-observation-weight", type=float, default=0.25)
    parser.add_argument("--trajectory-tau-stagnation-weight", type=float, default=0.15)
    parser.add_argument("--trajectory-tau-interaction-goal-weight", type=float, default=0.55)
    parser.add_argument("--trajectory-tau-interaction-user-weight", type=float, default=0.45)
    parser.add_argument("--trajectory-tau-history-decay", type=float, default=0.80)
    parser.add_argument("--trajectory-tau-uncertainty-decay", type=float, default=0.75)
    parser.add_argument("--trajectory-tau-recent-window", type=int, default=5)
    parser.add_argument("--trajectory-tau-novelty-window", type=int, default=4)
    parser.add_argument("--trajectory-tau-graph-weight", type=float, default=0.30)
    parser.add_argument("--trajectory-tau-graph-uncertainty-weight", type=float, default=0.35)
    parser.add_argument("--trajectory-tau-graph-edge-decay", type=float, default=0.85)
    parser.add_argument("--trajectory-tau-graph-max-neighbors", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = (args.root_flag or args.root or DEFAULT_ROOT).expanduser().resolve()
    result_path = root / "result.json"
    if not result_path.exists():
        raise FileNotFoundError(f"Missing result.json: {result_path}")

    prefix_percents = parse_prefix_percents(args.prefix_percents)
    tracer_config = TRACERConfig(
        top_k_percentile=args.tracer_top_k_percentile,
        ensemble_weight_max=args.tracer_ensemble_weight_max,
    )
    saup_config = SAUPConfig(
        alpha=args.saup_alpha,
        beta=args.saup_beta,
        gamma=args.saup_gamma,
    )
    uprop_config = UPropApproxConfig(
        pmi_weight=args.uprop_pmi_weight,
        kernel_tau=args.uprop_kernel_tau,
        max_history=args.uprop_max_history,
        min_kernel=args.uprop_min_kernel,
    )
    tau_config = TAUConfig(
        alpha=args.tau_alpha,
        beta=args.tau_beta,
        decay=args.tau_decay,
        lambda_goal=args.tau_lambda_goal,
    )
    trajectory_tau_config = TrajectoryTAUConfig(
        alpha=args.trajectory_tau_alpha,
        beta=args.trajectory_tau_beta,
        momentum_weight=args.trajectory_tau_momentum_weight,
        repetition_weight=args.trajectory_tau_repetition_weight,
        observation_weight=args.trajectory_tau_observation_weight,
        stagnation_weight=args.trajectory_tau_stagnation_weight,
        interaction_goal_weight=args.trajectory_tau_interaction_goal_weight,
        interaction_user_weight=args.trajectory_tau_interaction_user_weight,
        history_decay=args.trajectory_tau_history_decay,
        uncertainty_decay=args.trajectory_tau_uncertainty_decay,
        recent_window=args.trajectory_tau_recent_window,
        novelty_window=args.trajectory_tau_novelty_window,
        graph_weight=args.trajectory_tau_graph_weight,
        graph_uncertainty_weight=args.trajectory_tau_graph_uncertainty_weight,
        graph_edge_decay=args.trajectory_tau_graph_edge_decay,
        graph_max_neighbors=args.trajectory_tau_graph_max_neighbors,
    )

    result_data = load_json(result_path)
    eval_key, eval_payload = get_eval_payload(result_data, args.eval_key)
    reward_mapping = extract_reward_mapping(eval_payload)
    exception_mapping = extract_exception_mapping(eval_payload)

    detail_rows: list[dict[str, Any]] = []
    counters = {
        "rewarded_trials": len(reward_mapping),
        "skipped_exception": 0,
        "missing_trajectory": 0,
        "used_trials_any_prefix": 0,
    }
    used_trial_names: set[str] = set()

    for trial_dir in sorted(root.iterdir()):
        if not trial_dir.is_dir():
            continue
        trial_name = trial_dir.name
        if trial_name not in reward_mapping:
            continue

        exception_type = trial_exception_type(trial_dir, exception_mapping)
        if exception_type and not args.include_exceptions:
            counters["skipped_exception"] += 1
            continue

        traj_path = trial_dir / "agent" / "trajectory.json"
        if not traj_path.exists():
            counters["missing_trajectory"] += 1
            continue

        try:
            trajectory = load_json(traj_path)
        except Exception as exc:
            print(f"[WARN] failed to load {traj_path}: {exc}")
            continue

        steps = trajectory.get("steps", [])
        reward = reward_mapping[trial_name]
        success = int(reward > 0.5)
        trial_used = False

        for prefix_percent in prefix_percents:
            prefix_steps, agent_total, prefix_agent_count = truncate_to_prefix(steps, prefix_percent)
            if agent_total == 0 or not prefix_steps:
                continue

            entropy_info = calculate_entropy_prefix(prefix_steps)
            tracer_info = calculate_tracer_prefix(prefix_steps, tracer_config)
            saup_info = calculate_saup_prefix(prefix_steps, saup_config)
            uprop_info = calculate_uprop_prefix(prefix_steps, uprop_config)
            tau_info = calculate_tau_prefix(prefix_steps, trial_dir, tau_config)
            trajectory_tau_info = calculate_trajectory_tau_prefix(prefix_steps, trial_dir, trajectory_tau_config)

            detail_rows.append(
                {
                    "trial": trial_name,
                    "reward": reward,
                    "success": success,
                    "failure": 1 - success,
                    "exception_type": exception_type,
                    "prefix_percent": prefix_percent,
                    "total_agent_steps": agent_total,
                    "prefix_agent_steps": prefix_agent_count,
                    **entropy_info,
                    **tracer_info,
                    **saup_info,
                    **uprop_info,
                    **tau_info,
                    **trajectory_tau_info,
                }
            )
            trial_used = True

        if trial_used:
            used_trial_names.add(trial_name)

    counters["used_trials_any_prefix"] = len(used_trial_names)

    if not detail_rows:
        raise RuntimeError("No valid prefix samples found.")

    detail_df = pd.DataFrame(detail_rows)

    summary_rows: list[dict[str, Any]] = []
    method_to_score = {
        "entropy": "entropy_confidence",
        "tracer": "tracer_confidence",
        "saup": "saup_confidence",
        "uprop": "uprop_confidence",
        "tau": "tau_confidence",
        "trajectory_tau": "trajectory_tau_confidence",
    }

    for prefix_percent in prefix_percents:
        prefix_df = detail_df[detail_df["prefix_percent"] == prefix_percent].copy()
        for method_name, score_col in method_to_score.items():
            try:
                metrics = evaluate_confidence_scores(prefix_df, score_col)
            except ValueError as exc:
                metrics = {
                    "score_column": score_col,
                    "error": str(exc),
                    "n": int(len(prefix_df)),
                }

            summary_rows.append(
                {
                    "prefix_percent": prefix_percent,
                    "method": method_name,
                    **metrics,
                }
            )

    summary_df = pd.DataFrame(summary_rows)

    detail_csv = root / f"{args.output_prefix}_details.csv"
    summary_csv = root / f"{args.output_prefix}_summary.csv"
    summary_json = root / f"{args.output_prefix}_summary.json"

    detail_df.to_csv(detail_csv, index=False)
    summary_df.to_csv(summary_csv, index=False)
    save_json(
        summary_json,
        {
            "root": str(root),
            "eval_key": eval_key,
            "prefix_percents": prefix_percents,
            "include_exceptions": bool(args.include_exceptions),
            "tracer_config": asdict(tracer_config),
            "saup_config": asdict(saup_config),
            "uprop_config": asdict(uprop_config),
            "tau_config": asdict(tau_config),
            "trajectory_tau_config": asdict(trajectory_tau_config),
            "counters": counters,
            "summary": summary_rows,
            "outputs": {
                "detail_csv": str(detail_csv),
                "summary_csv": str(summary_csv),
                "summary_json": str(summary_json),
            },
        },
    )

    print("=" * 80)
    print("GAIA Prefix Confidence evaluation")
    print("=" * 80)
    print(f"Root: {root}")
    print(f"Eval key: {eval_key}")
    print(f"Prefix percents: {', '.join(f'{x:.2f}' for x in prefix_percents)}")
    print(f"Rewarded trials: {counters['rewarded_trials']}")
    print(f"Skipped exception trials: {counters['skipped_exception']}")
    print(f"Used trials: {counters['used_trials_any_prefix']}")
    print()
    print(summary_df.to_string(index=False))
    print()
    print("Saved:")
    print(f"  Details: {detail_csv}")
    print(f"  Summary CSV: {summary_csv}")
    print(f"  Summary JSON: {summary_json}")


if __name__ == "__main__":
    main()
