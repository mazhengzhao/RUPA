#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Method-agnostic empirical analysis of failure-signal sources.

This script does not use Trajectory_TAU per-step risk, graph uncertainty, or
interaction-gap outputs. It computes simple structural diagnostics directly
from raw agent trajectories and visualizes where high-anomaly steps appear in
failed trajectories.

Outputs:
    <root>/failure_structural_sources/structural_failure_signals_summary.json
    <root>/failure_structural_sources/structural_signal_step_positions.pdf/.svg
    <root>/failure_structural_sources/structural_signal_source_categories.pdf/.svg
    <root>/failure_structural_sources/structural_signal_component_profile.pdf/.svg
"""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from evaluate_trajectory_tau_graph_experiments import DEFAULT_ROOT, finite_float, load_dataset


STOPWORDS = {
    "the", "and", "for", "that", "this", "with", "from", "have", "will", "need",
    "into", "then", "than", "what", "when", "where", "which", "there", "their",
    "about", "using", "first", "next", "let", "try", "run", "command", "output",
    "analysis", "plan", "task", "current", "step",
}

ERROR_HINTS = (
    "error", "failed", "failure", "not found", "no such", "command not found",
    "permission denied", "traceback", "exception", "invalid", "cannot", "can't",
    "unable", "timeout", "timed out", "empty", "no output", "didn't return",
    "doesn't return", "not useful", "mixed output", "lingering output",
)

CORRECTION_HINTS = (
    "try again", "different approach", "another", "instead", "however", "but",
    "maybe", "seems", "fresh", "more carefully", "not useful", "didn't return",
    "doesn't return", "let me", "fallback",
)

SOURCE_ORDER = [
    "Repeated action",
    "Stagnation",
    "Env feedback",
    "Goal drift",
    "Correction/retry",
    "Other",
]

COMPONENT_KEYS = [
    ("repetition", "Repetition"),
    ("stagnation", "Stagnation"),
    ("feedback_conflict", "Feedback conflict"),
    ("goal_drift", "Goal drift"),
    ("correction_retry", "Correction/retry"),
]

COMPONENT_COLORS = {
    "Repetition": "#F28E2B",
    "Stagnation": "#B07AA1",
    "Feedback conflict": "#E15759",
    "Goal drift": "#4C78A8",
    "Correction/retry": "#59A14F",
}


@dataclass
class StructuralConfig:
    recent_window: int = 4
    repetition_weight: float = 0.30
    stagnation_weight: float = 0.30
    feedback_weight: float = 0.20
    goal_drift_weight: float = 0.10
    correction_weight: float = 0.07
    user_feedback_weight: float = 0.03


def configure_matplotlib() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 9,
            "legend.fontsize": 7,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.75,
            "grid.linewidth": 0.45,
        }
    )


def save_json(path: Path, data: Any) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(json_safe(data), f, indent=2, ensure_ascii=False)


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


def parse_formats(raw: str) -> list[str]:
    formats = []
    for part in raw.split(","):
        fmt = part.strip().lower().lstrip(".")
        if fmt:
            if fmt not in {"pdf", "svg", "eps"}:
                raise ValueError(f"Unsupported vector format: {fmt}")
            formats.append(fmt)
    if not formats:
        raise ValueError("At least one output format is required")
    return formats


def source_role(step: dict[str, Any]) -> str:
    raw = str(step.get("source", "")).lower()
    if raw in {"agent", "assistant"}:
        return "assistant"
    if raw in {"environment", "tool", "observation"}:
        return "environment"
    if raw == "user":
        return "user"
    return raw


def text_of(step: dict[str, Any]) -> str:
    chunks = []
    for key in ("message", "observation"):
        value = step.get(key)
        if isinstance(value, str):
            chunks.append(value)
        elif value is not None:
            chunks.append(json.dumps(value, ensure_ascii=False, sort_keys=True))
    return "\n".join(chunks)


def assistant_text(step: dict[str, Any]) -> str:
    value = step.get("message")
    if isinstance(value, str):
        return value
    if value is None:
        return ""
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def observation_text(step: dict[str, Any]) -> str:
    value = step.get("observation")
    if isinstance(value, str):
        return value
    if value is None:
        return ""
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def tokenize(text: str) -> set[str]:
    tokens = set()
    for token in re.findall(r"[A-Za-z0-9_./-]+", text.lower()):
        token = token.strip("._-/")
        if len(token) < 3 or token in STOPWORDS:
            continue
        tokens.add(token)
    return tokens


def jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def contains_hint(text: str, hints: tuple[str, ...]) -> bool:
    lowered = text.lower()
    return any(hint in lowered for hint in hints)


def tool_signatures(step: dict[str, Any]) -> set[str]:
    signatures = set()
    for call in step.get("tool_calls") or []:
        if not isinstance(call, dict):
            continue
        fn = str(call.get("function_name", "")).strip()
        args = call.get("arguments") or {}
        if isinstance(args, dict):
            keystrokes = str(args.get("keystrokes", "")).strip().splitlines()
            command_head = keystrokes[0].strip().split(" ")[0] if keystrokes else ""
            signature = "::".join(part for part in (fn, command_head) if part)
        else:
            signature = fn
        if signature:
            signatures.add(signature)
    return signatures


def clamp01(value: float) -> float:
    return float(max(0.0, min(1.0, value)))


def structural_features(
    step: dict[str, Any],
    assistant_history: list[dict[str, Any]],
    previous_role: str | None,
    previous_observation: str,
    goal_tokens: set[str],
    config: StructuralConfig,
) -> dict[str, float]:
    text = assistant_text(step)
    tokens = tokenize(text)
    tools = tool_signatures(step)
    recent = assistant_history[-max(1, config.recent_window):]

    max_text_overlap = 0.0
    max_tool_overlap = 0.0
    for prev in recent:
        max_text_overlap = max(max_text_overlap, jaccard(tokens, prev["tokens"]))
        max_tool_overlap = max(max_tool_overlap, jaccard(tools, prev["tools"]))

    repetition = clamp01(max(max_tool_overlap, max_text_overlap))
    novelty = 1.0 - max_text_overlap if recent else 0.0
    no_new_tool = 1.0 if tools and any(tools == prev["tools"] for prev in recent) else 0.0
    stagnation = clamp01(0.55 * repetition + 0.30 * no_new_tool + 0.15 * (1.0 - novelty))

    feedback_text = previous_observation
    feedback_conflict = 1.0 if contains_hint(feedback_text, ERROR_HINTS) else 0.0
    if not feedback_text.strip() and recent:
        feedback_conflict = max(feedback_conflict, 0.35)

    user_feedback = 1.0 if previous_role == "user" else 0.0
    goal_overlap = jaccard(tokens, goal_tokens)
    goal_drift = clamp01(1.0 - goal_overlap) if goal_tokens else 0.0
    correction_retry = 1.0 if contains_hint(text, CORRECTION_HINTS) else 0.0

    structural_score = clamp01(
        config.repetition_weight * repetition
        + config.stagnation_weight * stagnation
        + config.feedback_weight * feedback_conflict
        + config.goal_drift_weight * goal_drift
        + config.correction_weight * correction_retry
        + config.user_feedback_weight * user_feedback
    )
    return {
        "structural_score": structural_score,
        "repetition": repetition,
        "stagnation": stagnation,
        "feedback_conflict": feedback_conflict,
        "goal_drift": goal_drift,
        "correction_retry": correction_retry,
        "user_feedback": user_feedback,
        "text_overlap": max_text_overlap,
        "tool_overlap": max_tool_overlap,
    }


def classify_source(features: dict[str, float]) -> str:
    if features["repetition"] >= 0.50:
        return "Repeated action"
    if features["stagnation"] >= 0.45:
        return "Stagnation"
    if features["feedback_conflict"] >= 0.50:
        return "Env feedback"
    if features["goal_drift"] >= 0.85:
        return "Goal drift"
    if features["correction_retry"] >= 0.50:
        return "Correction/retry"
    return "Other"


def structural_steps_for_item(item: dict[str, Any], config: StructuralConfig) -> list[dict[str, Any]]:
    goal_tokens = tokenize(item.get("goal_text", ""))
    assistant_history: list[dict[str, Any]] = []
    previous_role: str | None = None
    previous_observation = ""
    assistant_index = 0
    rows = []

    for raw_step_index, step in enumerate(item["steps"]):
        role = source_role(step)
        if role == "assistant":
            assistant_index += 1
            features = structural_features(
                step,
                assistant_history,
                previous_role,
                previous_observation,
                goal_tokens,
                config,
            )
            rows.append(
                {
                    "raw_step_index": raw_step_index,
                    "assistant_step_index": assistant_index,
                    **features,
                }
            )
            assistant_history.append(
                {
                    "tokens": tokenize(assistant_text(step)),
                    "tools": tool_signatures(step),
                }
            )
            previous_observation = observation_text(step)
        elif role == "user":
            previous_observation = ""
        elif role == "environment":
            previous_observation = text_of(step)
        previous_role = role
    return rows


def analyze_failures(dataset: list[dict[str, Any]], config: StructuralConfig) -> tuple[pd.DataFrame, dict[str, int]]:
    rows = []
    skipped = {"success": 0, "no_assistant_steps": 0}
    for item in dataset:
        if int(item["failure"]) != 1:
            skipped["success"] += 1
            continue
        steps = structural_steps_for_item(item, config)
        if not steps:
            skipped["no_assistant_steps"] += 1
            continue
        selected = max(steps, key=lambda row: row["structural_score"])
        n_steps = len(steps)
        rel_pos = 0.0 if n_steps <= 1 else (selected["assistant_step_index"] - 1) / float(n_steps - 1)
        source = classify_source(selected)
        rows.append(
            {
                "trial": item["trial"],
                "reward": item["reward"],
                "assistant_step_index": selected["assistant_step_index"],
                "num_assistant_steps": n_steps,
                "relative_position": rel_pos,
                "source_category": source,
                **selected,
            }
        )
    return pd.DataFrame(rows), skipped


def save_figure(fig: Any, output_base: Path, formats: list[str]) -> list[str]:
    paths = []
    for fmt in formats:
        path = output_base.with_suffix(f".{fmt}")
        fig.savefig(path, format=fmt, bbox_inches="tight")
        paths.append(str(path))
    plt.close(fig)
    return paths


def plot_position_distribution(df: pd.DataFrame, output_base: Path, formats: list[str]) -> list[str]:
    fig, ax = plt.subplots(figsize=(3.25, 2.2))
    bins = np.linspace(0.0, 1.0, 11)
    ax.hist(
        df["relative_position"].to_numpy(dtype=float),
        bins=bins,
        weights=np.ones(len(df), dtype=float) * 100.0 / max(1, len(df)),
        color="#4C78A8",
        edgecolor="white",
        linewidth=0.6,
        alpha=0.9,
    )
    median = float(df["relative_position"].median())
    ax.axvline(median, color="#E15759", linestyle="--", linewidth=1.0, label=f"Median={median:.2f}")
    ax.set_xlabel("Normalized position of highest-anomaly step")
    ax.set_ylabel("Failed trajectories (%)")
    ax.set_xlim(0.0, 1.0)
    ax.yaxis.grid(True, alpha=0.28)
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout(pad=0.35)
    return save_figure(fig, output_base, formats)


def plot_source_categories(df: pd.DataFrame, output_base: Path, formats: list[str]) -> list[str]:
    counts = df["source_category"].value_counts().to_dict()
    labels = [label for label in SOURCE_ORDER if counts.get(label, 0) > 0]
    values = np.asarray([counts[label] for label in labels], dtype=float)
    percentages = 100.0 * values / max(1.0, values.sum())

    fig_height = max(2.1, 0.28 * len(labels) + 0.8)
    fig, ax = plt.subplots(figsize=(3.25, fig_height))
    y = np.arange(len(labels))
    ax.barh(y, percentages, color="#59A14F", edgecolor="white", linewidth=0.6)
    for yi, pct in zip(y, percentages):
        ax.text(pct + 1.0, yi, f"{pct:.1f}%", va="center", fontsize=7)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Share of failed trajectories (%)")
    ax.set_xlim(0.0, max(20.0, float(percentages.max()) + 12.0))
    ax.xaxis.grid(True, alpha=0.28)
    fig.tight_layout(pad=0.35)
    return save_figure(fig, output_base, formats)


def plot_component_profile(df: pd.DataFrame, output_base: Path, formats: list[str]) -> list[str]:
    counts = df["source_category"].value_counts().to_dict()
    labels = [label for label in SOURCE_ORDER if counts.get(label, 0) > 0]
    profile = df.groupby("source_category")[[key for key, _ in COMPONENT_KEYS]].mean(numeric_only=True)

    fig_height = max(2.25, 0.32 * len(labels) + 0.95)
    fig, ax = plt.subplots(figsize=(4.35, fig_height))
    y = np.arange(len(labels))
    left = np.zeros(len(labels), dtype=float)
    for key, label in COMPONENT_KEYS:
        values = np.asarray([float(profile.loc[src, key]) if src in profile.index else 0.0 for src in labels])
        ax.barh(
            y,
            values,
            left=left,
            label=label,
            color=COMPONENT_COLORS[label],
            edgecolor="white",
            linewidth=0.35,
        )
        left += values
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Mean structural diagnostic intensity")
    ax.xaxis.grid(True, alpha=0.28)
    ax.legend(
        frameon=False,
        ncols=1,
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        handlelength=1.4,
    )
    fig.tight_layout(pad=0.35)
    return save_figure(fig, output_base, formats)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze method-agnostic structural failure signals.")
    parser.add_argument("root", nargs="?", type=Path, default=None)
    parser.add_argument("--root", dest="root_flag", type=Path, default=None)
    parser.add_argument("--eval-key", default=None)
    parser.add_argument("--include-exceptions", action="store_true")
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--formats", default="pdf,svg")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_matplotlib()
    root = (args.root_flag or args.root or DEFAULT_ROOT).expanduser().resolve()
    if not (root / "result.json").exists():
        raise FileNotFoundError(f"Missing result.json: {root / 'result.json'}")

    out_dir = (args.out_dir or (root / "failure_structural_sources")).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    formats = parse_formats(args.formats)
    config = StructuralConfig()

    dataset, dataset_info = load_dataset(root, args.eval_key, args.include_exceptions)
    if not dataset:
        raise RuntimeError("No usable trials found")
    df, skipped = analyze_failures(dataset, config)
    if df.empty:
        raise RuntimeError("No failed trajectories with assistant steps found")

    position_paths = plot_position_distribution(df, out_dir / "structural_signal_step_positions", formats)
    category_paths = plot_source_categories(df, out_dir / "structural_signal_source_categories", formats)
    component_paths = plot_component_profile(df, out_dir / "structural_signal_component_profile", formats)

    source_counts = df["source_category"].value_counts().to_dict()
    source_percentages = {key: float(100.0 * value / len(df)) for key, value in source_counts.items()}
    summary = {
        "root": str(root),
        "eval_key": dataset_info["eval_key"],
        "analysis_type": "method_agnostic_structural_diagnostics",
        "include_exceptions": bool(args.include_exceptions),
        "structural_config": asdict(config),
        "counters": dataset_info["counters"],
        "skipped": skipped,
        "n_failed_analyzed": int(len(df)),
        "position_summary": {
            "mean": finite_float(df["relative_position"].mean()),
            "median": finite_float(df["relative_position"].median()),
            "std": finite_float(df["relative_position"].std()),
            "min": finite_float(df["relative_position"].min()),
            "max": finite_float(df["relative_position"].max()),
        },
        "source_counts": source_counts,
        "source_percentages": source_percentages,
        "component_means": {
            label: {
                key: finite_float(value)
                for key, value in group[[k for k, _ in COMPONENT_KEYS]].mean(numeric_only=True).to_dict().items()
            }
            for label, group in df.groupby("source_category")
        },
        "highest_structural_anomaly_steps": df.to_dict(orient="records"),
        "outputs": {
            "summary_json": str(out_dir / "structural_failure_signals_summary.json"),
            "position_figures": position_paths,
            "source_category_figures": category_paths,
            "component_profile_figures": component_paths,
        },
    }
    save_json(out_dir / "structural_failure_signals_summary.json", summary)

    print("=" * 80)
    print("Method-agnostic structural failure-signal analysis")
    print("=" * 80)
    print(f"Root: {root}")
    print(f"Eval key: {dataset_info['eval_key']}")
    print(f"Failed trajectories analyzed: {len(df)}")
    print()
    display = (
        pd.DataFrame(
            {
                "source_category": list(source_counts.keys()),
                "percentage": [source_percentages[key] for key in source_counts.keys()],
            }
        )
        .sort_values("percentage", ascending=False)
        .reset_index(drop=True)
    )
    print(display.to_string(index=False))
    print()
    print(f"Saved summary JSON: {out_dir / 'structural_failure_signals_summary.json'}")
    print("Saved figures:")
    for path in position_paths + category_paths + component_paths:
        print(f"  {path}")


if __name__ == "__main__":
    main()
