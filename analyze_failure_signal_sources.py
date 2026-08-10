#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Analyze where failure signals emerge in failed agent trajectories.

For each failed trajectory, this script identifies the assistant step with the
highest Trajectory_TAU per-step score and summarizes its position, structural
source category, and uncertainty-signal profile.

Outputs:
    <root>/failure_signal_sources/failure_signal_sources_summary.json
    <root>/failure_signal_sources/failure_signal_step_positions.pdf/.svg
    <root>/failure_signal_sources/failure_signal_source_categories.pdf/.svg
    <root>/failure_signal_sources/failure_signal_component_profile.pdf/.svg

Usage:
    python analyze_failure_signal_sources.py jobs/2026-07-01__10-00-09
    python analyze_failure_signal_sources.py jobs/2026-07-01__10-00-09 --score-key graph_uncertainty
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from evaluate_trajectory_tau_graph_experiments import (
    DEFAULT_ROOT,
    TrajectoryTAUConfig,
    finite_float,
    load_dataset,
    trajectory_tau_module,
)


calculate_trajectory_tau_score = trajectory_tau_module.calculate_trajectory_tau_score

SOURCE_ORDER = [
    "Env feedback",
    "Repeated action",
    "Stagnation",
    "User feedback",
    "Progression",
    "Parallel branch",
    "Token overlap",
    "Sequential",
    "Other",
]

COMPONENT_KEYS = [
    ("ui", "Base U"),
    ("momentum", "Momentum"),
    ("repetition", "Repetition"),
    ("observation", "Observation"),
    ("stagnation", "Stagnation"),
    ("graph_uncertainty", "Graph U"),
    ("interaction_gap", "Interaction gap"),
]

COMPONENT_COLORS = {
    "Base U": "#4C78A8",
    "Momentum": "#59A14F",
    "Repetition": "#F28E2B",
    "Observation": "#E15759",
    "Stagnation": "#B07AA1",
    "Graph U": "#9C755F",
    "Interaction gap": "#76B7B2",
}


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
        if not fmt:
            continue
        if fmt not in {"pdf", "svg", "eps"}:
            raise ValueError(f"Unsupported vector format: {fmt}")
        formats.append(fmt)
    if not formats:
        raise ValueError("At least one output format is required")
    return formats


def last_path_segment(raw: Any) -> str:
    text = str(raw).strip().rstrip("/")
    return text.split("/")[-1] if text else ""


def title_context(eval_key: str) -> str:
    parts = str(eval_key).split("__")
    if len(parts) >= 3:
        model = last_path_segment(parts[1])
        dataset = last_path_segment(parts[2])
        if model and dataset:
            return f"{model} on {dataset}"
    return last_path_segment(eval_key)


def dominant_edge(step: dict[str, Any]) -> dict[str, Any] | None:
    edges = step.get("graph_edges") or []
    if not edges:
        return None
    return max(edges, key=lambda edge: float(edge.get("weight", 0.0) or 0.0))


def edge_type_counter(step: dict[str, Any]) -> Counter[str]:
    counter: Counter[str] = Counter()
    for edge in step.get("graph_edges") or []:
        edge_type = edge.get("edge_type")
        if edge_type:
            counter[str(edge_type)] += 1
    return counter


def classify_source(step: dict[str, Any]) -> str:
    edges = step.get("graph_edges") or []
    edge_types = {str(edge.get("edge_type")) for edge in edges if edge.get("edge_type")}
    source_roles = {str(edge.get("source_role")) for edge in edges if edge.get("source_role")}

    if {"feedback_response", "feedback_instability"} & edge_types and "environment" in source_roles:
        return "Env feedback"
    if {"tool_repetition", "repetition"} & edge_types:
        return "Repeated action"
    if float(step.get("stagnation", 0.0) or 0.0) >= 0.25:
        return "Stagnation"
    if "latest_user" in edge_types or "user" in source_roles:
        return "User feedback"
    if "progression" in edge_types:
        return "Progression"
    if "parallel" in edge_types:
        return "Parallel branch"
    if "token_overlap" in edge_types:
        return "Token overlap"
    if "sequential" in edge_types:
        return "Sequential"
    return "Other"


def highest_uncertainty_step(per_step: list[dict[str, Any]], score_key: str) -> tuple[int, dict[str, Any]] | None:
    valid = [
        (idx, step)
        for idx, step in enumerate(per_step)
        if finite_float(step.get(score_key)) is not None
    ]
    if not valid:
        return None
    return max(valid, key=lambda item: float(item[1].get(score_key, 0.0) or 0.0))


def analyze_failures(
    dataset: list[dict[str, Any]],
    config: TrajectoryTAUConfig,
    score_key: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    skipped = {"success": 0, "no_per_step": 0, "missing_score": 0}

    for item in dataset:
        if int(item["failure"]) != 1:
            skipped["success"] += 1
            continue

        tau_info = calculate_trajectory_tau_score(item["steps"], item["goal_text"], config)
        per_step = tau_info.get("per_step") or []
        if not per_step:
            skipped["no_per_step"] += 1
            continue

        selected = highest_uncertainty_step(per_step, score_key)
        if selected is None:
            skipped["missing_score"] += 1
            continue

        idx, step = selected
        n_steps = len(per_step)
        rel_pos = 0.0 if n_steps <= 1 else idx / float(n_steps - 1)
        edge = dominant_edge(step)
        source = classify_source(step)
        edge_counts = edge_type_counter(step)

        row: dict[str, Any] = {
            "trial": item["trial"],
            "reward": item["reward"],
            "assistant_step_index": idx + 1,
            "num_assistant_steps": n_steps,
            "relative_position": rel_pos,
            "source_category": source,
            "score_key": score_key,
            "score": finite_float(step.get(score_key)),
            "risk": finite_float(step.get("risk")),
            "ui": finite_float(step.get("ui")),
            "effective_ui": finite_float(step.get("effective_ui")),
            "propagation": finite_float(step.get("propagation")),
            "combined_propagation": finite_float(step.get("combined_propagation")),
            "graph_uncertainty": finite_float(step.get("graph_uncertainty")),
            "graph_edge_count": finite_float(step.get("graph_edge_count")),
            "interaction_gap": finite_float(step.get("interaction_gap")),
            "momentum": finite_float(step.get("momentum")),
            "repetition": finite_float(step.get("repetition")),
            "observation": finite_float(step.get("observation")),
            "stagnation": finite_float(step.get("stagnation")),
            "dominant_edge_type": edge.get("edge_type") if edge else None,
            "dominant_edge_weight": finite_float(edge.get("weight")) if edge else None,
            "dominant_edge_source_role": edge.get("source_role") if edge else None,
            "edge_type_counts": dict(edge_counts),
        }
        rows.append(row)

    return pd.DataFrame(rows), skipped


def save_figure(fig: Any, output_base: Path, formats: list[str]) -> list[str]:
    paths = []
    for fmt in formats:
        path = output_base.with_suffix(f".{fmt}")
        fig.savefig(path, format=fmt, bbox_inches="tight")
        paths.append(str(path))
    plt.close(fig)
    return paths


def plot_position_distribution(df: pd.DataFrame, output_base: Path, formats: list[str], title: str | None) -> list[str]:
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
    ax.set_xlabel("Normalized position of highest-risk step")
    ax.set_ylabel("Failed trajectories (%)")
    ax.set_xlim(0.0, 1.0)
    ax.yaxis.grid(True, alpha=0.28)
    if title:
        ax.set_title(title)
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout(pad=0.35)
    return save_figure(fig, output_base, formats)


def plot_source_categories(df: pd.DataFrame, output_base: Path, formats: list[str], title: str | None) -> list[str]:
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
    ax.invert_yaxis()
    ax.set_xlabel("Share of failed trajectories (%)")
    ax.set_xlim(0.0, max(20.0, float(percentages.max()) + 12.0))
    ax.xaxis.grid(True, alpha=0.28)
    if title:
        ax.set_title(title)
    fig.tight_layout(pad=0.35)
    return save_figure(fig, output_base, formats)


def plot_component_profile(df: pd.DataFrame, output_base: Path, formats: list[str], title: str | None) -> list[str]:
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
    ax.invert_yaxis()
    ax.set_xlabel("Mean signal intensity at highest-risk step")
    ax.xaxis.grid(True, alpha=0.28)
    if title:
        ax.set_title(title)
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
    parser = argparse.ArgumentParser(description="Analyze highest-risk steps in failed trajectories.")
    parser.add_argument("root", nargs="?", type=Path, default=None)
    parser.add_argument("--root", dest="root_flag", type=Path, default=None)
    parser.add_argument("--eval-key", default=None)
    parser.add_argument("--include-exceptions", action="store_true")
    parser.add_argument(
        "--score-key",
        default="risk",
        choices=["risk", "graph_uncertainty", "effective_ui", "combined_propagation", "interaction_gap"],
        help="Per-step score used to locate the highest-uncertainty step.",
    )
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--formats", default="pdf,svg")
    parser.add_argument("--no-title", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_matplotlib()

    root = (args.root_flag or args.root or DEFAULT_ROOT).expanduser().resolve()
    if not (root / "result.json").exists():
        raise FileNotFoundError(f"Missing result.json: {root / 'result.json'}")

    formats = parse_formats(args.formats)
    out_dir = (args.out_dir or (root / "failure_signal_sources")).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    config = TrajectoryTAUConfig()
    dataset, dataset_info = load_dataset(root, args.eval_key, args.include_exceptions)
    if not dataset:
        raise RuntimeError("No usable trials found")

    df, skipped = analyze_failures(dataset, config, args.score_key)
    if df.empty:
        raise RuntimeError("No failed trajectories with valid per-step scores found")

    context = title_context(dataset_info["eval_key"])
    title_suffix = None if args.no_title else context

    position_paths = plot_position_distribution(
        df,
        out_dir / "failure_signal_step_positions",
        formats,
        f"{title_suffix}: highest-risk step positions" if title_suffix else None,
    )
    category_paths = plot_source_categories(
        df,
        out_dir / "failure_signal_source_categories",
        formats,
        f"{title_suffix}: high-risk source categories" if title_suffix else None,
    )
    component_paths = plot_component_profile(
        df,
        out_dir / "failure_signal_component_profile",
        formats,
        f"{title_suffix}: high-risk signal profile" if title_suffix else None,
    )

    source_counts = df["source_category"].value_counts().to_dict()
    source_percentages = {
        key: float(100.0 * value / len(df))
        for key, value in source_counts.items()
    }
    summary = {
        "root": str(root),
        "eval_key": dataset_info["eval_key"],
        "include_exceptions": bool(args.include_exceptions),
        "score_key": args.score_key,
        "trajectory_tau_config": asdict(config),
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
        "highest_risk_steps": df.to_dict(orient="records"),
        "outputs": {
            "summary_json": str(out_dir / "failure_signal_sources_summary.json"),
            "position_figures": position_paths,
            "source_category_figures": category_paths,
            "component_profile_figures": component_paths,
        },
    }
    save_json(out_dir / "failure_signal_sources_summary.json", summary)

    print("=" * 80)
    print("Failure signal source analysis")
    print("=" * 80)
    print(f"Root: {root}")
    print(f"Eval key: {dataset_info['eval_key']}")
    print(f"Score key: {args.score_key}")
    print(f"Failed trajectories analyzed: {len(df)}")
    print()
    print("Highest-risk source categories:")
    display = (
        pd.DataFrame(
            {
                "source_category": list(source_counts.keys()),
                "count": list(source_counts.values()),
                "percentage": [source_percentages[key] for key in source_counts.keys()],
            }
        )
        .sort_values("count", ascending=False)
        .reset_index(drop=True)
    )
    print(display.to_string(index=False))
    print()
    print(f"Saved summary JSON: {out_dir / 'failure_signal_sources_summary.json'}")
    print("Saved figures:")
    for path in position_paths + category_paths + component_paths:
        print(f"  {path}")


if __name__ == "__main__":
    main()
