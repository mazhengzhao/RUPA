#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Plot prefix UQ curves from JSON produced by evaluate_prefix_uq_curves.py.

The default figure size is suitable for one subfigure in a paper figure.

Examples:
    python plot_prefix_uq_curves.py jobs/2026-07-01__10-00-09/prefix_uq_curves_percent.json
    python plot_prefix_uq_curves.py jobs/2026-07-01__10-00-09/prefix_uq_curves_steps.json --metric auprc
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


DEFAULT_STYLE = {
    "entropy": {"label": "Entropy", "color": "#4C78A8", "marker": "o", "linestyle": "-"},
    "tracer": {"label": "TRACER", "color": "#59A14F", "marker": "s", "linestyle": "-"},
    "saup": {"label": "SAUP", "color": "#B07AA1", "marker": "^", "linestyle": "-"},
    "uprop": {"label": "UProp", "color": "#E15759", "marker": "v", "linestyle": "-"},
    "tau": {"label": "TAU", "color": "#9C755F", "marker": "D", "linestyle": "-"},
    "trajectory_tau": {"label": "Trajectory TAU", "color": "#F28E2B", "marker": "P", "linestyle": "-"},
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


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_csv_list(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def parse_key_value_map(raw: str) -> dict[str, str]:
    mapping: dict[str, str] = {}
    if not raw.strip():
        return mapping
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise ValueError(f"Expected key=value item, got {part!r}")
        key, value = part.split("=", 1)
        mapping[key.strip()] = value.strip().strip("\"'")
    return mapping


def last_path_segment(raw: Any) -> str:
    text = str(raw).strip()
    if not text:
        return ""
    return text.rstrip("/").split("/")[-1]


def title_from_eval_key(data: dict[str, Any], metric: str) -> str | None:
    eval_key = str(data.get("eval_key", "")).strip()
    if not eval_key:
        return None

    parts = eval_key.split("__")
    if len(parts) >= 3:
        model = last_path_segment(parts[1])
        dataset = last_path_segment(parts[2])
        if model and dataset:
            return f"{dataset}:{model}"

    compact_key = last_path_segment(eval_key)
    if compact_key:
        return f"{compact_key}: {metric.upper()} vs. Prefix"
    return None


def selected_point_indices(data: dict[str, Any], max_step: int | None) -> list[int]:
    point_items = data.get("points", [])
    mode = data.get("mode", "percent")
    indices = []
    for idx, item in enumerate(point_items):
        point = item.get("point") if isinstance(item, dict) else item
        if mode == "steps" and max_step is not None and float(point) > max_step:
            continue
        indices.append(idx)
    return indices


def x_values(
    data: dict[str, Any],
    indices: list[int],
    uniform_x_spacing: bool,
) -> tuple[np.ndarray, list[str], str]:
    mode = data.get("mode", "percent")
    point_items = data.get("points", [])
    points = [
        item.get("point") if isinstance(item, dict) else item
        for item in point_items
    ]
    points = [points[idx] for idx in indices]
    x = np.arange(len(points), dtype=float)

    if mode == "percent":
        labels = [f"{int(round(float(point) * 100.0))}" for point in points]
        return x, labels, "Prefix (%)"

    labels = [f"{int(float(point))}" for point in points]
    return x, labels, "Prefix agent steps"


def metric_values(data: dict[str, Any], method: str, metric: str, indices: list[int]) -> np.ndarray:
    values = []
    points = data.get("points", [])
    for idx in indices:
        point = points[idx]
        metrics = point.get("method_metrics", {}).get(method, {})
        value = metrics.get(metric)
        values.append(np.nan if value is None else float(value))
    return np.asarray(values, dtype=float)


def plot_curve(
    data: dict[str, Any],
    metric: str,
    methods: list[str],
    label_map: dict[str, str],
    output_base: Path,
    formats: list[str],
    figsize: tuple[float, float],
    title: str | None,
    legend: bool,
    max_step: int | None,
    uniform_x_spacing: bool,
    ylim: tuple[float, float] | None,
    tight_y: bool,
    y_padding: float,
    chance_line: float | None,
) -> list[Path]:
    indices = selected_point_indices(data, max_step)
    x, xtick_labels, xlabel = x_values(data, indices, uniform_x_spacing)
    fig, ax = plt.subplots(figsize=figsize)

    for method in methods:
        style = dict(DEFAULT_STYLE.get(method, {}))
        style.setdefault("label", method)
        style.setdefault("color", None)
        style.setdefault("marker", "o")
        style.setdefault("linestyle", "-")
        label = label_map.get(method, style["label"])
        y = metric_values(data, method, metric, indices)
        if np.all(np.isnan(y)):
            continue
        ax.plot(
            x,
            y,
            label=label,
            color=style["color"],
            marker=style["marker"],
            linestyle=style["linestyle"],
            linewidth=1.35,
            markersize=3.6,
            markeredgewidth=0.4,
        )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(metric.upper())
    if chance_line is not None:
        ax.axhline(chance_line, color="#777777", linestyle="--", linewidth=0.8, zorder=0)
    if ylim is not None:
        ax.set_ylim(*ylim)
    elif tight_y:
        all_values = []
        for method in methods:
            y = metric_values(data, method, metric, indices)
            all_values.extend([float(value) for value in y if not np.isnan(value)])
        if all_values:
            low = max(0.0, min(all_values) - y_padding)
            high = min(1.02, max(all_values) + y_padding)
            if high - low < 0.08:
                center = (high + low) / 2.0
                low = max(0.0, center - 0.04)
                high = min(1.02, center + 0.04)
            ax.set_ylim(low, high)
        else:
            ax.set_ylim(0.0, 1.02)
    else:
        ax.set_ylim(0.0, 1.02)
    ax.yaxis.grid(True, alpha=0.28)
    ax.xaxis.grid(False)
    if len(x) > 0:
        ax.set_xticks(x)
        ax.set_xticklabels(xtick_labels)
    if title:
        ax.set_title(title)
    if legend:
        ax.legend(frameon=False, loc="lower right", handlelength=1.8)
    fig.tight_layout(pad=0.35)

    paths = []
    for fmt in formats:
        path = output_base.with_suffix(f".{fmt}")
        fig.savefig(path, format=fmt, bbox_inches="tight")
        paths.append(path)
    plt.close(fig)
    return paths


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
        raise ValueError("At least one format is required")
    return formats


def parse_figsize(raw: str) -> tuple[float, float]:
    if "," not in raw:
        raise ValueError("--figsize must be WIDTH,HEIGHT")
    width, height = raw.split(",", 1)
    return float(width), float(height)


def parse_ylim(raw: str) -> tuple[float, float] | None:
    if not raw.strip():
        return None
    if "," not in raw:
        raise ValueError("--ylim must be MIN,MAX")
    low, high = raw.split(",", 1)
    low_value = float(low)
    high_value = float(high)
    if high_value <= low_value:
        raise ValueError("--ylim max must be greater than min")
    return low_value, high_value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot prefix UQ curves from saved JSON.")
    parser.add_argument("json", type=Path, help="Path to prefix_uq_curves_*.json")
    parser.add_argument(
        "--metric",
        default="auroc",
        help=(
            "Metric or comma-separated metrics to plot. Available: "
            "auroc,auprc,f1_at_threshold,accuracy_at_threshold."
        ),
    )
    parser.add_argument(
        "--methods",
        default="entropy,tracer,saup,uprop,tau,trajectory_tau",
        help="Comma-separated methods to include and order.",
    )
    parser.add_argument(
        "--label-map",
        default="",
        help="Comma-separated method=label overrides, e.g. trajectory_tau=Ours.",
    )
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--formats", default="pdf,svg")
    parser.add_argument(
        "--figsize",
        default="3.25,2.25",
        help="Figure size in inches. Default is suitable for a single paper subfigure.",
    )
    parser.add_argument("--title", default="")
    parser.add_argument("--no-legend", action="store_true")
    parser.add_argument(
        "--ylim",
        default="",
        help="Manual y-axis limits as MIN,MAX, e.g. 0.5,0.75.",
    )
    parser.add_argument(
        "--tight-y",
        action="store_true",
        help="Automatically tighten y-axis limits around plotted curves.",
    )
    parser.add_argument(
        "--y-padding",
        type=float,
        default=0.025,
        help="Padding used by --tight-y.",
    )
    parser.add_argument(
        "--chance-line",
        type=float,
        default=None,
        help="Optional horizontal reference line, e.g. 0.5 for AUROC.",
    )
    parser.add_argument(
        "--max-step",
        type=int,
        default=None,
        help="For step-mode JSON, plot only prefix points <= this step count.",
    )
    parser.add_argument(
        "--uniform-x-spacing",
        action="store_true",
        help="Retained for backward compatibility. X-axis spacing is always uniform.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_matplotlib()
    path = args.json.expanduser().resolve()
    data = load_json(path)
    metrics = parse_csv_list(args.metric)
    allowed_metrics = {"auroc", "auprc", "f1_at_threshold", "accuracy_at_threshold"}
    unknown_metrics = sorted(set(metrics) - allowed_metrics)
    if unknown_metrics:
        raise ValueError(f"Unknown metrics: {unknown_metrics}. Available: {sorted(allowed_metrics)}")
    methods = parse_csv_list(args.methods)
    label_map = parse_key_value_map(args.label_map)
    formats = parse_formats(args.formats)
    figsize = parse_figsize(args.figsize)
    ylim = parse_ylim(args.ylim)
    out_dir = (args.out_dir or path.parent).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Saved vector figures:")
    for metric in metrics:
        title = args.title.strip() or title_from_eval_key(data, metric)
        output_base = out_dir / f"{path.stem}_{metric}"
        paths = plot_curve(
            data=data,
            metric=metric,
            methods=methods,
            label_map=label_map,
            output_base=output_base,
            formats=formats,
            figsize=figsize,
            title=title,
            legend=not args.no_legend,
            max_step=args.max_step,
            uniform_x_spacing=args.uniform_x_spacing,
            ylim=ylim,
            tight_y=args.tight_y,
            y_padding=args.y_padding,
            chance_line=args.chance_line,
        )
        for out in paths:
            print(f"  {out}")


if __name__ == "__main__":
    main()
