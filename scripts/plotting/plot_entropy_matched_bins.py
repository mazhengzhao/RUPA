#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Plot entropy-matched-bin experiment results as publication-ready vector figures.

Input is the summary JSON or entropy-bin CSV produced by:
    evaluate_trajectory_tau_graph_experiments.py

Example:
    python plot_entropy_matched_bins.py \\
      jobs/2026-06-26__09-42-41/trajectory_tau_graph_experiments_summary.json

    python plot_entropy_matched_bins.py \\
      jobs/2026-06-26__09-42-41/trajectory_tau_graph_experiments_summary.json \\
      --methods entropy_mean_ui,tracer_score,saup_score,trajectory_tau_score \\
      --label-map trajectory_tau_score=RUPA

Outputs by default:
    <input_stem>_auroc.pdf
    <input_stem>_auroc.svg
    <input_stem>_auprc.pdf
    <input_stem>_auprc.svg
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

import matplotlib as mpl
import matplotlib.pyplot as plt


DEFAULT_METHODS = [
    {
        "label": "Entropy",
        "prefix": "entropy_mean_ui",
        "color": "#4C78A8",
        "hatch": "",
    },
    {
        "label": "TRACER",
        "prefix": "tracer_score",
        "color": "#59A14F",
        "hatch": "",
    },
    {
        "label": "SAUP",
        "prefix": "saup_score",
        "color": "#B07AA1",
        "hatch": "",
    },
    {
        "label": "UProp",
        "prefix": "uprop_score",
        "color": "#E15759",
        "hatch": "",
    },
    {
        "label": "RUPA",
        "prefix": "trajectory_tau_score",
        "color": "#F28E2B",
        "hatch": "",
    },
]


def configure_matplotlib() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 10,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "grid.linewidth": 0.5,
        }
    )


def load_bins(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if "entropy_bin_metrics" not in data:
            raise ValueError(f"JSON input does not contain entropy_bin_metrics: {path}")
        df = pd.DataFrame(data["entropy_bin_metrics"])
    else:
        df = pd.read_csv(path)

    required = {
        "entropy_bin",
        "n",
        "n_success",
        "n_failure",
        "entropy_min",
        "entropy_max",
        "failure_rate",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns in {path}: {missing}")
    return df.sort_values("entropy_bin").reset_index(drop=True)


def bin_labels(df: pd.DataFrame) -> list[str]:
    labels = []
    for _, row in df.iterrows():
        labels.append(
            f"Q{int(row['entropy_bin']) + 1}\n"
            f"[{row['entropy_min']:.3f}, {row['entropy_max']:.3f}]"
        )
    return labels


def parse_key_value_map(raw: str) -> dict[str, str]:
    mapping: dict[str, str] = {}
    if not raw.strip():
        return mapping

    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise ValueError(f"Expected key=value item in mapping, got: {part!r}")
        key, value = part.split("=", 1)
        mapping[key.strip()] = value.strip().strip("\"'")
    return mapping


def default_method_by_prefix() -> dict[str, dict[str, str]]:
    methods = {spec["prefix"]: dict(spec) for spec in DEFAULT_METHODS}
    methods.update(
        {
            "seq_prob_score": {
                "label": "SP",
                "prefix": "seq_prob_score",
                "color": "#76B7B2",
                "hatch": "",
            },
            "sp_score": {
                "label": "SP",
                "prefix": "sp_score",
                "color": "#76B7B2",
                "hatch": "",
            },
            "no_graph_tau_score": {
                "label": "No-Graph TAU",
                "prefix": "no_graph_tau_score",
                "color": "#9C755F",
                "hatch": "",
            },
        }
    )
    return methods


def method_specs(
    selected_prefixes: list[str] | None,
    label_map: dict[str, str],
    color_map: dict[str, str],
) -> list[dict[str, str]]:
    defaults = default_method_by_prefix()

    if selected_prefixes:
        prefixes = selected_prefixes
    else:
        prefixes = [spec["prefix"] for spec in DEFAULT_METHODS]

    methods = []
    fallback_colors = [
        "#4C78A8",
        "#F28E2B",
        "#59A14F",
        "#B07AA1",
        "#9C755F",
        "#E15759",
        "#76B7B2",
        "#EDC948",
    ]

    for idx, prefix in enumerate(prefixes):
        spec = dict(defaults.get(prefix, {}))
        spec.setdefault("prefix", prefix)
        spec.setdefault("label", prefix)
        spec.setdefault("color", fallback_colors[idx % len(fallback_colors)])
        spec.setdefault("hatch", "")
        if prefix in label_map:
            spec["label"] = label_map[prefix]
        if prefix in color_map:
            spec["color"] = color_map[prefix]
        methods.append(spec)

    return methods


def metric_column(prefix: str, metric: str) -> str:
    return f"{prefix}_{metric}"


def available_methods(
    df: pd.DataFrame,
    metric: str,
    selected_prefixes: list[str] | None,
    label_map: dict[str, str],
    color_map: dict[str, str],
) -> list[dict[str, str]]:
    methods = []
    for spec in method_specs(selected_prefixes, label_map, color_map):
        col = metric_column(spec["prefix"], metric)
        if col in df.columns:
            methods.append(spec)
        elif selected_prefixes:
            raise ValueError(f"Selected method prefix {spec['prefix']!r} has no column {col!r}")
    if not methods:
        raise ValueError(f"No method columns found for metric={metric}")
    return methods


def plot_metric(
    df: pd.DataFrame,
    metric: str,
    output_base: Path,
    formats: Iterable[str],
    selected_prefixes: list[str] | None,
    label_map: dict[str, str],
    color_map: dict[str, str],
    title: str | None,
    ylim: tuple[float, float] | None,
    tight_y: bool,
    y_padding: float,
    min_y_span: float,
) -> list[Path]:
    methods = available_methods(df, metric, selected_prefixes, label_map, color_map)
    labels = bin_labels(df)
    x = np.arange(len(df))
    width = min(0.18, 0.76 / max(1, len(methods)))

    fig_width = max(6.2, 1.05 * len(df) + 2.4)
    fig, ax = plt.subplots(figsize=(fig_width, 3.2))

    plotted_values: list[float] = []
    for idx, spec in enumerate(methods):
        col = metric_column(spec["prefix"], metric)
        values = df[col].to_numpy(dtype=float)
        plotted_values.extend([float(value) for value in values if np.isfinite(value)])
        offset = (idx - (len(methods) - 1) / 2.0) * width
        ax.bar(
            x + offset,
            values,
            width=width,
            label=spec["label"],
            color=spec["color"],
            edgecolor="black",
            linewidth=0.45,
            hatch=spec["hatch"],
            alpha=0.92,
        )

    if ylim is not None:
        ax.set_ylim(*ylim)
    elif tight_y and plotted_values:
        y_min = min(plotted_values)
        y_max = max(plotted_values)
        span = max(y_max - y_min, min_y_span)
        pad = max(span * y_padding, 0.01)
        center = 0.5 * (y_min + y_max)
        lower = y_min - pad
        upper = y_max + pad
        if upper - lower < span:
            lower = center - 0.5 * span
            upper = center + 0.5 * span
        ax.set_ylim(max(0.0, lower), min(1.02, upper))
    else:
        ax.set_ylim(0.0, 1.04)
    ax.set_ylabel(metric.upper())
    ax.set_xlabel("Entropy-matched bin")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.yaxis.grid(True, linestyle="-", alpha=0.25)

    handles, handle_labels = ax.get_legend_handles_labels()
    ax.legend(
        handles,
        handle_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.23),
        ncol=min(5, len(methods)),
        frameon=False,
        columnspacing=1.2,
        handlelength=1.5,
    )

    if title:
        ax.set_title(title)
    else:
        ax.set_title(f"Entropy-matched bins: {metric.upper()} within bins")

    # Mark bins without both classes. They naturally produce NaN bars.
    single_class = df[(df["n_success"] == 0) | (df["n_failure"] == 0)]
    for _, row in single_class.iterrows():
        idx = int(row["entropy_bin"])
        ax.text(
            idx,
            0.06,
            "single\nclass",
            ha="center",
            va="bottom",
            fontsize=7,
            color="#555555",
        )

    fig.tight_layout(pad=0.8)

    output_paths = []
    for fmt in formats:
        path = output_base.with_suffix(f".{fmt}")
        fig.savefig(path, format=fmt, bbox_inches="tight")
        output_paths.append(path)
    plt.close(fig)
    return output_paths


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
    parser = argparse.ArgumentParser(
        description="Plot entropy-matched-bin experiment results as vector figures."
    )
    parser.add_argument(
        "input",
        type=Path,
        help="Path to trajectory_tau_graph_experiments_summary.json or entropy_bins.csv.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to the CSV directory.",
    )
    parser.add_argument(
        "--metrics",
        default="auroc,auprc",
        help="Comma-separated metrics to plot: auroc,auprc.",
    )
    parser.add_argument(
        "--methods",
        default="",
        help=(
            "Comma-separated method prefixes to include and order. "
            "Example: entropy_mean_ui,tracer_score,saup_score,trajectory_tau_score. "
            "Defaults to available baseline methods plus RUPA."
        ),
    )
    parser.add_argument(
        "--label-map",
        default="",
        help=(
            "Comma-separated prefix=label overrides. "
            "Example: trajectory_tau_score=Ours,graph_mean_uncertainty='Graph signal'."
        ),
    )
    parser.add_argument(
        "--color-map",
        default="",
        help="Comma-separated prefix=#RRGGBB color overrides.",
    )
    parser.add_argument(
        "--formats",
        default="pdf,svg",
        help="Comma-separated vector formats: pdf,svg,eps.",
    )
    parser.add_argument(
        "--title-prefix",
        default="",
        help="Optional title prefix, e.g. dataset name.",
    )
    parser.add_argument(
        "--ylim",
        default="",
        help="Manual y-axis limits as MIN,MAX, e.g. 0.3,0.85.",
    )
    parser.add_argument(
        "--no-tight-y",
        action="store_true",
        help="Disable adaptive y-axis tightening and use 0,1.04.",
    )
    parser.add_argument(
        "--y-padding",
        type=float,
        default=0.08,
        help="Relative padding for adaptive y-axis. Default: 0.08.",
    )
    parser.add_argument(
        "--min-y-span",
        type=float,
        default=0.18,
        help="Minimum adaptive y-axis span. Smaller values make differences more visible. Default: 0.18.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_matplotlib()

    input_path = args.input.expanduser().resolve()
    df = load_bins(input_path)
    out_dir = (args.out_dir or input_path.parent).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    formats = parse_formats(args.formats)
    selected_prefixes = [item.strip() for item in args.methods.split(",") if item.strip()] or None
    label_map = parse_key_value_map(args.label_map)
    color_map = parse_key_value_map(args.color_map)
    ylim = parse_ylim(args.ylim)
    metrics = [item.strip().lower() for item in args.metrics.split(",") if item.strip()]
    for metric in metrics:
        if metric not in {"auroc", "auprc"}:
            raise ValueError(f"Unsupported metric: {metric}")

    created = []
    for metric in metrics:
        title = None
        if args.title_prefix:
            title = f"{args.title_prefix}: entropy-matched {metric.upper()}"
        output_base = out_dir / f"{input_path.stem}_{metric}"
        created.extend(
            plot_metric(
                df=df,
                metric=metric,
                output_base=output_base,
                formats=formats,
                selected_prefixes=selected_prefixes,
                label_map=label_map,
                color_map=color_map,
                title=title,
                ylim=ylim,
                tight_y=not args.no_tight_y,
                y_padding=args.y_padding,
                min_y_span=args.min_y_span,
            )
        )

    print("Saved vector figures:")
    for path in created:
        print(f"  {path}")


if __name__ == "__main__":
    main()
