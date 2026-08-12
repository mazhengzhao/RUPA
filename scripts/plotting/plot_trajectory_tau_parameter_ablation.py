#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Plot Trajectory_TAU parameter ablation results as publication-ready line charts.

The script reads the summary JSON produced by
    evaluate_trajectory_tau_extended_ablation.py
and generates one vector figure per parameter group.

Each figure plots AUROC against the swept parameter values.

Example:
    python plot_trajectory_tau_parameter_ablation.py \\
      jobs/2026-07-01__10-00-09/trajectory_tau_extended_ablation_summary.json

Outputs by default:
    <summary_dir>/trajectory_tau_parameter_ablation_plots/*.pdf
    <summary_dir>/trajectory_tau_parameter_ablation_plots/*.svg
    <summary_dir>/trajectory_tau_parameter_ablation_plots/manifest.json
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METRIC_SPECS = [
    ("auroc", "AUROC", "#4C78A8"),
]

EDGE_WEIGHT_DEFAULTS = {
    "sequential": 0.35,
    "latest_user": 0.75,
    "tool_repetition": 0.95,
    "repetition": 0.95,
    "feedback_instability": 0.85,
    "feedback": 0.90,
    "progression": 0.65,
    "parallel": 0.45,
    "token_overlap": 0.70,
}

EDGE_FAMILY_DEFAULTS = {
    "sequential": 0.35,
    "latest_user": 0.75,
    "repetition": 0.95,
    "feedback": 0.875,
    "progression": 0.65,
    "parallel": 0.45,
    "token_overlap": 0.70,
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


def save_json(path: Path, data: Any) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(json_safe(data), f, indent=2, ensure_ascii=False)


def load_summary(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if "parameter_ablation_metrics" not in data:
        raise ValueError(f"Summary JSON does not contain parameter_ablation_metrics: {path}")
    return data


def safe_slug(text: str) -> str:
    text = text.strip().replace("::", "__")
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "parameter"


def display_name(param_name: str) -> str:
    if param_name == "full":
        return "Full"
    if param_name == "alpha":
        return r"lambda_h"
    if param_name.startswith("edge_family::"):
        family = param_name.split("::", 1)[1]
        return f"Edge weight: {family}"
    if param_name.startswith("edge_weight::"):
        edge_type = param_name.split("::", 1)[1]
        return f"Edge weight: {edge_type}"
    return param_name.replace("_", " ").title()


def default_value_from_config(param_name: str, base_config: dict[str, Any]) -> Any:
    if param_name == "full":
        return None
    if param_name.startswith("edge_weight::"):
        edge_type = param_name.split("::", 1)[1]
        return EDGE_WEIGHT_DEFAULTS.get(edge_type, 1.0)
    if param_name.startswith("edge_family::"):
        family = param_name.split("::", 1)[1]
        return EDGE_FAMILY_DEFAULTS.get(family, 1.0)
    return base_config.get(param_name)


def parse_params(raw: str | None) -> list[str] | None:
    if raw is None:
        return None
    items = [part.strip() for part in raw.split(",") if part.strip()]
    return items or None


def plot_parameter_group(
    df: pd.DataFrame,
    param_name: str,
    base_config: dict[str, Any],
    out_dir: Path,
    formats: list[str],
    fixed_ylim: bool,
    ylim_pad: float,
    min_y_span: float,
    show_default: bool,
    filename_suffix: str,
) -> dict[str, Any]:
    group = df[df["param_name"] == param_name].copy()
    if group.empty:
        raise ValueError(f"No rows found for param_name={param_name!r}")

    if param_name == "full":
        return {"param_name": param_name, "skipped": True}

    group = group[group["param_value"].notna()].copy()
    if group.empty:
        return {"param_name": param_name, "skipped": True}

    group["param_value"] = group["param_value"].astype(float)
    group = group.sort_values("param_value", kind="mergesort").reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(3.35, 2.2))
    plotted_values: list[float] = []
    for metric_col, metric_label, color in METRIC_SPECS:
        if metric_col not in group.columns:
            continue
        metric_values = pd.to_numeric(group[metric_col], errors="coerce")
        plotted_values.extend([float(v) for v in metric_values if pd.notna(v)])
        ax.plot(
            group["param_value"],
            metric_values,
            marker="o",
            markersize=3.2,
            linewidth=1.4,
            color=color,
            label=metric_label,
        )

    ax.set_title(display_name(param_name))
    ax.set_xlabel("Parameter value")
    ax.set_ylabel("Score")
    if fixed_ylim:
        ax.set_ylim(0.0, 1.0)
    elif plotted_values:
        y_min = min(plotted_values)
        y_max = max(plotted_values)
        raw_span = y_max - y_min
        span = max(raw_span, min_y_span)
        pad = max(span * ylim_pad, 0.004)
        center = 0.5 * (y_max + y_min)
        lower = y_min - pad
        upper = y_max + pad
        if upper - lower < span:
            lower = center - 0.5 * span
            upper = center + 0.5 * span
        lower = max(0.0, lower)
        upper = min(1.02, upper)
        ax.set_ylim(lower, upper)

    if show_default:
        default_value = default_value_from_config(param_name, base_config)
        if default_value is not None:
            try:
                default_x = float(default_value)
                ax.axvline(default_x, color="#777777", linestyle="--", linewidth=0.8, alpha=0.75)
            except (TypeError, ValueError):
                pass
    ax.grid(True, alpha=0.22)

    if len(group) <= 7:
        ax.set_xticks(group["param_value"].tolist())
    else:
        ax.locator_params(axis="x", nbins=6)

    fig.tight_layout()

    slug = safe_slug(param_name)
    saved: list[str] = []
    for fmt in formats:
        path = out_dir / f"{slug}_parameter_ablation{filename_suffix}.{fmt}"
        fig.savefig(path, bbox_inches="tight")
        saved.append(str(path))
    plt.close(fig)

    return {
        "param_name": param_name,
        "display_name": display_name(param_name),
        "show_default": show_default,
        "rows": int(len(group)),
        "saved": saved,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot Trajectory_TAU parameter ablation line charts.")
    parser.add_argument("summary_json", type=Path, help="Path to trajectory_tau_extended_ablation_summary.json")
    parser.add_argument(
        "--params",
        default=None,
        help="Comma-separated param_name filter. Default plots all parameter groups.",
    )
    parser.add_argument(
        "--formats",
        default="pdf,svg",
        help="Comma-separated output formats, e.g. pdf,svg.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to save figures. Default: <summary_dir>/trajectory_tau_parameter_ablation_plots",
    )
    parser.add_argument(
        "--fixed-ylim",
        action="store_true",
        help="Use a fixed [0, 1] y-axis. Default uses adaptive y-limits to highlight variation.",
    )
    parser.add_argument(
        "--ylim-pad",
        type=float,
        default=0.08,
        help="Relative padding for adaptive y-limits. Default: 0.08.",
    )
    parser.add_argument(
        "--min-y-span",
        type=float,
        default=0.02,
        help="Minimum adaptive y-axis span. Smaller values make subtle changes more visible. Default: 0.02.",
    )
    parser.add_argument(
        "--default-versions",
        choices=["both", "with", "without"],
        default="both",
        help="Whether to draw plots with the default marker, without it, or both. Default: both.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_matplotlib()

    summary_path = args.summary_json.expanduser().resolve()
    data = load_summary(summary_path)
    df = pd.DataFrame(data["parameter_ablation_metrics"])
    if df.empty:
        raise RuntimeError("No parameter ablation metrics found in summary JSON.")

    selected_params = parse_params(args.params)
    if selected_params is not None:
        missing = sorted(set(selected_params) - set(df["param_name"].astype(str)))
        if missing:
            raise ValueError(f"Unknown param_name values: {missing}")
        param_names = selected_params
    else:
        param_names = [name for name in df["param_name"].dropna().astype(str).unique().tolist() if name != "full"]
        param_names.sort()

    out_dir = (args.output_dir or (summary_path.parent / "trajectory_tau_parameter_ablation_plots")).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    formats = [fmt.strip().lower().lstrip(".") for fmt in args.formats.split(",") if fmt.strip()]
    if not formats:
        raise ValueError("At least one output format is required.")

    manifest = {
        "summary_json": str(summary_path),
        "output_dir": str(out_dir),
        "params": [],
        "formats": formats,
    }

    version_specs = []
    if args.default_versions in {"both", "without"}:
        version_specs.append((False, ""))
    if args.default_versions in {"both", "with"}:
        version_specs.append((True, "_with_default"))

    for param_name in param_names:
        for show_default, suffix in version_specs:
            entry = plot_parameter_group(
                df,
                param_name,
                data.get("trajectory_tau_base_config", {}),
                out_dir,
                formats,
                fixed_ylim=args.fixed_ylim,
                ylim_pad=args.ylim_pad,
                min_y_span=args.min_y_span,
                show_default=show_default,
                filename_suffix=suffix,
            )
            manifest["params"].append(entry)

    save_json(out_dir / "manifest.json", manifest)
    print(f"Saved parameter ablation figures to {out_dir}")


if __name__ == "__main__":
    main()
