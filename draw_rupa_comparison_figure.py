#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Draw a comparison figure: traditional UQ vs. RUPA structure-aware modeling."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch


OUT_BASE = Path("rupa_vs_traditional_structure")

TEXT = "#2F2F2F"
MUTED = "#6B6B6B"
GRAY = "#8A8A8A"
LIGHT_GRAY = "#F3F3F3"
TRAD = "#BDBDBD"

COLORS = {
    "user": "#4C78A8",
    "assistant": "#F28E2B",
    "env": "#59A14F",
    "risk": "#E15759",
    "logic": "#2F8F6B",
    "prop": "#7A5195",
    "rupa_bg": "#E8F5E9",
    "gap_bg": "#F3E5F5",
    "out_bg": "#FBE9E7",
}


def configure() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "mathtext.fontset": "stix",
            "font.size": 7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def box(ax, x, y, w, h, fc, ec="#D5D5D5", lw=0.8, radius=0.016, z=1):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0.006,rounding_size={radius}",
        facecolor=fc,
        edgecolor=ec,
        linewidth=lw,
        zorder=z,
    )
    ax.add_patch(patch)
    return patch


def arrow(ax, p0, p1, color=GRAY, lw=0.9, rad=0.0, style="-|>", ls="-", alpha=1.0, z=5):
    patch = FancyArrowPatch(
        p0,
        p1,
        arrowstyle=style,
        mutation_scale=8,
        linewidth=lw,
        color=color,
        linestyle=ls,
        connectionstyle=f"arc3,rad={rad}",
        alpha=alpha,
        zorder=z,
    )
    ax.add_patch(patch)
    return patch


def node(ax, x, y, color, label, r=0.015, alpha=1.0, z=6):
    ax.add_patch(Circle((x, y), r, facecolor=color, edgecolor="white", linewidth=0.8, alpha=alpha, zorder=z))
    ax.text(x, y, label, ha="center", va="center", fontsize=5.8, color="white", fontweight="bold", zorder=z + 1)


def row_label(ax, x, y, title, subtitle, color):
    ax.text(x, y + 0.045, title, ha="left", va="center", fontsize=7.3, fontweight="bold", color=color)
    ax.text(x, y + 0.012, subtitle, ha="left", va="center", fontsize=5.6, color=MUTED)


def draw_sequence(ax, x, y, colors, compact=False):
    labels = [("u", colors["user"]), ("y", colors["assistant"]), ("o", colors["env"]), ("y", colors["assistant"]), ("o", colors["env"])]
    dx = 0.036 if compact else 0.043
    for i, (lab, color) in enumerate(labels):
        xi = x + i * dx
        node(ax, xi, y, color, lab, r=0.014 if compact else 0.016)
        if i > 0:
            arrow(ax, (x + (i - 1) * dx + 0.016, y), (xi - 0.016, y), color=GRAY, lw=0.75)


def draw_flat_uq(ax, x, y):
    for i in range(4):
        xi = x + i * 0.035
        ax.add_patch(Circle((xi, y), 0.014, facecolor="#FFFFFF", edgecolor=TRAD, linewidth=1.1, zorder=4))
        ax.text(xi, y, r"$U$", ha="center", va="center", fontsize=5.5, color=MUTED, zorder=5)
    ax.text(x + 0.052, y - 0.045, "local scores", ha="center", va="center", fontsize=5.5, color=MUTED)


def draw_no_structure(ax, x, y):
    box(ax, x - 0.055, y - 0.035, 0.11, 0.070, "#FFFFFF", ec=TRAD, lw=0.9, radius=0.010)
    ax.text(x, y + 0.011, "flat aggregate", ha="center", va="center", fontsize=5.7, color=TEXT)
    ax.text(x, y - 0.015, r"$\frac{1}{T}\sum_t U_t$", ha="center", va="center", fontsize=6.1, color=MUTED)
    ax.text(x, y - 0.065, "no relation edges", ha="center", va="center", fontsize=5.2, color=COLORS["risk"])


def draw_graph(ax, x, y):
    pts = {
        "u": (x, y + 0.045),
        "y1": (x + 0.055, y + 0.055),
        "o": (x + 0.110, y + 0.045),
        "y2": (x + 0.055, y - 0.030),
        "t": (x + 0.120, y - 0.035),
    }
    node(ax, *pts["u"], COLORS["user"], "u")
    node(ax, *pts["y1"], COLORS["assistant"], "y")
    node(ax, *pts["o"], COLORS["env"], "o")
    node(ax, *pts["y2"], COLORS["assistant"], "y")
    node(ax, *pts["t"], COLORS["assistant"], "t")
    ax.add_patch(Circle(pts["t"], 0.024, fill=False, edgecolor=COLORS["risk"], linewidth=1.2, zorder=7))
    arrow(ax, pts["u"], pts["y1"], color="#555555", lw=0.75)
    arrow(ax, pts["y1"], pts["o"], color="#555555", lw=0.75)
    arrow(ax, pts["y1"], pts["y2"], color=COLORS["risk"], lw=1.0, ls="--")
    arrow(ax, pts["y2"], pts["t"], color=COLORS["logic"], lw=1.0, ls="-.")
    arrow(ax, pts["o"], pts["t"], color=COLORS["risk"], lw=1.0, ls="--", rad=0.12)
    arrow(ax, pts["u"], pts["t"], color=COLORS["user"], lw=0.8, rad=-0.25)
    ax.text(x + 0.06, y - 0.080, r"$G_t=(V_t,E_t)$", ha="center", va="center", fontsize=6.5, color=TEXT)


def draw_prop(ax, x, y):
    ax.text(x, y + 0.050, "incoming edges", ha="center", va="center", fontsize=5.5, color=MUTED)
    for i, (c, lw) in enumerate([(COLORS["user"], 1.1), (COLORS["risk"], 2.0), (COLORS["prop"], 2.8)]):
        yy = y + 0.020 - i * 0.028
        ax.plot([x - 0.050, x - 0.015], [yy, yy], color=c, lw=lw, solid_capstyle="round")
        arrow(ax, (x - 0.012, yy), (x + 0.030, y - 0.005), color=COLORS["prop"], lw=0.8, alpha=0.8)
    node(ax, x + 0.048, y - 0.005, COLORS["assistant"], "t", r=0.017)
    ax.text(x + 0.048, y - 0.055, r"$G_t \rightarrow H_t$", ha="center", va="center", fontsize=6.3, color=COLORS["prop"])


def draw_gap_and_fusion(ax, x, y):
    ax.plot([x - 0.040, x + 0.050], [y + 0.025, y + 0.025], color="#D0D0D0", lw=0.8)
    arrow(ax, (x, y + 0.025), (x + 0.045, y + 0.025), color=COLORS["user"], lw=1.0)
    arrow(ax, (x, y + 0.025), (x + 0.018, y + 0.062), color=COLORS["risk"], lw=1.1)
    ax.text(x + 0.018, y + 0.075, r"$y_t$", fontsize=5.7, color=COLORS["risk"], ha="center")
    ax.text(x + 0.055, y + 0.013, r"$x$", fontsize=5.7, color=COLORS["user"], ha="center")
    ax.text(x, y - 0.032, r"$C_t$: goal drift", ha="center", va="center", fontsize=5.7, color=TEXT)

    gx = x + 0.145
    ax.add_patch(Circle((gx, y + 0.018), 0.045, facecolor="none", edgecolor="#CFCFCF", linewidth=3.0, zorder=3))
    ax.add_patch(Circle((gx, y + 0.018), 0.045, facecolor="none", edgecolor=COLORS["risk"], linewidth=3.0, alpha=0.55, zorder=4))
    ax.plot([gx, gx + 0.030], [y + 0.018, y + 0.045], color=TEXT, lw=1.0, zorder=5)
    ax.text(gx, y - 0.050, r"$R_t=U_t(1+\alpha H_t+\beta C_t)$", ha="center", va="center", fontsize=5.8, color=COLORS["risk"])


def main() -> None:
    configure()
    fig, ax = plt.subplots(figsize=(7.4, 2.8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.5, 0.955, "RUPA introduces reasoning-structure modeling beyond traditional UQ", ha="center", va="top", fontsize=9.0, fontweight="bold", color=TEXT)
    ax.text(0.5, 0.910, "Traditional UQ: local/flat uncertainty   vs.   RUPA: relation graph + uncertainty propagation + alignment-aware risk", ha="center", va="top", fontsize=6.2, color=MUTED)

    # Row backgrounds.
    box(ax, 0.035, 0.590, 0.93, 0.235, LIGHT_GRAY, ec="#D4D4D4", lw=0.8, radius=0.018)
    box(ax, 0.035, 0.270, 0.93, 0.255, "#F8FBF8", ec="#CFCFCF", lw=0.8, radius=0.018)

    row_label(ax, 0.055, 0.708, "Traditional UQ", "flat local scores", MUTED)
    row_label(ax, 0.055, 0.393, "RUPA", "structure-aware UQ", COLORS["risk"])

    # Traditional pipeline.
    draw_sequence(ax, 0.235, 0.710, {"user": TRAD, "assistant": TRAD, "env": TRAD}, compact=True)
    arrow(ax, (0.435, 0.710), (0.485, 0.710), color=GRAY, lw=1.0)
    draw_flat_uq(ax, 0.515, 0.710)
    arrow(ax, (0.665, 0.710), (0.715, 0.710), color=GRAY, lw=1.0)
    draw_no_structure(ax, 0.785, 0.710)
    ax.text(0.905, 0.710, "confidence", ha="center", va="center", fontsize=6.0, color=MUTED)

    # RUPA pipeline.
    draw_sequence(ax, 0.210, 0.395, COLORS, compact=True)
    arrow(ax, (0.400, 0.395), (0.438, 0.395), color=GRAY, lw=1.0)
    draw_graph(ax, 0.455, 0.405)
    arrow(ax, (0.600, 0.395), (0.635, 0.395), color=GRAY, lw=1.0)
    draw_prop(ax, 0.690, 0.405)
    arrow(ax, (0.765, 0.395), (0.800, 0.395), color=GRAY, lw=1.0)
    draw_gap_and_fusion(ax, 0.795, 0.390)

    # Contribution callouts.
    ax.text(0.525, 0.295, "C1 relation graph", ha="center", va="center", fontsize=5.8, color=COLORS["risk"], fontweight="bold")
    ax.text(0.690, 0.295, "C2 propagation", ha="center", va="center", fontsize=5.8, color=COLORS["prop"], fontweight="bold")
    ax.text(0.855, 0.295, "C3 alignment-aware risk", ha="center", va="center", fontsize=5.8, color=COLORS["risk"], fontweight="bold")

    # Role legend.
    for i, (key, label) in enumerate([("user", "User"), ("assistant", "Assistant"), ("env", "Environment")]):
        x = 0.055 + i * 0.095
        ax.add_patch(Circle((x, 0.110), 0.007, facecolor=COLORS[key], edgecolor="none"))
        ax.text(x + 0.012, 0.110, label, ha="left", va="center", fontsize=5.8, color=MUTED)

    fig.savefig(OUT_BASE.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(OUT_BASE.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(OUT_BASE.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
