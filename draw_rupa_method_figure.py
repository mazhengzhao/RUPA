#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Draw an ICLR-style horizontal overview figure for RUPA."""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Arc, Circle, FancyArrowPatch, FancyBboxPatch, Polygon


OUT_BASE = Path("rupa_method_overview")

TEXT = "#2F2F2F"
MUTED = "#6B6B6B"
GRAY = "#8A8A8A"

PANELS = [
    ("M1 Trajectory", "#E1F5FE"),
    ("M2 Relation Graph", "#E8F5E9"),
    ("M3 Propagation", "#FFFDE7"),
    ("M4 Interaction Gap", "#F3E5F5"),
    ("M5 Fusion Output", "#FBE9E7"),
]

ROLE_COLORS = {
    "user": "#4C78A8",
    "assistant": "#F28E2B",
    "env": "#59A14F",
    "risk": "#E15759",
    "logic": "#2F8F6B",
    "prop": "#7A5195",
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


def box(ax, x, y, w, h, fc, ec="#D0D0D0", lw=0.8, radius=0.018, z=1):
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
        mutation_scale=8.5,
        linewidth=lw,
        color=color,
        linestyle=ls,
        connectionstyle=f"arc3,rad={rad}",
        alpha=alpha,
        zorder=z,
    )
    ax.add_patch(patch)
    return patch


def node(ax, x, y, color, label, r=0.017, shape="circle", alpha=1.0, z=6):
    if shape == "square":
        patch = box(ax, x - r, y - r, 2 * r, 2 * r, color, ec="white", lw=0.8, radius=0.006, z=z)
    else:
        patch = Circle((x, y), r, facecolor=color, edgecolor="white", linewidth=0.9, alpha=alpha, zorder=z)
        ax.add_patch(patch)
    ax.text(x, y, label, ha="center", va="center", color="white", fontsize=6.2, fontweight="bold", zorder=z + 1)
    return patch


def panel_title(ax, x, y, title):
    ax.text(x, y, title, ha="center", va="top", fontsize=5.8, fontweight="bold", color=TEXT, zorder=10)


def contribution_tag(ax, x, y, text, color):
    box(ax, x, y, 0.063, 0.026, "#FFFFFF", ec=color, lw=0.8, radius=0.010, z=10)
    ax.text(x + 0.0315, y + 0.013, text, ha="center", va="center", fontsize=4.8, fontweight="bold", color=color, zorder=11)


def draw_backbone(ax, xs, y):
    arrow(ax, (xs[0], y), (xs[-1], y), color="#9A9A9A", lw=1.5, style="-|>", z=3)
    labels = ["Trajectory", "Graph (C1)", "Propagation (C2)", "Gap (C3)", "Fusion (C3)"]
    for x, label in zip(xs, labels):
        ax.plot([x, x], [y - 0.012, y + 0.012], color="#9A9A9A", lw=0.8, zorder=3)
        ax.text(x, y - 0.032, label, ha="center", va="top", fontsize=5.9, color=MUTED)


def draw_m1(ax, x, y, w, h):
    panel_title(ax, x + w / 2, y + h - 0.025, "Trajectory")
    labels = [("User", ROLE_COLORS["user"]), ("Reason", ROLE_COLORS["assistant"]), ("Tool", ROLE_COLORS["assistant"]), ("Obs", ROLE_COLORS["env"])]
    y0 = y + 0.33
    step_w, step_h = 0.032, 0.056
    xs = [x + 0.030, x + 0.068, x + 0.106, x + 0.144]
    for i, ((lab, color), xi) in enumerate(zip(labels, xs)):
        box(ax, xi, y0, step_w, step_h, color, ec="white", lw=0.8, radius=0.009, z=5)
        ax.text(xi + step_w / 2, y0 + step_h / 2, lab, ha="center", va="center", fontsize=4.6, color="white", fontweight="bold", zorder=6)
        if i > 0:
            arrow(ax, (xs[i - 1] + step_w, y0 + step_h / 2), (xi, y0 + step_h / 2), color=GRAY, lw=0.8)
    ax.text(x + w / 2, y + 0.15, r"$s_t=(y_t,o_t)$", ha="center", va="center", fontsize=7.0, color=TEXT)
    ax.text(x + w / 2, y + 0.095, r"action $y_t$ + observation $o_t$", ha="center", va="center", fontsize=5.0, color=MUTED)


def draw_m2(ax, x, y, w, h):
    panel_title(ax, x + w / 2, y + h - 0.025, "Graph")
    ax.text(x + w / 2, y + h - 0.078, r"$G_t=(V_t,E_t)$", ha="center", va="center", fontsize=7.2, color=TEXT)

    p = {
        "u": (x + 0.038, y + 0.265),
        "a1": (x + 0.082, y + 0.315),
        "e": (x + 0.130, y + 0.293),
        "a2": (x + 0.082, y + 0.205),
        "t": (x + 0.137, y + 0.185),
    }
    node(ax, *p["u"], ROLE_COLORS["user"], "u")
    node(ax, *p["a1"], ROLE_COLORS["assistant"], "y")
    node(ax, *p["e"], ROLE_COLORS["env"], "o")
    node(ax, *p["a2"], ROLE_COLORS["assistant"], "y")
    node(ax, *p["t"], ROLE_COLORS["assistant"], "t")
    ax.add_patch(Circle(p["t"], 0.024, fill=False, edgecolor=ROLE_COLORS["risk"], linewidth=1.2, zorder=7))

    arrow(ax, p["u"], p["a1"], color="#555555", lw=0.8)
    arrow(ax, p["a1"], p["e"], color="#555555", lw=0.8)
    arrow(ax, p["a1"], p["a2"], color=ROLE_COLORS["risk"], lw=1.0, ls="--", rad=0.0)
    arrow(ax, p["a2"], p["t"], color=ROLE_COLORS["logic"], lw=1.0, ls="-.", rad=0.0)
    arrow(ax, p["e"], p["t"], color=ROLE_COLORS["risk"], lw=1.0, ls="--", rad=0.08)
    arrow(ax, p["u"], p["t"], color=ROLE_COLORS["user"], lw=0.9, rad=-0.22)

    ly = y + 0.065
    ax.plot([x + 0.018, x + 0.043], [ly, ly], color="#555555", lw=0.8)
    ax.text(x + 0.048, ly, "seq/last", ha="left", va="center", fontsize=4.5, color=MUTED)
    ax.plot([x + 0.018, x + 0.043], [ly - 0.026, ly - 0.026], color=ROLE_COLORS["risk"], lw=1.0, ls="--")
    ax.text(x + 0.048, ly - 0.026, "repeat/fb", ha="left", va="center", fontsize=4.5, color=ROLE_COLORS["risk"])
    ax.plot([x + 0.018, x + 0.043], [ly - 0.052, ly - 0.052], color=ROLE_COLORS["logic"], lw=1.0, ls="-.")
    ax.text(x + 0.048, ly - 0.052, "prog/parallel", ha="left", va="center", fontsize=4.5, color=ROLE_COLORS["logic"])


def draw_m3(ax, x, y, w, h):
    panel_title(ax, x + w / 2, y + h - 0.025, "Propagation")
    center = (x + 0.105, y + 0.245)
    for dx, alpha in [(-0.030, 0.18), (-0.017, 0.28)]:
        Circle((center[0] + dx, center[1]), 0.030, facecolor=ROLE_COLORS["risk"], edgecolor="none", alpha=alpha, zorder=3)
        ax.add_patch(Circle((center[0] + dx, center[1]), 0.030, facecolor=ROLE_COLORS["risk"], edgecolor="none", alpha=alpha, zorder=3))
    node(ax, *center, ROLE_COLORS["assistant"], "t", r=0.025)

    sources = [
        (x + 0.033, y + 0.305, 2.0, 0.75),
        (x + 0.033, y + 0.245, 1.2, 0.55),
        (x + 0.033, y + 0.185, 2.8, 0.90),
    ]
    for sx, sy, lw, risk in sources:
        c = (risk, 0.25, 0.25)
        ax.add_patch(Circle((sx, sy), 0.013, facecolor=c, edgecolor="white", linewidth=0.7, zorder=5))
        arrow(ax, (sx + 0.014, sy), (center[0] - 0.027, center[1]), color=ROLE_COLORS["prop"], lw=lw, alpha=0.85, rad=0.08 * (sy - center[1]))

    ax.text(x + w / 2, y + 0.105, r"$G_t=\frac{\sum w_{it}P_i}{\sum w_{it}}$", ha="center", va="center", fontsize=6.5, color=ROLE_COLORS["prop"])
    ax.text(x + w / 2, y + 0.055, r"$H_t=\eta_gG_t+w_m m_t$", ha="center", va="center", fontsize=6.1, color=TEXT)
    ax.text(center[0] - 0.025, center[1] + 0.055, r"$m_t$", ha="center", va="center", fontsize=6.4, color=ROLE_COLORS["risk"])


def draw_m4(ax, x, y, w, h):
    panel_title(ax, x + w / 2, y + h - 0.025, "Gap")
    cx, cy = x + w / 2, y + 0.215
    ax.add_patch(Arc((cx, cy), 0.116, 0.116, theta1=0, theta2=180, edgecolor="#A0A0A0", linewidth=0.8, zorder=3))
    ax.plot([cx - 0.065, cx + 0.065], [cy, cy], color="#D0D0D0", lw=0.7, zorder=2)
    arrow(ax, (cx, cy), (cx + 0.052, cy), color=ROLE_COLORS["user"], lw=1.0)
    arrow(ax, (cx, cy), (cx + 0.022, cy + 0.050), color=ROLE_COLORS["risk"], lw=1.2)
    arrow(ax, (cx, cy), (cx - 0.038, cy + 0.030), color=ROLE_COLORS["env"], lw=1.0)
    ax.text(cx + 0.058, cy - 0.020, r"$x$", fontsize=6.4, color=ROLE_COLORS["user"])
    ax.text(cx + 0.020, cy + 0.061, r"$y_t$", fontsize=6.4, color=ROLE_COLORS["risk"])
    ax.text(cx - 0.055, cy + 0.040, r"$o_t$", fontsize=6.4, color=ROLE_COLORS["env"])
    ax.text(cx, y + 0.105, "Goal Drift", ha="center", va="center", fontsize=6.0, fontweight="bold", color=TEXT)
    ax.text(cx, y + 0.073, "(Semantic Gap)", ha="center", va="center", fontsize=5.1, color=MUTED)
    ax.text(cx, y + 0.030, r"$C_t=\eta_{goal}(1-D)+\eta_o(1-D)$", ha="center", va="center", fontsize=5.0, color=TEXT)


def draw_gauge(ax, cx, cy, r, value=0.72):
    ax.add_patch(Arc((cx, cy), 2 * r, 2 * r, theta1=0, theta2=180, edgecolor="#CFCFCF", linewidth=4.5, zorder=3))
    ax.add_patch(Arc((cx, cy), 2 * r, 2 * r, theta1=0, theta2=180 * value, edgecolor=ROLE_COLORS["risk"], linewidth=4.5, zorder=4))
    theta = math.radians(180 * (1 - value))
    tip = (cx + r * 0.74 * math.cos(theta), cy + r * 0.74 * math.sin(theta))
    ax.plot([cx, tip[0]], [cy, tip[1]], color=TEXT, lw=1.0, zorder=5)
    ax.add_patch(Circle((cx, cy), 0.006, facecolor=TEXT, edgecolor="none", zorder=6))


def draw_m5(ax, x, y, w, h):
    panel_title(ax, x + w / 2, y + h - 0.025, "Fusion")
    ax.text(x + w / 2, y + 0.315, r"$R_t=U_t(1+\alpha H_t+\beta C_t)$", ha="center", va="center", fontsize=5.7, color=ROLE_COLORS["risk"])
    draw_gauge(ax, x + w / 2, y + 0.205, 0.055)
    ax.text(x + 0.030, y + 0.145, "Low", ha="center", va="center", fontsize=5.6, color=MUTED)
    ax.text(x + w - 0.030, y + 0.145, "High", ha="center", va="center", fontsize=5.6, color=MUTED)
    ax.text(x + w / 2, y + 0.080, "Confidence Gauge", ha="center", va="center", fontsize=5.7, fontweight="bold", color=TEXT)
    ax.text(x + w / 2, y + 0.040, r"$RUPA(\tau)=\frac{1}{T}\sum_tR_t$", ha="center", va="center", fontsize=5.4, color=TEXT)


def draw_role_legend(ax, x, y):
    items = [("user", "User"), ("assistant", "Assistant"), ("env", "Environment")]
    for i, (key, label) in enumerate(items):
        xi = x + i * 0.095
        ax.add_patch(Circle((xi, y), 0.007, facecolor=ROLE_COLORS[key], edgecolor="none", zorder=4))
        ax.text(xi + 0.012, y, label, ha="left", va="center", fontsize=5.8, color=MUTED)


def main() -> None:
    configure()
    fig, ax = plt.subplots(figsize=(7.4, 2.45))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.5, 0.955, "RUPA: Relational Uncertainty Propagation for Agents", ha="center", va="top", fontsize=8.8, fontweight="bold", color=TEXT)
    ax.text(0.5, 0.905, "Trajectory $\\rightarrow$ Graph $\\rightarrow$ Propagation $\\rightarrow$ Gap $\\rightarrow$ Confidence", ha="center", va="top", fontsize=6.0, color=MUTED)

    x0, y0, w, h, gap = 0.025, 0.245, 0.17, 0.57, 0.018
    xs = [x0 + i * (w + gap) for i in range(5)]
    for i, (title, fc) in enumerate(PANELS):
        box(ax, xs[i], y0, w, h, fc, ec="#D5D5D5", lw=0.8, radius=0.016, z=1)

    draw_m1(ax, xs[0], y0, w, h)
    draw_m2(ax, xs[1], y0, w, h)
    draw_m3(ax, xs[2], y0, w, h)
    draw_m4(ax, xs[3], y0, w, h)
    draw_m5(ax, xs[4], y0, w, h)

    centers = [x + w / 2 for x in xs]
    draw_backbone(ax, centers, 0.155)
    draw_role_legend(ax, 0.035, 0.055)

    fig.savefig(OUT_BASE.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(OUT_BASE.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(OUT_BASE.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
