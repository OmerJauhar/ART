"""
Pilot Plots
===========

Accuracy-vs-cost chart and gate calibration chart.

Style: minimal and solid. No titles (captions belong in the text), no gridlines, no top or
right spines, black text and axes, fully opaque marks, and a colorblind-checked categorical
palette. Every label is placed so it overlaps no other label or marker.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import patheffects  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import FuncFormatter, PercentFormatter  # noqa: E402

# Ink and categorical colors (adjacent pairs checked for color-vision deficiency)
INK = "#000000"
SURFACE = "#ffffff"

GROUPS = {
    "single": ("Single model", "#000000"),
    "ensemble": ("Ensemble baseline", "#2a78d6"),
    "art": ("ART", "#eb6834"),
    "cascade": ("Cascade", "#1baf7a"),
}

# Readable names for the methods in the pilot; anything else falls back to its id
DISPLAY = {
    "majority_vote": "Majority vote",
    "fusion": "Fusion",
    "judge": "LLM judge",
    "jev_pick": "Jev pick",
    "art_r1": "ART, one round",
    "art_r2": "ART, two rounds",
    "elo_router": "ART ratings as a router",
    "cascade_llm": "Cascade (LLM gate)",
    "cascade_jev": "Cascade (Jev gate)",
}
MODEL_NAMES = {
    "deepseek-v4-pro": "DeepSeek V4 Pro",
    "gemini-3.8-flash": "Gemini 3.8 Flash",
    "claude-sonnet-5.5": "Claude Sonnet 5.5",
    "gpt-6.1-sol": "GPT-6.1",
}

STYLE = {
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.edgecolor": INK,
    "axes.linewidth": 0.8,
    "axes.labelcolor": INK,
    "axes.titlecolor": INK,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": False,
    "xtick.color": INK,
    "ytick.color": INK,
    "xtick.labelcolor": INK,
    "ytick.labelcolor": INK,
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "xtick.minor.size": 0,
    "ytick.minor.size": 0,
    "legend.frameon": False,
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "svg.fonttype": "none",
}


def _group(method: str) -> str:
    """Color group for a method name."""
    if method.startswith("single:") or method.startswith("best_single"):
        return "single"
    if method.startswith("art_") or method == "elo_router":
        return "art"
    if method.startswith("cascade"):
        return "cascade"
    return "ensemble"


def _display(method: str) -> str:
    """Readable label for a method id."""
    if method.startswith("single:"):
        model = method.split(":", 1)[1]
        return MODEL_NAMES.get(model, model)
    return DISPLAY.get(method, method)


def _merge(names: list[str]) -> str:
    """One label for methods that share a point: 'Cascade (A)' + 'Cascade (B)' -> 'Cascades (A, B)'."""
    if len(names) > 1 and all(n.startswith("Cascade (") and n.endswith(")") for n in names):
        return "Cascades (" + ", ".join(n[len("Cascade ("):-1] for n in sorted(names)) + ")"
    return " · ".join(names)


def _dollars(value: float, _pos: Any = None) -> str:
    """Tick label for a dollar amount on a log axis."""
    if value >= 0.01:
        return f"${value:.2f}"
    return f"${value:.3f}"


def _pareto(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Points no other point beats on both cost (lower) and accuracy (higher), cheapest first."""
    frontier: list[tuple[float, float]] = []
    best = -1.0
    for cost, acc in sorted(points, key=lambda p: (p[0], -p[1])):
        if acc > best + 1e-12:
            frontier.append((cost, acc))
            best = acc
    return frontier


def _place_labels(
    ax: Any,
    fig: Any,
    items: list[dict[str, Any]],
    marker_pt: float,
    obstacles: list[tuple[tuple[float, float, float, float], float]],
    marker_points: list[tuple[float, float]],
) -> None:
    """
    Place each label beside its point. Candidates are tried in a fixed order and the first
    that overlaps no other label, no marker and no line (whiskers, frontier) wins; otherwise
    the least-overlapping one, weighting labels and markers above lines. Labels never leave
    the axes. Boxes are in display pixels; offsets are written in points so PNG and SVG agree.
    """
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    ax_box = ax.get_window_extent(renderer)
    to_px = ax.transData.transform
    k = fig.dpi / 72.0
    r = marker_pt * k
    marker_boxes = [(mx - r, my - r, mx + r, my + r)
                    for mx, my in (to_px(p) for p in marker_points)]
    placed: list[tuple[float, float, float, float]] = []
    pad = 2.0 * k
    for it in items:
        px, py = to_px((it["x"], it["y"]))
        span = to_px((it.get("x_max", it["x"]), it["y"]))[0] - px  # merged points side by side
        text = ax.text(0, 0, it["label"], fontsize=9, transform=None)
        bb = text.get_window_extent(renderer)
        w, h = bb.width, bb.height
        text.remove()
        g = 6.5 * k  # clearance from the marker center, in pixels
        lift = 2.5 * k
        candidates = [
            (span + g, -h / 2),        # right
            (span + g, lift),          # just above the right-hand run
            (span + g, -h - lift),     # just below it
            (-g - w, -h / 2),          # left
            (-g - w, lift),            # upper left
            (-g - w, -h - lift),       # lower left
            (-w / 2, g + 1 * k),       # above
            (-w / 2, -g - h - 1 * k),  # below
        ]
        best = None
        for rank, (dx, dy) in enumerate(candidates):
            box = (px + dx - pad, py + dy - pad, px + dx + w + pad, py + dy + h + pad)
            if not (box[0] >= ax_box.x0 and box[2] <= ax_box.x1
                    and box[1] >= ax_box.y0 and box[3] <= ax_box.y1):
                continue
            score = (sum(_overlap(box, b) for b in placed + marker_boxes)
                     + sum(weight * _overlap(box, b) for b, weight in obstacles)
                     + 4.0 * rank)  # prefer earlier candidates when overlaps tie
            if best is None or score < best[0]:
                best = (score, box, dx, dy)
        if best is None:  # nothing fits inside the axes: fall back to the right side
            best = (0, (px, py, px + w, py + h), 9.0 * k, -h / 2)
        _, box, dx, dy = best
        placed.append(box)
        ax.annotate(it["label"], (it["x"], it["y"]), textcoords="offset points",
                    xytext=(dx / k, dy / k), ha="left", va="bottom", fontsize=9, color=INK,
                    zorder=6, path_effects=[patheffects.withStroke(linewidth=3.5,
                                                                   foreground=SURFACE)])


def _overlap(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    """Overlap area of two (x0, y0, x1, y1) boxes."""
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    return w * h if w > 0 and h > 0 else 0.0


def accuracy_vs_cost(
    summary: list[dict[str, Any]],
    sweeps: dict[str, list[dict[str, float]]],
    out_base: Path,
    title: str,
    split: str = "test",
) -> None:
    """
    Accuracy (with 95% CI whiskers) against mean USD per question on a log axis, colored by
    method type, with the Pareto frontier drawn through the methods nothing else beats.
    Writes out_base.png and out_base.svg. The cascade tau sweeps stay in tau_sweep.csv; they
    are not drawn here.
    """
    del sweeps  # kept for the call signature; see the docstring
    with plt.rc_context(STYLE):
        rows = [
            r for r in summary
            if r["mean_usd_per_question"] > 0 and not math.isnan(r["accuracy"])
            and r["method"] != "best_single_oracle"
        ]
        calib_row = next((r for r in rows if r["method"] == "best_single_calibration"), None)
        rows = [r for r in rows if r["method"] != "best_single_calibration"]

        fig, ax = plt.subplots(figsize=(9.6, 6.0))
        fig.subplots_adjust(left=0.09, right=0.97, top=0.97, bottom=0.11)

        # Pareto frontier first, so the markers sit on top of it
        frontier = _pareto([(r["mean_usd_per_question"], r["accuracy"]) for r in rows])
        costs = [r["mean_usd_per_question"] for r in rows]
        x_lo, x_hi = min(costs) / 1.6, max(costs) * 1.9
        if len(frontier) > 1:
            # Best accuracy reachable at each cost: a staircase that holds its level to the right
            xs = [p[0] for p in frontier] + [x_hi]
            ys = [p[1] for p in frontier] + [frontier[-1][1]]
            ax.step(xs, ys, where="post", color=INK, linewidth=1.2,
                    solid_joinstyle="round", solid_capstyle="round", zorder=3)

        # Group methods that land on (almost) the same point into one label
        items: list[dict[str, Any]] = []
        for r in sorted(rows, key=lambda r: (r["mean_usd_per_question"], r["accuracy"])):
            x, y = r["mean_usd_per_question"], r["accuracy"]
            group = _group(r["method"])
            color = GROUPS[group][1]
            ax.errorbar([x], [y], yerr=[[y - r["ci_low"]], [r["ci_high"] - y]], fmt="none",
                        ecolor=color, elinewidth=1.0, capsize=3, capthick=1.0, zorder=2)
            ax.plot(x, y, "o", markersize=8, color=color, markeredgecolor=SURFACE,
                    markeredgewidth=1.5, zorder=4)
            label = _display(r["method"])
            if calib_row is not None and r["method"].startswith("single:") and \
                    math.isclose(x, calib_row["mean_usd_per_question"]) and \
                    math.isclose(y, calib_row["accuracy"]):
                label += " (best on calibration)"
            near = next((it for it in items
                         if abs(math.log10(it["x"]) - math.log10(x)) < 0.03
                         and abs(it["y"] - y) < 0.005), None)
            if near is not None:
                near["names"].append(label)
                near["label"] = _merge(near["names"])
                near["x_max"] = max(near.get("x_max", near["x"]), x)
            else:
                items.append({"x": x, "y": y, "label": label, "names": [label]})

        ax.set_xscale("log")
        ax.set_xlim(x_lo, x_hi)
        ticks = [t for t in (0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0)
                 if x_lo <= t <= x_hi]
        ax.set_xticks(ticks)
        ax.xaxis.set_major_formatter(FuncFormatter(_dollars))
        ax.xaxis.set_minor_formatter(FuncFormatter(lambda *_: ""))
        lows = [r["ci_low"] for r in rows]
        ax.set_ylim(max(0.0, math.floor(min(lows) * 20) / 20 - 0.02), 1.02)
        ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
        ax.set_xlabel("Mean cost per question (USD, log scale)")
        ax.set_ylabel("Accuracy")

        fig.canvas.draw()
        to_px = ax.transData.transform
        k = fig.dpi / 72.0
        obstacles = []
        for r in rows:
            (x0, y0), (_, y1) = (to_px((r["mean_usd_per_question"], r["ci_low"])),
                                 to_px((r["mean_usd_per_question"], r["ci_high"])))
            obstacles.append(((x0 - 1.5 * k, y0, x0 + 1.5 * k, y1), 0.15))
        if len(frontier) > 1:
            path = list(zip(xs, ys))
            for (ax0, ay0), (ax1, ay1) in zip(path, path[1:]):
                (p0x, p0y), (p1x, _) = to_px((ax0, ay0)), to_px((ax1, ay0))
                obstacles.append(((p0x, p0y - 1.5 * k, p1x, p0y + 1.5 * k), 0.5))  # run
                (q0x, q0y), (_, q1y) = to_px((ax1, ay0)), to_px((ax1, ay1))
                obstacles.append(((q0x - 1.5 * k, min(q0y, q1y), q0x + 1.5 * k,
                                   max(q0y, q1y)), 0.5))  # riser
        _place_labels(ax, fig, items, marker_pt=5, obstacles=obstacles,
                      marker_points=[(r["mean_usd_per_question"], r["accuracy"]) for r in rows])

        # No title: the caption lives in the text that shows the figure
        del title, split
        present = []
        for key, (name, color) in GROUPS.items():
            if any(_group(r["method"]) == key for r in rows):
                present.append(Line2D([], [], marker="o", linestyle="", markersize=8,
                                      color=color, markeredgecolor=SURFACE, label=name))
        if len(frontier) > 1:
            present.append(Line2D([], [], color=INK, linewidth=1.2,
                                  solid_capstyle="round", label="Best accuracy at each cost"))
        ax.legend(handles=present, loc="lower right", fontsize=9, labelcolor=INK,
                  handlelength=1.6, borderaxespad=0.2)

        fig.savefig(out_base.with_suffix(".png"), dpi=200)
        fig.savefig(out_base.with_suffix(".svg"))
        plt.close(fig)


def gate_reliability(
    calibration: dict[str, dict[str, Any]], out_path: Path, title: str, split: str = "test"
) -> None:
    """Reliability diagram and risk-coverage curve for each gate."""
    names = {"llm_judge": "LLM judge", "jev": "Jev"}
    colors = {"llm_judge": "#2a78d6", "jev": "#eb6834"}
    with plt.rc_context(STYLE):
        fig, (left, right) = plt.subplots(1, 2, figsize=(11.5, 4.8))
        fig.subplots_adjust(left=0.07, right=0.98, top=0.96, bottom=0.13, wspace=0.26)

        left.plot([0, 1], [0, 1], color=INK, linewidth=0.8, zorder=1)
        handles = []
        for gate, data in calibration.items():
            rows = [r for r in data["reliability"] if r["count"] > 0]
            color = colors.get(gate, INK)
            xs = [r["mean_p"] for r in rows]
            ys = [r["frac_correct"] for r in rows]
            left.plot(xs, ys, color=color, linewidth=1.2, zorder=2)
            left.scatter(xs, ys, s=[22 + 14 * r["count"] for r in rows], color=color,
                         edgecolors=SURFACE, linewidths=1.5, zorder=3)
            handles.append(Line2D([], [], marker="o", linestyle="", markersize=8, color=color,
                                  markeredgecolor=SURFACE,
                                  label=f"{names.get(gate, gate)} (Brier {data['brier']:.3f})"))
            curve = data["risk_coverage"]
            right.plot([r["coverage"] for r in curve], [r["risk"] for r in curve],
                       color=color, linewidth=2, solid_capstyle="round", solid_joinstyle="round",
                       label=names.get(gate, gate))
        handles.append(Line2D([], [], color=INK, linewidth=0.8, label="Perfectly calibrated"))

        left.set_xlim(0, 1.02)
        left.set_ylim(-0.04, 1.04)
        for axis in (left.xaxis, left.yaxis, right.xaxis, right.yaxis):
            axis.set_major_formatter(PercentFormatter(1.0, decimals=0))
        left.set_xlabel("Gate's predicted chance the cheap answer is right")
        left.set_ylabel("Cheap answers actually right")
        left.legend(handles=handles, loc="lower right", fontsize=9, labelcolor=INK,
                    handlelength=1.6)

        top = max([r["risk"] for d in calibration.values() for r in d["risk_coverage"]] + [0.1])
        right.set_xlim(0, 1.02)
        right.set_ylim(0, math.ceil(top * 10) / 10 + 0.05)
        right.set_xlabel("Cheap answers accepted (coverage)")
        right.set_ylabel("Error rate among accepted answers")
        right.legend(loc="upper left", fontsize=9, labelcolor=INK, handlelength=1.6)

        del title, split  # no title: the caption lives in the text that shows the figure
        fig.savefig(out_path, dpi=200)
        plt.close(fig)
