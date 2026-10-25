"""
Pilot Plots
===========

Accuracy-vs-cost chart and gate reliability chart.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Okabe-Ito palette (colorblind-safe)
COLORS = {
    "single": "#0072B2",
    "ensemble": "#009E73",
    "art": "#D55E00",
    "router": "#CC79A7",
    "jev": "#E69F00",
    "cascade_llm": "#56B4E9",
    "cascade_jev": "#000000",
}


def _group(method: str) -> str:
    """Color group for a method name."""
    if method.startswith("single:") or method.startswith("best_single"):
        return "single"
    if method.startswith("art_"):
        return "art"
    if method in ("cascade_llm", "cascade_jev"):
        return method
    if method == "jev_pick":
        return "jev"
    if method == "elo_router":
        return "router"
    return "ensemble"


def accuracy_vs_cost(
    summary: list[dict[str, Any]],
    sweeps: dict[str, list[dict[str, float]]],
    out_base: Path,
    title: str,
) -> None:
    """
    Scatter of accuracy (with 95% CI) against mean USD per question on a log axis, plus the
    cascade tau sweeps as lines. Writes out_base.png and out_base.svg.
    """
    fig, ax = plt.subplots(figsize=(10, 6.5))
    for name, rows in sweeps.items():
        points = sorted(
            (r["mean_cost_usd"], r["accuracy"]) for r in rows
            if r["mean_cost_usd"] > 0 and not math.isnan(r["accuracy"])
        )
        if points:
            ax.plot(
                [p[0] for p in points], [p[1] for p in points],
                color=COLORS[name], alpha=0.45, linewidth=1.5,
                label=f"{name} tau sweep (test split, not tuned)",
            )
    # Methods landing on the same point (e.g. best_single_* and the model they picked) share
    # one marker and one label
    points: dict[tuple[float, float], list[dict[str, Any]]] = {}
    for row in summary:
        cost = row["mean_usd_per_question"]
        if cost <= 0 or math.isnan(row["accuracy"]):
            continue
        points.setdefault((round(cost, 12), round(row["accuracy"], 12)), []).append(row)
    placed: list[tuple[float, float, int]] = []
    for (cost, accuracy), rows in sorted(points.items()):
        row = rows[0]
        color = COLORS[_group(row["method"])]
        yerr = [
            [max(0.0, accuracy - row["ci_low"])],
            [max(0.0, row["ci_high"] - accuracy)],
        ]
        marker = "s" if row["method"] in ("cascade_llm", "cascade_jev") else "o"
        ax.errorbar(
            cost, accuracy, yerr=yerr, fmt=marker, color=color,
            ecolor=color, elinewidth=1, capsize=3, markersize=6,
        )
        names = [
            r["method"] + (" (oracle)" if r["method"] == "best_single_oracle" else "")
            for r in rows
        ]
        # Stagger labels of nearby points (within ~0.3 decades and 2 points of accuracy)
        x = math.log10(cost)
        slot = sum(1 for px, py, _ in placed if abs(px - x) < 0.3 and abs(py - accuracy) < 0.02)
        placed.append((x, accuracy, slot))
        ax.annotate(" = ".join(names), (cost, accuracy), textcoords="offset points",
                    xytext=(6, 4 - 11 * slot), fontsize=8)
    ax.set_xscale("log")
    ax.set_xlabel("Mean USD per question (log scale)")
    ax.set_ylabel("Accuracy (95% bootstrap CI)")
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.25)
    if sweeps:
        ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_base.with_suffix(".png"), dpi=150)
    fig.savefig(out_base.with_suffix(".svg"))
    plt.close(fig)


def gate_reliability(calibration: dict[str, dict[str, Any]], out_path: Path, title: str) -> None:
    """Reliability diagram and risk-coverage curve for each gate."""
    fig, (left, right) = plt.subplots(1, 2, figsize=(12, 5))
    colors = {"llm_judge": COLORS["cascade_llm"], "jev": COLORS["cascade_jev"]}
    left.plot([0, 1], [0, 1], linestyle="--", color="#888888", linewidth=1, label="perfect")
    for gate, data in calibration.items():
        rows = [r for r in data["reliability"] if r["count"] > 0]
        left.plot(
            [r["mean_p"] for r in rows], [r["frac_correct"] for r in rows],
            marker="o", color=colors[gate],
            label=f"{gate} (Brier {data['brier']:.3f}, n={data['n']})",
        )
        for r in rows:
            left.annotate(str(r["count"]), (r["mean_p"], r["frac_correct"]),
                          textcoords="offset points", xytext=(4, -10), fontsize=7,
                          color=colors[gate])
        curve = data["risk_coverage"]
        right.plot([r["coverage"] for r in curve], [r["risk"] for r in curve],
                   color=colors[gate], label=gate)
    left.set_xlabel("Gate p (decile bin mean)")
    left.set_ylabel("Fraction of cheap answers correct")
    left.set_title("Reliability (labels = bin counts)")
    left.set_xlim(0, 1)
    left.set_ylim(0, 1)
    left.legend(fontsize=8)
    right.set_xlabel("Coverage (fraction of cheap answers accepted)")
    right.set_ylabel("Risk (error rate among accepted)")
    right.set_title("Risk-coverage")
    right.set_xlim(0, 1)
    right.set_ylim(0, 1)
    right.legend(fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
