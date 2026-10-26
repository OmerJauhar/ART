"""
Pilot Publish
=============

Copies the shareable part of a run's results to experiments/pilot/published/<run_name>/,
which is tracked: aggregate tables, plots, a sanitized config, per-question correctness and
cost, and the ELO router ratings.

Never published: question text, answer options, correct letters and model outputs, nor the
files holding them (questions.jsonl, calls.jsonl, per_question.jsonl, run.log). Questions are
identified only by their LAB-Bench ID. Before finishing, every published file is checked for
the run's question text, options and model outputs, and for the OpenRouter key.

    python -m experiments.pilot.publish --run-name pilot-v1
    python -m experiments.pilot.publish --run-name pilot-v1 --print-table  # Markdown summary
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import shutil
import sys
from pathlib import Path
from typing import Any

from experiments.pilot import data as pilot_data
from experiments.pilot.settings import PILOT_DIR, RESULTS_DIR, load_api_key

PUBLISHED_DIR = PILOT_DIR / "published"
REPO_ROOT = PILOT_DIR.parents[1]

COPIED = (
    "summary.csv",
    "accuracy_vs_cost.png",
    "accuracy_vs_cost.svg",
    "gate_reliability.png",
    "gate_calibration.csv",
    "gate_risk_coverage.csv",
    "tau_sweep.csv",
)

CITATION = "Laurent et al., 2024, arXiv:2407.10362"
DATA_LICENSE = "CC BY-SA 4.0"

# Shorter options (numbers, single words) also occur in ordinary numbers and labels, so the
# leak check only looks for options at least this long; question text is always checked
MIN_OPTION_CHARS = 12
# Model outputs are checked by their first characters
OUTPUT_PREFIX_CHARS = 60


def dataset_canary(subsets: list[str]) -> str:
    """The LAB-Bench canary string, read from the pinned dataset files (never retyped)."""
    import pandas as pd

    values: set[str] = set()
    for subset in subsets:
        frame = pd.read_parquet(pilot_data.download_subset(subset), columns=["canary"])
        values |= {str(v) for v in frame["canary"].unique()}
    if len(values) != 1:
        raise ValueError(f"Expected one canary string in {subsets}, found {len(values)}")
    return values.pop()


def sanitize_config(config: dict[str, Any]) -> dict[str, Any]:
    """
    config.json without local paths.

    The command line becomes a module invocation, paths inside the repository become relative
    and other absolute paths on the command line keep only their file name.

    Raises:
        ValueError: If an absolute path is left
    """
    clean = copy.deepcopy(config)
    root = f"{REPO_ROOT}/"
    argv = [token.replace(root, "") for token in clean.get("command", "").split(" ")]
    if argv[0].endswith("run.py"):
        argv[0] = "python -m experiments.pilot.run"
    # Any other absolute path (outside the repository) keeps only its file name
    clean["command"] = " ".join(Path(t).name if t.startswith("/") else t for t in argv)

    def walk(value: Any) -> Any:
        if isinstance(value, dict):
            return {k: walk(v) for k, v in value.items()}
        if isinstance(value, list):
            return [walk(v) for v in value]
        if isinstance(value, str):
            value = value.replace(root, "")
            if str(Path.home()) in value or any(t.startswith("/") for t in value.split()):
                raise ValueError(f"config.json still holds a local path: {value!r}")
        return value

    return walk(clean)


def per_question_table(rows: list[dict[str, Any]], methods: list[str]) -> list[dict[str, Any]]:
    """Correctness (0/1), cost and calls per method and question; the ELO router's choice."""
    table = []
    for row in rows:
        out: dict[str, Any] = {
            "question_id": row["id"], "subset": row["subset"], "split": row["split"],
        }
        for method in methods:
            outcome = row["methods"].get(method)
            out[f"{method}.correct"] = int(outcome["correct"]) if outcome else ""
            out[f"{method}.cost_usd"] = outcome["cost_usd"] if outcome else ""
            out[f"{method}.calls"] = outcome["calls"] if outcome else ""
            if method == "elo_router":
                out["elo_router.routed_to"] = outcome.get("routed_to", "") if outcome else ""
        table.append(out)
    return table


def elo_router_json(elo_router: dict[str, Any] | None) -> dict[str, Any]:
    """The frozen per-subset ratings and top model; the trajectory only if it is IDs and ELO."""
    if not elo_router:
        return {}
    out = {"ratings": elo_router["ratings"], "top_model": elo_router["top_model"]}
    trajectory = elo_router.get("trajectory") or {}
    plain = all(
        set(step) == {"question_id", "elo"}
        and isinstance(step["question_id"], str)
        and all(isinstance(v, (int, float)) for v in step["elo"].values())
        for steps in trajectory.values() for step in steps
    )
    if plain:
        out["trajectory"] = trajectory
    return out


def published_readme(readme: str, config: dict[str, Any], canary: str) -> str:
    """The results README with the published file list and a data and attribution section."""
    lines = [line for line in readme.rstrip("\n").split("\n") if not line.startswith("Files: ")]
    if config["status"]["live_calls"] == 0:
        at = next(i for i, line in enumerate(lines) if line.startswith("- Live spend this run"))
        lines.insert(at + 1, (
            "- These outputs were regenerated from the run's call cache (no new API calls); "
            "the cost of the calls used is what they cost when they were made"
        ))
    dataset = config["dataset"]
    lines += [
        "Files: README.md, config.json, summary.csv, per_question.csv (correct, cost_usd and "
        "calls per method and question; the ELO router's choice), elo_router.json, "
        "tau_sweep.csv, gate_calibration.csv, gate_risk_coverage.csv, "
        "accuracy_vs_cost.png/.svg, gate_reliability.png. Question text, answer options, "
        "correct answers and model outputs are not published; questions are identified by "
        "their LAB-Bench ID.",
        "",
        "## Data and attribution",
        "",
        f"Questions are from LAB-Bench ({CITATION}): the dataset `{dataset['name']}` on "
        f"HuggingFace at commit `{dataset['revision']}`, licensed {DATA_LICENSE}. Subsets used "
        f"in this run: {', '.join(config['selection']['subsets'] or dataset['subsets_used'])}.",
        "",
        "LAB-Bench asks that its data never appear in training corpora. Its canary string:",
        "",
        f"`{canary}`",
    ]
    return "\n".join(lines) + "\n"


def markdown_table(summary_path: Path) -> str:
    """Method, accuracy with 95% CI and USD per question, as a Markdown table."""
    lines = ["| method | accuracy [95% CI] | $ per question |", "|---|---|---|"]
    with summary_path.open() as f:
        for row in csv.DictReader(f):
            lines.append(
                f"| {row['method']} | {float(row['accuracy']):.3f} "
                f"[{float(row['ci_low']):.3f}, {float(row['ci_high']):.3f}] "
                f"| {float(row['mean_usd_per_question']):.4f} |"
            )
    return "\n".join(lines)


def forbidden_strings(run_dir: Path) -> list[str]:
    """Question text, longer options and model output prefixes of a run: none may be published."""
    strings: list[str] = []
    for line in (run_dir / "questions.jsonl").read_text().splitlines():
        question = json.loads(line)
        strings.append(question["stem"])
        strings += [o for o in question["options"] if len(o) >= MIN_OPTION_CHARS]
    for line in (run_dir / "per_question.jsonl").read_text().splitlines():
        for answer in json.loads(line)["answers"].values():
            content = (answer.get("content") or "").strip()
            if len(content) >= OUTPUT_PREFIX_CHARS:
                strings.append(content[:OUTPUT_PREFIX_CHARS])
    return [s for s in strings if s.strip()]


def check_published(out_dir: Path, forbidden: list[str], secret: str | None) -> None:
    """
    Raise if any published file contains a forbidden string or the API key.

    Raises:
        ValueError: Naming the file (never the matched string)
    """
    needles = [s.encode() for s in forbidden]
    for path in sorted(out_dir.iterdir()):
        content = path.read_bytes()
        if secret and secret.encode() in content:
            raise ValueError(f"{path.name} contains the API key")
        if any(needle in content for needle in needles):
            raise ValueError(f"{path.name} contains question text, an option or a model output")


def publish(
    run_name: str,
    results_dir: Path = RESULTS_DIR,
    published_dir: Path = PUBLISHED_DIR,
    canary: str | None = None,
) -> Path:
    """
    Write the published copy of a run.

    Args:
        run_name: Folder under results_dir
        results_dir: Where runs are written
        published_dir: Where the published copies go (one folder per run, replaced)
        canary: The LAB-Bench canary; read from the dataset files when not given

    Returns:
        The published folder
    """
    run_dir = results_dir / run_name
    out_dir = published_dir / run_name
    config = json.loads((run_dir / "config.json").read_text())
    if canary is None:
        canary = dataset_canary(config["selection"]["subsets"] or config["dataset"]["subsets_used"])

    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    for name in COPIED:
        shutil.copyfile(run_dir / name, out_dir / name)

    clean = sanitize_config(config)
    (out_dir / "config.json").write_text(json.dumps(clean, indent=2) + "\n")
    (out_dir / "elo_router.json").write_text(
        json.dumps(elo_router_json(config["elo_router"]), indent=2) + "\n"
    )
    with (run_dir / "summary.csv").open() as f:
        methods = [row["method"] for row in csv.DictReader(f)]
    rows = [json.loads(line) for line in (run_dir / "per_question.jsonl").read_text().splitlines()]
    table = per_question_table(rows, methods)
    with (out_dir / "per_question.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    readme = (run_dir / "README.md").read_text()
    (out_dir / "README.md").write_text(published_readme(readme, clean, canary))

    check_published(out_dir, forbidden_strings(run_dir), load_api_key())
    return out_dir


def main(argv: list[str] | None = None) -> int:
    """Entry point."""
    parser = argparse.ArgumentParser(description="Publish a run's shareable results")
    parser.add_argument("--run-name", required=True,
                        help="Folder under experiments/pilot/results/")
    parser.add_argument("--print-table", action="store_true",
                        help="Print the published summary as a Markdown table and exit")
    args = parser.parse_args(argv)
    if args.print_table:
        print(markdown_table(PUBLISHED_DIR / args.run_name / "summary.csv"))
        return 0
    out_dir = publish(args.run_name)
    print(f"Published {len(list(out_dir.iterdir()))} files to {out_dir.relative_to(REPO_ROOT)}:")
    for path in sorted(out_dir.iterdir()):
        print(f"  {path.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
