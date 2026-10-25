"""
Pilot Runner
============

Command-line entry point for the real-model pilot.

    python -m experiments.pilot.run --dry-run                 # fake provider, 100 questions
    python -m experiments.pilot.run --smoke --run-name smoke  # 5 real questions + projection
    python -m experiments.pilot.run --run-name pilot-v1       # full real run

See experiments/pilot/README.md.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import math
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from loguru import logger

from experiments.pilot import data as pilot_data
from experiments.pilot.analysis import NEVER_ACCEPT_TAU, Analysis
from experiments.pilot.budget import BudgetExceeded, BudgetGuard
from experiments.pilot.cache import CallCache
from experiments.pilot.data import Question
from experiments.pilot.fake_provider import FakeTransport
from experiments.pilot.gateway import QUESTION_ID, Gateway, HttpTransport
from experiments.pilot.jev import JevClient
from experiments.pilot.methods import QuestionRunner
from experiments.pilot.plots import accuracy_vs_cost, gate_reliability
from experiments.pilot.prompts import format_question
from experiments.pilot.settings import (
    PILOT_DIR,
    RESULTS_DIR,
    PilotModels,
    load_api_key,
    load_models,
)

SAMPLE_IDS_PATH = PILOT_DIR / "sample_ids.json"
CACHE_DIR = PILOT_DIR / "cache"
SMOKE_SIZE = 5


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Command-line arguments."""
    parser = argparse.ArgumentParser(description="ART real-model pilot")
    parser.add_argument("--run-name", help="Output folder under experiments/pilot/results/")
    parser.add_argument("--dry-run", action="store_true",
                        help="Use the fake provider for LLM and Jev calls (no OpenRouter traffic)")
    parser.add_argument("--smoke", action="store_true",
                        help=f"Run {SMOKE_SIZE} questions and print the projected full-run cost")
    parser.add_argument("--limit", type=int, help="Run only the first N sampled questions")
    parser.add_argument("--max-usd", type=float, default=40.0,
                        help="Hard cap on live spend recorded in the cache file (default 40)")
    parser.add_argument("--concurrency", type=int, default=8, help="Max concurrent API calls")
    parser.add_argument("--question-concurrency", type=int, default=4,
                        help="Questions processed at once")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cache", type=Path, help="Cache file (default depends on --dry-run)")
    parser.add_argument("--prepare-data", action="store_true",
                        help="Download LAB-Bench, draw the sample, write sample_ids.json, exit")
    return parser.parse_args(argv)


def git_commit() -> str:
    """Current git commit, with -dirty if the tree has uncommitted changes."""
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True,
            cwd=PILOT_DIR,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"], capture_output=True,
            text=True, check=True, cwd=PILOT_DIR,
        ).stdout.strip()
        return commit + ("-dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def load_sample(seed: int) -> tuple[list[Question], dict[str, int]]:
    """Draw the sample and check it against the committed sample_ids.json."""
    rows = pilot_data.load_rows()
    usable = pilot_data.usable_counts(rows)
    if sum(usable.values()) < pilot_data.SAMPLE_SIZE:
        raise SystemExit(
            f"Only {sum(usable.values())} usable text-only LAB-Bench questions; the plan's "
            "MMLU-Pro fallback is not implemented."
        )
    questions = pilot_data.sample_questions(rows, seed=seed)
    if SAMPLE_IDS_PATH.exists():
        expected = json.loads(SAMPLE_IDS_PATH.read_text())
        if seed == expected["seed"] and [q.id for q in questions] != expected["ids"]:
            raise SystemExit(
                "The drawn sample does not match sample_ids.json; the dataset or sampling "
                "code changed. Rerun --prepare-data deliberately if this is intended."
            )
    return questions, usable


def prepare_data(seed: int) -> None:
    """Write sample_ids.json and print the sample's composition."""
    rows = pilot_data.load_rows()
    questions = pilot_data.sample_questions(rows, seed=seed)
    payload = {
        "dataset": pilot_data.DATASET,
        "revision": pilot_data.REVISION,
        "seed": seed,
        "calibration_size": pilot_data.CALIBRATION_SIZE,
        "usable": pilot_data.usable_counts(rows),
        "ids": [q.id for q in questions],
        "subsets": [q.subset for q in questions],
    }
    SAMPLE_IDS_PATH.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"Wrote {SAMPLE_IDS_PATH}")
    print("Usable text-only questions:", payload["usable"])
    for split in ("calibration", "test"):
        counts: dict[str, int] = {}
        for q in questions:
            if q.split == split:
                counts[q.subset] = counts.get(q.subset, 0) + 1
        print(f"{split}: {counts}")


async def fit_elo_router(
    runner: QuestionRunner,
    calibration: list[Question],
    models: PilotModels,
) -> dict[str, Any]:
    """
    Run art_r1 over the calibration split without resetting ELO, separately per subset, and
    freeze the ratings. All calls are normally cache hits from the per-question pass.
    """
    ratings: dict[str, dict[str, float]] = {}
    trajectory: dict[str, list[dict[str, Any]]] = {}
    for subset in sorted({q.subset for q in calibration}):
        agents = runner.make_agents()
        trajectory[subset] = []
        for question in [q for q in calibration if q.subset == subset]:
            async def one(q: Question = question) -> None:
                QUESTION_ID.set(q.id)
                try:
                    await runner.run_tournament(q, rounds=1, agents=agents)
                except Exception as e:  # an unscored final round; ratings stay as they are
                    logger.warning(f"ELO router tournament failed on {q.id}: {e}")
            await asyncio.create_task(one())
            trajectory[subset].append({
                "question_id": question.id,
                "elo": {m: a.elo_score for m, a in agents.items()},
            })
        ratings[subset] = {m: a.elo_score for m, a in agents.items()}
    top_model = {
        subset: max(models.ids, key=lambda m: (r[m], -models.ids.index(m)))
        for subset, r in ratings.items()
    }
    return {"ratings": ratings, "top_model": top_model, "trajectory": trajectory}


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write rows as JSON lines."""
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, default=_json_default) + "\n")


def _json_default(value: Any) -> Any:
    """JSON fallback for sets and non-finite floats."""
    if isinstance(value, set):
        return sorted(value)
    raise TypeError(f"Not JSON serializable: {type(value).__name__}")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write dict rows as CSV."""
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def projected_cost(analysis: Analysis, done: list[Question], everything: list[Question]) -> float:
    """
    Full-run cost projected from the questions run so far.

    Every call carries the question text, so cost is scaled by question length: total cost
    so far per character of question text, times the characters in the full sample.
    """
    raw_by_id = {r["id"]: r for r in analysis.raw}
    spent = sum(analysis.unique_cost(raw_by_id[q.id]) for q in done if q.id in raw_by_id)
    done_chars = sum(len(format_question(q)) for q in done if q.id in raw_by_id)
    all_chars = sum(len(format_question(q)) for q in everything)
    return spent / done_chars * all_chars if done_chars else float("nan")


def per_question_rows(analysis: Analysis, questions: list[Question]) -> list[dict[str, Any]]:
    """Merge raw method details with outcomes for per_question.jsonl."""
    rows = []
    by_id = {q.id: q for q in questions}
    for record in analysis.raw:
        outcomes = analysis.outcomes[record["id"]]
        methods = {}
        for name, outcome in outcomes.items():
            details = record["methods"].get(name, {})
            extra = {k: v for k, v in details.items() if k not in ("answer", "groups")}
            methods[name] = {**extra, **outcome}
        rows.append({
            "id": record["id"],
            "index": record["index"],
            "subset": record["subset"],
            "subtask": record["subtask"],
            "split": record["split"],
            "correct_letter": record["correct_letter"],
            "options": by_id[record["id"]].options,
            "answers": record["answers"],
            "methods": methods,
        })
    return rows


def results_readme(
    run_name: str,
    config: dict[str, Any],
    summary: list[dict[str, Any]],
    calibration: dict[str, dict[str, Any]],
) -> str:
    """Short results README."""
    lines = [f"# Pilot results: {run_name}", ""]
    if config["dry_run"]:
        lines += [
            "**DRY RUN.** Every model and Jev response here is fake, from the deterministic fake "
            "provider. Accuracy numbers are meaningless; costs use real list prices with "
            "estimated prompt sizes and assumed completion lengths.",
            "",
        ]
    status = config["status"]
    lines += [
        f"- Date: {config['date']}",
        f"- Git commit: `{config['git_commit']}`",
        f"- Status: {status['state']} ({status['completed_questions']} of "
        f"{status['requested_questions']} questions completed)",
        f"- Evaluation split: {config['evaluation_split']}",
        f"- Live spend this run: ${status['live_spend_usd']:.4f} over {status['live_calls']} calls",
        f"- Cost of all calls used by this run (including cache hits): "
        f"${status['cost_of_calls_used_usd']:.4f}",
        f"- Projected cost of the full {pilot_data.SAMPLE_SIZE}-question run: "
        f"${status['projected_full_run_usd']:.2f}",
        "",
        "## Summary",
        "",
        "| method | n | accuracy [95% CI] | USD/q | calls/q | latency ms | unparseable | coverage "
        "| notes |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for row in summary:
        coverage = f"{row['coverage']:.2f}" if row["coverage"] != "" else ""
        lines.append(
            f"| {row['method']} | {row['n']} | {row['accuracy']:.3f} "
            f"[{row['ci_low']:.3f}, {row['ci_high']:.3f}] | {row['mean_usd_per_question']:.4f} "
            f"| {row['mean_calls_per_question']:.1f} | {row['mean_latency_ms']:.0f} "
            f"| {row['unparseable']} | {coverage} | {row['notes']} |"
        )
    lines += ["", "## Cascade thresholds (fitted on the calibration split)", ""]
    for name, tau in config["taus"].items():
        note = " (never accept: always escalate)" if tau == NEVER_ACCEPT_TAU else ""
        lines.append(f"- {name}: tau = {tau:.4f}{note}")
    lines += ["", "## Gate calibration (evaluation split)", ""]
    for gate, data in calibration.items():
        lines.append(f"- {gate}: Brier = {data['brier']:.4f} (n = {data['n']})")
    lines += [
        "",
        "Latency is idealized: calls in the same step are assumed to run in parallel, using "
        "each call's recorded latency (cache hits keep the original latency).",
        "",
        "Files: config.json, questions.jsonl, calls.jsonl, per_question.jsonl, summary.csv, "
        "tau_sweep.csv, gate_calibration.csv, gate_risk_coverage.csv, accuracy_vs_cost.png/.svg, "
        "gate_reliability.png, run.log.",
    ]
    return "\n".join(lines) + "\n"


async def run(args: argparse.Namespace) -> int:
    """Run the pilot; return a process exit code."""
    models = load_models()
    all_questions, usable = load_sample(args.seed)
    questions = all_questions
    if args.smoke:
        questions = questions[:SMOKE_SIZE]
    if args.limit is not None:
        questions = questions[: args.limit]

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    run_name = args.run_name or f"{'dryrun' if args.dry_run else 'run'}-{stamp}"
    out_dir = RESULTS_DIR / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.remove()
    logger.add(out_dir / "run.log", level="INFO")

    default_cache = "calls_dryrun.sqlite" if args.dry_run else "calls.sqlite"
    cache_path = args.cache or CACHE_DIR / default_cache
    cache = CallCache(cache_path)
    budget = BudgetGuard(args.max_usd, already_spent=cache.total_cost())
    if args.dry_run:
        transport: Any = FakeTransport(all_questions, models)
    else:
        api_key = load_api_key()
        if not api_key:
            print("OPENROUTER_API_KEY is not set (environment or experiments/pilot/.env).",
                  file=sys.stderr)
            return 2
        transport = HttpTransport(api_key)
    gateway = Gateway(transport, cache, budget, models, concurrency=args.concurrency)
    jev = JevClient(gateway, models.jev.model, models.jev.fallback_model)
    runner = QuestionRunner(gateway, models, jev, seed=args.seed)

    mode = "DRY RUN (fake provider)" if args.dry_run else "LIVE (OpenRouter)"
    print(f"{mode}: {len(questions)} questions -> {out_dir}")
    print(f"Budget ${args.max_usd:.2f}; already recorded in {cache_path.name}: "
          f"${budget.spent:.4f}")

    semaphore = asyncio.Semaphore(args.question_concurrency)
    finished = 0

    async def process(question: Question) -> dict[str, Any] | None:
        nonlocal finished
        async with semaphore:
            if budget.exhausted:
                return None
            try:
                record = await runner.run(question)
            except BudgetExceeded:
                return None
            if budget.exhausted:
                # A call for this question may have been refused; drop it as incomplete
                return None
            finished += 1
            print(f"  [{finished}/{len(questions)}] {question.subset:<16} {question.split:<11} "
                  f"live calls {gateway.live_calls:>5}  spend ${gateway.live_spend:.4f}")
            return record

    try:
        results = await asyncio.gather(*[process(q) for q in questions])
    finally:
        if isinstance(transport, HttpTransport):
            await transport.close()
    raw = [r for r in results if r is not None]
    done = [q for q, r in zip(questions, results) if r is not None]
    stopped = budget.exhausted
    if stopped:
        print(f"Budget guard stopped the run after {len(raw)} complete questions.")

    calibration_done = [q for q in done if q.split == "calibration"]
    elo_router = None
    if calibration_done and not budget.exhausted:
        try:
            elo_router = await fit_elo_router(runner, calibration_done, models)
        except BudgetExceeded:
            print("Budget guard stopped the ELO router fit.")

    analysis = Analysis(raw, gateway.records, models, elo_router, seed=args.seed)
    summary = analysis.summary()
    sweeps = analysis.sweeps()
    calibration = analysis.gate_calibration()
    projection = projected_cost(analysis, done, all_questions)
    used_cost = sum(analysis.unique_cost(r) for r in raw)

    versions: dict[str, set[str]] = {}
    for record in gateway.records.values():
        versions.setdefault(record.model, set()).add(record.model_version)
    jev_versions = sorted({
        v for m, vs in versions.items() if m.startswith("typesafe/") for v in vs
    })
    split_counts: dict[str, dict[str, int]] = {}
    for q in all_questions:
        split_counts.setdefault(q.split, {}).setdefault(q.subset, 0)
        split_counts[q.split][q.subset] += 1

    config = {
        "run_name": run_name,
        "date": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": git_commit(),
        "command": " ".join(sys.argv),
        "dry_run": args.dry_run,
        "dataset": {
            "name": pilot_data.DATASET,
            "revision": pilot_data.REVISION,
            "subsets_used": list(pilot_data.USED_SUBSETS),
            "subsets_excluded": pilot_data.EXCLUDED_SUBSETS,
            "seqqa_subtasks_excluded": list(pilot_data.EXCLUDED_SEQQA_SUBTASKS),
            "usable_text_only": usable,
            "sample_size": pilot_data.SAMPLE_SIZE,
            "split_counts": split_counts,
        },
        "seed": args.seed,
        "split": {"calibration": "first 30 sampled", "test": "last 70 sampled"},
        "evaluation_split": analysis.eval_split,
        "models": [
            {
                "id": m.id, "label": m.label, "family": m.family,
                "prompt_usd_per_mtok": m.prompt_usd_per_mtok,
                "completion_usd_per_mtok": m.completion_usd_per_mtok,
                "supports_temperature": m.supports_temperature,
                "reasoning": m.reasoning,
                "versions_returned": sorted(versions.get(m.id, set())),
            }
            for m in models.models
        ],
        "prices_fetched": models.fetched,
        "roles": models.roles,
        "temperature": models.temperature,
        "max_tokens": models.max_tokens,
        "jev": {
            "requested_model": models.jev.model,
            "fallback_model": models.jev.fallback_model,
            "model_used": jev.model,
            "versions_returned": jev_versions,
            "input_usd_per_mtok": models.jev.input_usd_per_mtok,
        },
        "taus": analysis.taus,
        "elo_router": elo_router,
        "best_single_calibration": analysis.best_single_calibration,
        "best_single_oracle": analysis.best_single_oracle,
        "budget": {"max_usd": args.max_usd, "cache_file": str(cache_path.name)},
        "status": {
            "state": "stopped by budget guard" if stopped else "complete",
            "requested_questions": len(questions),
            "completed_questions": len(raw),
            "live_calls": gateway.live_calls,
            "live_spend_usd": gateway.live_spend,
            "cost_of_calls_used_usd": used_cost,
            "projected_full_run_usd": projection,
            "total_recorded_in_cache_usd": cache.total_cost(),
        },
    }

    (out_dir / "config.json").write_text(json.dumps(config, indent=2, default=_json_default))
    write_jsonl(out_dir / "questions.jsonl", [q.to_dict() for q in done])
    write_jsonl(out_dir / "calls.jsonl", [r.to_log() for r in gateway.records.values()])
    write_jsonl(out_dir / "per_question.jsonl", per_question_rows(analysis, done))
    write_csv(out_dir / "summary.csv", summary)
    write_csv(out_dir / "tau_sweep.csv", [
        {"cascade": name, "split": analysis.eval_split, "kind": "sweep (not tuned)", **row}
        for name, rows in sweeps.items() for row in rows
    ])
    write_csv(out_dir / "gate_calibration.csv", [
        {"gate": gate, **row, "brier": data["brier"]}
        for gate, data in calibration.items() for row in data["reliability"]
    ])
    write_csv(out_dir / "gate_risk_coverage.csv", [
        {"gate": gate, **row}
        for gate, data in calibration.items() for row in data["risk_coverage"]
    ])
    title_prefix = "[DRY RUN, fake responses] " if args.dry_run else ""
    accuracy_vs_cost(summary, sweeps, out_dir / "accuracy_vs_cost",
                     f"{title_prefix}Accuracy vs cost ({analysis.eval_split} split)")
    gate_reliability(calibration, out_dir / "gate_reliability.png",
                     f"{title_prefix}Gate calibration ({analysis.eval_split} split)")
    (out_dir / "README.md").write_text(results_readme(run_name, config, summary, calibration))
    cache.close()

    print()
    print(f"{'method':<28}{'n':>4}{'acc':>7}{'USD/q':>10}{'calls/q':>9}{'unparse':>9}")
    for row in summary:
        print(f"{row['method']:<28}{row['n']:>4}{row['accuracy']:>7.3f}"
              f"{row['mean_usd_per_question']:>10.4f}{row['mean_calls_per_question']:>9.1f}"
              f"{row['unparseable']:>9}")
    print()
    print(f"Taus (calibration): " + ", ".join(f"{k}={v:.3f}" for k, v in analysis.taus.items()))
    print(f"Live spend this run: ${gateway.live_spend:.4f} ({gateway.live_calls} calls)")
    if not math.isnan(projection):
        print(f"Projected cost of the full {pilot_data.SAMPLE_SIZE}-question run: "
              f"${projection:.2f}")
    print(f"Outputs: {out_dir}")
    return 1 if stopped else 0


def main(argv: list[str] | None = None) -> int:
    """Entry point."""
    args = parse_args(argv)
    if args.prepare_data:
        prepare_data(args.seed)
        return 0
    return asyncio.run(run(args))


def ensure_hash_seed() -> None:
    """
    Re-execute with PYTHONHASHSEED=0 if needed.

    The tournament engine iterates sets of agent IDs, and ranking ties fall back to that
    order; a fixed hash seed makes it the same in every process.
    """
    if os.environ.get("PYTHONHASHSEED") != "0":
        env = {**os.environ, "PYTHONHASHSEED": "0"}
        os.execve(sys.executable, [sys.executable, "-m", "experiments.pilot.run", *sys.argv[1:]],
                  env)


if __name__ == "__main__":
    ensure_hash_seed()
    sys.exit(main())
