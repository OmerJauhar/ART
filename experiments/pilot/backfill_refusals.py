"""
Backfill Provider Refusals
==========================

Refusal caching (see is_provider_refusal in gateway.py) came after the pilot-v1 run, so that
run's provider refusals never reached the cache and a rerun would send those requests again.
This script puts them in the cache as the run saw them, without any API call:

1. Replay the run's selection from the cache and collect every missing call. per_question.jsonl
   must record each one as failed in the run (an answer error with the 403, or a critique
   missing from the round although critic and target both answered).
2. Match the missing calls to the engine's failure lines in the run's run.log: "Agent <model>
   failed to generate response" for an answer, "Critique from <critic> to <target> failed" for
   a critique. The log names models but not questions, so a line is tied to a question through
   its tournament, whose start line quotes the question's first 50 characters. An answer
   failure can belong to any tournament in progress; a critique failure belongs to the
   tournament completed right after it (for one round, the critique warnings, the round
   summary and the completion are logged with no await in between). Only the per-question
   pass counts: the first N tournaments of the log, N being the run's questions times its ART
   methods. The ELO router fit after it replays the calibration tournaments.
3. Missing calls with the same stage, model(s) and question prefix form a group. A group
   matches when the per-question pass logged exactly as many failure lines of that kind under
   that prefix, in distinct tournaments, and every one is an HTTP 403 "flagged for possible
   biological risk". Within a group, lines are assigned in log order to questions in sample
   order (questions start in sample order).
4. Only if every group matches, replay again and send just the matched calls through the
   gateway's refusal path, which caches each one as refused at cost 0, marked as backfilled
   from run.log with the log line.

The results folder is only read. Nothing is written unless every missing call matches.

    python -m experiments.pilot.backfill_refusals --run-name pilot-v1 --check  # match only
    python -m experiments.pilot.backfill_refusals --run-name pilot-v1          # match, write
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from experiments.pilot import data as pilot_data
from experiments.pilot.cache import CallCache, request_key
from experiments.pilot.data import Question
from experiments.pilot.estimate import CacheOnlyTransport, replay
from experiments.pilot.gateway import QUESTION_ID, REFUSAL_KEY, STAGE, ApiError
from experiments.pilot.prompts import format_question
from experiments.pilot.run import CACHE_DIR, load_sample
from experiments.pilot.settings import PILOT_DIR, RESULTS_DIR, PilotModels, load_models

REFUSED_TEXT = "flagged for possible biological risk"
# The engine logs the first 50 characters of the query when a tournament starts
QUERY_PREFIX_CHARS = 50
ART_METHODS = ("art_r1", "art_r2")

_RECORD = re.compile(r"^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d\.\d{3}) \| \w+\s*\| \S+ - ")
_START = re.compile(r"Starting tournament (\S+) for query: (.*)\.\.\.$", re.S)
_END = re.compile(r"Tournament (\S+) (?:completed|failed)")
_ROUND = re.compile(r"Round \d+: \d+ of \d+ critiques failed and were dropped$")
_ANSWER = re.compile(r"Agent (\S+) failed to generate response: (.*)$", re.S)
_CRITIQUE = re.compile(r"Critique from (\S+) to (\S+) failed: (.*)$", re.S)
_API_403 = "API error 403: "


@dataclass
class LogRecord:
    """One log record (continuation lines included)."""

    line: int
    time: str
    text: str


@dataclass
class Failure:
    """An engine failure line for an answer or critique, and its candidate tournaments."""

    record: LogRecord
    stage: str
    model: str
    target: str | None
    error: str
    tournaments: set[str]

    @property
    def refused(self) -> bool:
        """Whether the failure is a 403 flagged for possible biological risk."""
        return self.error.startswith(_API_403) and REFUSED_TEXT in self.error

    @property
    def message(self) -> str:
        """The provider's message."""
        return self.error.removeprefix(_API_403)


@dataclass
class MissingCall:
    """A call the selection needs that is not in the cache."""

    key: str
    question: Question
    stage: str
    model: str
    target: str | None = None
    match: Failure | None = None
    group_size: int = 1


@dataclass
class Selection:
    """The questions and methods of a recorded run."""

    questions: list[Question]
    skip: set[str]
    seed: int
    models: PilotModels


class RecordingTransport(CacheOnlyTransport):
    """Refuses every request, like CacheOnlyTransport, keeping its question, stage and body."""

    def __init__(self) -> None:
        """Create the transport."""
        super().__init__()
        self.requests: dict[str, tuple[str, str, dict[str, Any]]] = {}

    async def post(self, url: str, body: dict[str, Any]) -> dict[str, Any]:
        """Record the request, then refuse it."""
        endpoint = "decisions" if url.rstrip("/").endswith("/decisions") else "chat"
        self.requests.setdefault(
            request_key(endpoint, body), (QUESTION_ID.get(), STAGE.get(), body)
        )
        return await super().post(url, body)


class BackfillTransport(CacheOnlyTransport):
    """Answers the matched calls with their logged refusal; refuses everything else."""

    def __init__(self, refusals: dict[str, ApiError]) -> None:
        """Create the transport."""
        super().__init__()
        self.refusals = refusals
        self.sent: set[str] = set()

    async def post(self, url: str, body: dict[str, Any]) -> dict[str, Any]:
        """Raise the logged refusal for a matched call."""
        key = request_key("chat", body)
        if key in self.refusals and not url.rstrip("/").endswith("/decisions"):
            self.sent.add(key)
            raise self.refusals[key]
        return await super().post(url, body)


def load_selection(run_dir: Path) -> Selection:
    """Rebuild a run's selection from its config.json."""
    config = json.loads((run_dir / "config.json").read_text())
    selection = config["selection"]
    sample, _ = load_sample(config["seed"])
    questions = pilot_data.select_questions(
        sample, selection["subsets"], selection["test_per_subset"]
    )
    return Selection(questions, set(selection["skip_methods"]), config["seed"], load_models())


def read_log(path: Path) -> list[LogRecord]:
    """Split a loguru log into records; lines without a timestamp continue the last one."""
    records: list[LogRecord] = []
    for number, line in enumerate(path.read_text().splitlines(), start=1):
        match = _RECORD.match(line)
        if match:
            records.append(LogRecord(line=number, time=match.group(1), text=line[match.end():]))
        elif records:
            records[-1].text += "\n" + line
    return records


def log_failures(records: list[LogRecord], n_tournaments: int) -> tuple[list[Failure], dict]:
    """
    Every answer and critique failure logged in the first n_tournaments tournaments.

    Returns:
        (failures, query prefix by tournament ID)
    """
    prefixes: dict[str, str] = {}
    running: set[str] = set()
    failures: list[Failure] = []
    for i, record in enumerate(records):
        if start := _START.search(record.text):
            if len(prefixes) == n_tournaments:
                break  # the per-question pass is over
            prefixes[start.group(1)] = start.group(2)
            running.add(start.group(1))
        elif end := _END.search(record.text):
            running.discard(end.group(1))
        elif answer := _ANSWER.search(record.text):
            failures.append(Failure(record, "answer", answer.group(1), None, answer.group(2),
                                    set(running)))
        elif critique := _CRITIQUE.search(record.text):
            tournament = _block_tournament(records, i)
            failures.append(Failure(
                record, "critique", critique.group(1), critique.group(2), critique.group(3),
                {tournament} if tournament in running else set(),
            ))
    return failures, prefixes


def _block_tournament(records: list[LogRecord], index: int) -> str | None:
    """The tournament completed right after the critique warnings starting at index."""
    for record in records[index:]:
        if _CRITIQUE.search(record.text) or _ROUND.search(record.text):
            continue
        end = _END.search(record.text)
        return end.group(1) if end else None
    return None


def critique_target(body: dict[str, Any], answers: dict[str, Any]) -> str | None:
    """The model whose answer a critique request quotes, if exactly one."""
    prompt = body["messages"][-1]["content"]
    found = [m for m, a in answers.items() if a.get("content") and a["content"] in prompt]
    return found[0] if len(found) == 1 else None


def recorded_in_results(call: MissingCall, row: dict[str, Any]) -> bool:
    """Whether per_question.jsonl agrees that this call failed in the run."""
    first_round = row["methods"]["art_r1"]["rounds"][0]
    if call.stage == "answer":
        error = row["answers"][call.model].get("error") or ""
        return (_API_403 in error and REFUSED_TEXT in error
                and call.model not in first_round["responses"])
    if call.model not in first_round["responses"] or call.target not in first_round["responses"]:
        return False
    critics = {c["critic"] for c in first_round["critiques"].get(call.target, [])}
    return call.model not in critics


def describe(call: MissingCall) -> str:
    """Short description of a missing call (no question text)."""
    q = call.question
    target = f" -> {call.target}" if call.target else ""
    return f"index {q.index} ({q.subset}, {q.split}) {call.stage} {call.model}{target}"


async def find_missing(
    run_dir: Path,
    selection: Selection,
    cache: CallCache,
) -> tuple[list[MissingCall], list[str]]:
    """
    The selection's missing calls, matched to log lines where possible.

    Returns:
        (missing calls, problems that rule out backfilling)
    """
    rows = {
        row["id"]: row
        for row in map(json.loads, (run_dir / "per_question.jsonl").read_text().splitlines())
    }
    by_id = {q.id: q for q in selection.questions}

    transport = RecordingTransport()
    await replay(selection.questions, selection.models, cache, selection.seed, selection.skip,
                 transport=transport)
    missing = []
    problems = []
    for key, (question_id, stage, body) in transport.requests.items():
        if question_id not in by_id:
            continue
        call = MissingCall(key, by_id[question_id], stage, str(body.get("model", "")))
        missing.append(call)
        if stage not in ("answer", "critique"):
            problems.append(f"{describe(call)}: only answers and critiques can be matched")
            continue
        if stage == "critique":
            call.target = critique_target(body, rows[question_id]["answers"])
            if call.target is None:
                problems.append(f"{describe(call)}: cannot tell which answer it critiques")
                continue
        if question_id not in rows or not recorded_in_results(call, rows[question_id]):
            problems.append(f"{describe(call)}: per_question.jsonl does not record this failure")

    n_art = sum(1 for name in ART_METHODS if name not in selection.skip)
    failures, prefixes = log_failures(read_log(run_dir / "run.log"),
                                      len(selection.questions) * n_art)
    groups: dict[tuple[Any, ...], list[MissingCall]] = {}
    for call in missing:
        prefix = format_question(call.question)[:QUERY_PREFIX_CHARS]
        groups.setdefault((call.stage, call.model, call.target, prefix), []).append(call)
    for (stage, model, target, prefix), calls in groups.items():
        lines = [
            f for f in failures
            if (f.stage, f.model, f.target) == (stage, model, target)
            and any(prefixes[t] == prefix for t in f.tournaments)
        ]
        tournaments = [{t for t in f.tournaments if prefixes[t] == prefix} for f in lines]
        names = ", ".join(describe(c) for c in calls)
        if len(lines) != len(calls):
            problems.append(f"{names}: {len(lines)} matching failure lines in run.log")
        elif not all(f.refused for f in lines):
            bad = [f.record.line for f in lines if not f.refused]
            problems.append(f"{names}: run.log lines {bad} are not 403 flagged refusals")
        elif any(len(t) != 1 for t in tournaments) or len(set().union(*tournaments)) != len(lines):
            problems.append(f"{names}: run.log lines are not in one distinct tournament each")
        else:
            ordered = sorted(calls, key=lambda c: c.question.index)
            for call, line in zip(ordered, sorted(lines, key=lambda f: f.record.line)):
                call.match = line
                call.group_size = len(calls)
    return missing, problems


async def backfill(run_name: str, cache_path: Path, check: bool) -> int:
    """Match and (unless check) write the refusals; return a process exit code."""
    run_dir = RESULTS_DIR / run_name
    log_path = (run_dir / "run.log").relative_to(PILOT_DIR.parents[1])
    selection = load_selection(run_dir)
    cache = CallCache(cache_path)
    try:
        missing, problems = await find_missing(run_dir, selection, cache)
        for call in missing:
            if call.match:
                shared = (f"; one of {call.group_size} identical-prefix questions, assigned in "
                          "sample order" if call.group_size > 1 else "")
                print(f"{describe(call)}: run.log line {call.match.record.line} "
                      f"({call.match.record.time}{shared})")
        if problems:
            print("Not backfilling; unmatched:", file=sys.stderr)
            for problem in problems:
                print(f"  {problem}", file=sys.stderr)
            return 1
        if not missing:
            print("Nothing to backfill: every call of the selection is in the cache.")
            return 0
        if check:
            print(f"{len(missing)} missing calls, all matched (--check: nothing written).")
            return 0

        refusals = {}
        for call in missing:
            assert call.match is not None
            note = {
                "backfilled_from": str(log_path),
                "log_line": call.match.record.line,
                "logged_at": call.match.record.time,
            }
            if call.group_size > 1:
                note["log_line_assignment"] = (
                    f"one of {call.group_size} questions with the same 50-character prefix, "
                    "each refused; lines assigned in sample order"
                )
            refusals[call.key] = ApiError(403, call.match.message, note=note)
        before = len(cache)
        transport = BackfillTransport(refusals)
        await replay(selection.questions, selection.models, cache, selection.seed,
                     selection.skip, transport=transport)
        written = len(cache) - before
        stored = [cache.get(key) for key in refusals]
        assert transport.sent == set(refusals), "a matched call was not requested"
        assert written == len(refusals), f"expected {len(refusals)} new rows, got {written}"
        assert all(s and REFUSAL_KEY in s.response and s.cost_usd == 0.0 for s in stored)
        _, still_missing = await replay(selection.questions, selection.models, cache,
                                        selection.seed, selection.skip)
        assert not any(still_missing[q.id] for q in selection.questions), "calls still missing"
        print(f"Backfilled {written} refusals into {cache_path.name} (cost 0, marked as "
              f"backfilled from {log_path}); the selection is now fully cached.")
        return 0
    finally:
        cache.close()


def main(argv: list[str] | None = None) -> int:
    """Entry point."""
    parser = argparse.ArgumentParser(description="Backfill a run's logged refusals into the cache")
    parser.add_argument("--run-name", required=True,
                        help="Folder under experiments/pilot/results/")
    parser.add_argument("--cache", type=Path, default=CACHE_DIR / "calls.sqlite")
    parser.add_argument("--check", action="store_true", help="Match only; write nothing")
    args = parser.parse_args(argv)
    return asyncio.run(backfill(args.run_name, args.cache, args.check))


if __name__ == "__main__":
    # Same hash seed as run.py, so replays build the same requests
    if os.environ.get("PYTHONHASHSEED") != "0":
        os.execve(sys.executable, [sys.executable, "-m", "experiments.pilot.backfill_refusals",
                                   *sys.argv[1:]], {**os.environ, "PYTHONHASHSEED": "0"})
    sys.exit(main())
