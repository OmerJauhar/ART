"""
Pilot Cost Estimate
===================

Projects the cost of a run from calls already in the cache (normally the smoke test's).

The cache does not store what each call was for, so cached questions are replayed through
the normal pipeline with a transport that refuses every request. Cache hits come back
labelled with their stage, nothing is spent, and each refused request is a missing call.
From the cached calls, per stage and model:

    cost per call = multiplier * (completion_tokens * output_price
                                  + (overhead_tokens + question_tokens) * input_price)

completion_tokens (visible plus reasoning) and overhead_tokens (everything in the prompt
besides the question: templates, the responses being critiqued) are means over the cached
calls. question_tokens is the question's length in characters times the model's measured
tokens per character for that subset (from its answer calls). The multiplier is billed cost
over list-price cost for the model, since OpenRouter may route to a provider priced above
list. The run total counts each distinct call once; per-method figures count each method as
if it ran alone.

New spend prices only the missing calls. A missing answer also changes every call built on it
(critiques by and of that model, revisions, fusion, Jev), which the replay cannot see, so those
planned calls are priced with it. A question with no cached calls is priced whole that way.
"""

from __future__ import annotations

import asyncio
import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from loguru import logger

from experiments.pilot.analysis import all_methods
from experiments.pilot.budget import BudgetGuard
from experiments.pilot.cache import CallCache, request_key
from experiments.pilot.data import Question
from experiments.pilot.gateway import QUESTION_ID, STAGE, ApiError, CallRecord, Gateway
from experiments.pilot.jev import JevClient
from experiments.pilot.methods import QuestionRunner
from experiments.pilot.prompts import format_question
from experiments.pilot.settings import PilotModels

JEV = "jev"
STAGES = ("answer", "critique", "revision", "judge", "fusion", "jev")

# A planned call: (kind, model, detail). kind is answer, critique, critique_r2, revision,
# fusion, jev_pick or jev_gate; detail tells apart calls of the same kind and model (the
# critique target, or the Jev ordering).
CallId = tuple[str, str, str]


# Distinct missing calls of one question: request key -> (kind, model), as in Observation
Missing = dict[str, tuple[str, str]]


class CacheOnlyTransport:
    """Transport for replays: every request reaching it is a cache miss, refused and recorded."""

    def __init__(self) -> None:
        """Create the transport."""
        self.missing: defaultdict[str, Missing] = defaultdict(dict)

    async def post(self, url: str, body: dict[str, Any]) -> dict[str, Any]:
        """Record the request as missing and refuse it."""
        if url.rstrip("/").endswith("/decisions"):
            types = {q.get("type") for q in (body.get("questions") or {}).values()}
            kind, model, endpoint = ("jev_pick" if "choice" in types else "jev_gate"), JEV, (
                "decisions"
            )
        else:
            kind, model, endpoint = STAGE.get(), str(body.get("model", "")), "chat"
        self.missing[QUESTION_ID.get()][request_key(endpoint, body)] = (kind, model)
        raise ApiError("not_cached", "replays make no API calls")


@dataclass
class Observation:
    """One cached call, labelled by the replay."""

    kind: str
    model: str
    question_id: str
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float


@dataclass
class StageFit:
    """Mean token counts for one stage and model."""

    n: int
    completion_tokens: float
    overhead_tokens: float


@dataclass
class CostModel:
    """Per-call cost model fitted from cached calls."""

    models: PilotModels
    stages: dict[tuple[str, str], StageFit]
    tokens_per_char: dict[tuple[str, str], float]
    multiplier: dict[str, float]
    n_calls: int
    question_ids: set[str] = field(default_factory=set)

    def question_tokens(self, model: str, question: Question) -> float:
        """Estimated tokens of the question text for a model."""
        ratio = self.tokens_per_char.get((model, question.subset))
        if ratio is None:
            ratio = self.tokens_per_char[(model, "*")]
        return len(format_question(question)) * ratio

    def stage_fit(self, kind: str, model: str) -> StageFit:
        """The fit for a stage and model, else that stage pooled over models."""
        kind = "critique" if kind == "critique_r2" else kind
        fit = self.stages.get((kind, model)) or self.stages.get((kind, "*"))
        if fit is None:
            raise KeyError(f"no cached {kind} calls to estimate from")
        return fit

    def call_cost(self, kind: str, model: str, question: Question) -> float:
        """Estimated cost in USD of one call."""
        fit = self.stage_fit(kind, model)
        prompt = fit.overhead_tokens + self.question_tokens(model, question)
        if model == JEV:
            list_cost = self.models.jev.cost(round(prompt))
        else:
            list_cost = self.models.get(model).cost(prompt, fit.completion_tokens)
        return self.multiplier.get(model, 1.0) * list_cost


def observations(records: dict[str, CallRecord]) -> list[Observation]:
    """Label replayed calls by stage; Jev calls are keyed by the question name they asked."""
    result = []
    for record in records.values():
        if not record.question_ids:
            continue
        if record.endpoint == "decisions":
            kind = "jev_pick" if "pick" in (record.response.get("answers") or {}) else "jev_gate"
            model = JEV
        else:
            kind, model = record.stage, record.model
        result.append(Observation(
            kind=kind,
            model=model,
            question_id=sorted(record.question_ids)[0],
            prompt_tokens=record.prompt_tokens,
            completion_tokens=record.completion_tokens,
            cost_usd=record.cost_usd,
        ))
    return result


def _mean(values: list[float]) -> float:
    """Mean of a non-empty list."""
    return sum(values) / len(values)


def fit_cost_model(
    obs: list[Observation],
    questions: dict[str, Question],
    models: PilotModels,
) -> CostModel:
    """
    Fit the per-call cost model.

    Args:
        obs: Labelled cached calls
        questions: Questions by ID (every observation's question must be here)
        models: Model prices

    Raises:
        ValueError: If there are no cached answer calls to measure question tokens from
    """
    chars = {qid: len(format_question(q)) for qid, q in questions.items()}
    answers = [o for o in obs if o.kind == "answer"]
    if not answers:
        raise ValueError("No cached answer calls to estimate from; run --smoke first")

    # Tokens per character of question text, per model and subset, from the answer calls
    # (whose prompt is the question plus a short system prompt)
    sums: dict[tuple[str, str], list[float]] = {}
    for o in answers:
        for key in ((o.model, questions[o.question_id].subset), (o.model, "*")):
            total = sums.setdefault(key, [0.0, 0.0])
            total[0] += o.prompt_tokens
            total[1] += chars[o.question_id]
    tokens_per_char = {key: t / c for key, (t, c) in sums.items() if c}
    # Jev's tokenizer is unknown; use the chat models' mean ratio
    for subset in {key[1] for key in tokens_per_char}:
        ratios = [r for (m, s), r in tokens_per_char.items() if s == subset and m != JEV]
        tokens_per_char[(JEV, subset)] = _mean(ratios)
    measured = {(o.model, o.question_id): o.prompt_tokens for o in answers}

    def question_tokens(o: Observation) -> float:
        if (o.model, o.question_id) in measured:
            return measured[(o.model, o.question_id)]
        subset = questions[o.question_id].subset
        ratio = tokens_per_char.get((o.model, subset), tokens_per_char.get((o.model, "*")))
        if ratio is None:
            ratio = _mean([r for (m, s), r in tokens_per_char.items() if s == "*"])
        return chars[o.question_id] * ratio

    groups: dict[tuple[str, str], list[Observation]] = {}
    for o in obs:
        groups.setdefault((o.kind, o.model), []).append(o)
        groups.setdefault((o.kind, "*"), []).append(o)
    stages = {
        key: StageFit(
            n=len(group),
            completion_tokens=_mean([o.completion_tokens for o in group]),
            overhead_tokens=max(0.0, _mean([o.prompt_tokens - question_tokens(o) for o in group])),
        )
        for key, group in groups.items()
    }

    multiplier = {}
    for model in {o.model for o in obs}:
        billed = sum(o.cost_usd for o in obs if o.model == model)
        if model == JEV:
            listed = sum(models.jev.cost(o.prompt_tokens) for o in obs if o.model == model)
        else:
            spec = models.get(model)
            listed = sum(
                spec.cost(o.prompt_tokens, o.completion_tokens) for o in obs if o.model == model
            )
        multiplier[model] = billed / listed if listed > 0 else 1.0

    return CostModel(
        models=models,
        stages=stages,
        tokens_per_char=tokens_per_char,
        multiplier=multiplier,
        n_calls=len(obs),
        question_ids={o.question_id for o in obs},
    )


def plan_calls(models: PilotModels, skip: set[str]) -> dict[str, set[CallId]]:
    """
    The calls each kept method needs on one question, as if it ran alone.

    elo_router and best_single_* make one answer call whose model is only known at run time,
    and the cascades escalate only on some questions, so those are given by their most
    expensive case here (see method_bounds for the cheapest).
    """
    ids = models.ids
    cheap, judge = models.roles["cheap"], models.roles["judge"]
    answers: set[CallId] = {("answer", m, "") for m in ids}
    r1 = answers | {("critique", c, t) for c in ids for t in ids if c != t}
    plan: dict[str, set[CallId]] = {f"single:{m.label}": {("answer", m.id, "")}
                                    for m in models.models}
    plan.update({
        "majority_vote": answers,
        "art_r1": r1,
        "art_r2": r1 | {("revision", m, "") for m in ids}
        | {("critique_r2", c, t) for c in ids for t in ids if c != t},
        "judge": answers | {("critique", judge, t) for t in ids},
        "fusion": answers | {("fusion", judge, "")},
        "jev_pick": answers | {("jev_pick", JEV, "1"), ("jev_pick", JEV, "2")},
        # The LLM gate (judge critiquing the cheap answer) is one of art_r1's critiques
        "cascade_llm": r1 | {("critique", judge, cheap)},
        "cascade_jev": r1 | {("jev_gate", JEV, "")},
    })
    for name in ("elo_router", "best_single_calibration", "best_single_oracle"):
        plan[name] = answers  # placeholder; costed through method_bounds
    return {name: calls for name, calls in plan.items() if name not in skip}


def method_bounds(models: PilotModels) -> dict[str, tuple[set[CallId], ...]]:
    """Alternative call sets for methods whose calls depend on run-time choices."""
    cheap, judge = models.roles["cheap"], models.roles["judge"]
    singles = tuple({("answer", m, "")} for m in models.ids)
    return {
        "elo_router": singles,
        "best_single_calibration": singles,
        "best_single_oracle": singles,
        "cascade_llm": ({("answer", cheap, ""), ("critique", judge, cheap)},),
        "cascade_jev": ({("answer", cheap, ""), ("jev_gate", JEV, "")},),
    }


def answer_dependents(
    run_calls: set[CallId],
    models: PilotModels,
    missing_answers: set[str],
) -> set[CallId]:
    """
    Planned calls that change when these models' answers are missing from the cache.

    A model without an answer is neither critiqued nor critiques (the engine only seats
    models that answered), and its answer feeds fusion, the Jev picks, every revision and so
    every round-two critique; the Jev gate reads only the cheap model's answer.
    """
    if not missing_answers:
        return set()
    cheap = models.roles["cheap"]
    dependents = set()
    for call in run_calls:
        kind, model, detail = call
        if kind == "answer":
            hit = model in missing_answers
        elif kind == "critique":
            hit = model in missing_answers or detail in missing_answers
        elif kind == "jev_gate":
            hit = cheap in missing_answers
        else:  # revision, critique_r2, fusion, jev_pick
            hit = True
        if hit:
            dependents.add(call)
    return dependents


def call_stage(call: CallId, plan: dict[str, set[CallId]]) -> str:
    """Stage a planned call is reported under; judge-only critiques count as judge."""
    kind = call[0]
    if kind in ("jev_pick", "jev_gate"):
        return "jev"
    if kind == "critique_r2":
        return "critique"
    if kind == "critique":
        art = plan.get("art_r1", set()) | plan.get("art_r2", set())
        return "critique" if call in art else "judge"
    return kind


@dataclass
class Estimate:
    """Projected cost of a run."""

    total_usd: float
    new_usd: float
    n_questions: int
    missing_calls: int
    partial_questions: list[str]
    partial_whole_usd: float
    cached_questions: list[str]
    cached_billed_usd: float
    cached_estimated_usd: float
    by_stage: dict[str, float]
    by_model: dict[str, float]
    methods: list[dict[str, Any]]
    cost_model: CostModel


def estimate_cost(
    cost_model: CostModel,
    questions: list[Question],
    skip: set[str],
    cached: set[str] | None = None,
    cached_billed_usd: float = 0.0,
    missing: dict[str, Missing] | None = None,
) -> Estimate:
    """
    Project the cost of running the kept methods on the given questions.

    Args:
        cost_model: Fitted per-call cost model
        questions: Questions the run will cover
        skip: Methods left out of the run
        cached: IDs of questions whose calls are all in the cache already (they cost nothing);
            any other question is priced whole. Ignored when missing is given.
        cached_billed_usd: What the cached questions' calls actually cost when they were made
        missing: Each question's missing calls from a replay; a question with none is fully
            cached, and the others pay only for their missing calls (see the module docstring)
    """
    if missing is not None:
        cached = {q.id for q in questions if not missing.get(q.id)}
    cached = cached or set()
    models = cost_model.models
    plan = plan_calls(models, skip)
    bounds = method_bounds(models)
    run_calls = set().union(*plan.values()) | {("answer", m, "") for m in models.ids}
    labels = {m.id: m.label for m in models.models}

    total = new = cached_estimated = partial_whole = 0.0
    missing_calls = 0
    partial: list[str] = []
    by_stage = {stage: 0.0 for stage in STAGES}
    by_model: dict[str, float] = {}
    per_method = {name: [0.0, 0.0] for name in plan}  # [most expensive, cheapest] case
    for question in questions:
        costs = {c: cost_model.call_cost(c[0], c[1], question) for c in run_calls}
        question_total = sum(costs.values())
        total += question_total
        if question.id in cached:
            cached_estimated += question_total
        elif missing is None:
            new += question_total
        else:
            calls = missing[question.id].values()
            missing_calls += len(calls)
            partial.append(question.id)
            partial_whole += question_total
            lost = {model for kind, model in calls if kind == "answer"}
            dependents = answer_dependents(run_calls, models, lost)
            new += sum(costs[c] for c in dependents)
            for kind, model in calls:
                # Missing calls other than critiques are covered by the dependents whenever
                # an answer is missing (the replay built them around the absent answer)
                if kind == "critique" or (kind != "answer" and not lost):
                    new += cost_model.call_cost(kind, model, question)
        for call, cost in costs.items():
            by_stage[call_stage(call, plan)] += cost
            name = "jev" if call[1] == JEV else labels[call[1]]
            by_model[name] = by_model.get(name, 0.0) + cost
        for name, calls in plan.items():
            options = [sum(costs[c] for c in option) for option in bounds.get(name, (calls,))]
            high = max(options) if name not in ("cascade_llm", "cascade_jev") else (
                sum(costs[c] for c in calls)
            )
            per_method[name][0] += high
            per_method[name][1] += min(options)

    n = len(questions)
    methods = []
    for name in all_methods(models):
        if name not in plan:
            continue
        high, low = per_method[name]
        note = ""
        if not n:
            pass
        elif name in ("elo_router", "best_single_calibration", "best_single_oracle"):
            note = f"one answer, model chosen at run time: ${low / n:.4f} to ${high / n:.4f}/q"
        elif name in ("cascade_llm", "cascade_jev"):
            note = f"always escalating; accepting every cheap answer: ${low / n:.4f}/q"
        methods.append({
            "method": name,
            "usd_per_question": high / n if n else float("nan"),
            "usd_total": high,
            "note": note,
        })
    return Estimate(
        total_usd=total,
        new_usd=new,
        n_questions=n,
        missing_calls=missing_calls,
        partial_questions=partial,
        partial_whole_usd=partial_whole,
        cached_questions=sorted(cached & {q.id for q in questions}),
        cached_billed_usd=cached_billed_usd,
        cached_estimated_usd=cached_estimated,
        by_stage=by_stage,
        by_model=by_model,
        methods=methods,
        cost_model=cost_model,
    )


async def replay(
    questions: list[Question],
    models: PilotModels,
    cache: CallCache,
    seed: int,
    skip: set[str],
) -> tuple[dict[str, CallRecord], defaultdict[str, Missing]]:
    """
    Run questions through the pipeline using only the cache.

    Returns:
        (call records of the cache hits, distinct missing calls per question ID)
    """
    transport = CacheOnlyTransport()
    gateway = Gateway(transport, cache, BudgetGuard(math.inf), models, concurrency=16)
    jev = JevClient(gateway, models.jev.model, models.jev.fallback_model)
    runner = QuestionRunner(gateway, models, jev, seed=seed, skip=skip)
    # Every refused request would otherwise log an agent error
    logger.disable("art_framework")
    try:
        await asyncio.gather(*[runner.run(q) for q in questions])
    finally:
        logger.enable("art_framework")
    return gateway.records, transport.missing


async def estimate_from_cache(
    sample: list[Question],
    selected: list[Question],
    models: PilotModels,
    cache: CallCache,
    seed: int,
    skip: set[str],
) -> Estimate:
    """
    Fit the cost model on every cached call for the sample and project the selected run.

    Args:
        sample: The whole sample (any of it may have cached calls to learn from)
        selected: Questions the run will cover
        models: Model settings
        cache: The cache to read (never written: replays make no calls)
        seed: Run seed (it shapes the fusion and Jev orderings, so the request bodies)
        skip: Methods left out of the run
    """
    records, _ = await replay(sample, models, cache, seed, skip=set())
    cost_model = fit_cost_model(observations(records), {q.id: q for q in sample}, models)
    selected_records, missing = await replay(selected, models, cache, seed, skip)
    cached = {q.id for q in selected if not missing[q.id]}
    billed = sum(
        r.cost_usd for r in selected_records.values() if r.question_ids & cached
    )
    return estimate_cost(cost_model, selected, skip, cached, billed,
                         missing={q.id: missing[q.id] for q in selected})


def format_estimate(estimate: Estimate, max_usd: float, already_recorded: float) -> str:
    """Human-readable estimate."""
    cm = estimate.cost_model
    models = cm.models
    labels = {m.id: m.label for m in models.models}
    labels[JEV] = "jev"
    lines = [
        f"Fitted on {cm.n_calls} cached calls from {len(cm.question_ids)} questions.",
        "",
        "Per call: cost = multiplier x (completion tokens x output price + (overhead + "
        "question tokens) x input price)",
        f"{'stage':<10}{'model':<20}{'n':>4}{'completion':>12}{'overhead':>10}{'multiplier':>12}",
    ]
    for kind in ("answer", "critique", "revision", "fusion", "jev_pick", "jev_gate"):
        for model in [*models.ids, JEV]:
            fit = cm.stages.get((kind, model))
            if fit is None:
                continue
            lines.append(
                f"{kind:<10}{labels[model]:<20}{fit.n:>4}{fit.completion_tokens:>12.0f}"
                f"{fit.overhead_tokens:>10.0f}{cm.multiplier.get(model, 1.0):>12.2f}"
            )
    n = estimate.n_questions
    lines += [
        "",
        f"Per method, as if run alone ({n} questions):",
        f"{'method':<28}{'USD/q':>9}{'USD total':>11}  note",
    ]
    for row in estimate.methods:
        lines.append(
            f"{row['method']:<28}{row['usd_per_question']:>9.4f}{row['usd_total']:>11.2f}  "
            f"{row['note']}"
        )
    stage_text = ", ".join(f"{k} ${v:.2f}" for k, v in estimate.by_stage.items() if v)
    model_text = ", ".join(
        f"{k} ${v:.2f}" for k, v in sorted(estimate.by_model.items(), key=lambda kv: -kv[1])
    )
    n_cached = len(estimate.cached_questions)
    per_question = f" (${estimate.total_usd / n:.4f}/q)" if n else ""
    lines += [
        "",
        f"Total, each distinct call once: ${estimate.total_usd:.2f}{per_question}",
        f"  by stage: {stage_text}",
        f"  by model: {model_text}",
        f"Already in the cache: {n_cached} of {n} questions (billed "
        f"${estimate.cached_billed_usd:.2f}; this model estimates "
        f"${estimate.cached_estimated_usd:.2f} for them)",
        f"New spend needed: ${estimate.new_usd:.2f}" + (
            f" ({estimate.missing_calls} missing calls in {len(estimate.partial_questions)} "
            f"questions, with the calls built on any missing answer; "
            f"${estimate.partial_whole_usd:.2f} if those questions were priced whole)"
            if estimate.partial_questions else ""
        ),
        f"Budget: the cache already records ${already_recorded:.2f} and --max-usd caps that "
        f"total, so --max-usd {max_usd:.2f} leaves ${max(0.0, max_usd - already_recorded):.2f} "
        f"for new calls.",
    ]
    return "\n".join(lines)
