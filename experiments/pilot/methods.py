"""
Pilot Methods
=============

Runs every method on one question. Methods share calls through the Gateway cache; each
method's own call trace records exactly which calls it needed, so its cost can be counted as
if it ran alone.

Each method result stores "groups": lists of call keys that run one after another (calls in
the same group run in parallel). Cost, call count and tokens come from the union of keys;
latency is the sum over groups of the slowest call in each group.
"""

from __future__ import annotations

import asyncio
import random
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, TypeVar

from art_framework.agents.base import AgentResponse, Critique
from art_framework.tournament.engine import RoundResult, TournamentConfig, TournamentEngine
from experiments.pilot.agents import PilotAgent
from experiments.pilot.data import Question
from experiments.pilot.gateway import QUESTION_ID, STAGE, TRACE, CallRecord, Gateway
from experiments.pilot.jev import JevClient
from experiments.pilot.prompts import (
    FUSION_SYSTEM,
    extract_answer,
    format_question,
    fusion_prompt,
)
from experiments.pilot.settings import PilotModels

T = TypeVar("T")

NO_RESPONSE = "(no response)"

JEV_PICK_INSTRUCTIONS = "Which response gives the correct final answer to the question?"
JEV_GATE_INSTRUCTIONS = "The final answer choice in this response is correct."
JEV_GATE_CRITERIA = {
    "true": "The option named on the response's final ANSWER line is the correct answer.",
    "false": (
        "The option named on the response's final ANSWER line is not the correct answer, "
        "or the response gives no final answer."
    ),
}


def group_trace(trace: list[CallRecord]) -> list[list[str]]:
    """Split a call trace into consecutive runs of the same stage (one group per step)."""
    groups: list[list[str]] = []
    last_stage = None
    for record in trace:
        if record.stage != last_stage or not groups:
            groups.append([])
            last_stage = record.stage
        groups[-1].append(record.key)
    return groups


async def traced(factory: Callable[[], Awaitable[T]]) -> tuple[T | None, list[CallRecord], str]:
    """
    Run a coroutine in its own task with a fresh call trace.

    Returns:
        (result or None on error, trace, error message or "")
    """
    async def runner() -> tuple[T | None, list[CallRecord], str]:
        trace: list[CallRecord] = []
        TRACE.set(trace)
        try:
            return await factory(), trace, ""
        except Exception as e:  # recorded per method; budget stops are checked by the caller
            return None, trace, f"{type(e).__name__}: {e}"

    return await asyncio.create_task(runner())


@dataclass
class EloSnapshot:
    """ELO ratings by model before and after one round."""

    before: dict[str, float]
    after: dict[str, float]


class QuestionRunner:
    """Runs all methods for single questions."""

    def __init__(
        self,
        gateway: Gateway,
        models: PilotModels,
        jev: JevClient,
        seed: int = 0,
    ) -> None:
        """Create the runner."""
        self.gateway = gateway
        self.models = models
        self.jev = jev
        self.seed = seed

    def make_agents(self) -> dict[str, PilotAgent]:
        """Fresh agents (ELO 1500) for every model, keyed by model ID."""
        return {m.id: PilotAgent(m, self.gateway, self.models) for m in self.models.models}

    async def run_tournament(
        self,
        question: Question,
        rounds: int,
        agents: dict[str, PilotAgent] | None = None,
    ) -> dict[str, Any]:
        """
        Run ART on one question and return its answer plus the round-by-round record.

        Args:
            question: The question
            rounds: rounds_per_tournament
            agents: Agents to use; pass the same agents to keep ELO across questions
        """
        agents = agents or self.make_agents()
        model_of = {agent.id: model_id for model_id, agent in agents.items()}
        snapshots: list[EloSnapshot] = []
        before = {model_of[a.id]: a.elo_score for a in agents.values()}

        def on_round(_: RoundResult) -> None:
            nonlocal before
            after = {model_of[a.id]: a.elo_score for a in agents.values()}
            snapshots.append(EloSnapshot(before=before, after=after))
            before = after

        engine = TournamentEngine(
            agents=list(agents.values()),
            config=TournamentConfig(rounds_per_tournament=rounds),
            on_round_complete=on_round,
        )
        result = await engine.run_tournament(format_question(question), query_id=question.id)
        winner = model_of[result.winner_agent_id]
        return {
            "answer": extract_answer(result.best_response.content, question.letters),
            "winner_model": winner,
            "final_elo": {model_of[aid]: elo for aid, elo in result.final_rankings},
            "rounds": [
                serialize_round(r, model_of, question, snap)
                for r, snap in zip(result.rounds, snapshots)
            ],
        }

    async def run(self, question: Question) -> dict[str, Any]:
        """Run every method on one question; return the raw per-question record."""
        task = asyncio.create_task(self._run(question))
        return await task

    async def _run(self, question: Question) -> dict[str, Any]:
        """Body of run(), inside its own task so QUESTION_ID stays local."""
        QUESTION_ID.set(question.id)
        text = format_question(question)
        cheap = self.models.roles["cheap"]
        judge_id = self.models.roles["judge"]
        record: dict[str, Any] = {
            "id": question.id,
            "index": question.index,
            "subset": question.subset,
            "subtask": question.subtask,
            "split": question.split,
            "correct_letter": question.correct_letter,
            "answers": {},
            "methods": {},
        }
        methods = record["methods"]

        # 1. Each model answers once; every other method reuses these calls
        answer_agents = self.make_agents()
        answer_results = await asyncio.gather(*[
            traced(lambda a=agent: a.generate_response(text, question.id))
            for agent in answer_agents.values()
        ])
        answers: dict[str, AgentResponse | None] = {}
        answer_keys: dict[str, str] = {}
        for model_id, (response, trace, error) in zip(answer_agents, answer_results):
            answers[model_id] = response
            answer_keys[model_id] = trace[-1].key if trace else ""
            letter = extract_answer(response.content, question.letters) if response else None
            record["answers"][model_id] = {
                "letter": letter,
                "content": response.content if response else None,
                "error": error or None,
                "call": answer_keys[model_id] or None,
            }
        answer_group = [k for k in answer_keys.values() if k]

        for spec in self.models.models:
            entry = record["answers"][spec.id]
            methods[f"single:{spec.label}"] = {
                "answer": entry["letter"],
                "groups": [[answer_keys[spec.id]]] if answer_keys[spec.id] else [],
            }

        # 2. ART, one round then two rounds (round 1 of art_r2 is served from cache)
        for name, rounds in (("art_r1", 1), ("art_r2", 2)):
            art, trace, error = await traced(lambda r=rounds: self.run_tournament(question, r))
            methods[name] = {
                "answer": art["answer"] if art else None,
                "groups": group_trace(trace),
                "error": error or None,
                **({k: v for k, v in art.items() if k != "answer"} if art else {}),
            }

        # 3. Methods that build on the four answers, in parallel
        judge_task = self._judge(question, text, answers, answer_group, judge_id)
        fusion_task = self._fusion(question, text, answers, answer_group, judge_id)
        jev_pick_task = self._jev_pick(question, text, answers, answer_group)
        jev_gate_task = self._jev_gate(text, answers[cheap], answer_keys[cheap])
        judge, fusion, jev_pick, jev_gate = await asyncio.gather(
            judge_task, fusion_task, jev_pick_task, jev_gate_task
        )
        methods["judge"] = judge
        methods["fusion"] = fusion
        methods["jev_pick"] = jev_pick

        # 4. Cascade components; accept/escalate is decided once tau is fitted
        cheap_group = [answer_keys[cheap]] if answer_keys[cheap] else []
        llm_gate = judge["per_target"].get(cheap, {})
        methods["cascade_llm"] = {
            "cheap_answer": record["answers"][cheap]["letter"],
            "p": llm_gate.get("p", 0.0),
            "gate_error": llm_gate.get("error"),
            "accept_groups": [cheap_group, [llm_gate["call"]] if llm_gate.get("call") else []],
        }
        methods["cascade_jev"] = {
            "cheap_answer": record["answers"][cheap]["letter"],
            "p": jev_gate["p"],
            "gate_error": jev_gate["error"],
            "accept_groups": [cheap_group, jev_gate["groups"]],
        }
        return record

    async def _judge(
        self,
        question: Question,
        text: str,
        answers: dict[str, AgentResponse | None],
        answer_group: list[str],
        judge_id: str,
    ) -> dict[str, Any]:
        """Judge model scores each answer with ART's critique prompt; pick the top OVERALL."""
        judge = self.make_agents()[judge_id]
        targets = [m for m in self.models.ids if answers[m] is not None]
        results = await asyncio.gather(*[
            traced(lambda m=m: judge.critique_response(text, answers[m]))  # type: ignore
            for m in targets
        ])
        per_target: dict[str, dict[str, Any]] = {}
        calls = []
        for model_id, (critique, trace, error) in zip(targets, results):
            call = trace[-1].key if trace else None
            if call:
                calls.append(call)
            per_target[model_id] = {
                "overall": critique.score if critique else None,
                "p": critique.score / 100 if critique else 0.0,
                "error": error or None,
                "call": call,
                "critique": serialize_critique(critique) if critique else None,
            }
        ids = self.models.ids
        scored = [m for m in ids if per_target.get(m, {}).get("overall") is not None]
        # Ties go to the earlier model in models.yaml order
        best = max(
            scored, key=lambda m: (per_target[m]["overall"], -ids.index(m)), default=None
        )
        best_answer = answers[best] if best else None
        letter = extract_answer(best_answer.content, question.letters) if best_answer else None
        return {
            "answer": letter,
            "picked_model": best,
            "per_target": per_target,
            "groups": [answer_group, calls],
        }

    async def _fusion(
        self,
        question: Question,
        text: str,
        answers: dict[str, AgentResponse | None],
        answer_group: list[str],
        judge_id: str,
    ) -> dict[str, Any]:
        """Judge model reads all four answers and gives its own final answer."""
        order = list(self.models.ids)
        random.Random(f"{self.seed}:fusion:{question.id}").shuffle(order)
        responses = [answers[m].content if answers[m] else NO_RESPONSE for m in order]
        judge = self.make_agents()[judge_id]

        async def call() -> str:
            STAGE.set("fusion")
            return await judge._call_llm([
                {"role": "system", "content": FUSION_SYSTEM},
                {"role": "user", "content": fusion_prompt(text, responses)},
            ])

        content, trace, error = await traced(call)
        return {
            "answer": extract_answer(content, question.letters),
            "order": order,
            "content": content,
            "error": error or None,
            "groups": [answer_group, [r.key for r in trace]],
        }

    async def _jev_pick(
        self,
        question: Question,
        text: str,
        answers: dict[str, AgentResponse | None],
        answer_group: list[str],
    ) -> dict[str, Any]:
        """One Jev Choice question per ordering; average probabilities per model."""
        order = list(self.models.ids)
        random.Random(f"{self.seed}:jev:{question.id}").shuffle(order)
        orderings = [order, list(reversed(order))]

        async def ask(ordering: list[str]) -> Any:
            labels = {f"R{i}": model for i, model in enumerate(ordering, start=1)}
            state = {
                "question": text,
                "responses": {
                    label: answers[m].content if answers[m] else NO_RESPONSE
                    for label, m in labels.items()
                },
            }
            criteria = {
                label: f"Response {label} gives the correct final answer." for label in labels
            }
            return await self.jev.choice(state, JEV_PICK_INSTRUCTIONS, criteria)

        results = await asyncio.gather(*[traced(lambda o=o: ask(o)) for o in orderings])
        per_ordering = []
        sums = {m: 0.0 for m in self.models.ids}
        calls = []
        errors = []
        for ordering, (answer, trace, error) in zip(orderings, results):
            calls += [r.key for r in trace]
            if answer is None:
                errors.append(error)
                continue
            probs = {
                ordering[int(label[1:]) - 1]: p for label, p in answer.probabilities.items()
                if label[1:].isdigit() and 1 <= int(label[1:]) <= len(ordering)
            }
            for m, p in probs.items():
                sums[m] += p
            ids = self.models.ids
            top = max(ids, key=lambda m: (probs.get(m, 0.0), -ids.index(m)))
            per_ordering.append({
                "ordering": ordering,
                "probabilities": probs,
                "top_model": top,
                "model_version": answer.model_version,
            })

        picked = None
        if per_ordering:
            averaged = {m: sums[m] / len(per_ordering) for m in self.models.ids}
            picked = max(
                self.models.ids, key=lambda m: (averaged[m], -self.models.ids.index(m))
            )
        else:
            averaged = {}
        tops = [o["top_model"] for o in per_ordering]
        disagree = len(tops) == 2 and tops[0] != tops[1]
        return {
            "answer": (
                extract_answer(answers[picked].content, question.letters)
                if picked and answers[picked] else None
            ),
            "picked_model": picked,
            "probabilities": averaged,
            "orderings": per_ordering,
            "orderings_disagree": disagree,
            "error": "; ".join(errors) or None,
            "groups": [answer_group, calls],
        }

    async def _jev_gate(
        self,
        text: str,
        cheap_answer: AgentResponse | None,
        cheap_key: str,
    ) -> dict[str, Any]:
        """One Jev Noul question on the cheap model's answer; p = P(yes)."""
        if cheap_answer is None:
            return {"p": 0.0, "error": "no cheap answer", "groups": []}
        state = {"question": text, "response": cheap_answer.content}
        answer, trace, error = await traced(
            lambda: self.jev.noul(state, JEV_GATE_INSTRUCTIONS, JEV_GATE_CRITERIA)
        )
        return {
            "p": answer.p_yes if answer else 0.0,
            "error": error or None,
            "model_version": answer.model_version if answer else None,
            "groups": [r.key for r in trace],
        }


def serialize_critique(critique: Critique) -> dict[str, Any]:
    """JSON form of a critique."""
    return {
        "overall": critique.score,
        "accuracy": critique.accuracy_score,
        "coherence": critique.coherence_score,
        "completeness": critique.completeness_score,
        "relevance": critique.relevance_score,
        "pros": critique.pros,
        "cons": critique.cons,
        "suggestions": critique.suggestions,
        "text": critique.reasoning,
    }


def serialize_round(
    round_result: RoundResult,
    model_of: dict[str, str],
    question: Question,
    elo: EloSnapshot,
) -> dict[str, Any]:
    """JSON form of one ART round, with agent IDs replaced by model IDs."""
    return {
        "round": round_result.round_number,
        "responses": {
            model_of[aid]: {
                "letter": extract_answer(r.content, question.letters),
                "content": r.content,
                "confidence": r.confidence,
                "confidence_source": r.metadata.get("confidence_source"),
            }
            for aid, r in round_result.responses.items()
        },
        "critiques": {
            model_of[target]: [
                {"critic": model_of[c.critic_agent_id], **serialize_critique(c)}
                for c in critiques
            ]
            for target, critiques in round_result.critiques.items()
        },
        "scores": {model_of[aid]: s for aid, s in round_result.scores.items()},
        "elo_before": elo.before,
        "elo_after": elo.after,
        "failed_critiques": round_result.failed_critiques,
        "best_model": model_of[round_result.best_response_agent_id],
    }
