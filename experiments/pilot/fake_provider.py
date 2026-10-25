"""
Fake Provider
=============

Deterministic stand-in for OpenRouter chat completions and the Jev Decisions API, used by
--dry-run and the tests. No network. It recognizes each request type from its prompt,
returns pseudo-answers (some correct, some wrong, some unparseable), fake probabilities, and
usage priced at the real per-token rates in models.yaml.

Prompt tokens are estimated from characters. Completion tokens use the assumed typical
lengths in TYPICAL_COMPLETION_TOKENS, so a dry run doubles as a cost projection; the smoke
test replaces these assumptions with measured numbers.
"""

from __future__ import annotations

import hashlib
import math
from typing import Any

from experiments.pilot.data import Question
from experiments.pilot.prompts import FUSION_SYSTEM, extract_answer, format_question
from experiments.pilot.settings import PilotModels

# Characters per token for prompt-size estimates (DNA-heavy text tokenizes poorly)
CHARS_PER_TOKEN = 3.2

# Assumed visible completion tokens per call type, before reasoning tokens
TYPICAL_COMPLETION_TOKENS = {"answer": 350, "critique": 450, "revision": 400, "fusion": 300}

# Assumed reasoning tokens per call at each OpenRouter reasoning effort
REASONING_TOKENS = {"none": 0, "minimal": 150, "low": 500, "medium": 1200, "high": 2500}

UNPARSEABLE_ANSWER_RATE = 0.06
UNPARSEABLE_CRITIQUE_RATE = 0.04


def _unit(*parts: Any) -> float:
    """Deterministic pseudo-random number in [0, 1) from the given parts."""
    digest = hashlib.sha256("|".join(str(p) for p in parts).encode()).hexdigest()
    return int(digest[:12], 16) / float(16**12)


def _between(text: str, start: str, end: str) -> str:
    """Substring of text between two markers (or to the end)."""
    if start not in text:
        return ""
    after = text.split(start, 1)[1]
    return after.split(end, 1)[0] if end in after else after


class FakeTransport:
    """Transport that answers chat and Decisions requests without any network."""

    def __init__(self, questions: list[Question], models: PilotModels) -> None:
        """Index the questions so prompts can be matched to their correct answers."""
        self.models = models
        self._questions = [(format_question(q), q) for q in questions]
        ranked = sorted(models.models, key=lambda m: m.blended_price)
        skills = [0.45, 0.55, 0.62, 0.70]
        self.skill = {m.id: skills[i] for i, m in enumerate(ranked)}
        self.calls = 0

    def _find(self, text: str) -> Question | None:
        """The question whose formatted text appears in text."""
        for formatted, question in self._questions:
            if formatted in text:
                return question
        return None

    def _wrong_letter(self, question: Question, *seed: Any) -> str:
        """A deterministic wrong letter."""
        wrong = [x for x in question.letters if x != question.correct_letter]
        return wrong[int(_unit("wrong", *seed) * len(wrong))]

    async def post(self, url: str, body: dict[str, Any]) -> dict[str, Any]:
        """Dispatch on the endpoint."""
        self.calls += 1
        if url.rstrip("/").endswith("/decisions"):
            return self._decision(body)
        return self._chat(body)

    def _chat(self, body: dict[str, Any]) -> dict[str, Any]:
        """Fake chat completion."""
        model = body["model"]
        messages = body["messages"]
        system = messages[0]["content"] if messages[0]["role"] == "system" else ""
        user = messages[-1]["content"]
        question = self._find(user)

        if "evaluator" in system:
            kind, content = "critique", self._critique(model, user, question)
        elif "previous response has" in system:
            kind, content = "revision", self._revision(model, user, question)
        elif system == FUSION_SYSTEM:
            kind, content = "fusion", self._fusion(model, user, question)
        else:
            kind, content = "answer", self._answer(model, user, question)

        spec = self.models.get(model)
        effort = (spec.reasoning or {}).get("effort", "none")
        jitter = 0.75 + 0.5 * _unit("len", model, user)
        reasoning = int(REASONING_TOKENS.get(effort, 0) * jitter)
        visible = int(TYPICAL_COMPLETION_TOKENS[kind] * jitter)
        prompt_chars = sum(len(m["content"]) for m in messages)
        prompt_tokens = math.ceil(prompt_chars / CHARS_PER_TOKEN)
        completion_tokens = reasoning + visible
        return {
            "id": f"gen-fake-{self.calls}",
            "model": f"{model}-dryrun",
            "choices": [{
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "completion_tokens_details": {"reasoning_tokens": reasoning},
                "cost": spec.cost(prompt_tokens, completion_tokens),
            },
            "_fake_latency_ms": 600 + 12 * completion_tokens * jitter,
        }

    def _answer(self, model: str, user: str, question: Question | None) -> str:
        """First answer: correct with the model's skill probability, sometimes unparseable."""
        if question is None:
            return "I could not find the options.\nANSWER: A"
        u = _unit("answer", model, user)
        if u < UNPARSEABLE_ANSWER_RATE:
            return "This depends on details not given, so I cannot pick one option."
        if u < UNPARSEABLE_ANSWER_RATE + self.skill[model] * (1 - UNPARSEABLE_ANSWER_RATE):
            letter = question.correct_letter
        else:
            letter = self._wrong_letter(question, model, user)
        return f"[dry run] {model} reasoned briefly about the options.\nANSWER: {letter}"

    def _critique(self, model: str, user: str, question: Question | None) -> str:
        """Critique scoring correct answers higher, with noise; sometimes unparseable."""
        u = _unit("critique", model, user)
        if u < UNPARSEABLE_CRITIQUE_RATE:
            return "The response seems reasonable overall but I would not score it."
        response = _between(user, "RESPONSE TO EVALUATE:", "Please evaluate using")
        letter = extract_answer(response, question.letters) if question else None
        if letter is None:
            base = 20.0
        elif question is not None and letter == question.correct_letter:
            base = 72.0
        else:
            base = 50.0
        scores = {}
        for name in ("ACCURACY", "COHERENCE", "COMPLETENESS", "RELEVANCE", "OVERALL"):
            noise = (_unit("score", name, model, user) - 0.5) * 36
            scores[name] = int(min(100, max(0, base + noise)))
        lines = [f"{name}: {value}" for name, value in scores.items()]
        return (
            "\n".join(lines)
            + "\n\nPROS:\n- States a final answer\n\nCONS:\n- Brief justification\n\n"
            "SUGGESTIONS:\n- Check each option against the question\n\n"
            "REASONING:\n[dry run] fake critique"
        )

    def _revision(self, model: str, user: str, question: Question | None) -> str:
        """Revision: keeps the previous letter, sometimes switches to the correct one."""
        previous = _between(user, "YOUR PREVIOUS ANSWER:", "CRITIQUES:")
        if question is None:
            return "Revised.\nANSWER: A"
        letter = extract_answer(previous, question.letters)
        if letter is None or _unit("revise", model, user) < 0.3:
            letter = question.correct_letter if _unit("fix", model, user) < 0.6 else (
                self._wrong_letter(question, "rev", model, user)
            )
        return f"[dry run] {model} revised its answer.\nANSWER: {letter}"

    def _fusion(self, model: str, user: str, question: Question | None) -> str:
        """Fusion: usually correct, otherwise the most common letter among responses."""
        if question is None:
            return "ANSWER: A"
        if _unit("fusion", model, user) < 0.68:
            letter = question.correct_letter
        else:
            letters = [
                extract_answer(block, question.letters)
                for block in user.split("--- Response")[1:]
            ]
            letters = [x for x in letters if x]
            letter = max(sorted(set(letters)), key=letters.count) if letters else "A"
        return f"[dry run] fused the responses.\nANSWER: {letter}"

    def _decision(self, body: dict[str, Any]) -> dict[str, Any]:
        """Fake Decisions response for one Choice or Noul question."""
        state = body["state"]
        name, spec = next(iter(body["questions"].items()))
        question = self._find(str(state.get("question", "")))
        seed = str(body)

        if spec["type"] == "choice":
            weights = {}
            for i, option in enumerate(spec["criteria"]):
                text = str(state.get("responses", {}).get(option, ""))
                letter = extract_answer(text, question.letters) if question else None
                good = question is not None and letter == question.correct_letter
                position_bias = 0.6 if i == 0 else 0.0
                noise = _unit("jev", option, seed) * 1.2
                weights[option] = math.exp((1.8 if good else 0.0) + position_bias + noise)
            total = sum(weights.values())
            probabilities = {k: round(v / total, 4) for k, v in weights.items()}
            choice = max(probabilities, key=probabilities.get)
            answer = {
                "type": "choice",
                "choice": choice,
                "confidence": max(probabilities.values()),
                "probabilities": probabilities,
            }
        else:
            text = str(state.get("response", ""))
            letter = extract_answer(text, question.letters) if question else None
            good = question is not None and letter == question.correct_letter
            p = (0.72 if good else 0.35) + (_unit("noul", seed) - 0.5) * 0.5
            answer = {"type": "noul", "noul": round(min(0.99, max(0.01, p)), 4)}

        input_tokens = math.ceil(len(str(body)) / CHARS_PER_TOKEN)
        return {
            "id": f"gen-dec-fake-{self.calls}",
            "model": f"{body['model']}-dryrun",
            "provider": "FakeTypeSafe",
            "answers": {name: answer},
            "usage": {
                "input_tokens": input_tokens,
                "output_tokens": 8,
                "cost": self.models.jev.cost(input_tokens),
            },
            "_fake_latency_ms": 250 + 400 * _unit("lat", seed),
        }
