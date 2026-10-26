"""
ART as ScreamingFace recipes (https://docs.screamingface.ai).

Two ways to express one round of ART with the current ScreamingFace client:

- ``art_one_judge``: best-of-N with judge scores. Every model answers; one judge model
  scores all answers in a single call and the highest-scored answer is returned verbatim.
- ``art_peer_panel``: every model is also a judge. Each peer scores all N answers in one
  call (N judge calls instead of ART's N(N-1) pairwise critiques), and the peers' picks are
  combined by plurality vote.

Known gaps against ART as published (asks for the toolkit): self-scores are not masked,
there is no ELO state carried across questions, and judges see answers in a fixed order.

Requires the ScreamingFace client (``pip install screamingface``, Python >= 3.12).
Evaluate with, e.g.:

    import screamingface as sf
    from art_recipes import build
    sf.connect()
    report = sf.evaluate(list(build().values()), benchmark="lab_bench_seqqa", limit=20)
"""

from __future__ import annotations

import screamingface as sf

PILOT_MODELS = (
    "openrouter/deepseek/deepseek-v4-pro",
    "openrouter/google/gemini-3.8-flash",
    "openrouter/anthropic/claude-sonnet-5.5",
    "openrouter/openai/gpt-6.1",
)

# The ART critique rubric, condensed for a judge that scores every answer at once.
ART_RUBRIC = (
    "You are reviewing candidate answers to the same question. Score each answer from 0 to 100 "
    "on accuracy, coherence, completeness and relevance, weighting accuracy most, and give each "
    "answer one overall score. A wrong final answer cannot score above 40."
)


def art_one_judge(models: tuple[str, ...] = PILOT_MODELS, judge: str | None = None) -> sf.Fusion:
    """One round of ART with a single judge: best-of-N by judge scores."""
    judge = judge or models[2]
    return sf.Fusion(
        [sf.Model(m) for m in models],
        name="art-one-judge",
        synthesizer=sf.Judge(judge, returns="scores", rubric=ART_RUBRIC),
    )


def art_peer_panel(models: tuple[str, ...] = PILOT_MODELS) -> sf.Fusion:
    """One round of ART where every model judges: one Judge Fusion per peer, then a vote."""
    members = [sf.Model(m) for m in models]
    peers = [
        sf.Fusion(
            members,
            name=f"peer-{i}",
            synthesizer=sf.Judge(judge, returns="scores", rubric=ART_RUBRIC),
        )
        for i, judge in enumerate(models)
    ]
    return sf.Fusion(peers, name="art-peer-panel", synthesizer=sf.Vote())


def build(models: tuple[str, ...] = PILOT_MODELS) -> dict[str, sf.Fusion]:
    """All ART recipes for one model set, keyed by name."""
    return {
        "art_one_judge": art_one_judge(models),
        "art_peer_panel": art_peer_panel(models),
    }


if __name__ == "__main__":
    for name, recipe in build().items():
        print(name, recipe)
