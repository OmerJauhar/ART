# Real-model pilot: when is peer critique worth its cost?

This harness measures accuracy against cost per question for ART, simpler ensembles, and cheap
gates that decide when the full ART tournament is needed. Every method runs on the same
questions and reuses the same cached model calls.

Nothing here has been run against paid APIs yet. Everything below was built and tested with the
fake provider (`--dry-run`).

## Quick start

From the repository root, with the project's dependencies installed (`pip install -e ".[dev]"`
plus `pandas`, `pyarrow`, `matplotlib` and `pyyaml`):

```bash
# 0. Offline rehearsal: fake provider for every LLM and Jev call, all 100 questions
python -m experiments.pilot.run --dry-run --run-name dryrun

# 1. Put your key in experiments/pilot/.env (gitignored) or the environment
echo 'OPENROUTER_API_KEY=sk-or-...' > experiments/pilot/.env

# 2. Smoke test: 5 real questions, prints the projected cost of the full run
python -m experiments.pilot.run --smoke --run-name smoke --max-usd 5

# 3. Full run (reuses every call the smoke test already paid for)
python -m experiments.pilot.run --run-name pilot-v1 --max-usd 40
```

Other flags: `--limit N` (first N sampled questions), `--concurrency 8` (API calls in flight),
`--question-concurrency 4`, `--seed 0`, `--cache PATH`, `--prepare-data` (redraw the sample and
rewrite `sample_ids.json`).

Before a paid run, check that the prices in `models.yaml` still match
<https://openrouter.ai/api/v1/models>, and read the smoke test's output for empty answers or
unparseable critiques per model. Both show up in `per_question.jsonl` and `run.log`.

## Data

LAB-Bench (`futurehouse/lab-bench`), pinned to HuggingFace commit `5c77cec`. Only questions
answerable from the text supplied with them are used:

| Subset | Used | Why |
|---|---|---|
| SeqQA | 11 of 15 subtasks (440 questions) | Sequences are in the question. The four `PCR-gene-*` / `PCR-geneprimers-*` subtasks name an E. coli gene without its sequence, so they need a lookup and are excluded. |
| ProtocolQA | all 108 | The protocol text ships with each question and is inlined into the prompt. |
| CloningScenarios | 32 of 33 | Plasmid sequences are in the question. One question cites Addgene catalog numbers instead and is excluded. |
| FigQA, TableQA | no | Need figure or table images. |
| LitQA2, SuppQA | no | Need literature or supplementary-material retrieval. |
| DbQA | no | Needs database lookups (DisGeNet, OMIM). |

Sample: 100 questions, seed 0, as equal across the three subsets as their sizes allow
(SeqQA 34, ProtocolQA 34, CloningScenarios 32). The first 30 are the calibration split
(10 per subset); the last 70 are the test split (SeqQA 24, ProtocolQA 24, CloningScenarios 22).
Answer options are shuffled with a seed per question. `sample_ids.json` holds the exact IDs and
correct letters; the run refuses to start if a fresh draw does not match it.

Headline numbers are on the 70 test questions. Thresholds, the ELO router's ratings and
`best_single_calibration` are fitted on the calibration split only.

Question text is not committed: LAB-Bench carries a canary asking that it not appear in training
corpora, so `data/`, `cache/` and `results/` are gitignored.

## Models

Picked on 2026-10-25 from OpenRouter's public models endpoint: four families, each at its mid
tier (capable, below that family's frontier model). Details and reasons are in `models.yaml`.

| Model | Family | USD per M input / output | Role |
|---|---|---|---|
| `deepseek/deepseek-v4-pro` | DeepSeek (open weights) | 0.2088 / 0.4176 | cheap (first answerer in cascades) |
| `google/gemini-3.8-flash` | Google | 0.825 / 4.125 | judge (judge, fusion, LLM gate) |
| `anthropic/claude-sonnet-5.5` | Anthropic | 2.00 / 10.00 | |
| `openai/gpt-6.1-sol` | OpenAI | 2.00 / 10.00 | |

Roles are derived from prices: cheap is the cheapest model and judge the second-cheapest, so the
judge never grades its own answer in a cascade.

Jev: `typesafe/jev-1.14` through OpenRouter's alpha Decisions API, falling back to
`typesafe/jev-1.13` if 1.14 is unavailable. Input is billed at $0.042 per million tokens and
output is free. The dated snapshot each response reports is recorded in `calls.jsonl` and
`config.json`.

Settings that differ from a plain chat call:

- Temperature 0 is sent to every model, but `claude-sonnet-5.5` and `gpt-6.1-sol` do not accept
  `temperature` (it is missing from their supported parameters), so OpenRouter drops it for
  them. Their answers are not guaranteed deterministic; the cache makes reruns reproducible.
- Gemini, Sonnet and GPT-6.1 always reason, and reasoning tokens count against `max_tokens`.
  They run at reasoning effort `low` with `max_tokens` 4096. DeepSeek runs with reasoning off
  (its options are `xhigh`, `high` or `none`).

## Methods

All on the same questions, all reusing cached calls.

1. `single:<model>`: each model's own answer.
2. `majority_vote`: over the four answers; unparseable answers do not vote; ties broken by a seeded
   draw per question (tie count reported).
3. `art_r1`: ART `TournamentEngine` with four `LLMAgent`s, one round (critique + score). Fresh
   agents (ELO 1500) per question.
4. `art_r2`: two rounds (one revision round). Round 1 is served from `art_r1`'s cache.
5. `judge`: the judge model scores each answer with ART's critique prompt (4 calls) and the
   highest OVERALL wins.
6. `fusion`: the judge model reads the question and all four answers and gives its own answer.
7. `elo_router`: `art_r1` tournaments over the calibration split without resetting ELO, separately
   per subset; on the test split each question goes to the top-rated model for its subset
   (1 call). `best_single_calibration` (fixed on calibration) and `best_single_oracle` (picked on
   the test split, labelled as an oracle) are reported alongside.
8. `jev_pick`: one Jev Choice question over the four answers labelled R1..R4 in a seeded order,
   asked twice (that order and its reverse). Probabilities are averaged per model and the highest
   wins. How often the two orderings disagree is reported as position bias.
9. `cascade_llm`: the cheap model answers; the judge model scores that answer with ART's critique
   prompt; p = OVERALL/100. Accept if p >= tau, otherwise escalate to `art_r1`.
10. `cascade_jev`: the same with a Jev Noul gate on the cheap answer; p = P(yes).

Cascade thresholds maximize cascade accuracy on the calibration split, ties going to higher
coverage. The test result is reported at that tau. A tau sweep from 0 to 1 on the test split is
saved separately (`tau_sweep.csv`, and lines on the accuracy-vs-cost plot) and is labelled as a
sweep, not a tuned result.

The ART methods use the real framework: `PilotAgent` subclasses `LLMAgent` and overrides only
`_call_llm` to send requests through the cache and budget guard; ART's prompts, critique parsing
and usage accounting are unchanged. The question text given to agents includes the options and
the `ANSWER: <letter>` instruction, so revised answers keep that format.

Answers are read from the last `ANSWER: X` line (uppercase letter, valid for the question).
Anything else counts as wrong and is counted as unparseable per method.

## Cost accounting

Every call, LLM or Jev, is stored with its model, the model version returned, a prompt hash,
the full response, usage (tokens and OpenRouter's `usage.cost`) and latency.

A method's cost per question is the full cost of every distinct call it needs, as if it ran
alone: a call shared by several methods counts in full for each of them. A cascade costs the
cheap answer plus the gate, plus all of `art_r1` when it escalates (the cheap answer counted
once). Calls and tokens per question are counted the same way.

The LLM gate (the judge critiquing the cheap answer) is the same request as one of `art_r1`'s
critiques, so an escalated `cascade_llm` costs exactly what `art_r1` costs.

Latency is idealized: calls within one step are assumed to run in parallel and steps run in
sequence, using each call's recorded latency.

## Projected cost

A full run makes 37 distinct calls per question: 4 answers, 24 critiques (12 per ART round),
4 revisions, 1 judge self-critique, 1 fusion call and 3 Jev calls. The dry run prices these at
the real per-token rates and projects **$28.70** for the 100 questions (Sonnet $11.22,
GPT-6.1 $11.22, Gemini $5.70, DeepSeek $0.53, Jev $0.03). Critiques are about 75% of it.

That figure rests on assumptions the smoke test replaces with measurements: prompt tokens at
3.2 characters per token, and completions of 350 to 450 visible tokens plus about 500 reasoning
tokens per call at effort `low`. Completion tokens are about two thirds of the projected cost,
so the real number moves roughly with reasoning length. If every call used all 4096 output
tokens the run would cost $93.67; the budget guard stops it at `--max-usd` first.

## Reproducibility

- Requests are deterministic, so reruns hit the cache: agents get stable IDs (their model IDs),
  critiques are sorted by critic before the revision prompt is built, and the CLI re-executes
  itself with `PYTHONHASHSEED=0` because the tournament engine iterates sets of agent IDs and
  falls back to that order when two agents tie on ELO.
- Option shuffles, majority-vote tie breaks, fusion and Jev orderings use seeds derived from
  `--seed` and the question ID, so they do not depend on processing order.

## Engineering

- Cache: SQLite at `cache/calls.sqlite` (dry runs use `cache/calls_dryrun.sqlite`), keyed by a
  hash of the endpoint and request body. Reruns are free.
- Budget guard: `--max-usd` (default 40) caps all live spend recorded in the cache file, across
  runs, for LLM and Jev calls together. Before each uncached call the guard reserves that call's
  worst case (prompt size at 2 characters per token plus `max_tokens` of output); if that would
  cross the cap, the call is refused, unfinished questions are dropped, and the outputs cover
  the questions that finished.
- Async, at most 8 concurrent calls, exponential backoff on 429 and 5xx (honouring
  `Retry-After`), 120 s timeout.
- The API key is read from `OPENROUTER_API_KEY` or `experiments/pilot/.env` and is only ever
  placed in the request header. `.env` is gitignored.
- The Decisions API is alpha; all of its request and response handling is in `jev.py`.

## Outputs (`results/<run_name>/`)

| File | Contents |
|---|---|
| `config.json` | models, prices, roles, Jev version, seed, split, taus, ELO router ratings, date, git commit, spend |
| `questions.jsonl` | ID, subset, options, correct letter, split |
| `calls.jsonl` | every call: model, version, prompt hash, response, usage, latency |
| `per_question.jsonl` | each method's answer, correctness, cost, calls; ART round-by-round records (answers, critiques, scores, ELO before/after); cascade p and accept/escalate |
| `summary.csv` | per method: accuracy with bootstrap 95% CI, USD/question, calls/question, tokens, latency, unparseable count, coverage |
| `tau_sweep.csv` | both cascades' accuracy, cost and coverage for tau 0..1 on the test split |
| `gate_calibration.csv`, `gate_risk_coverage.csv`, `gate_reliability.png` | reliability deciles, Brier score and risk-coverage for both gates |
| `accuracy_vs_cost.png` / `.svg` | accuracy (with CI) vs mean USD/question on a log axis, plus the cascade sweeps |
| `README.md`, `run.log` | short results summary with total spend; full log |

## Tests

```bash
pytest -q experiments/pilot/tests
```

No test touches the network: the fake provider and in-memory transports stand in for
OpenRouter, and the Jev parser is checked against a fixture copied from the documented response.
