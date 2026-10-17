# Open Referee ⚖️

**An open-source AI referee for academic papers.** Open Referee reads your
manuscript, grounds itself in the relevant literature (OpenAlex, Crossref,
web search), stress-tests every section the way a referee for a top journal
would, and returns a structured review: an overall report plus inline comments
anchored to the exact passages they criticize, each with a severity score.

It is a free, self-hosted, private alternative to hosted tools like
[refine.ink](https://www.refine.ink): you bring your own LLM keys (cloud or
local), your data never leaves your machine except to the LLM providers you
configure, and the full pipeline is inspectable.

> ⚠️ Open Referee is a tool *for authors* to stress-test drafts before
> submission. It complements — never replaces — actual peer review.

## Features

- **Multi-format ingestion** — PDF, DOCX, LaTeX (with `\input` resolution), Markdown
- **Anchored feedback** — every comment pins to exact manuscript text via
  fuzzy quote matching (with confidence), degrading gracefully to section level
- **Multi-model roles** — `strong`, `small`, and `vision` roles, each bound to
  any provider: *openai-compatible, Anthropic, Gemini, Ollama, vLLM* — mix
  cloud and local models freely
- **Literature grounding** — [openalex-py](https://pypi.org/project/openalex-py)
  for OpenAlex, Crossref polite pool, and pluggable web search
  (**Tavily**, **TinyFish**, **Brave** — in your own priority order, with failover)
- **Review-community signals** — scans PubPeer, OpenReview, and PREreview for
  prior discussion of the paper and its key citations
- **User-supplied literature** — attach related-work PDFs to steer the review
- **Adversarial pipeline** — section verifiers → challenger ("Referee 2") →
  bibliography audit → meta-review → **validation pass** that drops
  unverifiable comments before delivery
- **Depth presets** — `fast` / `standard` / `deep` (deep ≈ up to 150 claim
  verifications + every artifact + full defense rounds)
- **Budget-driven mode (`auto`)** — set a budget and a planner designs the
  review to fit it: upfront feasibility check (aborts with a suggested
  minimum budget when a meaningful review is not affordable), a cost model
  per pipeline step, live spend-vs-plan tracking with adaptive estimates,
  and deterministic steering (LLM replanning only for genuine judgment
  calls — and the planner's own calls are charged to the same budget)
- **Desk-reject estimator** — a second product: describe a journal (aim,
  scope, metrics, institutional considerations — free-form) and get the
  probability of passing the desk screen, from evidence-grounded factor
  scores plus a transparent calibrated decision model
- **Cost control** — per-call usage ledger, cost estimation from a pricing
  table, hard spend cap per review (pricing entries are **required** in
  `auto` mode and for the fit estimate; local models can be priced at `0.0`)
- **Attribution & metadata** — every report (review and fit) carries a
  build-metadata table (model, calls, in/out tokens, cost per role), the
  budget story (planned vs spent, adjustments), and a
  *built with open-referee* link
- **PWA** — installable web app (htmx + Alpine, no build step), live SSE
  progress with a budget panel, resumable runs, JSON/Markdown/**PDF** report
  export

## Quickstart

```bash
# run from PyPI (once published)
uvx open-referee serve          # or: pip install open-referee && open-referee serve

# run from source
uv sync
uv run open-referee serve       # http://127.0.0.1:8410

# run with Docker
docker compose up --build
```

Open `http://127.0.0.1:8410`, configure providers under **Settings** (or edit
`~/.open-referee/config.yaml`), upload a paper, and start a review.

### Configuration

Secrets live in environment variables; the config file only references their
names. See [`config.example.yaml`](config.example.yaml) for every option.

```yaml
llm:
  providers:
    openrouter: {type: openai_compatible, base_url: https://openrouter.ai/api/v1, api_key_env: OPENROUTER_API_KEY}
    ollama:     {type: ollama, base_url: http://localhost:11434}
  roles:
    strong: {provider: openrouter, model: anthropic/claude-sonnet-4.5}
    small:  {provider: ollama, model: llama3.1:8b}
    vision: {provider: gemini, model: gemini-2.5-flash}
search:
  order: [tavily, tinyfish, brave]   # your priority, failover top-down
review:
  max_cost_usd: 10.0
```

For budget-driven mode (`depth: auto`) and the desk-reject estimator, every
configured model needs a price under `llm.pricing` — the run fails fast with
the exact YAML to add otherwise. An explicit `0.0` entry counts as priced
(local models).

## How a review runs

| Stage | Role | What it does |
|---|---|---|
| Triage | strong | domain, math/statistics density, novelty claims, search queries |
| Claim inventory | strong | exhaustive catalogue of atomically checkable claims (theorems+proofs, equations, numbers, table/figure-text pairs, citations, cross-references) |
| Per-claim verifiers | strong/small | one dedicated call per claim: math re-derivation with an isolated definitions pack, number checks, table/figure-vs-prose, citation support, cross-reference targets |
| Literature survey | small | builds the context pack (OpenAlex/Crossref/web + your PDFs) |
| Community scout | small | PubPeer / OpenReview / PREreview signals |
| Section verifiers | small + vision | specialized lenses per section: math/theory, statistics/econometrics, prose coherence, literature coherence (+ figures via vision) |
| Whole-paper verifiers | strong | cross-section coherence: abstract vs results, numbers/notation drift across sections, contradictions between results |
| Artifact verifiers | vision | EVERY figure and EVERY table read from the page image: plotted values vs prose, caption accuracy, significance stars vs SEs, axis/unit errors |
| Challenger | strong | per-section adversarial re-read + whole-paper challenge: validates/adjusts/drops candidates, hunts missed global weaknesses |
| Citation audit | small | reference existence + in-text quotation consistency, missing key work |
| Meta-review | strong | dedupe, severity calibration, overall report |
| Defense–adjudication | small + strong | per comment: authors' best defense argued, adjudicator upholds/adjusts/dismisses |
| **Review validator** | strong | final gate: quotes verified, severities recalibrated |
| Assembler | — | anchor resolution (fuzzy + confidence), exports |

Interrupted runs resume from `~/.open-referee/runs/<id>/` — stage artifacts,
the budget plan, and the full spend ledger are persisted after every step.

### Budget-driven mode (`auto`)

Pick `auto` on the review form (or in settings) and set a budget. The run
then works like this:

1. **Preflight (free)** — after ingestion, a cost model built from document
   statistics and your `llm.pricing` table checks feasibility. If the budget
   cannot buy a meaningful minimum (control calls + 6 central claim
   verifications + core synthesis), the run aborts immediately and suggests
   the minimum — and what a standard-quality review would cost.
2. **Plan v0 (deterministic, free)** — a greedy quality ordering produces a
   starting plan; the budget is a hard ceiling and the plan target is soft.
3. **Planner call (budgeted)** — after the claim inventory, one cheap
   planner LLM call designs the strategy with full knowledge of the actual
   claim count. Simple papers get cheap plans — a planner may target $25 of
   a $100 budget, and that is respected (no spending money just because it
   is available). Its output is validated and repaired deterministically;
   LLM arithmetic is never trusted.
4. **Tracking + steering** — every call (including triage, the claim
   inventory, and the planner itself) is charged to the budget. Unit costs
   adapt to observed reality. At stage boundaries the harness steers
   deterministically (a forced repair applies without any LLM call); only
   genuine multi-option judgment calls go back to the planner, throttled to
   a small reserve. Fan-out loops gate per call, so claim verification
   degrades gracefully (importance-first) instead of crashing into the cap.

### Run history (estimation dataset)

Every run — review or fit, completed or failed — appends one JSON line to
`~/.open-referee/run_history.jsonl` (one line per run; resumed runs update
theirs in place). Each record pairs the paper's characteristics (length,
sections, figures/tables, references, claims, lenses) with the full budget
story at three levels of detail:

- **per stage**: status, timing, and planned vs actual calls/tokens/cost
  for every phase of the review (triage, claims, verify, challenge, …);
- **per call type**: the calibration pair — estimated in/out tokens per
  call vs the observed average, plus planned/actual counts and costs;
- **per call**: an ordered spend timeline with cumulative cost, and the
  plan-version trajectory (`preflight-v0 → claims-milestone → …`) with the
  spend at each change.

Tokens are recorded alongside dollars so the dataset stays valid when
prices change; it is the base for improving the cost model's estimates in
future versions.

### Desk-reject estimator

The *Journal fit estimate* card takes a paper plus a free-form venue
profile — anything from a journal's aims-and-scope to a working-paper
collection's institutional considerations. One strong-model call produces
six evidence-grounded factor scores (scope fit, aim alignment,
methods/format fit, novelty threshold, presentation readiness,
policy/ethics); a transparent weighted-logistic calibration (weights
printed in the report) maps them to the probability of *not* being desk
rejected. The same profile can optionally be attached to a full review as
venue context.

## Development

```bash
uv sync --extra dev
uv run pytest -q        # 114 tests: budget/fit units + pipeline e2e + regressions
uv run ruff check src tests
uv run black --check src tests
```

## License

MIT — see [LICENSE](LICENSE).
