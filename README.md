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
- **Cost control** — per-call usage ledger, cost estimation from a pricing
  table, hard spend cap per review
- **PWA** — installable web app (htmx + Alpine, no build step), live SSE
  progress, resumable runs, JSON/Markdown/**PDF** report export

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

Interrupted runs resume from `~/.open-referee/runs/<id>/` — stage artifacts
are persisted after every step.

## Development

```bash
uv sync --extra dev
uv run pytest -q        # 45 tests incl. full-pipeline e2e + regression suite
uv run ruff check src tests
uv run black --check src tests
```

## License

MIT — see [LICENSE](LICENSE).
