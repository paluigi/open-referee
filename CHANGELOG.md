# Changelog

## [Unreleased]

### Added — verification-mechanics rework (decomposition approach)

- **Claim inventory stage** (strong model): exhaustive catalogue of
  atomically checkable claims — theorem/proof pairs, equations, numerical
  statements, table/figure–text pairs, citation–claim pairs, cross
  references, assumption uses — each with verbatim components and anchors.
- **Per-claim verification workers**: one dedicated LLM call per claim
  (strong model re-derives math/theory; small model handles mechanical
  checks), each with an isolated *definitions pack* built from extracted
  theorem environments (statement + only the definitions its symbols match)
  instead of whole-section context.
- **Theorem/proof extraction**: theorem environments parsed out of the
  document and paired with their proofs.
- **Table extraction**: PDF tables via pymupdf `find_tables` (parsed cells +
  rendered region image) and markdown pipe tables; block→page mapping.
- **Artifact verification**: EVERY figure and EVERY table verified by the
  vision model against the prose passages mentioning it — plotted values vs
  claims, caption accuracy, significance stars vs reported SEs (t-ratios
  recomputed where possible), axis/unit errors, internal table consistency.
- **Defense–adjudication gate**: each surviving comment gets an authors'
  defense (small model) and an adjudication (strong model) that
  upholds/adjusts/dismisses and recalibrates severity — precision-first.
- **Depth presets** (`fast`/`standard`/`deep`) controlling claim budgets,
  lenses, whole-paper passes, defense rounds, artifact caps; selector in the
  settings UI.
- CLAIMS stage is resumable like all others.

(Evaluation against a reference corpus happens offline in a separate
private harness; not part of this repository.)

### Added — whole-paper review passes

Two new passes ensure consistency *across* sections (single-section lenses
cannot see cross-section drift):
- **Cross-section coherence referee** (strong model, full manuscript) after
  the section lenses: abstract/intro vs results overclaiming, contribution
  vs delivery, numbers/notation drifting across sections, setup assumptions
  honored in later analyses, results contradicting other results,
  conclusion vs evidence, cross-section promises.
- **Whole-paper challenger** (strong model, full manuscript) after the
  section challengers: validates the coherence candidates (those not claimed
  by any section challenger) and hunts missed global weaknesses
  (overclaiming gap, silent assumption violations, the single change that
  would most undermine the contribution).

### Changed — specialized referee prompts

- All agents now share an explicit **referee persona** (severe, rigorous, fair
  — top-journal referee) and a common **severity rubric** (0.8+ only for
  result-invalidating flaws).
- Triage additionally classifies `mathematical_density` and
  `statistical_content` and inventories `novelty_claims`.
- Verification now runs **specialized lenses per section**, selected by triage:
  - *math/theory*: statement–proof fit, hidden assumptions, WLOG claims, case
    coverage, index/dimension bookkeeping, algebra verification, circularity,
    limiting arguments;
  - *statistics/econometrics*: identification, design-specific threats
    (DiD/IV/RDD/panel/event studies), inference (clustering, MHT), data
    quality, table-vs-text consistency, robustness, overreach;
  - *prose coherence*: contradictions, cross-reference errors, drifting
    quantities, definitions, promises kept, logical flow;
  - *literature coherence*: attribution accuracy vs context-pack evidence,
    novelty-claim checks, missing engagement with author-supplied work,
    citation-claim mismatch.
- Citation audit upgraded to two dimensions: **existence** (verified /
    year/title/author mismatch / not_found / suspicious) and **quotation
    consistency** (in-text citation claims vs cited work's abstract), fed by a
    new in-text citation extractor (author-year, numeric, natbib styles).
- Meta-reviewer now writes thematic sections ordered by importance, includes a
  strengths paragraph, and ends with a calibrated recommendation
  (accept / minor / major / reject).
- Validator drops unsubstantiated comments ("a fabricated criticism is worse
  than a missed one"), merges duplicates, and checks the report's
  recommendation against the severity distribution.

### Fixed (PR #1 review round)

- **`open-referee serve` production crash**: `create_app()` never set
  `app.state.config_path`, so every page returned 500 outside tests. It now
  defaults to `DEFAULT_CONFIG_PATH`.
- **Survey stage crash**: the surveyor's Crossref missing-reference lookups ran
  after the `async with CrossrefClient` block had exited (`RuntimeError:
  client has been closed`). The surveyor step now runs inside the client
  contexts; OpenAlex/Crossref/SearchRouter are opened once per stage via
  context managers.
- **OpenAlex client**: `_work_to_item` read nonexistent `openalexpy.Work`
  attributes (`authors`, `year`, `citations`) — authors/year/citation counts
  were silently dropped. Now maps `authorships`, `publication_year`,
  `cited_by_count`. The REST fallback client is always created (previously it
  existed only when openalexpy was missing, so an openalexpy failure crashed
  with `AttributeError`). DOIs are no longer truncated to their suffix.
- **SSE replay**: `RunState.events` was excluded from the persisted state, so
  replaying finished runs yielded nothing and always emitted a synthetic
  "done" — even for failed runs. Events are now persisted; replay ends with
  the run's real terminal status; late subscribers to finished runs get the
  buffered events instead of hanging; finished background-task references are
  bounded (max 50).
- **Budget cap behavior**: `BudgetExceeded` now drains into a partial report
  (with validator notes listing completed stages) instead of discarding
  everything. Output-token cost estimation is proportional to input size
  instead of pre-charging `max_tokens // 2` per call, which tripped the cap
  prematurely on large-context models.
- **Settings save**: hand-edited custom providers in the YAML are preserved
  (the form only manages the five well-known provider names). Settings save
  also reads the app's config path instead of the global default.
- **Upload security**: upload filenames are sanitized (basename + character
  allowlist), neutralizing path traversal via crafted multipart filenames.
- **Sections**: `Section.path` was built from a string title (`list(title)` →
  char list). Now a proper hierarchical path list.
- **Service worker**: registered from `/static/sw.js`, its default scope
  (`/static/`) controlled no app pages, so offline caching never worked.
  Now served at `/sw.js` with `Service-Worker-Allowed: /`.
- **Figure context**: figure verification used the last 60 blocks regardless
  of figure position; now centered on the caption block when available.
- Removed dead config (`ReviewConfig.language`, `SECTIONS_PER_VERIFIER_CHUNK`),
  a duplicate `ModelSpec` import, and an odd `cfg and name` expression.

### Tests

- New `tests/test_regression_fixes.py` covering every fix above (17 tests,
  total suite: 45 passing), including the production `create_app()` path with
  no test-injected state.
