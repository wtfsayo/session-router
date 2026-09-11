# session-router

**Session-aware LLM router: route each turn between a local small LLM
(Ollama / llama.cpp) and cloud LLMs (OpenAI / Anthropic / OpenRouter) —
balancing quality, cost, latency, prompt-cache reuse, and PII privacy.**

An OpenAI-compatible inference gateway that decides, per turn, whether the
cheap local model suffices or the request must escalate — and stays sticky
within a session so model switches don't keep busting warm KV caches.

Built for agentic / multi-turn workloads where per-request routers lose:
tool loops, long sessions, prompt-caching economics, and privacy gating.

- **4-tier model ladder** — local → cloud tiers, priced per model
  (incl. cache read/write pricing), not just a binary cheap/expensive split
- **PII gate before any egress** — STRICT turns never leave the device;
  redactable PII is replaced with session-stable placeholders and restored
  locally on return
- **Cache-aware switching costs** — prompt-cache TTL, warm-token estimates,
  and re-prefill cost are priced into every route decision
- **Decision rules**: greedy margin, work-function (MTS), or satisficing
  (`P(suffices) ≥ τ`) with sticky escalation, idle/drift resets, and
  tool-loop hard locks
- **Cascade acceptance gate** — validators + answer-span logprob
  confidence; escalate to a stronger model on rejection, paying only the
  divergence prefill
- **Drop-in proxy** — OpenAI-compatible `POST /v1/chat/completions`;
  sessions via `metadata.session_id` or `X-Session-Id`
- **Zero dependencies** — core is stdlib-only Python

## Quickstart

```bash
uv run --with-editable . session-router demo          # simulated session
uv run --with-editable . session-router serve --port 8400 \
    --cloud-key $OPENAI_API_KEY                        # OpenAI-compatible API
uv run --with-editable . session-router chat "what is 2+2?"
uv run --with-editable . --with pytest pytest tests   # test suite
```

Responses include a `_router` diagnostics block (chosen model, reason,
privacy tier, candidate scores, cache estimate) — shadow-evaluate routing
decisions without changing behavior.

## How it decides

Per turn: scan for PII → estimate per-tier sufficiency
`P(tier t suffices)` → apply session policy (hard locks for outstanding
tool calls and non-portable provider state; idle + topic-drift reset
boundaries; sticky escalation; `agentic:hold` on mid-loop observation
steps) → pick lowest sufficient tier (satisfice), min cost+switch
(margin), or work-function state (WFA) → generate → accept or escalate.

## Trained scorer

The default `HeuristicScorer` is a cold-start placeholder. Train a real
head on labeled routing data:

```bash
uv run --with pandas --with pyarrow --with huggingface_hub \
    --with scikit-learn python bench/train_scorer.py \
    --out artifacts/scorer.pkl
session-router serve --scorer artifacts/scorer.pkl
```

`bench/train_scorer.py` trains on TwinRouterBench step labels (optionally
merged with RouterBench outcome rows). Any estimator with
`predict_proba` over `sessionrouter.features.extract_features` works;
bring your own labels by training on your traffic.

## Benchmarks

Static track evaluated against **TwinRouterBench**
([Amorph/TwinRouterBench](https://huggingface.co/datasets/Amorph/TwinRouterBench),
official `compute_v2_scores` — RowPass / RowExact / TrajPass / CostSave,
session-level 70/30 split, 290 held-out steps):

| Router | RowPass | RowExact | TrajPass | CostSave | Combined |
|---|---|---|---|---|---|
| trained head → τ-satisfice (this repo) | 96.6 | 78.3 | 92.1 | 57.0 | **81.1** |
| real `Router` w/ session policy | 96.6 | 77.6 | 92.1 | 56.1 | **80.6** |
| paper best (SR-KNN, *in-sample* UB) | 91.9 | 78.8 | 84.7 | 56.2 | 77.9 |
| always-low / always-high | 74.1 / 100 | 74.1 / 16.2 | 59.7 / 100 | 55.3 / 0 | 65.8 / 54.1 |

`bench/hillclimb.py` sweeps (features, head, decision rule, τ) against the
official scorer and logs every run to `bench/experiments.sqlite`.
`bench/dynamic_router.py` adapts the trained scorer to the live
mini-SWE-agent track (`miniswerouterbench run --router-import ...`);
validated end-to-end, not yet run at the 100-case scale.

```bash
TRB_REPO=/path/to/TwinRouterBench \
uv run --with pandas --with pyarrow --with huggingface_hub \
    --with scikit-learn --with tiktoken --with scipy \
    python bench/twinrouter.py                       # 4-tier official metrics
uv run --with pandas --with scikit-learn \
    python bench/run_bench.py                        # RouterBench + session sim
```

## Backends

Ollama (`keep_alive`, logprobs) · llama.cpp server (`id_slot`,
`cache_prompt`, slot save/restore) · OpenAI-compatible
(`prompt_cache_key`, `cached_tokens`) · Anthropic (`cache_control`
breakpoints, read/write usage fields) · OpenRouter via OpenAI-compat.

## Privacy notes

- The built-in detector is regex+Luhn heuristics (email, phone, SSN,
  payment card, IBAN, API keys, IPs). It is a floor, not a ceiling —
  plug a real recognizer (Presidio, GLiNER-PII) via the `detectors` hook
  in `pii.detect`.
- STRICT turns never reach a cloud backend, and a transcript that has
  ever contained strict PII keeps the whole session local (the history
  itself would leak it). Recovery requires bounded-context escalation —
  not yet implemented.
- Restore maps live only in `Sanitizer` memory; they are never sent or
  logged.

## Honest limitations

- No commercial API accepts externally supplied KV state — cross-model
  cache transfer is only viable between self-hosted same-family models
  (ridge-mapper style). The default handoff is transcript + prefix cache.
- Gate logprob thresholds are placeholders; fit them on your local model.
- `quality_weight` is a dollars-vs-quality scale knob: ~0.05 for
  ~$0.005-per-call clouds, lower for cheaper endpoints.

## License

MIT
