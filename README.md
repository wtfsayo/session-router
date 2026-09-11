# session-router

Session-aware, privacy-first router between a local small LLM and cloud
LLMs. Routes each turn to local or cloud while balancing quality, cost,
latency, prompt-cache reuse, and PII — and stays sticky within a session
so model switches don't keep busting warm caches.

## Design in one paragraph

Per turn: scan for PII (strict tiers never leave the device; redactable
PII is replaced with session-stable placeholders before any cloud call and
restored on return) → estimate `P(local suffices)` → apply the session
policy (hard locks for tool loops, idle/drift reset boundaries, sticky
escalation, margin or work-function decision rule priced with cache
warmth) → local-first generate → composite acceptance gate (validators +
answer-span logprob confidence) → escalate to cloud on rejection, with
transcript handoff that only pays the divergence prefill.

## Install / run

```bash
uv run --with-editable . session-router demo          # simulated session
uv run --with-editable . session-router serve --port 8400 \
    --cloud-key $OPENAI_API_KEY                        # OpenAI-compatible API
uv run --with-editable . session-router chat "what is 2+2?"
uv run --with-editable . --with pytest pytest tests   # test suite
```

Sessions: pass `metadata.session_id` (or `X-Session-Id`) on
`POST /v1/chat/completions`. Responses include a `_router` diagnostics
block (chosen model, reason, privacy tier, scores, cache estimate).

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
