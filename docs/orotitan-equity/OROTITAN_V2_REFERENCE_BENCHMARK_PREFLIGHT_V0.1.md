# OROTITAN V2 REFERENCE BENCHMARK — PREFLIGHT V0.1

STATUS = BENCHMARK_INFRASTRUCTURE_READY  
DATE = 2026-10-04  
ENGINE = `1116ca12dce2d30ddbb4b699945d92ae235c940cbf69d104a01019fc21efbf5e`

## 1. CORPUS

The first reference corpus contains 20 published V2 analyses covering all 10 sectors represented in the current published universe.

Stress coverage includes:

- banking;
- insurance;
- asset management;
- payment networks/processors;
- range-valued valuation outputs;
- recurring subscription;
- semiconductors;
- luxury;
- medical devices;
- pharma/biotech;
- energy;
- real estate;
- transaction-fee models;
- both ALLOWED and CONDITIONAL score permission;
- LOW and MEDIUM valuation reliability.

OQS range in the corpus:

- minimum: STMicroelectronics = 61.5;
- maximum: BELIMO Holding AG = 85.75.

## 2. HISTORICAL I2 PREFLIGHT

Authoritative I2 reconciliation reports found: **20/20**.

Historical reconciliation result:

```text
PASS = 20
FAIL = 0
PASS RATE = 100%
```

This is a consistency check, not yet an external calculation-accuracy benchmark.

Historical report formats differ. Depending on the run, PASS may be represented by:

- exact reconciliation;
- exact match;
- all deterministic outputs equal;
- candidate field PASS;
- zero absolute deltas;
- tolerance-based numerical reconciliation.

The benchmark normalization layer now handles these variants explicitly.

## 3. WHAT IS NOT YET MEASURED

True LLM reproducibility is **not yet measured**.

Current repeated independent executions:

```text
0 / 200
```

Phase A requires:

```text
20 companies
× 10 frozen-artifact reruns
= 200 executions
```

The repository currently contains no model execution runner. There is no existing OpenAI / AI SDK / generateText / streamText invocation path to reuse.

Therefore no reproducibility percentage is reported yet.

## 4. METRICS READY

The evaluator can already compute:

- score mean absolute deviation;
- exact score agreement rate;
- per-dimension stability;
- certification agreement / flip rate;
- terminal-verdict agreement / flip rate;
- I2 pass rate;
- evidence precision / recall / F1;
- forecast MAE / bias / RMSE;
- Spearman rank correlation.

## 5. TWO BENCHMARK LAYERS

### Phase A — Frozen artifact replay

Purpose: isolate judgment/scoring variability.

Inputs:

- frozen authoritative Deep Dive artifacts;
- no new web research;
- no publication;
- no canonical mutation.

### Phase B — Frozen source replay

Purpose: measure retrieval/extraction/evidence variability.

Blocked until a point-in-time raw source pack and human-reviewed gold labels are frozen.

## 6. CURRENT CONCLUSION

The benchmark infrastructure is ready, but the V2 engine's LLM reproducibility has not yet been empirically measured.

The next blocker is operational, not methodological:

> build or connect an independent model execution runner capable of performing the 200 isolated benchmark reruns while recording model/version/configuration and artifact hashes.

No V3 scoring methodology should be declared superior until this V2 baseline is executed.
