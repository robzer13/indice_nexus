# OROTITAN_ENGINE_BENCHMARK_SPEC_V0.1

STATUS = DRAFT_NON_AUTHORITATIVE  
PURPOSE = MEASURE ENGINE RELIABILITY, NOT COMPANY QUALITY

---

## 1. CENTRAL QUESTION

OroTitan must be able to answer:

```text
How reproducible is this engine?
How often are its factual claims correct?
How often are its calculations correct?
How well are its forecasts calibrated?
```

No single composite "engine score" is authorized in v0.1.

---

## 2. FOUR SEPARATE BENCHMARK AXES

### A. REPRODUCIBILITY

Same point-in-time evidence package is executed repeatedly.

Required metrics:

- exact primitive agreement rate;
- categorical-state agreement;
- dimension-score mean absolute deviation;
- OQS mean absolute deviation;
- verdict flip rate;
- certification flip rate;
- evidence-selection overlap;
- source-attribution overlap.

For categorical states use an agreement statistic suitable for multiple runs (for example Fleiss' kappa where applicable).

Target principle:

```text
IDENTICAL INPUT
→ IDENTICAL PRIMITIVES
→ IDENTICAL DETERMINISTIC SCORE
```

If deterministic scoring is implemented correctly, numerical score variance after primitive lock should be zero.

### B. EVIDENCE ACCURACY

Create a frozen gold dataset of claims and sources.

Measure:

- factual precision;
- factual recall;
- supported-claim rate;
- unsupported-claim rate;
- source-attribution precision;
- source-class accuracy;
- independence-group accuracy;
- contradiction-detection precision/recall;
- material-evidence omission rate.

Every benchmark item must preserve point-in-time availability.

### C. CALCULATION ACCURACY

Gold calculation cases cover:

- ROIC;
- ROIIC;
- FCF;
- owner-earnings states;
- valuation math;
- expected return;
- score caps;
- I2;
- price ladders.

Measure:

- exact-match rate;
- absolute numerical error;
- formula-selection error;
- invalid-state handling;
- unit/currency errors.

For deterministic calculations the target is 100% exact-match on valid benchmark cases.

### D. PREDICTIVE CALIBRATION

Historical forecasts are evaluated only after the outcome horizon matures.

Possible metrics:

- expected-return MAE;
- expected-return bias;
- calibration slope/intercept;
- realized-return rank correlation (Spearman);
- monotonicity by OVS bucket;
- monotonicity by Investment Score bucket;
- threshold hit rate;
- drawdown / downside-event calibration when forecasted;
- moat/runway state persistence.

No hindsight rewriting of the original forecast is allowed.

---

## 3. BENCHMARK CORPUS

Start with:

```text
PHASE 1
20 companies
× multiple sectors
× 10 repeated runs
= reproducibility corpus

PHASE 2
50 companies
= evidence + calculation gold set

PHASE 3
100+ historical point-in-time snapshots
= predictive backtest where legitimate
```

The corpus must include:

- easy disclosure;
- low disclosure;
- banks/insurers;
- serial acquirers;
- high-growth businesses;
- cyclicals;
- companies with known negative evidence;
- ambiguous/insufficient-data cases.

---

## 4. RUN MATRIX

For each benchmark case record:

- engine fingerprint;
- model/provider;
- model version;
- reasoning configuration where observable;
- prompt artifact version;
- evidence package hash;
- randomization/repetition index;
- execution timestamp;
- primitive outputs;
- calculation outputs;
- final scores;
- terminal verdict.

This makes model changes separable from methodology changes.

---

## 5. ENGINE COMPARISON

Compare generations only on the same frozen benchmark pack.

Example reporting:

| Metric | Engine A | Engine B |
|---|---:|---:|
| Primitive exact agreement | 84% | 97% |
| OQS MAE across repeats | 4.2 | 0.0 |
| Unsupported claim rate | 3.8% | 0.9% |
| Calculation exact match | 98.1% | 100% |
| Contradiction recall | 76% | 92% |

A newer engine is not "better" merely because its scores are higher.

---

## 6. CONFIDENCE INTERVALS AND SAMPLE SIZE

Every published benchmark metric must display:

- N cases;
- N executions;
- benchmark version;
- engine fingerprint;
- confidence interval where meaningful.

Small samples must be marked INSUFFICIENT_SAMPLE.

---

## 7. PREDICTIVE BENCHMARK GUARDRAILS

Predictive testing is especially vulnerable to leakage.

Required:

- freeze the historical data cutoff;
- exclude evidence unavailable at that cutoff;
- freeze the historical reference price;
- preserve the original valuation assumptions;
- never use later restatements silently;
- report survivorship and selection bias;
- separate backtest from genuinely forward-held predictions.

Forward-held predictions should receive greater evidentiary weight than reconstructed backtests.

---

## 8. FIRST ENGINE QUALITY DASHBOARD

Initially expose four separate measures:

```text
REPRODUCIBILITY
EVIDENCE_ACCURACY
CALCULATION_ACCURACY
PREDICTIVE_CALIBRATION
```

Example:

```text
Engine fingerprint: 1116ca12…
Reproducibility: 98.7% (N=200 runs)
Evidence accuracy: 97.4% (N=1,240 claims)
Calculation accuracy: 100% (N=410 calculations)
Predictive calibration: INSUFFICIENT_HISTORY
```

Do not average these into one number in v0.1.

---

## 9. FAILURE TAXONOMY

Every benchmark failure should be classed:

```text
RETRIEVAL_ERROR
SOURCE_SELECTION_ERROR
FACT_EXTRACTION_ERROR
TEMPORAL_LEAKAGE
PRIMITIVE_CLASSIFICATION_ERROR
CONFLICT_HANDLING_ERROR
CALCULATION_ERROR
RULE_ENGINE_ERROR
SCHEMA_ERROR
TRACEABILITY_ERROR
FORECAST_ERROR
```

This makes engine improvement actionable.

---

## 10. INITIAL SUCCESS CRITERIA FOR NEXT ENGINE FREEZE

Candidate targets to validate, not yet frozen:

- deterministic numerical scoring after primitive lock: 100%;
- calculation exact-match: 100% on valid gold cases;
- unsupported material claim rate: <1%;
- material source-attribution accuracy: >99%;
- primitive agreement: >95%;
- terminal-verdict flip rate on identical inputs: <1%.

Targets must be reviewed after the first benchmark to avoid arbitrary thresholds.

---

## 11. RELATION TO CURRENT OROTITAN

Current canonical snapshots remain valid historical records.

The benchmark evaluates the engine that produced them; it does not retroactively rewrite them.

When a new engine is frozen:

```text
OLD SNAPSHOT
= HISTORICAL OUTPUT OF OLD ENGINE

NEW REFRESH
= NEW OUTPUT OF NEW ENGINE
```

Both remain comparable through their fingerprints.
