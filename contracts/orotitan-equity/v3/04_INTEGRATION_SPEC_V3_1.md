# 04_INTEGRATION_SPEC_V3.1 - VALUATION DATE ALIGNMENT OVERLAY

**Status:** FROZEN DESIGN - V3.1
**Base:** 04_INTEGRATION_SPEC_V3 @ SHA256 2ab3923f18973cad469510c13906a0641275f2d112c43cbdf7a9047975395e39
**Method authorities:** OROTITAN_ECONOMIC_SHARE_COUNT_METHODOLOGY_V1_FREEZE_V1.0 + OROTITAN_VALUATION_DATE_ALIGNMENT_AUTHORITY_V1_FREEZE_V1.0
**Integration creates methodology:** NO

All V3.0 projection rules remain unchanged except the date relationship superseded below.

## 1. Economic-share-count projection

`v3_methodology.economic_share_count.share_count_as_of_date` must equal the snapshot `DATA_CUTOFF` / governed `VALUATION_DATE`, not `REFERENCE_PRICE_DATE`.

For `BOUNDED`, the bound effective date is the same valuation date. Exact/bounded/unknown representation semantics, no-scalarization rules and range propagation remain unchanged.

## 2. Valuation timing projection

A V3.1 analysis-capable snapshot additionally carries `v3_methodology.valuation_timing` with:
- methodology_version;
- valuation_date;
- time_origin;
- reference_price_date;
- economic_share_count_date;
- share_count_as_of_date;
- ev_to_equity_bridge_date;
- market_cap_date;
- per_share_output_date.

Cross-field equality is validated by the I3-B successor, not inferred by JSON Schema.

## 3. Required date relationships

For every analysis-capable V3.1 snapshot:

```text
valuation_date = data_cutoff
time_origin = valuation_date
economic_share_count_date = valuation_date
share_count_as_of_date = valuation_date
per_share_output_date = valuation_date
reference_price_date <= valuation_date
market_cap_date = reference_price_date when a numeric reference price exists
ev_to_equity_bridge_date = valuation_date for FCFF
```

For FCFE or supported Owner Earnings equity DCF, `ev_to_equity_bridge_date = NOT_APPLICABLE`.

## 4. Fail-closed semantics

Integration does not invent a missing date, denominator bridge or EV bridge.

Any mismatch, hidden stale-denominator transport, hidden scalarization or unresolved required date maps to upstream block; Integration may not patch it to make the payload validate.

## 5. Historical compatibility

Historical snapshots and runs remain immutable. V3.1 fields are never backfilled into prior snapshots solely for shape compatibility.
