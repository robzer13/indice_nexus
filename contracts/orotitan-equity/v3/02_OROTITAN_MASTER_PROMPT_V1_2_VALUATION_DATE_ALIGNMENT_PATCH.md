# 02_OROTITAN_MASTER_PROMPT_V1.2 - VALUATION DATE ALIGNMENT PATCH

**Status:** FROZEN - V1.2
**Base incorporated by reference:** 02_OROTITAN_MASTER_PROMPT_V1_1_ECONOMIC_SHARE_COUNT_PATCH @ SHA256 2447ad836d9507eb82db884b1bdb1e0062d1512526d12afa4593339ce51dd0df
**Scope:** VALUATION_DATE_ALIGNMENT only
**Methodology change:** YES - narrow global timing/date repair
**Company-specific exception:** FORBIDDEN

The V1.1 Master Prompt remains executable in full except for date clauses explicitly superseded by `OROTITAN_VALUATION_DATE_ALIGNMENT_AUTHORITY_V1_FREEZE_V1.0`.

For valuation-date alignment, authority order is:

1. `OROTITAN_VALUATION_DATE_ALIGNMENT_AUTHORITY_V1_FREEZE_V1.0`;
2. `OROTITAN_DCF_TIMING_AUTHORITY_V1_FREEZE_V1.0` for DCF temporal mechanics not superseded by the alignment authority;
3. `OROTITAN_ECONOMIC_SHARE_COUNT_METHODOLOGY_V1_FREEZE_V1.0` for denominator definition, exactness, bounds and propagation, excluding its superseded date clauses;
4. incorporated base methodology clauses not superseded above;
5. execution, stage, integration and schema layers.

Canonical valuation denominator timing is:

```text
VALUATION_DATE = DATA_CUTOFF
SHARE_COUNT_AS_OF_DATE = VALUATION_DATE
PER_SHARE_OUTPUT_DATE = VALUATION_DATE
```

`REFERENCE_PRICE_DATE` remains a separate admitted observed-market date and never moves `VALUATION_DATE`.

All formulas, thresholds, scores, Certification rules, Investment Policy and terminal gates remain unchanged.
