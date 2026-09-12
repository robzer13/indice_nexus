# `04_INTEGRATION_SPEC_V1.md`
## OroTitan Equity Research - Phase 4 / Screener Data Contract
### Status

```text
PHASE 4
= DATA CONTRACT ONLY

PHYSICAL SCREENER IMPLEMENTATION
= NOT STARTED
```
## 1. Governing contract

The contract implements only the projection boundary:

```text
MASTER PROMPT / CERTIFIED DOSSIER
= SOURCE OF ANALYTICAL TRUTH

SCREENER
= STRUCTURED PROJECTION
```

No score, gate, valuation method, portfolio rule, trading rule or research method is created here. The Phase-3 regression established that the patched Master Prompt preserves the frozen scoring, certification, sector and terminal systems. The canonical patched artifact hash is recorded in the JSON Schema metadata.
## 2. Four-layer model

```text
L1 IDENTITY
L2 RESEARCH / FUNDAMENTALS
L3 INVESTMENT / VALUATION
L4 OPERATIONAL STATE
```

L1 is shared across snapshots. L2-L4 are versioned inside each research snapshot. `CURRENT_MARKET_PRICE` is deliberately outside the research snapshot because a market tick must never rewrite the research reference price.
## 3. Minimal canonical objects

V1 uses seven implementation objects, not eleven separate tables or services:

```text
L1_IDENTITY
  issuer
  securities[]
  research_dossier

CURRENT_SNAPSHOT
HISTORICAL_SNAPSHOTS[]
CURRENT_MARKET_PRICE
ACTIVATION_EVENTS[]
```

Inside each analysis snapshot, L2 contains fundamentals/certification/business quality, L3 contains valuation/investment, and L4 contains terminal/readiness/next-action state. Price Ladder and OroTitan gates remain nested in the snapshot because they are version-bound analytical outputs. This is an implementation grouping only.
## 4. Point-in-time and versioning

Every snapshot requires `DATA_CUTOFF`. A numeric research reference price requires `REFERENCE_PRICE_DATE`. `REPORT_VERSION`, `METHOD_VERSION`, `CALCULATION_VERSION`, `EVIDENCE_LEDGER_VERSION` and the full frozen method-version set are retained. Historical snapshots are append-only. `research_dossier.current_snapshot_id` points to the latest valid projection, while prior snapshots remain immutable.

A price feed update changes `CURRENT_MARKET_PRICE`, not `current_snapshot.data_lock.reference_price`. A market-zone crossing can create an `activation_event`, but the event explicitly carries `market_event_is_certified_snapshot = false`.
## 5. Null / unknown / N/A semantics

V1 deliberately avoids overloaded JSON `null` in canonical semantic fields. Where a scalar may legitimately be unavailable, the contract uses the exact explicit tokens:

```text
UNKNOWN
NOT_APPLICABLE
NOT_ASSESSABLE
MISSING
NOT_AVAILABLE
```

No such state is encoded as `0` or an empty string. Optional whole objects are omitted rather than silently null-filled. This preserves the distinction between unknown evidence, an inapplicable sector metric, an unassessable valuation, missing required work and a price zone that is structurally unavailable.
## 6. Stored versus recomputable

```text
STORED / TRACEABLE JUDGMENT
= dimension judgments, fundamental states, certification, Elite gate judgments, readiness, opportunity path, next action

DETERMINISTICALLY RECOMPUTABLE
= OQS_RAW, WEAK_LINK_CAP, OQS, OVS, INVESTMENT_RAW, INVESTMENT_SCORE, OROTITAN_STATUS, potential OroTitan max price when all four components exist
```

The schema embeds the frozen formulas as read-only annotations. JSON Schema validates structure and logical state compatibility. Arithmetic equality is validated by the single canonical server-side computation layer because standard JSON Schema does not express cross-field arithmetic. No frontend formula is authoritative.
## 7. Canonical mapping table

| MASTER_PROMPT_OUTPUT | SCHEMA_FIELD | TYPE / UNIT | STORAGE RULE | UPDATE RULE | HISTORY RULE |
|---|---|---|---|---|---|
| ISSUER_ID | l1_identity.issuer.issuer_id | string | STORED FROM RESEARCH / identity registry | Update only by authoritative identity resolution; never by ticker guess | Issuer identity persistent; dated corporate changes outside snapshot |
| SECURITY_ID | l1_identity.securities[].security_id | string | STORED FROM IDENTITY REGISTRY | Append/retire security records; do not merge with issuer | Security identity historical |
| LEGAL_NAME | l1_identity.issuer.legal_name | string | STORED | Update on authoritative legal-name change | Identity history preserved |
| DISPLAY_NAME | l1_identity.issuer.display_name | string | STORED | Presentation-safe update | Current identity field |
| TICKER / ISIN / EXCHANGE | l1_identity.securities[] | string | STORED | Security-level only | Historical security lines preserved |
| COUNTRY / REPORTING_CURRENCY | l1_identity.issuer.* | string | STORED | Authoritative issuer update | Snapshot uses issuer reference |
| PRIMARY_LISTING | l1_identity.securities[].primary_listing | boolean | STORED | Security normalization only | Historical security state preservable |
| LISTING_STATUS | l1_identity.issuer.listing_status + securities[].listing_status | string | STORED | Authoritative listing event | Never converts technical disposition into investment rejection |
| CANDIDATE_EPISODE | l1_identity.research_dossier.candidate_episode | string | STORED | New episode only on frozen IR-04 transformation logic | Old episode remains historical |
| DATA_CUTOFF | current_snapshot.data_lock.data_cutoff | date | STORED | Immutable within snapshot | Mandatory on every historical snapshot |
| REFERENCE_PRICE | current_snapshot.data_lock.reference_price | price/state | STORED | Immutable within snapshot | Historical research price retained |
| REFERENCE_PRICE_DATE | current_snapshot.data_lock.reference_price_date | date/state | STORED | Paired with research reference price | Historical |
| CALCULATION_DATE | current_snapshot.data_lock.calculation_date | date | STORED | New calculation -> new/versioned snapshot | Historical |
| LAST_FULL_RESEARCH_DATE | current_snapshot.data_lock.last_full_research_date | date/state | STORED | Changes only after full research/refresh | Historical |
| REPORT_VERSION / METHOD_VERSION / CALCULATION_VERSION / EVIDENCE_LEDGER_VERSION | current_snapshot.versions.* | string | STORED | Append/version, never silent overwrite | Historical |
| DISCOVERY_STATUS | ...l2_research_fundamentals.discovery_record.discovery_status | enum | STORED FROM DISCOVERY | New dated Discovery decision | Historical |
| DISCOVERY_METHODS[] | ...discovery_record.discovery_methods[] | enum[] | STORED | Append evidence routes; no arithmetic bonus | Historical |
| ARCHETYPE_HINTS[] | ...discovery_record.archetype_hints[] | enum[] | STORED | Hypotheses only | Historical |
| ATTENTION_STATE | ...discovery_record.attention_state | enum | STORED | May change research priority only | Historical |
| EVIDENCE_GRADE | ...discovery_record.evidence_grade | enum | STORED | New evidence may create new snapshot | Historical |
| DATA_SUFFICIENCY | ...discovery_record.data_sufficiency | enum | STORED | New evidence may change | Historical |
| HARD_KILL_STATUS | ...discovery_record.hard_kill_status.HK-A..HK-E | object of frozen enums | STORED | Causal evidence update only | Historical |
| VALUATION_SANITY | ...discovery_record.valuation_sanity | enum | STORED | Discovery only, no DCF | Historical |
| RESEARCH_PRIORITY | ...discovery_record.research_priority | enum | STORED | Expected value of further research only | Historical |
| WATCH_TYPE / REACTIVATION_TRIGGER / REJECTION_REASON | ...discovery_record.* | enum/string | STORED | Trigger/causal update only | Historical |
| MOAT_EVIDENCE_STATE / MOAT_TREND / MOAT_DURABILITY | ...fundamental_states.moat_* | enum | STORED JUDGMENTS | Refresh only with new evidence | Historical |
| RUNWAY_EVIDENCE_STATE / RUNWAY_MAGNITUDE / RUNWAY_HORIZON | ...fundamental_states.runway_* | enum | STORED JUDGMENTS | Refresh only with new evidence | Historical |
| ROIC_TREND | ...fundamental_states.roic_trend | enum | STORED JUDGMENT | New normalized history may change | Historical |
| FORENSIC_RELIABILITY | ...fundamental_states.forensic_reliability | enum | STORED JUDGMENT | Refresh on forensic evidence | Historical |
| SECTOR_METHOD_STATUS | ...fundamental_states.sector_method_results[].status | enum | STORED / TRACEABLE | Apply all economically relevant overlays | Historical |
| STANDARD_ROIC / ALL_IN_ROIC / ROIC_EX_GOODWILL / R&D_ADJUSTED_ROIC / ROIIC | ...analytical_metrics.* | number/range/state | STORED CALCULATION OUTPUT | Recompute only with method-consistent inputs | Historical |
| STANDARDIZED_FCF / OWNER_EARNINGS / FCF_PER_SHARE / SHARE_COUNT_CAGR | ...analytical_metrics.* | number/range/state | STORED CALCULATION OUTPUT | Recompute on affected block | Historical |
| BUSINESS_RESEARCH_STATUS / INVESTMENT_CONCLUSION_STATUS / SCORE_PERMISSION | ...certification.* | enum | STORED JUDGMENTS | New certification decision only | Historical |
| HARD_BLOCKERS[] / MATERIAL_LIMITATIONS[] / CRITICAL_UNKNOWNS[] / UNRESOLVED_CONFLICTS[] / REQUIRED_FIXES[] | ...certification.* | object[] | STORED | Append/resolve through new snapshot; never hide past blocker | Historical |
| MOAT_SCORE ... RESILIENCE_RISK_SCORE | ...business_quality.*_score | 0-100 in 5-point increments or state/range | STORED DIMENSION JUDGMENTS | Change requires evidence/rationale/version | Historical |
| OQS_RAW | ...business_quality.oqs_raw | 0-100 | DETERMINISTICALLY RECOMPUTABLE | Server recompute from frozen weights | Historical snapshot stores result |
| WEAK_LINK_CAP | ...business_quality.weak_link_cap | 0-100 | DETERMINISTICALLY RECOMPUTABLE | Server recompute | Historical |
| OQS | ...business_quality.oqs | 0-100 | DETERMINISTICALLY RECOMPUTABLE | Server recompute; price-only delta cannot change it | Historical |
| QUALITY_CLASS | ...business_quality.quality_class | enum | DETERMINISTIC FROM OQS when numeric | Server derive/display | Historical |
| PRIMARY_EXPECTED_RETURN / RETURN_HORIZON | ...l3_investment_valuation.valuation.* | return / years | STORED CALCULATION OUTPUT | Refresh valuation or price-only delta as required | Historical |
| NO_MULTIPLE_EXPANSION_RETURN / MATURE_NORMALIZATION_RETURN | ...valuation.* | return/state | STORED CALCULATION OUTPUT | N basis: valid numeric Mature Normalization first; valid numeric Same-Multiple only when Mature is legitimately NOT_ASSESSABLE / NOT_AVAILABLE; invalid or unreconciled Mature fails closed | Historical |
| MARGIN_OF_SAFETY | ...valuation.margin_of_safety | enum | STORED JUDGMENT | Valuation refresh | Historical |
| VALUATION_RELIABILITY | ...valuation.valuation_reliability | enum | STORED JUDGMENT | Valuation refresh | Historical |
| MARKET_EXPECTATION_GAP | ...valuation.market_expectation_gap | enum | STORED JUDGMENT / reverse-DCF output | Valuation refresh | Historical |
| OVS | ...valuation.ovs | 0-100/state | DETERMINISTICALLY RECOMPUTABLE | Server only; `SCORE_PERMISSION = SUSPENDED` requires `NOT_AVAILABLE` when valuation is assessable, or `NOT_ASSESSABLE` when `VALUATION_RELIABILITY = NOT_ASSESSABLE` | Historical |
| INVESTMENT_RAW / INVESTMENT_SCORE | ...l3_investment_valuation.investment.* | 0-100/state | DETERMINISTICALLY RECOMPUTABLE | Server only | Historical |
| INVESTMENT_CLASS | ...investment.investment_class | enum | DETERMINISTIC FROM INVESTMENT_SCORE when numeric | Server derive/display | Historical |
| PRICE_FOR_REQUIRED_RETURN_H / PRICE_FOR_STRONG_RETURN / PRICE_FOR_EXCEPTIONAL_RETURN | ...valuation.price_ladder.* | price/state | STORED CALCULATION OUTPUT | Recompute with current certified fundamentals / hurdle inputs | Historical ladder snapshot |
| REQUIRED_RETURN_H / STRONG_RETURN_THRESHOLD / EXCEPTIONAL_RETURN_THRESHOLD | ...valuation.price_ladder.* | % p.a. | STORED POLICY INPUT | Locked by `OROTITAN_INVESTMENT_POLICY_V1.0.0`: 10.0% / 12.5% / 15.0% | Historical |
| INVESTABLE_PRICE_ZONE / STRONG_OPPORTUNITY_ZONE / POTENTIAL_OROTITAN_PRICE_ZONE | ...valuation.price_ladder.*_zone | zone/state | STORED DERIVED OUTPUT | Potential OroTitan zone only when all non-valuation gates pass | Historical |
| POTENTIAL_OROTITAN_MAX_PRICE | ...valuation.price_ladder.potential_orotitan_max_price | price/state | DETERMINISTICALLY RECOMPUTABLE if four components available | min(P_ER,P_NO_EXPANSION,P_MOS,P_IMPLIED) | Historical |
| OROTITAN_STATUS | ...l4_operational_state.orotitan.orotitan_status | YES/NO | DETERMINISTIC FROM GATES | Re-run terminal gate after relevant change | Historical |
| OROTITAN_GATE_RESULTS[] | ...orotitan.orotitan_gate_results[] | object[] | STORED TRACEABLE JUDGMENTS | Judgment changes require new evidence/version | Historical |
| OROTITAN_GAPS[] | ...orotitan.orotitan_gaps[] | object[] | STORED | Derived from failed decision-useful gates, no proximity score | Historical |
| DOSSIER_READINESS | ...l4_operational_state.dossier_readiness | enum | STORED OPERATIONAL JUDGMENT | Refresh as dossier completeness/staleness changes | Historical |
| OPPORTUNITY_PATH | ...l4_operational_state.opportunity_path | DIRECT/PREPARED | STORED | Operational path only, not quality | Historical |
| NEXT_ACTION | ...l4_operational_state.next_action | enum | STORED DECISION OUTPUT | Update after certified/activation decision | Historical |
| PRICE_LADDER_STATUS | ...l4_operational_state.price_ladder_status | enum | STORED | Current/stale/invalidated/not built | Historical |
| THESIS_INVALIDATION_TRIGGERS[] | ...l4_operational_state.thesis_invalidation_triggers[] | string[] | STORED | Update only through research/refresh | Historical |
| ACTIVATION_CHECK_REQUIRED | ...l4_operational_state.activation_check_required | boolean | DETERMINISTIC/OPERATIONAL | True on price-zone entry; never auto-BUY | Event/history |
| REFRESH_CLASS | ...l4_operational_state.refresh_class | enum | STORED EXECUTION CLASSIFICATION | PRICE_ONLY / ROUTINE / FULL | Historical |
| CURRENT_MARKET_PRICE | current_market_price.price | price | MARKET DATA, NOT RESEARCH SNAPSHOT | Can refresh frequently | Not allowed to rewrite historical research price |
| EVIDENCE_IDS[] / CALCULATION_IDS[] / REPORT_ID | current_snapshot.traceability.* | id arrays/string | STORED REFERENCES | Append/version through dossier | Historical |

## 8. Deterministic scoring contract

The schema does not store modifiable weights. The read-only contract metadata reproduces the frozen formulas. In server-side recomputation:

```text
OQS_RAW
= 0.20*MOAT + 0.15*RUNWAY + 0.20*RETURN_QUALITY
+ 0.10*CASH_ECONOMICS + 0.15*CAPITAL_ALLOCATION
+ 0.10*MANAGEMENT_GOVERNANCE + 0.10*RESILIENCE_RISK

WEAK_LINK_CAP
= MIN(APPLICABLE_DIMENSION_SCORES) + 25
  capped at 100

OQS
= MIN(OQS_RAW, WEAK_LINK_CAP)

INVESTMENT_RAW
= 0.70*OQS + 0.30*OVS

INVESTMENT_SCORE
= MIN(INVESTMENT_RAW, OQS, OVS + 15)
```

OVS retains the exact frozen ΔER anchor curve, linear interpolation, `N + 15` return-component cap, MOS cap and Valuation Reliability cap. Numeric `OVS` is invalid when `VALUATION_RELIABILITY = NOT_ASSESSABLE`. `SCORE_PERMISSION = SUSPENDED` also prohibits numeric `OVS`: use `NOT_ASSESSABLE` when `VALUATION_RELIABILITY = NOT_ASSESSABLE`, otherwise use `NOT_AVAILABLE`. These states are never encoded as `0` or `null`. Numeric `INVESTMENT_SCORE` requires numeric OQS and OVS.

Execution policy for V1 is versioned separately from the frozen economic methodology:

```text
POLICY_VERSION
= OROTITAN_INVESTMENT_POLICY_V1.0.0

REQUIRED_RETURN_H
= 10.0%

STRONG_RETURN_THRESHOLD
= 12.5%

EXCEPTIONAL_RETURN_THRESHOLD
= 15.0%
```

For OVS normalization basis selection, the canonical I2 execution layer applies exactly:

```text
1. valid numeric MATURE_NORMALIZATION_RETURN
   → N basis

2. otherwise, Mature Normalization legitimately NOT_ASSESSABLE / NOT_AVAILABLE
   + valid numeric NO_MULTIPLE_EXPANSION_RETURN
   → Same-Multiple fallback as N basis

3. otherwise
   → preserve the applicable Mature-Normalization unavailable semantic state
   → numeric OVS prohibited

INVALID / UNRECONCILED MATURE_NORMALIZATION_RETURN
→ FAIL CLOSED
→ no fallback
```

This selector is neither `MIN` nor `MAX`; when both returns are valid and numeric, Mature Normalization has priority regardless of relative value. I3-B validates policy consistency and reconciles the submitted deterministic outputs against I2; it does not create a second economic selector. The Phase-4 JSON payload shape is unchanged and no `n_basis`, `selected_return`, or equivalent field is added.
## 9. OroTitan terminal contract

`OROTITAN_STATUS` is outside scoring. The schema requires exactly ten terminal gate records:

```text
CERTIFICATION_GATE
MOAT_ELITE
RUNWAY_ELITE
RETURN_QUALITY_ELITE
CASH_ECONOMICS_ELITE
CAPITAL_ALLOCATION_ELITE
MANAGEMENT_GOVERNANCE_ELITE
RESILIENCE_ELITE
MATERIAL_WEAK_LINK_GATE
VALUATION_ELITE
```

Each carries `STATE`, `RATIONALE`, and `EVIDENCE_IDS[]`. `YES` is structurally valid only when all ten gates are `PASS`, business and investment research are both `CERTIFIED`, `SCORE_PERMISSION = ALLOWED`, and valuation reliability is `HIGH`. There is no proximity score.

`POTENTIAL_OROTITAN_PRICE_ZONE` is represented by a price-zone object only when all non-valuation gates pass. Otherwise it must be exactly `NOT_AVAILABLE`; in that state `POTENTIAL_OROTITAN_MAX_PRICE` must also be `NOT_AVAILABLE`.
## 10. Readiness, Direct/Prepared and activation

`DOSSIER_READINESS` remains exactly `READY / PARTIALLY_READY / NOT_READY`, with no readiness score. `OPPORTUNITY_PATH` remains `DIRECT / PREPARED` and is not rank-ordered. `NEXT_ACTION` remains the five frozen actions.

`READY` is schema-invalid if hard blockers or critical unknowns remain, the Price Ladder is not `CURRENT`, or the last full research date is unavailable. This implements the rapid-decision readiness boundary without changing the research method.

A current-price zone crossing may set `ACTIVATION_CHECK_REQUIRED = true`; it can never write a `BUY` state because no such canonical output exists.
## 11. Current versus history

The contract's current view is a pointer/projection over `current_snapshot`. History is `historical_snapshots[]`. A refresh creates a new snapshot/version; it does not update the prior snapshot in place.

For `PRICE_ONLY_DELTA`, the contract validator checks that OQS is unchanged versus the preceding analytical snapshot while valuation-dependent fields may change. This is a transition invariant, not a new score.
## 12. Evidence and calculation references

The Screener does not duplicate the full Evidence Ledger. Every snapshot carries `REPORT_ID`, `EVIDENCE_IDS[]`, and `CALCULATION_IDS[]`, while dimension/gate records can carry more granular evidence references. The report/evidence store remains authoritative for underlying proof.
## 13. Filtering / sorting use cases

The contract supports without new scores:

```text
highest OQS
highest Investment Score
QUALITY_CLASS = EXCEPTIONAL
NEXT_ACTION = INVESTABLE_NOW
NEXT_ACTION = WAIT_FOR_PRICE
DOSSIER_READINESS = READY
POTENTIAL_OROTITAN_PRICE_ZONE != NOT_AVAILABLE
OROTITAN_STATUS = YES
VALUATION_RELIABILITY = LOW / NOT_ASSESSABLE
BUSINESS_RESEARCH_STATUS = CERTIFIED_WITH_LIMITATIONS
MATERIAL_WEAK_LINK_GATE = FAIL
MOAT_TREND = STRENGTHENING
RUNWAY_HORIZON = EXTENDED
REFRESH_CLASS = FULL_REFRESH_REQUIRED
```

The important Prepared search is implemented by filtering high OQS + `READY` + `WAIT_FOR_PRICE` and comparing `CURRENT_MARKET_PRICE` with the Price Ladder. No `PROXIMITY_SCORE` is required or allowed.
## 14. Legacy / migration classification

The current physical Screener schema was not part of the supplied canonical input for this Phase-4 contract. Therefore no existing field is silently classified as `DIRECT_MAP`, `TRANSFORM`, `DEPRECATE`, `LEGACY_ONLY`, or `CONFLICT` without evidence. The migration classification must be performed later against the actual legacy field inventory.

This absence is not an `INTEGRATION_CONFLICT` in the data contract itself. A future `CONFLICT` is declared only if an existing field cannot represent the frozen concept without semantic loss. Methodology must never be changed to preserve a legacy column.
## 15. Contract tests C1-C12

| TEST | RESULT | PURPOSE |
|---|---|---|
| C1 | PASS | Discovery candidate only; no Deep-Dive scoring leakage |
| C2 | PASS | Certified exceptional business with valuation NOT_ASSESSABLE |
| C3 | PASS | Excellent investable non-OroTitan |
| C4 | PASS | Exceptional business, READY, WAIT_FOR_PRICE |
| C5 | PASS | Actual OroTitan reachability |
| C6 | PASS | Bank with industrial metrics NOT_APPLICABLE and sector method APPLIED |
| C7 | PASS | Insurer with industrial metrics NOT_APPLICABLE and sector method APPLIED |
| C8 | PASS | Critical blocker with SCORE_PERMISSION = SUSPENDED |
| C9 | PASS | Historical snapshot update preserves PIT history |
| C10 | PASS | PRICE_ONLY_DELTA preserves OQS |
| C11 | PASS | Market activation event is not a certified snapshot |
| C12 | PASS | Potential OroTitan zone unavailable when a non-valuation gate fails |

All twelve intended-valid payloads passed both Draft 2020-12 structural validation and the deterministic contract invariants used for cross-field arithmetic/history checks.

## 16. Invalid payload tests

| INVALID TEST | RESULT | INVALID STATE |
|---|---|---|
| I1_OROTITAN_YES_RUNWAY_FAIL | PASS | OROTITAN_STATUS = YES with RUNWAY_ELITE = FAIL |
| I2_OVS_WITH_VALUATION_NOT_ASSESSABLE | PASS | numeric OVS with VALUATION_RELIABILITY = NOT_ASSESSABLE |
| I3_OQS_WITH_SCORE_SUSPENDED | PASS | numeric OQS with SCORE_PERMISSION = SUSPENDED |
| I4_READY_WITH_CRITICAL_UNKNOWN | PASS | READY with critical unresolved input |
| I5_POTENTIAL_ZONE_WITH_MOAT_FAIL | PASS | potential OroTitan zone populated while MOAT_ELITE fails |
| I6_HISTORY_WITHOUT_DATA_CUTOFF | PASS | historical snapshot without DATA_CUTOFF |
| I7_REFERENCE_PRICE_WITHOUT_DATE | PASS | numeric REFERENCE_PRICE without valid REFERENCE_PRICE_DATE |
| I8_INVESTMENT_SCORE_WITHOUT_OVS | PASS | numeric Investment Score without numeric OVS |
| I9_SCORE_SUSPENDED_WITH_NUMERIC_OVS | PASS | numeric OVS with SCORE_PERMISSION = SUSPENDED and VALUATION_RELIABILITY = HIGH |

`PASS` in this table means the payload was correctly rejected as invalid.

### 16.1 Targeted SCORE_PERMISSION regression

Only the authorized regression set was replayed against `04_SCREENER_SCHEMA_V1_PATCHED.json`:

```text
C2
= PASS / VALID

C8
= PASS / VALID

I2_OVS_WITH_VALUATION_NOT_ASSESSABLE
= PASS / REJECTED

I3_OQS_WITH_SCORE_SUSPENDED
= PASS / REJECTED

I8_INVESTMENT_SCORE_WITHOUT_OVS
= PASS / REJECTED

I9_SCORE_SUSPENDED_WITH_NUMERIC_OVS
= PASS / REJECTED
```

Positive state checks:

```text
SCORE_PERMISSION = SUSPENDED
VALUATION_RELIABILITY = HIGH
OVS = NOT_AVAILABLE
= VALID

SCORE_PERMISSION = SUSPENDED
VALUATION_RELIABILITY = NOT_ASSESSABLE
OVS = NOT_ASSESSABLE
= VALID
```

Deterministic non-regression:

```text
OQS FORMULA
= UNCHANGED

OVS FORMULA / ANCHORS / CAPS
= UNCHANGED

INVESTMENT SCORE FORMULA
= UNCHANGED

OROTITAN TERMINAL GATE
= UNCHANGED
```

No other schema definition or top-level contract metadata was modified.

## 17. Schema self-audit

```text
EVERY REQUIRED PHASE-4 OUTPUT MAPPABLE
= PASS

ANY FROZEN STATE LOST
= NO

UNKNOWN / N/A / MISSING PRESERVED
= PASS

BUSINESS / INVESTMENT SEPARATION
= PASS

SCORE / CERTIFICATION SEPARATION
= PASS

OROTITAN TERMINAL LOGIC
= PASS

DIRECT / PREPARED
= PASS

READINESS
= PASS

RESEARCH PRICE / CURRENT MARKET PRICE SEPARATION
= PASS

PIT / HISTORY
= PASS

NEW SCORE
= NO

NEW METHODOLOGY
= NO
```
## 18. Final validation status

```text
SCHEMA_VALIDATION
= PASS

CANONICAL_OUTPUT_COVERAGE
= PASS

PIT_HISTORY_SUPPORT
= PASS

NULL_SEMANTICS
= PASS

SCORING_INVARIANTS
= PASS

OROTITAN_GATE_INVARIANTS
= PASS

DIRECT_PREPARED_SUPPORT
= PASS

REFRESH_ACTIVATION_SUPPORT
= PASS

NEW_METHOD_INTRODUCED
= NO

INTEGRATION_CONFLICTS
= 0
```

```text
PHASE4_PATCH_INTEGRITY
= PASS

I9_NUMERIC_OVS_WHEN_SUSPENDED
= REJECTED

REGRESSION
= PASS

NEW_METHOD_INTRODUCED
= NO

INTEGRATION_CONFLICTS
= 0
```

```text
PHASE 4 DATA CONTRACT
= VALIDATED

PHASE 4 DATA CONTRACT FREEZE
= READY

PHYSICAL SCREENER IMPLEMENTATION
= NOT STARTED
```
