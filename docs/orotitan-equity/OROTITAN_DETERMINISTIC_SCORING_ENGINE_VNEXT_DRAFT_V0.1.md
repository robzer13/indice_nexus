# OROTITAN_DETERMINISTIC_SCORING_ENGINE_VNEXT_DRAFT_V0.1

STATUS = DRAFT_NON_AUTHORITATIVE  
METHODOLOGY_CHANGE = YES  
CURRENT_FROZEN_V2_METHOD = UNCHANGED  
PURPOSE = DESIGN_CANDIDATE_FOR_NEXT_ENGINE_GENERATION

---

## 1. OBJECTIVE

Replace discretionary numerical judgment by a traceable chain:

```text
EVIDENCE
→ CONTROLLED PRIMITIVES
→ SCOREABILITY GATES
→ DETERMINISTIC RULES
→ DIMENSION SCORE
→ I2 / OQS / OVS / INVESTMENT SCORE
```

The LLM may research, normalize, contradict, classify and explain evidence.

The LLM must not freely choose the final numerical dimension score.

---

## 2. CORE INVARIANT

For identical:

- evidence set and versions;
- primitive states;
- scoring-rule version;
- sector method;
- certification state;

the numerical output MUST be byte-for-byte deterministic.

```text
SAME INPUT STATE
→ SAME SCORE
```

No analyst or LLM score override is permitted.

---

## 3. PRIMITIVE RECORD

Every score-driving primitive should use a controlled record such as:

```json
{
  "primitive_id": "MOAT.ECONOMIC_CONSEQUENCE",
  "dimension": "MOAT",
  "state": "DEMONSTRATED",
  "evidence_ids": ["..."],
  "counterevidence_ids": ["..."],
  "source_independence": "SUFFICIENT",
  "materiality": "MATERIAL",
  "rule_version": "MOAT_RULESET_VNEXT_0.1"
}
```

Allowed values are frozen per primitive. Free-text may explain a state but cannot replace it.

---

## 4. SCOREABILITY FIRST

Each dimension must first determine whether it is scoreable.

Possible execution states:

```text
SCOREABLE
SCOREABLE_WITH_LIMITATIONS
NOT_SCOREABLE
NOT_APPLICABLE
```

UNKNOWN is never converted to zero.

If a mandatory primitive is missing or unresolved, the dimension follows its frozen fail-closed rule.

---

## 5. DIMENSION DESIGN PRINCIPLE

A dimension score must be decomposable into explicit rule contributions.

The UI must be able to answer:

```text
WHY 85?
WHY NOT 80?
WHY NOT 90?
```

Every score must expose:

- base rule;
- positive contributions;
- negative contributions;
- caps;
- floors;
- weak-link rules;
- missing-data effects;
- exact primitive states used.

Prefer 5-point increments unless a future frozen rule requires finer granularity.

False precision should be avoided.

---

## 6. CANDIDATE PRIMITIVE FAMILIES

This section defines research directions only. Exact weights and thresholds are NOT frozen by this draft.

### 6.1 MOAT

Candidate primitives:

- economic consequence demonstrated;
- independent corroboration;
- durability horizon;
- moat trend;
- switching-cost evidence;
- pricing-power evidence;
- network-effect evidence;
- scale/cost advantage evidence;
- IP/know-how evidence;
- ecosystem/installed-base advantage;
- customer concentration vulnerability;
- credible competitive entry;
- technological substitution risk;
- counterevidence materiality.

### 6.2 RUNWAY

Candidate primitives:

- addressable expansion evidence;
- penetration headroom;
- product/geography expansion;
- unit-economics support;
- reinvestment capacity;
- demand durability;
- competitive capture risk;
- regulatory/physical constraints;
- top-down TAM dependence.

### 6.3 RETURN QUALITY

Prefer quantitative primitives wherever possible:

- standard ROIC;
- all-in ROIC where applicable;
- ROIC trend;
- economic spread;
- ROIIC interpretability;
- ROIIC magnitude;
- return attribution quality;
- acquisition-return economics where applicable;
- data quality.

### 6.4 CASH ECONOMICS

Candidate primitives:

- standardized FCF;
- cash conversion;
- owner-earnings assessability;
- maintenance/growth capex separation;
- SBC dilution;
- working-capital burden;
- lease/inventory intensity;
- accounting-to-cash reconciliation;
- forensic reliability.

### 6.5 CAPITAL ALLOCATION

Candidate primitives:

- reinvestment returns;
- acquisition discipline;
- acquisition seasoning;
- buyback value creation/destruction;
- dilution;
- balance-sheet discipline;
- dividend rationality where relevant;
- management capital-allocation consistency.

### 6.6 MANAGEMENT / GOVERNANCE

Candidate primitives:

- governance structure;
- incentive alignment;
- disclosure quality;
- capital-allocation accountability;
- related-party risk;
- succession record;
- management tenure/seasoning;
- documented execution reliability.

### 6.7 RESILIENCE / RISK

Candidate primitives:

- balance-sheet resilience;
- cyclicality;
- customer/supplier concentration;
- regulation;
- technological disruption;
- product/quality risk;
- geopolitical exposure;
- litigation;
- key-person dependence;
- operational fragility;
- credible downside pathways.

---

## 7. LLM ROLE

Allowed:

- retrieve sources;
- identify claims;
- normalize evidence;
- classify evidence into controlled primitive states;
- identify conflicts;
- propose unresolved status;
- explain the deterministic result.

Not allowed:

- choose the final dimension number directly;
- override a deterministic rule;
- silently resolve material conflicts;
- invent missing primitives;
- transform UNKNOWN into zero;
- use stylistic confidence as evidence.

---

## 8. RULE ENGINE OUTPUT

Each dimension should produce a machine-readable trace:

```json
{
  "dimension": "MOAT",
  "ruleset_version": "MOAT_RULESET_VNEXT_1.0",
  "primitive_snapshot_sha256": "...",
  "base_score": 70,
  "adjustments": [
    {"rule_id": "M01", "delta": 10, "reason": "ECONOMIC_CONSEQUENCE=DEMONSTRATED"},
    {"rule_id": "M04", "delta": 5, "reason": "DURABILITY=LONG"}
  ],
  "caps": [],
  "final_score": 85
}
```

The numerical path must be reproducible without an LLM.

---

## 9. ENGINE FINGERPRINT

A scoring engine generation must be identified by a fingerprint over the exact authoritative set:

- process contract;
- stage contracts;
- primitive dictionaries;
- dimension rulesets;
- sector overlays;
- I2 implementation;
- valuation rules;
- certification rules.

Changing any score-driving component creates a new engine fingerprint.

A schema migration alone does not create a new scoring engine.

---

## 10. MIGRATION POLICY

Existing published snapshots remain immutable.

They retain their original:

- engine fingerprint;
- data cutoff;
- scores;
- certification;
- publication timestamp.

No silent mass recalculation.

Migration to the future engine occurs only through an explicit REFRESH / FULL_REFRESH or authorized replay campaign.

---

## 11. ACCEPTANCE TESTS BEFORE FREEZE

At minimum:

1. same primitives repeated 100 times → exact same score;
2. shuffled evidence order → exact same score;
3. LLM provider change with identical primitives → exact same score;
4. UNKNOWN primitive → correct fail-closed behavior;
5. counterevidence changes primitive → deterministic score change;
6. score cannot be manually overridden;
7. every point is attributable to a rule ID;
8. sector-specific methods do not leak into other sectors;
9. current V2 canonical snapshots are unchanged;
10. I2 reconciliation remains exact after dimension outputs are locked.

---

## 12. FREEZE REQUIREMENT

This draft cannot replace the current frozen methodology.

Before activation:

```text
PRIMITIVE DICTIONARY FREEZE
+ RULESET FREEZE
+ REGRESSION BENCHMARK
+ CROSS-SECTOR TESTS
+ USER METHODOLOGY AUTHORIZATION
= NEXT ENGINE GENERATION
```
