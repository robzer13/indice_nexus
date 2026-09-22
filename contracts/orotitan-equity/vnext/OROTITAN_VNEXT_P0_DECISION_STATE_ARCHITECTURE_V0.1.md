# OROTITAN_VNEXT_P0_DECISION_STATE_ARCHITECTURE_V0.1

**Project:** OroTitan Equity Research  
**Status:** CANDIDATE — GATE 15 / P0-2  
**Methodology change:** NO  
**Provider dependency:** NONE  
**Production mutation:** NONE  
**Shadow DB mutation:** NONE

## 0. Purpose

This module implements the second Gate 15 P0 analytical component: Decision State Architecture.

It preserves the canonical operational NEXT_ACTION vocabulary:

```text
INVESTABLE_NOW
WAIT_FOR_PRICE
WAIT_FOR_EVIDENCE
REFRESH_REQUIRED
REJECT
```

VNext additionally preserves UNKNOWN as an explicit non-final state while required analysis remains incomplete.

## 1. Frozen meanings

```text
INVESTABLE_NOW
Current setup satisfies configured investment policy and certification requirements.
OroTitan terminal status is not required.

WAIT_FOR_PRICE
The business is prepared, but current valuation is inadequate.

WAIT_FOR_EVIDENCE
A material economic uncertainty prevents the investment decision.

REFRESH_REQUIRED
The existing dossier or Price Ladder cannot support a current decision.

REJECT
Structural economics fail the investment philosophy or thesis-breaking evidence exists.
```

These meanings are consumed, not redefined, by VNext.

## 2. Architecture principle

The module is not an automatic investment-decision formula.

Analytical judgments establish the decision basis. Deterministic code may only:

- validate the canonical vocabulary;
- check minimum compatibility with the frozen definitions;
- preserve UNKNOWN;
- detect contradictions;
- surface multiple simultaneously plausible canonical states;
- aggregate exact evidence and assumption references;
- fail closed when a proposed state is incompatible.

When multiple canonical states remain plausible, deterministic code must not invent a priority rule.

```text
MULTIPLE PLAUSIBLE STATES
-> ANALYST RESOLUTION REQUIRED
```

## 3. Decision basis

The V0.1 execution envelope receives traceable assessments for:

```text
investmentPolicySatisfied
certificationRequirementsSatisfied
businessPrepared
valuationAdequate
materialEconomicUncertainty
currentDecisionSupport
structuralEconomicsFail
thesisBreakingEvidence
```

Each is represented as:

```text
YES | NO | UNKNOWN
+ rationale
+ evidence IDs
+ assumption IDs
```

These fields are implementation-level decision-support judgments. They do not create new score inputs or new analytical methodology.

## 4. Minimum compatibility rules

### INVESTABLE_NOW

Minimum deterministic compatibility requires:

```text
investment policy satisfied = YES
certification requirements satisfied = YES
current decision support = YES
material economic uncertainty = NO
structural economics fail = NO
thesis-breaking evidence = NO
```

Attractive price or score alone can never satisfy this state.

### WAIT_FOR_PRICE

Minimum compatibility requires:

```text
business prepared = YES
valuation adequate = NO
current decision support = YES
material economic uncertainty = NO
structural economics fail = NO
thesis-breaking evidence = NO
```

### WAIT_FOR_EVIDENCE

Minimum compatibility:

```text
material economic uncertainty = YES
```

UNKNOWN values in other fields may remain explicit because this state exists precisely to represent a material evidence blocker.

### REFRESH_REQUIRED

Minimum compatibility:

```text
current decision support = NO
```

This represents a dossier / Price Ladder that cannot support a current decision.

### REJECT

Minimum compatibility:

```text
structural economics fail = YES
OR
thesis-breaking evidence = YES
```

## 5. No invented priority

Some real states can overlap operationally.

Example:

```text
known thesis-breaking evidence
+
stale existing dossier
```

may make both REJECT and REFRESH_REQUIRED superficially compatible.

V0.1 does not decide that one mechanically dominates the other.

It emits:

```text
selectionAmbiguous = true
MULTIPLE_CANONICAL_STATES_REQUIRE_ANALYST_RESOLUTION
```

This preserves analyst judgment and prevents deterministic methodology drift.

## 6. UNKNOWN discipline

UNKNOWN is allowed in the VNext state vector but is not a final canonical NEXT_ACTION.

```text
proposedDecisionState = UNKNOWN
-> finalizable = false
```

UNKNOWN basis fields require traceable evidence or assumptions and may not be empty placeholders.

## 7. Traceability

Every YES decision-basis claim requires evidence.

Every UNKNOWN basis claim requires a traceable evidence or assumption basis.

The output aggregates and deterministically sorts:

```text
supportingEvidenceIds[]
assumptionIds[]
```

## 8. Authority boundary

The module must never:

- derive a decision from OQS / OVS / Investment Score alone;
- derive OROTITAN_STATUS;
- override Certification;
- override the configured investment policy;
- coerce UNKNOWN to NO;
- create a new NEXT_ACTION;
- invent priority among unresolved overlapping states;
- write production.

## 9. Module Contract

```text
contracts/orotitan-equity/vnext/modules/
  DECISION_STATE_ARCHITECTURE.module-contract.v0.1.json
```

The contract is ANALYTICAL / DEEP_DIVE and preserves the exact frozen Deep Dive execution lifecycle.

## 10. Runtime implementation

```text
runtime/vnext/modules/decision-state-architecture.ts
```

No model-provider or Azure dependency exists.

## 11. Golden fixtures

```text
tests/fixtures/vnext/decision-state-architecture.v0.1.json
```

Fixture classes:

```text
INVESTABLE_NOW
WAIT_FOR_PRICE
WAIT_FOR_EVIDENCE
REFRESH_REQUIRED
REJECT
ambiguous REJECT + REFRESH_REQUIRED
UNKNOWN / not final
```

## 12. Deterministic tests

```text
tests/vnext-decision-state-architecture.test.ts
```

Tests cover:

```text
Gate 7 Module Contract validation
canonical decision-state vocabulary
policy + certification guard for INVESTABLE_NOW
prepared-business / valuation guard for WAIT_FOR_PRICE
material uncertainty routing for WAIT_FOR_EVIDENCE
stale support routing for REFRESH_REQUIRED
structural/thesis-breaking basis for REJECT
no deterministic priority invention
UNKNOWN preservation
finalization guard
traceability requirements
```

## 13. Gate 15 relationship

```text
P0-1 Weak Link Taxonomy          PASS / MERGED
P0-2 Decision State Architecture CANDIDATE
P0-3 Return Normalization        NOT STARTED
P0-4 Capital Seasoning           NOT STARTED
P0-5 Owner Cash                  NOT STARTED
P0-6 Financing Consistency       NOT STARTED
P0-7 Valuation Assumption Integrity NOT STARTED
P0-8 Valuation Diagnostic Integrity NOT STARTED
```

Gate 15 remains open until all eight P0 modules pass isolated fixtures and are frozen.

## 14. Acceptance matrix

```text
P0-2-01 canonical vocabulary preserved                    PASS
P0-2-02 UNKNOWN preserved                                 PASS
P0-2-03 INVESTABLE_NOW requires policy + certification    PASS
P0-2-04 WAIT_FOR_PRICE preserves prepared-opportunity rule PASS
P0-2-05 WAIT_FOR_EVIDENCE preserves material uncertainty  PASS
P0-2-06 REFRESH_REQUIRED preserves stale-support meaning  PASS
P0-2-07 REJECT preserves structural/thesis-breaking basis PASS
P0-2-08 score alone cannot create investment decision     PASS
P0-2-09 no deterministic priority invention               PASS
P0-2-10 evidence traceability required                    PASS
P0-2-11 assumptions explicit                              PASS
P0-2-12 no direct OroTitan authority                      PASS
P0-2-13 no provider/Azure dependency                      PASS
P0-2-14 isolated golden fixtures                          PASS
P0-2-15 deterministic CI                                  PENDING
```

## 15. Current state

```text
P0 MODULE = DECISION_STATE_ARCHITECTURE
STATUS    = CANDIDATE
LIVE MODEL REQUIRED = NO
SHADOW RUN MUTATION = FORBIDDEN UNTIL GATE 15
```
