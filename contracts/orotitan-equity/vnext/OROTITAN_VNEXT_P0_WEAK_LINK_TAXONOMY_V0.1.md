# OROTITAN_VNEXT_P0_WEAK_LINK_TAXONOMY_V0.1

**Project:** OroTitan Equity Research  
**Status:** CANDIDATE — GATE 15 / P0-1  
**Methodology change:** NO  
**Provider dependency:** NONE  
**Production mutation:** NONE  
**Shadow DB mutation:** NONE

## 0. Purpose

This module implements the first Gate 15 P0 analytical component: Weak Link Taxonomy.

It does not discover weak links by formula. It structures three analytical judgments already authorized by frozen OroTitan methodology and applies the frozen conjunction deterministically.

```text
MATERIAL_WEAK_LINK = YES
iff
MATERIALITY = YES
AND CAUSALITY = YES
AND UNRESOLVEDNESS = YES
```

A material weak link may be cross-block even when no isolated score dimension is low.

## 1. Authority boundary

Component assessments remain analytical judgments:

```text
MATERIALITY
CAUSALITY
UNRESOLVEDNESS
```

The runtime owns only deterministic combination, validation, traceability aggregation and downstream signaling.

The module must never:

- invent a causal mechanism;
- convert a generic risk label into causality;
- fabricate evidence or assumptions;
- silently resolve contradictory evidence;
- change scoring weights or formulas;
- set OROTITAN_STATUS directly;
- write production.

## 2. Tri-state discipline

Each criterion is:

```text
YES | NO | UNKNOWN
```

Combination:

```text
all YES                    -> MATERIAL_WEAK_LINK = YES
any NO                     -> MATERIAL_WEAK_LINK = NO
otherwise                  -> MATERIAL_WEAK_LINK = UNKNOWN
```

This preserves the frozen UNKNOWN discipline. UNKNOWN is never coerced to NO merely to complete the module.

## 3. Terminal transmission

The module does not own terminal status.

It emits only:

```text
YES      -> OROTITAN_STATUS_NO_REQUIRED
NO       -> NO_TERMINAL_SIGNAL
UNKNOWN  -> UNRESOLVED
```

The later terminal gate remains authoritative for the actual OROTITAN_STATUS.

## 4. Traceability

Every criterion carries:

```text
state
rationale
evidenceIds[]
contradictingEvidenceIds[]
assumptionIds[]
```

Rules:

- YES requires supporting evidence;
- UNKNOWN requires a traceable evidentiary basis;
- supporting and contradicting evidence may not silently overlap;
- output evidence and assumption references are deterministically deduplicated and sorted.

## 5. Module Contract

```text
contracts/orotitan-equity/vnext/modules/
  WEAK_LINK_TAXONOMY.module-contract.v0.1.json
```

The contract is ANALYTICAL / DEEP_DIVE and uses the exact frozen Deep Dive block lifecycle:

```text
INSUFFICIENT
IN_PROGRESS
PROVISIONALLY_STABLE
LOCKED
```

## 6. Runtime implementation

```text
runtime/vnext/modules/weak-link-taxonomy.ts
```

The implementation has no model-provider import and no Azure dependency.

Therefore P0 module construction can continue while live inference providers remain unavailable.

## 7. Golden fixtures

```text
tests/fixtures/vnext/weak-link-taxonomy.v0.1.json
```

Initial fixture classes:

```text
all three criteria hold
generic risk without causality
material/causal but unresolved UNKNOWN
real mechanism resolved by mitigation
```

## 8. Deterministic tests

```text
tests/vnext-weak-link-taxonomy.test.ts
```

Coverage includes:

```text
Gate 7 Module Contract validation
frozen three-condition conjunction
UNKNOWN preservation
NO dominance in conjunction
YES requires evidence
UNKNOWN requires traceable basis
evidence contradiction overlap rejected
terminal signal without direct terminal authority
deterministic evidence/assumption aggregation
```

## 9. Gate 15 relationship

Gate 15 requires eight P0 modules to operate independently on fixtures before any may mutate a shadow run.

Order remains:

```text
1 Weak Link Taxonomy
2 Decision State Architecture
3 Return Normalization
4 Capital Seasoning
5 Owner Cash
6 Financing Consistency
7 Valuation Assumption Integrity
8 Valuation Diagnostic Integrity
```

This document covers P0-1 only.

## 10. Acceptance matrix

```text
P0-1-01 frozen materiality rule preserved                 PASS
P0-1-02 frozen causality rule preserved                   PASS
P0-1-03 frozen unresolvedness rule preserved              PASS
P0-1-04 all-three conjunction deterministic               PASS
P0-1-05 cross-block weak link supported                    PASS
P0-1-06 UNKNOWN preserved                                 PASS
P0-1-07 generic risk label not sufficient                 PASS
P0-1-08 evidence refs required for YES                    PASS
P0-1-09 assumptions explicit                              PASS
P0-1-10 counterevidence explicit                          PASS
P0-1-11 no score/weight change                            PASS
P0-1-12 no direct OROTITAN_STATUS authority               PASS
P0-1-13 no provider/Azure dependency                      PASS
P0-1-14 isolated golden fixtures                          PASS
P0-1-15 deterministic CI                                  PENDING
```

## 11. Current state

```text
P0 MODULE = WEAK_LINK_TAXONOMY
STATUS    = CANDIDATE
LIVE MODEL REQUIRED = NO
SHADOW RUN MUTATION = FORBIDDEN UNTIL GATE 15
```
