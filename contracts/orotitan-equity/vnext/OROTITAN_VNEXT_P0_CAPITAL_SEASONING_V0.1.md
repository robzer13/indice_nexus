# OROTITAN_VNEXT_P0_CAPITAL_SEASONING_V0.1

**Project:** OroTitan Equity Research  
**Status:** CANDIDATE — GATE 15 / P0-4  
**Methodology change:** NO  
**Provider dependency:** NONE  
**Production mutation:** NONE  
**Shadow DB mutation:** NONE

## 0. Purpose

This module implements the fourth Gate 15 P0 analytical component: Capital Seasoning.

Its purpose is to prevent recent capital deployment from being judged as if it had already reached mature economic return.

Capital Seasoning does not change ROIC/ROIIC methodology. It operationalizes the existing investment-lag and cohort logic around when a capital cohort is sufficiently mature for return evidence to be interpreted.

## 1. Core principle

```text
RECENT CAPITAL DEPLOYMENT
!=
MATURE RETURN EVIDENCE
```

and:

```text
UNSEASONED CAPITAL
!=
FAILED RETURN

UNSEASONED CAPITAL
!=
PROVEN HIGH RETURN
```

The module therefore separates maturity of the capital cohort from quality of the eventual return.

## 2. Capital maturity states

V0.1 uses the following execution states:

```text
COMMITTED
DEPLOYED
IN_SERVICE
RAMPING
STABILIZING
SEASONED
UNKNOWN
```

These are analytical execution states, not new score bands or terminal investment states.

## 3. Evidence-driven progression

The state sequence is tied to observable operating evidence:

```text
COMMITTED
↓
DEPLOYED
↓
IN_SERVICE
↓
RAMPING
↓
STABILIZING
↓
SEASONED
```

No fixed number of months or years is embedded in V0.1.

Seasoning duration depends on the economic capital type, business model, investment lag and evidence. Calibration may later study typical durations, but Gate 15 does not freeze a universal threshold.

## 4. Investment lag

The module preserves the frozen execution metadata:

```text
INVESTMENT_LAG
= LOW | MEDIUM | HIGH | UNKNOWN
```

Long-lag capital such as capacity projects, R&D programs, implementations or acquisition integration may require cohort-specific or longer-window evidence before mature return judgment.

## 5. Capital types

V0.1 supports:

```text
CAPEX
WORKING_CAPITAL
R_AND_D
SALESFORCE
IMPLEMENTATION
REGULATORY_CAPITAL
ACQUISITION_CAPITAL
OTHER
```

This mirrors the broad capital types already used in the runway-to-marginal-return bridge.

## 6. Return evidence use

Each cohort emits:

```text
SEASONED
-> MATURE_RETURN_EVIDENCE_ALLOWED

COMMITTED / DEPLOYED / IN_SERVICE / RAMPING / STABILIZING
-> UNSEASONED_DO_NOT_JUDGE

UNKNOWN
-> UNKNOWN
```

Observed early return evidence may still be stored for an unseasoned cohort. The module only prevents that evidence from being treated as mature return proof.

## 7. Portfolio seasoning state

For material cohorts, V0.1 reports:

```text
FULLY_SEASONED
MIXED_SEASONING
UNSEASONED
UNKNOWN
NOT_APPLICABLE
```

This is an execution state only. It is not a quality score.

## 8. UNKNOWN discipline

Unknown materiality, deployment state, in-service state, utilization, stabilization or mature-return evidence remains explicit.

```text
UNKNOWN
!=
SEASONED
```

The runtime rejects attempts to coerce missing maturity evidence into a seasoned classification.

## 9. Relationship with Return Normalization

Capital Seasoning consumes the Return Normalization output.

```text
RETURN NORMALIZATION
→ establishes economically valid return framework

CAPITAL SEASONING
→ establishes whether the relevant capital cohort is mature enough
  for return evidence to be interpreted as mature
```

The module cannot override Return Normalization.

## 10. Authority boundary

The module must never:

- create a fixed seasoning duration;
- create a return hurdle or threshold;
- score capital seasoning;
- label unseasoned capital as value destructive merely because return is not yet mature;
- label recent capital as high-return merely from early evidence;
- override Return Quality scoring or Elite Return Quality;
- write production.

## 11. Module Contract

```text
contracts/orotitan-equity/vnext/modules/
  CAPITAL_SEASONING.module-contract.v0.1.json
```

## 12. Runtime implementation

```text
runtime/vnext/modules/capital-seasoning.ts
```

No model-provider or Azure dependency exists.

## 13. Golden fixtures

```text
tests/fixtures/vnext/capital-seasoning.v0.1.json
```

Fixture classes:

```text
committed but not deployed
ramping capital
seasoned acquisition cohort
mixed seasoned/unseasoned portfolio
unsupported seasoned classification
explicit UNKNOWN
NOT_APPLICABLE
```

## 14. Deterministic tests

```text
tests/vnext-capital-seasoning.test.ts
```

Coverage includes:

```text
Gate 7 Module Contract validation
evidence-based state progression
no elapsed-time threshold
unseasoned return-evidence firewall
sequence-conflict detection
UNKNOWN preservation
mixed portfolio state
NOT_APPLICABLE handling
no return threshold / score
fail-closed finalization
```

## 15. Gate 15 relationship

```text
P0-1 Weak Link Taxonomy              PASS / MERGED
P0-2 Decision State Architecture     PASS / MERGED
P0-3 Return Normalization            PASS / MERGED
P0-4 Capital Seasoning               CANDIDATE
P0-5 Owner Cash                      NOT STARTED
P0-6 Financing Consistency           NOT STARTED
P0-7 Valuation Assumption Integrity  NOT STARTED
P0-8 Valuation Diagnostic Integrity  NOT STARTED
```

## 16. Acceptance matrix

```text
P0-4-01 recent capital != mature return                         PASS
P0-4-02 unseasoned capital != failed return                     PASS
P0-4-03 maturity states traceable                               PASS
P0-4-04 no fixed seasoning duration                             PASS
P0-4-05 investment-lag metadata preserved                       PASS
P0-4-06 cohort-specific analysis                                PASS
P0-4-07 early return evidence may exist without mature judgment PASS
P0-4-08 UNKNOWN preserved                                       PASS
P0-4-09 sequence conflicts fail closed                          PASS
P0-4-10 mixed portfolio state supported                         PASS
P0-4-11 no score / return threshold                             PASS
P0-4-12 no Return Normalization override                        PASS
P0-4-13 no provider/Azure dependency                            PASS
P0-4-14 isolated golden fixtures                                PASS
P0-4-15 deterministic CI                                        PENDING
```

## 17. Current state

```text
P0 MODULE = CAPITAL_SEASONING
STATUS    = CANDIDATE
LIVE MODEL REQUIRED = NO
SHADOW RUN MUTATION = FORBIDDEN UNTIL GATE 15
```
