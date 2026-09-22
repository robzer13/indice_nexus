# OROTITAN_VNEXT_ADAPTIVE_ANALYTICAL_DEPTH_V0.1

**Project:** OroTitan Equity Research  
**Status:** CANDIDATE — GATE 16  
**Methodology change:** NO  
**Provider dependency:** NONE  
**Production mutation:** NONE  
**Shadow DB mutation:** NONE

## 0. Purpose

Gate 16 adds a deterministic policy for selecting how much analytical depth a VNext task requires.

The policy is independent of any physical LLM provider or model. It decides analytical depth from observable dossier state and records the reason for that depth.

```text
ANALYTICAL DEPTH
!=
MODEL NAME
```

A later routing policy may map analytical depth or task class to a calibrated provider/model. Gate 16 does not freeze that mapping.

## 1. Depth states

```text
DEPTH_1
DEPTH_2
DEPTH_3
```

### DEPTH_1

Simple dossier/task. No material uncertainty, weak link, material source conflict, marginal-return uncertainty, low valuation reliability, or sensitive decision boundary requires escalation.

### DEPTH_2

Additional analytical work is required because a significant but not maximum-severity uncertainty exists.

Examples:

```text
SIGNIFICANT_UNCERTAINTY
CRITICAL_UNKNOWN
EXECUTION_CONFIDENCE = MEDIUM
DECISION_BOUNDARY_SENSITIVITY = MEDIUM
```

### DEPTH_3

Maximum analytical depth is required when a hard trigger is present.

Hard triggers in V0.1:

```text
MATERIAL_WEAK_LINK
WEAK_LINK_UNRESOLVED
MARGINAL_RETURN_UNCERTAIN
VALUATION_RELIABILITY = LOW
VALUATION_RELIABILITY = NOT_ASSESSABLE
MATERIAL_SOURCE_CONFLICT
EXECUTION_CONFIDENCE = LOW
DECISION_BOUNDARY_SENSITIVITY = HIGH
MATERIAL_COUNTEREVIDENCE_RISK
```

This preserves the roadmap requirement that material weak links, uncertain marginal returns, low valuation reliability, source contradiction, or a decision close to a sensitive boundary force deeper analysis.

## 2. Deterministic minimum

The system derives a minimum permitted depth from the exact trigger set.

```text
no escalation trigger
-> DEPTH_1

any DEPTH_2 trigger and no DEPTH_3 trigger
-> DEPTH_2

any DEPTH_3 trigger
-> DEPTH_3
```

No operator, runtime or budget policy may silently downgrade below this deterministic minimum.

## 3. Deeper manual request

A user/operator may explicitly request more depth than the deterministic minimum.

Example:

```text
minimum = DEPTH_1
requested = DEPTH_2
selected = DEPTH_2
trigger = MANUAL_DEEPER_REQUEST
```

The request is recorded as a trigger. A manual request may increase depth but never reduce a required depth.

## 4. Second analyst

DEPTH_3 makes an independent second analyst eligible.

```text
DEPTH_1 -> second analyst not eligible through this policy
DEPTH_2 -> second analyst not eligible through this policy
DEPTH_3 -> second analyst eligible
```

Gate 16 does not require a second analyst on every DEPTH_3 task. The architecture preserves the roadmap wording that DEPTH_3 *may* execute an independent second analyst.

If a second analyst is executed:

```text
SECOND ANALYST
-> RECONCILIATION ARTIFACT REQUIRED
```

No voting rule is introduced. Evidence remains authoritative.

## 5. Traceability

Every executed analytical-depth decision must be traceable through a consumption record containing at least:

```text
RUN_ID
STAGE_CODE
MODULE_ID
EXECUTION_ID
SELECTED_DEPTH
TRIGGER_CODES[]
SECOND_ANALYST_EXECUTED
RECONCILIATION_ARTIFACT_ID
PROVIDER_REQUEST_IDS[]
```

`PROVIDER_REQUEST_IDS[]` may be empty while no live inference provider is configured.

The trace validator requires:

- selected depth matches the exact deterministic decision;
- at least one exact trigger is recorded;
- trigger codes belong to that exact decision;
- second analyst is used only when eligible;
- second analyst execution has a reconciliation artifact;
- no reconciliation artifact exists without a second-analyst execution.

## 6. Provider neutrality

Gate 16 deliberately contains no:

```text
AzureProvider
OpenAIProvider
AnthropicProvider
GeminiProvider
model name
endpoint
API key
pricing rule
```

This allows OroTitan to operate without Azure as an inference engine and keeps analytical depth separate from physical model routing.

## 7. Budget boundary

Budget cannot reduce the deterministic minimum depth.

```text
REQUIRED DEPTH_3
+ insufficient provider/budget capability
!=
silently run DEPTH_1 or DEPTH_2
```

The correct downstream behavior is to block, use an authorized calibrated alternative, or escalate to human handling under later routing/budget policy.

Gate 16 itself does not implement provider selection or a Budget Governor.

## 8. Relationship with Gate 15

Gate 15 P0 modules provide structured states that can become depth triggers, including:

```text
Weak Link Taxonomy
Return Normalization
Valuation Assumption Integrity
Valuation Diagnostic Integrity
Decision State Architecture
```

Gate 16 does not change any Gate 15 analytical judgment. It consumes resulting state/metadata to choose execution depth.

## 9. Runtime

```text
runtime/vnext/adaptive-analytical-depth.ts
```

## 10. Golden fixtures

```text
tests/fixtures/vnext/adaptive-analytical-depth.v0.1.json
```

Fixture classes:

```text
simple DEPTH_1
significant uncertainty DEPTH_2
material weak link DEPTH_3
marginal-return uncertainty DEPTH_3
low valuation reliability DEPTH_3
material source conflict DEPTH_3
high-sensitivity decision boundary DEPTH_3
manual deeper request
```

## 11. Tests

```text
tests/vnext-adaptive-analytical-depth.test.ts
```

Tests cover:

```text
deterministic minimum-depth selection
no silent depth downgrade
manual deeper request traceability
DEPTH_3 second-analyst eligibility
mandatory reconciliation when second analyst runs
trace/depth identity
exact trigger provenance
provider neutrality
no Azure dependency
```

## 12. Acceptance matrix

```text
G16-01 DEPTH_1 baseline defined                         PASS
G16-02 DEPTH_2 significant-uncertainty routing         PASS
G16-03 DEPTH_3 material weak-link routing              PASS
G16-04 DEPTH_3 marginal-return uncertainty             PASS
G16-05 DEPTH_3 low/not-assessable valuation reliability PASS
G16-06 DEPTH_3 material source conflict                PASS
G16-07 DEPTH_3 high decision-boundary sensitivity      PASS
G16-08 no downgrade below deterministic minimum        PASS
G16-09 second analyst eligible only at DEPTH_3         PASS
G16-10 second analyst requires reconciliation          PASS
G16-11 consumed depth traceable to exact triggers      PASS
G16-12 provider-neutral / Azure-independent            PASS
G16-13 no production mutation                          PASS
G16-14 deterministic CI                                PENDING
```

## 13. Current state

```text
GATE = 16
STATUS = CANDIDATE
LIVE MODEL REQUIRED = NO
PHYSICAL MODEL ROUTING = NOT FROZEN
PRODUCTION AUTHORITY = NONE
```
