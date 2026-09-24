# OROTITAN_GATE18_PHASE_B_SYNTHESIS_V2

**Gate:** 18  
**Phase:** B company calibration  
**Status:** UPDATED SYNTHESIS — NOT FROZEN  
**Methodology change:** NO  
**Publication authority:** DISABLED  
**Production mutation:** FORBIDDEN  
**Model winner:** NONE  
**Routing freeze:** NONE  
**Further paid execution:** NOT AUTHORIZED BY THIS SYNTHESIS

## 1. Purpose

This document supersedes the interpretation layer of
`OROTITAN_GATE18_PHASE_B_LUNA_FIVE_COMPANY_SYNTHESIS_V1.md`
without deleting or rewriting historical observations.

It incorporates:

- the original five-company LUNA sweep;
- the historical Adyen × TERRA targeted contrast;
- the Adyen v0.6 → v1.0 contract-learning sequence;
- the v1.0 normal Adyen canary;
- the v1.0 targeted claim-alignment probe.

Historical receipts remain authoritative for their exact execution contracts.

## 2. Historical five-company LUNA sweep

Under the earlier common moat-evidence contract:

| Company | Role | Engineering valid | Comparison-admissible |
| --- | --- | ---: | ---: |
| STMicroelectronics | CYCLICAL_SEGMENTED | PASS | PASS |
| RATIONAL AG | SIMPLE_CLEAN_COMPOUNDER | PASS | PASS |
| Constellation Software | SERIAL_ACQUIRER | PASS | PASS |
| Brookfield Corporation | ACCOUNTING_HEAVY | PASS | FAIL |
| Adyen | DIFFICULT_MOAT_CONFLICTING_EVIDENCE | PASS | FAIL |

Observed historical rate:

```text
engineering-valid = 5/5
comparison-admissible = 3/5
```

The two non-admissible LUNA cases shared the same broad failure class:

```text
canonical evidence IDs were present
+
schema / deterministic semantics passed
+
human evidence-role classification failed
```

### Brookfield historical failure

The v0.5 Brookfield output placed E-039 and E-042 in
`counterevidence_ids` against a peer-replicability / competition proposition.

Human adjudication found that both observations reinforced peer replicability
and competing-capital evidence rather than weakened it.

### Adyen historical failure

The v0.5 Adyen output placed E-046 in `counterevidence_ids` even though
E-046 was another issuer-reported uplift observation and therefore reinforced
the bounded claim plus its validation limitation.

## 3. Historical higher-reasoning contrast

Adyen × TERRA v0.5 was already executed.

Observed:

```text
schema_valid = true
semantic_valid = false
comparison_admissible = false
gateway_cost_usd = 0.022699
```

Failure modes:

```text
MALFORMED_EVIDENCE_REFERENCE_STRINGS
COUNTEREVIDENCE_ROLE_CLASSIFICATION
```

Therefore the historical evidence does **not** support the hypothesis:

```text
higher reasoning automatically fixes evidence-role classification
```

TERRA was materially more expensive than the corresponding historical LUNA
observation and did not solve the targeted semantic problem.

No inference is made here about TERRA under the later v1.0 contract.

## 4. Adyen contract-learning sequence

The Adyen sequence progressively isolated the real failure mode.

### v0.6

Improved deterministic reference integrity and counterevidence auditability,
but a supportive selected-case observation was still misclassified as
counterevidence.

### v0.7

Moved source limitations toward a qualification channel, but exposed a
support/qualification overlap contract defect.

### v0.8 / v0.9

Improved qualification handling and atomic claim syntax.

v0.9 still failed because atomic syntax did not guarantee semantic-target
alignment:

```text
low reported churn
!=
switching-friction inference
```

and:

```text
documented incident existence
!=
full-period availability
```

This led to the v1.0 law:

```text
ATOMICITY != CLAIM-TARGET ALIGNMENT
```

Counterevidence must weaken the exact proposition being asserted.

## 5. v1.0 normal Adyen observation

The normal v1.0 Adyen × LUNA canary produced:

```text
schema_valid = true
semantic_valid = true
human finding quality = PASS
comparison_admissible = true
observed polarity errors = NONE
```

The output correctly reframed multi-PSP / provider-substitution evidence as
support for a substitution-capability proposition rather than false
counterevidence to a low-churn observation.

However, the run did not spontaneously exercise:

- `counterevidence_ids`;
- a `MIXED` priority finding;
- E-055 / E-056 scope resolution;
- E-040 / E-043 directional qualification handling.

Therefore the receipt remained conservatively classified
`COMPLETE_PARTIAL_PASS` for regression coverage.

## 6. v1.0 targeted Adyen probe

The calibration-only probe
`ADYEN_CLAIM_TARGET_CORE_001` directly forced the previously unexercised
semantic paths.

Observed:

```text
schema_valid = true
semantic_valid = true
finish_reason = stop
retry_count = 0
gateway_cost_usd = 0.0022185
```

Targeted results:

```text
same-target switching-friction MIXED path = PASS
scope-resolved C-010 incident path = PASS
E-040 / E-043 qualification orthogonality = PASS
evidence-role inversion = NONE OBSERVED
```

Interpretation:

```text
the v1.0 contract can express the intended evidence-role semantics
when those paths are explicitly exercised
```

This does not prove:

- repeatability;
- cross-company generalization;
- a model winner;
- routing policy;
- Phase C completion.

## 7. Updated failure-model inference

Current evidence no longer supports packet density as the primary explanation.

RATIONAL had the largest historical packet and was admissible under LUNA.

The stronger explanation is:

```text
semantic evidence-role ambiguity
+
claim-target mismatch
+
qualification-versus-direction confusion
```

The Adyen v1.0 sequence provides direct evidence that prompt/schema semantics
can materially improve this failure class without changing the physical model.

This is contract-learning evidence, not a permanent routing rule.

## 8. Current physical-model evidence

### LUNA

Evidence now supports:

- reliable structured execution across all five historical pilot roles;
- historical comparison-admissibility on STMicroelectronics, RATIONAL and Constellation;
- historical role-classification failures on Brookfield and Adyen under v0.5;
- human-quality-admissible normal Adyen output under v1.0;
- direct targeted semantic-regression PASS on Adyen under v1.0.

Evidence does **not** yet prove:

- v1.0 cross-company generalization;
- v1.0 repeatability;
- v1.0 robustness across the diversified Phase C corpus.

### TERRA

Historical evidence includes:

- admissible STMicroelectronics observation;
- non-admissible RATIONAL observation;
- non-admissible Adyen v0.5 targeted contrast.

The old Adyen contrast must not be treated as evidence about TERRA under the
later v1.0 contract.

### SOL / ASTRA

Historical observations are mixed across STMicroelectronics and RATIONAL.

No current evidence justifies automatic escalation from LUNA to SOL or ASTRA.

## 9. Highest-information next observation

The next company-level calibration observation should be:

```text
Brookfield Corporation × LUNA
contract = v1.0 normal priority-selection path
probe = NONE
model = LUNA
reasoning = minimal
```

Why Brookfield:

1. Brookfield is the only second historical company with the same broad
   evidence-role classification failure class.
2. It belongs to a different archetype:
   `ACCOUNTING_HEAVY`, not `DIFFICULT_MOAT_CONFLICTING_EVIDENCE`.
3. A normal v1.0 run tests generalization without a targeted prompt.
4. Its historical failure is specific and auditable:
   peer replicability / competition evidence was assigned the wrong role.
5. A PASS would materially strengthen the hypothesis that v1.0 corrected a
   reusable semantic-contract defect rather than an Adyen-specific wording issue.
6. A FAIL would show that Adyen v1.0 success does not generalize and that Phase B
   requires another contract iteration before Phase C.

## 10. Why no new higher-model contrast yet

A new TERRA / SOL / ASTRA call is not currently the highest-information action.

Reason:

```text
historical Adyen TERRA already failed the old semantic problem
+
v1.0 changed the analytical contract materially
+
cross-company generalization of v1.0 LUNA is still unproven
```

Testing Brookfield × LUNA v1.0 first separates:

```text
contract generalization
from
physical-model escalation
```

Only after that observation should another physical-model contrast be proposed.

## 11. Decision logic after Brookfield v1.0

If Brookfield × LUNA v1.0 is:

### PASS

Then Phase B will have evidence that the v1.0 role-alignment contract works:

```text
Adyen normal path
+
Adyen targeted semantic path
+
Brookfield independent-company normal path
```

At that point the preferred next move is to prepare Phase C rather than
automatically purchase another model contrast.

### FAIL with the same role-classification class

Then:

```text
Phase B remains open
contract learning resumes
no higher-model escalation is automatic
```

### FAIL for a new model-capability reason

Then one targeted physical-model contrast may be justified, but only after a
new explicit synthesis and authorization.

## 12. Proposed Phase B exit boundary

Phase B should not be declared complete merely because one targeted probe passed.

A reasonable minimum evidence boundary before proposing transition to Phase C is:

```text
Phase A physical smoke = PASS
historical five-company pilot = complete
human-quality contract-learning trail = complete
v1.0 Adyen normal path = human-quality admissible
v1.0 targeted semantic probe = PASS
v1.0 Brookfield normal path = observed and adjudicated
no unresolved critical semantic-contract defect
separate Phase B closure artifact
```

This is an execution proposal, not a frozen Gate 18 law.

## 13. Current gate state

```text
Phase A = PASS
Phase B = IN PROGRESS

historical five-company LUNA sweep = COMPLETE
historical Adyen × TERRA contrast = COMPLETE / FAIL
Adyen v1.0 normal canary = HUMAN-QUALITY ADMISSIBLE / PARTIAL REGRESSION COVERAGE
Adyen v1.0 targeted probe = PASS

v1.0 cross-company generalization = NOT YET PROVEN
next proposed observation = Brookfield × LUNA v1.0 normal path
next paid execution authorized = NO

model winner = NONE
routing freeze = NONE
publication authority = FALSE
production mutation = FALSE
Gate 18 = IN PROGRESS / NOT FROZEN
```
