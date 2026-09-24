# OROTITAN_GATE18_PHASE_B_PROTOCOL_V1.0

**Gate:** 18  
**Phase:** B company calibration  
**Status:** CLAIM-TARGET ALIGNMENT CANDIDATE — NOT FROZEN  
**Paid execution:** NOT AUTHORIZED BY THIS DOCUMENT  
**Methodology:** UNCHANGED  
**Scope:** `MOAT_INPUTS`  
**Publication authority:** DISABLED  
**Production mutation:** FORBIDDEN  
**Model winner:** NONE  
**Routing freeze:** NONE  

## 1. Trigger

Protocol v1.0 follows the Adyen × LUNA v0.9 canary.

v0.9 achieved:

```text
schemaValid = true
semanticValid = true
claim atomicity = PASS
qualification attachment = PASS
```

but remained non-admissible under human adjudication because atomic syntax did not guarantee semantic alignment between an exact claim and its assigned counterevidence.

Two failures isolate the defect.

### Low reported churn

```text
claim:
"Adyen's customer relationships have low reported churn."

support:
E-036

assigned counterevidence:
E-037
E-041
```

E-037 / E-041 may weaken a switching-friction inference, but they do not make the exact descriptive proposition "reported churn is low" less true.

### Service disruption

```text
claim:
"Adyen's payment infrastructure has experienced service disruption during the measured period."

support:
E-055

assigned counterevidence:
E-056
```

High availability during another event can coexist with the documented incident. It therefore does not refute the exact existence claim.

## 2. Core v1.0 principle

```text
ATOMICITY
≠
CLAIM-TARGET ALIGNMENT
```

A finding is valid only when support and counterevidence bear on the same semantic target.

Before assigning an item to `counterevidence_ids`, apply:

```text
If this evidence is true,
does the exact claim become materially:

- less likely,
- less strong,
or
- less economically valid?

YES → counterevidence is possible
NO  → it is not counterevidence to that claim
```

This is the counterfactual counterevidence test.

## 3. Semantic-level rule

When opposition in the packet challenges an economic inference rather than a raw observation, the finding should be written at the inferential level if that is the decision-relevant proposition.

Preferred Adyen form:

```text
"Low churn and integration indicate meaningful customer switching friction."
```

Then:

```text
SUPPORT
E-036
E-042

COUNTEREVIDENCE
E-037
E-041 / E-051

CONFLICT
C-005
```

Support and opposition now address the same economic proposition.

If the analyst/model instead chooses a purely descriptive claim such as:

```text
"Reported churn is low."
```

then evidence that only challenges switching friction must not be placed in `counterevidence_ids`.

## 4. Coexistence rule

Evidence is not counterevidence merely because it points in a different qualitative direction.

If both statements can be fully true at the same time, opposition has not been established.

Examples:

```text
documented service incident
+
high uptime during another event
= compatible observations
```

and:

```text
low reported churn
+
customers can multi-source / shift volume
= compatible observations
```

The second observation may challenge an inference from low churn, but not the low-churn observation itself.

## 5. Scope-resolved conflict rule

A conflict resolved by scope must not automatically produce `MIXED`.

After respecting the scope distinction, ask whether residual evidence still weakens the exact claim.

```text
resolved-by-scope conflict
+
no residual weakening of exact claim
→ not a MIXED basis
```

## 6. Versioning

Execution protocol:

```text
protocol = 1.0
```

Prompt:

```text
promptTemplateId = GATE18_MOAT_EVIDENCE_AUDIT_V0_8
promptTemplateVersion = 0.8
```

Generation schema:

```text
generationSchemaId = GATE18_MOAT_EVIDENCE_AUDIT_OUTPUT_V0_6
generationSchemaVersion = 0.6
```

The schema remains unchanged by design.

```text
JSON shape = unchanged
packet = unchanged
qualification model = unchanged
4096-token envelope = unchanged
semantic prompt contract = changed
```

## 7. Preserved v0.9 controls

v1.0 preserves:

```text
atomic directional claims
contrastive-language guard
SUPPORT / COUNTEREVIDENCE exclusivity
orthogonal evidence_qualifications
canonical packet ID validation
conflict grounding
complete narrative sentence checks
weak-link short-label handling
zero automatic retry
zero automatic model escalation
```

## 8. Deterministic validator boundary

No new lexical heuristic attempts to prove claim-target alignment.

Reason:

```text
semantic target alignment
cannot be reliably inferred
from vocabulary alone
```

The deterministic validator continues to prove structural consistency.

The prompt contract plus human adjudication evaluate economic polarity.

## 9. Regression targets

The v1.0 no-cost regression pack must verify that the prompt explicitly requires:

```text
counterfactual counterevidence test
same semantic target for support/opposition
coexistence is insufficient for counterevidence
descriptive-vs-inferential level correction
resolved-by-scope conflict handling
```

It must also verify:

```text
protocol = 1.0
prompt = v0.8
schema = v0.6 unchanged
schema hash = v0.9 schema hash
runner routes through v1.0
maxOutputTokens = 4096
```

## 10. Adyen adjudication targets

A later Adyen execution, only if separately authorized after no-cost admission and a fresh budget snapshot, should satisfy all of the following.

### Switching-friction target

A MIXED finding should be framed at an economically contestable level such as switching friction or relationship persistence, not merely the descriptive fact of low reported churn.

### Service-reliability target

If the claim is only that an incident occurred, high uptime in a separate scoped event must not be counterevidence.

If the finding instead concerns broad infrastructure reliability, the claim must be framed so both incident evidence and uptime evidence genuinely bear on the same proposition.

### Qualification target

The v0.8/v0.9 correction remains mandatory:

```text
E-040
= SUPPORT
+ qualification when selected wins do not quantify revenue contribution
```

and issuer-hosted or selected evidence may remain SUPPORT when directionally supportive, with its limits recorded as qualifications.

## 11. Execution state

After merge and green tests:

```text
Gate 18 = IN_PROGRESS_NOT_FROZEN
Phase B = IN_PROGRESS
v1.0 contract = implementation candidate
paid v1.0 execution = NOT AUTHORIZED
automatic retry = NOT AUTHORIZED
TERRA escalation = NOT AUTHORIZED
SOL escalation = NOT AUTHORIZED
ASTRA escalation = NOT AUTHORIZED
model winner = NONE
routing freeze = NONE
```

The next permitted step is a no-cost Adyen v1.0 dry-run to pin prompt/schema hashes and cost ceilings.

Any paid canary requires a separate authorization after a fresh budget check.
