# OROTITAN_GATE18_PHASE_B_PROTOCOL_V0.6

**Gate:** 18  
**Phase:** B company calibration  
**Status:** CONTRACT HARDENING CANDIDATE — NOT FROZEN  
**Paid execution:** NOT AUTHORIZED BY THIS DOCUMENT  
**Methodology:** UNCHANGED  
**Scope:** `MOAT_INPUTS`  
**Publication authority:** DISABLED  
**Production mutation:** FORBIDDEN  
**Model winner:** NONE  
**Routing freeze:** NONE  

## 1. Trigger

Protocol v0.6 is a no-cost contract hardening response to observed Phase B failures.

Observed failure classes:

```text
RATIONAL x TERRA:
- malformed combined evidence references
- semanticValid=false

RATIONAL x SOL:
- non-canonical evidence reference
- semanticValid=false
- materially incomplete output

RATIONAL x ASTRA:
- non-canonical evidence reference
- semanticValid=false
- materially incomplete output

Brookfield x LUNA:
- canonical references
- semanticValid=true
- support/counterevidence role misclassification

Adyen x LUNA:
- canonical references
- semanticValid=true
- support/counterevidence role misclassification

Adyen x TERRA:
- malformed combined evidence references
- semanticValid=false
- support/counterevidence role misclassification
```

The evidence shows two distinct problems:

1. deterministic reference-format integrity;
2. analytical evidence-role polarity.

They must not be conflated.

## 2. Unchanged analytical boundary

v0.6 does not change the MOAT evidence-audit methodology.

Unchanged:

```text
module = MOAT_EVIDENCE_AUDIT_ASSISTED_V0_3
source scope = MOAT_INPUTS
point-in-time packet = unchanged
source-access policy = packet only
ASSIST-only authority = unchanged
publication authority = disabled
production mutation = forbidden
candidate physical model identities = unchanged
common maxOutputTokens = 4096
```

The packet builder remains the existing verified Gate 18 packet builder.

## 3. Versioned contract changes

Execution protocol:

```text
protocol = 0.6
```

Prompt:

```text
promptTemplateId = GATE18_MOAT_EVIDENCE_AUDIT_V0_4
promptTemplateVersion = 0.4
```

Generation schema:

```text
generationSchemaId = GATE18_MOAT_EVIDENCE_AUDIT_OUTPUT_V0_4
generationSchemaVersion = 0.4
```

The version bump is required because output semantics are stricter and one field is added.

## 4. Canonical ID hardening

Old contract:

```text
evidence ID pattern = ^E-
conflict ID pattern = ^C-
max length = 12
```

This admitted malformed strings such as:

```text
E-041','E-42
E-041','E-51
E-045','E-46
```

New contract:

```text
evidence ID pattern = ^E-\d{3}$
conflict ID pattern = ^C-\d{3}$
```

Each array element must therefore contain exactly one canonical ID.

## 5. Evidence-role hardening

For every priority finding:

```text
evidence_ids
= evidence that directly supports the exact wording of the claim

counterevidence_ids
= evidence that directly weakens, qualifies, or contradicts the exact wording of the claim
```

Polarity is relative to the claim, not to the company.

Example:

```text
claim:
"Low churn does not establish durable switching friction."

supports skeptical claim:
- evidence of multi-sourcing
- evidence of provider replacement
- evidence of volume shifting

counters skeptical claim:
- evidence of genuinely low churn
- evidence of deep integration
```

A limitation attached to supportive evidence is not automatically counterevidence.

If an item reports another issuer-measured uplift with the same validation limitation, it supports a bounded claim such as:

```text
"Reported uplift exists but broad independent validation is limited."
```

It does not counter that bounded claim merely because the underlying evidence has a limitation.

## 6. New counterevidence explanation

Each priority finding now includes:

```text
counterevidence_link: string | null
```

Rules:

```text
counterevidence_ids non-empty
=> counterevidence_link required

counterevidence_ids empty
=> counterevidence_link must be null
```

The link must state how the cited counterevidence weakens the exact claim.

This does not make polarity deterministically provable. It makes the model's role assignment explicit and auditable.

## 7. New deterministic invariants

v0.6 deterministic semantic validation now checks:

```text
exact E-NNN / C-NNN format
packet membership
no duplicate support refs
no duplicate counterevidence refs
no support/counterevidence overlap within a finding
counterevidence explanation present iff counterevidence exists
MIXED finding exposes a conflict or counterevidence
cited conflict touches at least one evidence ref in the finding when conflict evidence refs are available
unique material conflict IDs
existing narrative-completeness constraints
```

These checks are fail-closed.

## 8. Deliberate validator boundary

The deterministic validator does **not** claim to prove that an evidence item semantically weakens a natural-language claim.

That would require either:

1. a richer upstream Research artifact that explicitly labels evidence polarity relative to a normalized claim; or
2. another model judgment, which would merely move the calibration problem.

Therefore:

```text
canonical format = deterministic
membership = deterministic
overlap/duplication = deterministic
conflict linkage = deterministic
presence of role rationale = deterministic
truth of support/counterevidence polarity = human adjudication
```

Human quality adjudication remains required before an observation is comparison-admissible.

## 9. Regression tests

The v0.6 test pack includes explicit regression coverage for:

```text
E-041','E-51 style concatenated IDs
support/counterevidence overlap
missing counterevidence explanation
orphan counterevidence explanation
MIXED output with no exposed challenge
conflict citation disconnected from finding evidence
runner still using one common max-output envelope
existing paid-call spend guards
```

No model call is required to validate these invariants.

## 10. Execution state

After merge and green CI:

```text
Gate 18 = IN_PROGRESS_NOT_FROZEN
Phase B = IN_PROGRESS
v0.6 contract = implementation candidate
paid v0.6 canary = NOT AUTHORIZED
SOL escalation = NOT AUTHORIZED
ASTRA escalation = NOT AUTHORIZED
model winner = NONE
routing freeze = NONE
```

A future paid canary requires a separate authorization after the v0.6 code and tests are green.
