# OROTITAN_GATE18_PHASE_B_PROTOCOL_V0.7

**Gate:** 18  
**Phase:** B company calibration  
**Status:** STRUCTURAL EVIDENCE-ROLE HARDENING CANDIDATE — NOT FROZEN  
**Paid execution:** NOT AUTHORIZED BY THIS DOCUMENT  
**Methodology:** UNCHANGED  
**Scope:** `MOAT_INPUTS`  
**Publication authority:** DISABLED  
**Production mutation:** FORBIDDEN  
**Model winner:** NONE  
**Routing freeze:** NONE  

## 1. Trigger

Protocol v0.7 follows the Adyen × LUNA v0.6 canary.

v0.6 successfully fixed the deterministic reference-integrity problem:

```text
schemaValid = true
semanticValid = true
canonical E-NNN / C-NNN references = PASS
support/counter overlap = PASS
counterevidence rationale presence = PASS
```

But the human-quality target still failed.

The v0.6 Adyen finding stated, in bounded form:

```text
Adyen reports payment-performance benefits,
but broad validation is limited.
```

E-043 was then placed in `counterevidence_ids` even though it is another issuer-hosted selected customer case whose representativeness limitation reinforces the bounded claim rather than opposing it.

This exposes a structural problem:

```text
support
versus
counterevidence
```

is insufficient when some evidence supports the observation while limiting the strength or generalizability of the inference.

## 2. Unchanged analytical boundary

v0.7 does not change the MOAT methodology.

Unchanged:

```text
module = MOAT_EVIDENCE_AUDIT_ASSISTED_V0_3
source scope = MOAT_INPUTS
point-in-time packet = unchanged
packet lineage = unchanged
source-access policy = packet only
ASSIST-only authority = unchanged
publication authority = disabled
production mutation = forbidden
candidate physical model identities = unchanged
common maxOutputTokens = 4096
human adjudication remains mandatory
```

## 3. Versioned contract changes

Execution protocol:

```text
protocol = 0.7
```

Prompt:

```text
promptTemplateId = GATE18_MOAT_EVIDENCE_AUDIT_V0_5
promptTemplateVersion = 0.5
```

Generation schema:

```text
generationSchemaId = GATE18_MOAT_EVIDENCE_AUDIT_OUTPUT_V0_5
generationSchemaVersion = 0.5
```

## 4. Three-role evidence model

Every priority finding now separates three analytical roles.

### 4.1 Support

```text
evidence_ids
```

Contains evidence that directly supports the wording of the claim.

### 4.2 Qualification

```text
qualification_evidence_ids
qualification_link
```

Contains evidence that supports or contextualizes the claim but limits:

```text
strength
representativeness
generalizability
measurement quality
validation quality
denominator certainty
selection-bias confidence
```

Qualification is explicitly **not opposition**.

Typical examples:

```text
issuer-hosted selected customer case
issuer-measured outcome without independent audit
non-representative case study
unclear denominator
selected transaction sample
reported metric with incomplete definition
```

### 4.3 Counterevidence

```text
counterevidence_ids
counterevidence_link
```

Contains only observations that directly weaken or contradict the exact claim through:

```text
opposing mechanism
contrary observation
substitution
displacement
switching
economically inconsistent outcome
```

## 5. Adyen regression example

For a bounded claim such as:

```text
"Adyen reports payment-performance benefits,
but broad independent validation is limited."
```

the intended v0.7 classification is:

```text
support:
E-044
E-045
E-046

qualification:
E-043
because it is an issuer-hosted selected customer case
and is explicitly non-representative

counterevidence:
only evidence that actually shows the reported benefit does not exist,
reverses, disappears, or is economically contradicted
```

E-043 must no longer be forced into counterevidence merely because its evidentiary quality is limited.

## 6. Support state semantics

The existing support-state vocabulary remains:

```text
SUPPORTED
MIXED
UNRESOLVED
```

v0.7 clarifies:

```text
qualification alone does not make a claim MIXED
```

A `MIXED` finding must expose at least:

```text
counterevidence
or
a material conflict
```

A finding may therefore be:

```text
SUPPORTED + qualification
```

when the observation is supported but the inference must remain bounded.

## 7. Deterministic invariants

v0.7 preserves all v0.6 deterministic safeguards and adds:

```text
qualification_evidence_ids uses canonical E-NNN IDs
qualification refs are unique
qualification_link required iff qualification IDs exist
no orphan qualification_link
no support/qualification overlap
no qualification/counterevidence overlap
no support/counterevidence overlap
all three role channels count for conflict grounding
MIXED cannot be justified by qualification alone
```

The validator remains fail-closed.

## 8. Deliberate validator boundary

The deterministic validator still does not claim to understand natural-language economic truth.

It can prove:

```text
ID format
packet membership
role-channel exclusivity
presence of required rationale
conflict linkage
structural consistency
```

It cannot prove:

```text
that an item truly supports the claim
that an item truly qualifies rather than opposes the claim
that an item truly contradicts the claim
```

Those remain human-adjudicated until upstream Research artifacts encode normalized claim-relative polarity.

## 9. Regression tests

The v0.7 test pack explicitly covers:

```text
valid three-role finding
malformed qualification ID
support/qualification overlap
qualification/counterevidence overlap
missing qualification explanation
orphan qualification explanation
qualification alone cannot justify MIXED
SUPPORTED finding may contain qualification without opposition
conflict grounding recognizes all three evidence roles
runner routes through v0.7
common 4096 output envelope remains unchanged
existing spend guards remain unchanged
```

## 10. Execution state

After merge and green CI:

```text
Gate 18 = IN_PROGRESS_NOT_FROZEN
Phase B = IN_PROGRESS
v0.7 contract = implementation candidate
paid v0.7 canary = NOT AUTHORIZED
automatic retry = NOT AUTHORIZED
TERRA escalation = NOT AUTHORIZED
SOL escalation = NOT AUTHORIZED
ASTRA escalation = NOT AUTHORIZED
model winner = NONE
routing freeze = NONE
```

The next step after green CI is a no-cost Adyen dry-run under v0.7 to pin prompt/schema hashes and cost ceilings. A paid canary requires a separate authorization.
