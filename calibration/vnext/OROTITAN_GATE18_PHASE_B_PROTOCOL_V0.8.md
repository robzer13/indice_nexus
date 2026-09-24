# OROTITAN_GATE18_PHASE_B_PROTOCOL_V0.8

**Gate:** 18  
**Phase:** B company calibration  
**Status:** ORTHOGONAL EVIDENCE-QUALIFICATION HARDENING CANDIDATE — NOT FROZEN  
**Paid execution:** NOT AUTHORIZED BY THIS DOCUMENT  
**Methodology:** UNCHANGED  
**Scope:** `MOAT_INPUTS`  
**Publication authority:** DISABLED  
**Production mutation:** FORBIDDEN  
**Model winner:** NONE  
**Routing freeze:** NONE  

## 1. Trigger

Protocol v0.8 follows the Adyen × LUNA v0.7 canary.

The v0.7 targeted regression succeeded:

```text
E-043
v0.6 = counterevidence
v0.7 = qualification
```

That demonstrated that a separate qualification concept is necessary.

However, v0.7 also failed deterministic semantics because E-040 was used both as support and qualification.

That dual use is analytically legitimate:

```text
E-040 supports:
new enterprise wins exist

E-040 qualifies:
selected wins do not quantify their revenue contribution
```

Therefore, evidence direction and evidence qualification are not competing roles.

They are orthogonal dimensions.

## 2. Unchanged analytical boundary

v0.8 does not change the MOAT methodology.

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
protocol = 0.8
```

Prompt:

```text
promptTemplateId = GATE18_MOAT_EVIDENCE_AUDIT_V0_6
promptTemplateVersion = 0.6
```

Generation schema:

```text
generationSchemaId = GATE18_MOAT_EVIDENCE_AUDIT_OUTPUT_V0_6
generationSchemaVersion = 0.6
```

## 4. Direction is exclusive

Each evidence item cited in a finding has one directional role:

```text
evidence_ids
= SUPPORT

counterevidence_ids
= COUNTEREVIDENCE
```

The same evidence ID must never appear in both directional arrays for the same finding.

Direction answers:

```text
Does this observation support or weaken the exact claim?
```

## 5. Qualification is orthogonal

Qualification is represented through:

```text
evidence_qualifications: [
  {
    evidence_id: "E-NNN",
    qualification: "..."
  }
]
```

A qualification entry does not create a new directional role.

It answers:

```text
What limits the strength, scope, representativeness,
measurement quality, validation quality, denominator certainty,
or generalizability of this already directionally classified evidence?
```

Every qualification ID must already exist in either:

```text
evidence_ids
or
counterevidence_ids
```

An orphan qualification is invalid.

## 6. Adyen regression examples

### E-043

For a bounded payment-performance claim:

```text
direction:
SUPPORT

qualification:
issuer-hosted selected customer case;
not representative
```

### E-040

For a claim that Adyen continues to win new enterprises:

```text
direction:
SUPPORT

qualification:
selected wins;
revenue contribution not quantified
```

For the unresolved growth-attribution conflict, E-040 may remain SUPPORT while the qualification constrains the economic inference.

### E-037

For a claim that integration creates durable switching friction:

```text
direction:
COUNTEREVIDENCE

qualification:
optional if the packet contains a material scope limitation
```

## 7. Support-state semantics

The vocabulary remains:

```text
SUPPORTED
MIXED
UNRESOLVED
```

Qualification alone does not create MIXED status.

A MIXED finding must still expose:

```text
counterevidence
or
a material conflict
```

A finding may therefore be:

```text
SUPPORTED
+ supporting evidence
+ qualification metadata
+ no counterevidence
```

## 8. Deterministic invariants

v0.8 preserves canonical-ID and narrative checks and enforces:

```text
canonical E-NNN / C-NNN format
packet membership
unique support refs
unique counterevidence refs
no support/counterevidence overlap
unique qualification entry per evidence ID
each qualification references a directionally cited evidence ID
qualification narrative is complete
counterevidence rationale required iff counterevidence exists
MIXED requires counterevidence or material conflict
conflict grounding uses directional evidence refs
```

Unlike v0.7, v0.8 intentionally allows:

```text
support evidence + qualification metadata
counterevidence + qualification metadata
```

because qualification is not a direction.

## 9. Deliberate validator boundary

The validator can prove structural consistency.

It does not prove natural-language economic truth.

Deterministic:

```text
ID format
packet membership
direction exclusivity
qualification attachment
qualification uniqueness
conflict linkage
required rationale presence
```

Human-adjudicated:

```text
whether direction is economically correct
whether a stated qualification is materially relevant
whether claim wording is properly bounded
whether evidence quality supports the inference
```

## 10. Regression tests

The v0.8 test pack covers:

```text
qualification on SUPPORT
qualification on COUNTEREVIDENCE
E-040-style support + qualification
support/counter direction overlap rejection
orphan qualification rejection
duplicate qualification rejection
malformed qualification ID rejection
qualification alone cannot justify MIXED
SUPPORTED + qualification without opposition
runner routes through v0.8
common 4096 output envelope remains unchanged
existing paid-call spend guards remain unchanged
```

## 11. Execution state

After merge and green CI:

```text
Gate 18 = IN_PROGRESS_NOT_FROZEN
Phase B = IN_PROGRESS
v0.8 contract = implementation candidate
paid v0.8 canary = NOT AUTHORIZED
automatic retry = NOT AUTHORIZED
TERRA escalation = NOT AUTHORIZED
SOL escalation = NOT AUTHORIZED
ASTRA escalation = NOT AUTHORIZED
model winner = NONE
routing freeze = NONE
```

The next step after green CI is a no-cost Adyen dry-run to pin the v0.8 prompt/schema hashes and cost ceilings. Any paid canary requires a separate authorization.
