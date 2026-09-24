# OROTITAN_GATE18_PHASE_B_PROTOCOL_V0.9

**Gate:** 18  
**Phase:** B company calibration  
**Status:** ATOMIC-CLAIM HARDENING CANDIDATE — NOT FROZEN  
**Paid execution:** NOT AUTHORIZED BY THIS DOCUMENT  
**Methodology:** UNCHANGED  
**Scope:** `MOAT_INPUTS`  
**Publication authority:** DISABLED  
**Production mutation:** FORBIDDEN  
**Model winner:** NONE  
**Routing freeze:** NONE  

## 1. Trigger

Protocol v0.9 follows the Adyen × LUNA v0.8 canary.

v0.8 solved the support-plus-qualification problem:

```text
E-040
direction = SUPPORT
qualification = selected wins; revenue contribution not quantified
```

That confirmed the orthogonal qualification model.

However, v0.8 exposed a different failure mode.

The model wrote compound, contrastive claims that contained both the proposition and its challenge.

Examples from the failed Adyen v0.8 output:

```text
"Retention and integration indicators coexist with customers' ability to shift payment volume."

"Adyen demonstrates competitive displacement and enterprise win capability despite multi-provider market structure."
```

In those claims, evidence such as multi-PSP behavior and provider switching was placed in `counterevidence_ids`.

But relative to the exact compound wording, that evidence supported the challenge clause already embedded inside the claim.

The result was logically inconsistent polarity.

## 2. Core v0.9 principle

Each finding must contain one atomic directional proposition.

The claim must not contain its own challenge.

The structure becomes:

```text
atomic claim
  ├── SUPPORT evidence
  ├── COUNTEREVIDENCE
  ├── material conflicts
  └── evidence qualifications
```

The opposing fact stays outside the claim wording.

## 3. Good and bad examples

### Invalid compound claim

```text
"Integration and low churn indicate persistence,
but customers can shift volume."
```

This is invalid because the claim contains both proposition and challenge.

### Valid atomic claim

```text
"Integration and low reported churn indicate customer relationship persistence."
```

Then:

```text
SUPPORT:
E-036
E-042

COUNTEREVIDENCE:
E-037
E-041

CONFLICT:
C-005
```

Now counterevidence actually weakens the exact claim.

## 4. Contrastive claim language

v0.9 explicitly rejects contrastive claim framing such as:

```text
but
despite
although
though
while
whereas
yet
however
nevertheless
nonetheless
coexist / coexists with
```

This is a deterministic guardrail.

It does not prove full semantic atomicity, but it blocks the specific observed v0.8 failure class.

## 5. Versioning

Execution protocol:

```text
protocol = 0.9
```

Prompt:

```text
promptTemplateId = GATE18_MOAT_EVIDENCE_AUDIT_V0_7
promptTemplateVersion = 0.7
```

Generation schema shape:

```text
generationSchemaId = GATE18_MOAT_EVIDENCE_AUDIT_OUTPUT_V0_6
generationSchemaVersion = 0.6
```

The generation schema version intentionally remains unchanged.

Reason:

```text
JSON shape = unchanged
semantic contract = changed
prompt = changed
validator = changed
```

This preserves the distinction between structural schema evolution and semantic protocol evolution.

## 6. Unchanged v0.8 evidence model

v0.9 keeps the v0.8 directional and qualification model.

Direction remains exclusive:

```text
evidence_ids
= SUPPORT

counterevidence_ids
= COUNTEREVIDENCE
```

Qualification remains orthogonal:

```text
evidence_qualifications = optional metadata
attached to an already directionally cited evidence item
```

A qualification does not change direction.

## 7. Weak-link label correction

The v0.8 deterministic validator treated:

```text
weak_link_candidates.candidate
```

as if it were a narrative sentence and required terminal punctuation.

That generated a false `semanticValid=false` for short labels such as:

```text
Switching-friction inference from low churn and integrations
```

v0.9 classifies `candidate` as a short label.

Terminal punctuation is therefore optional.

The explanatory field:

```text
why_uncertain
```

remains a narrative sentence and must still be complete.

## 8. Deterministic invariants

v0.9 enforces:

```text
canonical E-NNN / C-NNN format
packet membership
unique support refs
unique counterevidence refs
no SUPPORT / COUNTEREVIDENCE overlap
qualification attachment to cited directional evidence
unique qualification entry per evidence ID
counterevidence rationale iff counterevidence exists
MIXED requires counterevidence or material conflict
conflict grounding
complete narrative sentences
atomic-claim contrastive-language guard
weak-link candidate labels do not require punctuation
```

## 9. Human-adjudication boundary

The validator still cannot prove economic truth.

Human adjudication remains required for:

```text
whether the claim is genuinely atomic beyond lexical checks
whether support truly supports the proposition
whether counterevidence truly weakens the proposition
whether qualification is materially relevant
whether conflict treatment is economically coherent
whether the judgment boundary is respected
```

## 10. Adyen regression targets

The next Adyen v0.9 canary, if separately authorized after dry-run admission and fresh budget check, should test:

### Switching-friction finding

Expected claim form:

```text
"Integration and low reported churn indicate customer relationship persistence."
```

Expected direction:

```text
SUPPORT:
E-036
E-042

COUNTEREVIDENCE:
E-037
E-041 or E-051

CONFLICT:
C-005
```

The claim itself must not say that customers can switch or shift volume.

### Enterprise-win finding

Expected claim form:

```text
"Adyen demonstrates enterprise win and displacement capability."
```

Expected direction:

```text
SUPPORT:
E-040
E-041

qualification:
E-040 selected wins; contribution not quantified
E-041 multi-provider scope may limit breadth of inference

COUNTEREVIDENCE:
only if an item actually weakens win/displacement capability
```

Multi-provider architecture alone must not be labeled counterevidence unless it actually weakens the exact atomic proposition.

### Payment-performance finding

The v0.8 structure remains acceptable:

```text
SUPPORT:
E-044
E-045
E-046

qualification:
issuer-measured / denominator / audit limitations
```

## 11. Regression tests

The v0.9 test pack covers:

```text
protocol 0.9
prompt v0.7
schema shape remains v0.6
atomic claim accepted
v0.8 coexist-style claim rejected
v0.8 despite-style claim rejected
contrastive atomicity breakers rejected
weak-link label without punctuation accepted
weak-link explanation still requires complete sentence
orthogonal qualification preserved
direction overlap still rejected
runner routes through v0.9
common 4096 output envelope unchanged
historical paid-call guards preserved
```

## 12. Execution state

After merge and green CI:

```text
Gate 18 = IN_PROGRESS_NOT_FROZEN
Phase B = IN_PROGRESS
v0.9 contract = implementation candidate
paid v0.9 execution = NOT AUTHORIZED
automatic retry = NOT AUTHORIZED
TERRA escalation = NOT AUTHORIZED
SOL escalation = NOT AUTHORIZED
ASTRA escalation = NOT AUTHORIZED
model winner = NONE
routing freeze = NONE
```

The next step after green CI is a no-cost Adyen v0.9 dry-run to pin the new prompt hash and cost ceilings.

Any paid canary requires a separate authorization after a fresh budget check.
