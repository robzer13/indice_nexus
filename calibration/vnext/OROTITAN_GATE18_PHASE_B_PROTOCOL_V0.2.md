# OROTITAN_GATE18_PHASE_B_PROTOCOL_V0.2

**Project:** OroTitan Equity Research  
**Gate:** 18  
**Phase:** B company-level calibration  
**Status:** EXECUTION CANDIDATE — NOT FROZEN  
**Methodology change:** NO  
**Production mutation:** FORBIDDEN  
**Publication authority:** DISABLED  

## 0. Why v0.2 exists

The first RATIONAL AG Phase B attempt used the broad
`EVIDENCE_AUDIT_ASSISTED_V0_1` packet and completed zero valid model outputs.

Observed paid attempt:

```text
models attempted = 4
valid model outputs = 0
runner-observed gateway cost = 0.5775745 USD
Vercel displayed project/team spend after attempt = 0.68 USD
```

The failure did not establish analytical model quality. It established that the
v0.1 execution envelope was too weakly bounded for a cost-sensitive first
company calibration.

v0.2 therefore changes only the calibration execution envelope. It does not
change frozen OroTitan analytical methodology.

## 1. Cost-discipline law

No paid multi-model call is a default action.

Real execution now requires:

```text
explicit --execute
+ explicit --model
+ explicit --max-case-spend-usd
+ valid OIDC
```

More than one physical model additionally requires:

```text
--allow-multi-model
```

Before every provider call:

```text
OBSERVED CASE SPEND
+ CONSERVATIVE LIVE-PRICE CEILING FOR NEXT MODEL
<= EXPLICIT CASE CAP
```

Otherwise the call is blocked before inference.

## 2. Module-specific packet

v0.1 sent a broad Research evidence projection.

v0.2 starts with one common, bounded calibration module:

```text
MOAT_EVIDENCE_AUDIT_ASSISTED_V0_2
scope = MOAT_INPUTS
```

The packet includes only:

```text
evidence explicitly tagged MOAT_INPUTS
+ ALL_DD_INPUTS evidence
+ evidence referenced by a MOAT-scoped conflict
+ MOAT-scoped conflicts
```

This is deterministic filtering. The same scoped packet is supplied to every
physical model compared for the same company.

The filter does not make a moat judgment. It only reduces irrelevant context.

## 3. Research-ledger normalization

The five pinned pilot dossiers do not expose one identical physical JSON shape.

Observed authoritative evidence-array variants include:

```text
evidence_items
items
evidence
```

Observed item identifier / module-tag variants include:

```text
evidence_id | id
used_in     | blocks
```

Observed conflict identifier / scope variants include:

```text
conflict_id | id
MOAT_INPUTS | MOAT
```

v0.2 normalizes these known physical representations into one provider-neutral
calibration packet only after:

```text
raw artifact SHA-256 verification
run-id verification
DATA_CUTOFF verification
artifact-type verification
artifact-version verification
```

Unknown or ambiguous physical representations fail closed.

## 4. Bounded output contract

The v0.1 generation schema allowed materially larger narrative output than was
appropriate for an admission-stage calibration call.

v0.2 limits the model to:

```text
priority findings       <= 3
material conflicts      <= 2
weak-link candidates    <= 2
unresolved points       <= 3
```

Narrative fields have explicit maximum lengths and evidence/conflict reference
arrays have explicit maximum item counts.

A deterministic maximal synthetic v0.2 output fixture must serialize to no more
than 3,600 characters.

The provider max-output setting remains:

```text
1536 tokens
```

until real diagnostics prove that a higher budget is necessary. It is not
increased speculatively.

## 5. Human-judgment boundary

v0.2 remains ASSIST-only.

It may surface:

```text
decision-relevant evidence findings
material conflicts
counter-evidence
candidate weak links
unresolved questions
```

It may not render:

```text
final moat mechanism judgment
final moat durability judgment
runway judgment
valuation conclusion
OQS / OVS
Investment Score
next action
publication decision
investment conclusion
```

## 6. Next paid-call rule

No new paid inference is authorized merely because v0.2 merges.

Before another paid call:

```text
1. deterministic CI must pass
2. all five pilot ledgers must dry-run locally without inference
3. packet sizes and live-price ceilings must be inspected
4. the first paid v0.2 execution must select exactly one low-cost model
5. only a valid structured + semantic result can justify testing another model
```

Gate 18 remains:

```text
IN PROGRESS
NOT FROZEN
MODEL WINNER = NONE
```
