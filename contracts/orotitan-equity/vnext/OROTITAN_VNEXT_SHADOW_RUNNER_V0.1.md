# OROTITAN_VNEXT_SHADOW_RUNNER_V0.1

**Project:** OroTitan Equity Research  
**Gate:** 17  
**Status:** IMPLEMENTATION CANDIDATE — NOT FROZEN  
**Methodology change:** NO  
**Production mutation:** FORBIDDEN  
**Publication authority:** DISABLED  
**Physical model calibration:** OUT OF SCOPE — GATE 18

## 0. Purpose

Gate 17 defines the VNext Shadow Runner used to execute the pinned
`AUDIT_CORPUS_VNEXT_V1` and compare VNext outputs against the frozen V2
published baseline without contaminating V2 or leaking V2 conclusions into
VNext execution.

Core sequence:

```text
PINNED POINT-IN-TIME MEMBER
-> EXECUTE VNEXT FROM SOURCE-RUN PROVENANCE
-> LOAD FROZEN V2 BASELINE AFTER VNEXT EXECUTION
-> COMPARE
-> RECORD EXPLICIT DIVERGENCES
-> ASSESS 33/33 EXECUTABILITY
-> VERIFY ZERO V2 POLLUTION
```

Gate 17 is runner/comparison infrastructure.

It does not select, calibrate, rank or freeze physical models.

## 1. Corpus authority

The Gate 17 corpus is exactly:

```text
calibration/vnext/AUDIT_CORPUS_VNEXT_V1.json
```

It contains exactly 33 published V2 dossiers pinned on 2026-09-22.

Each member pins at minimum:

```text
DISPLAY_NAME
ISSUER_ID
DOSSIER_ID
SECURITY_ID
V2_SNAPSHOT_ID
V2_SOURCE_RUN_ID
TICKER
EXCHANGE
DATA_CUTOFF
V2_PAYLOAD_SHA256
```

The corpus does not resolve "latest" at runtime.

A later production refresh must not silently replace any Gate 17 benchmark
member.

## 2. Point-in-time equality

The VNext execution input inherits the exact pinned:

```text
ISSUER_ID
DOSSIER_ID
SECURITY_ID
V2_SOURCE_RUN_ID
DATA_CUTOFF
```

`V2_SOURCE_RUN_ID` is the immutable provenance handle from which the exact
point-in-time Research / Deep Dive artifact lineage can be retrieved.

The runner must not advance `DATA_CUTOFF`.

## 3. Anti-answer-leakage firewall

The VNext executor interface must not receive:

```text
V2_STATE
V2 CANONICAL PAYLOAD
V2_PAYLOAD_SHA256
V2_SNAPSHOT_ID
V2 ANALYTICAL CONCLUSION
V2 SCORE
V2 VALUATION
V2 NEXT ACTION
```

as execution inputs.

Execution order is mandatory:

```text
1. EXECUTE VNEXT
2. LOAD V2 BASELINE
3. COMPARE
```

The baseline is comparison truth, not VNext analytical evidence.

## 4. V2 baseline admission

After VNext execution, the comparison baseline must resolve exactly:

```text
SNAPSHOT_ID = pinned V2_SNAPSHOT_ID
PAYLOAD_SHA256 = pinned V2_PAYLOAD_SHA256
```

Mismatch:

```text
-> MEMBER FAILED
-> DO NOT SILENTLY SUBSTITUTE ANOTHER SNAPSHOT
```

## 5. Divergence record

Every material comparison difference returned by the comparison layer uses
exactly these semantic fields:

```text
CHANGE_ID
V2_STATE
VNEXT_STATE
CAUSE
MODULE
EVIDENCE
ECONOMIC_INTERPRETATION
```

Rules:

```text
CHANGE_ID = non-blank and unique across one corpus execution
CAUSE = explicit, non-blank
MODULE = explicit, non-blank
EVIDENCE = one or more non-blank references
ECONOMIC_INTERPRETATION = explicit, non-blank
```

Gate 17 does not create a closed `CAUSE` taxonomy and does not score
divergences.

Zero divergences for a member is valid.

## 6. Failure isolation

One member failure must not erase the rest of the corpus run.

The runner attempts all 33 members and records each as:

```text
COMPLETE
or
FAILED
```

A member failure is explicit and prevents the Gate 17 exit condition.

No failed company may be silently omitted from the denominator.

## 7. Gate 17 exit condition

Gate 17 may pass only when:

```text
EXPECTED_MEMBERS = 33
ATTEMPTED_MEMBERS = 33
COMPLETED_MEMBERS = 33
FAILED_MEMBERS = 0
RUN_STATUS = COMPLETE
PRODUCTION_UNCHANGED = TRUE
```

Thus:

```text
33 / 33 EXECUTABLE
+
ZERO V2 POLLUTION
= GATE 17 PASS CANDIDATE
```

Passing Gate 17 does not validate analytical superiority of VNext.

## 8. Production pollution fingerprint

The pinned pre-Gate-17 production fingerprint records:

```text
research_dossiers
research_snapshots
orotitan_runs
orotitan_artifacts
dossier_pointer_sha256
snapshots_sha256
```

The same fingerprint must be observed after the Gate 17 acceptance execution.

Any difference means:

```text
VNEXT_SHADOW_PRODUCTION_POLLUTION_DETECTED
-> GATE 17 FAIL
```

This includes any change to a canonical current-snapshot pointer.

## 9. Shadow isolation

VNext target project:

```text
awgsurdyvsyolcgpnygh
orotitan-vnext-shadow
```

Production project:

```text
cugpgtzygqqlxetyeven
orotitan-screener
```

Gate 17 runtime contains no production write path.

The pure runner contains no:

```text
Supabase client
network call
environment secret
INSERT
UPDATE
UPSERT
DELETE
publication call
canonical writer call
```

Persistence adapters, if introduced later, must remain separately guarded by
the existing VNext shadow environment firewall.

## 10. No V2 reconstruction by convenience

The runner must compare against the exact pinned V2 snapshot.

It must not:

```text
recompute V2 from current data
resolve a newer V2 snapshot
rewrite old V2 conclusions
repair V2 retrospectively
normalize away a disagreement
treat VNext as canonical
```

The purpose is controlled comparison, not historical rewriting.

## 11. Model/provider boundary

Gate 17 is provider-neutral.

It does not define:

```text
LUNA
TERRA
SOL
ASTRA
Azure routing
OpenAI routing
Anthropic routing
Gemini routing
model quality ranking
model cost ranking
model latency ranking
```

Deterministic or synthetic executors are valid for runner assurance.

Real physical-model calibration belongs to Gate 18.

## 12. Analytical-methodology firewall

Gate 17 changes none of:

```text
frozen Research methodology
frozen Deep Dive methodology
OQS
OVS
Investment Score
Weak Link rules
valuation conventions
Certification
OroTitan terminal gate
GO PUBLISH authority
```

A divergence is evidence about implementation behavior, not a new analytical
state.

## 13. Implementation artifacts

```text
CONTRACT
= contracts/orotitan-equity/vnext/OROTITAN_VNEXT_SHADOW_RUNNER_V0.1.md

CORPUS
= calibration/vnext/AUDIT_CORPUS_VNEXT_V1.json

RUNTIME
= runtime/vnext/shadow-runner.ts

TESTS
= tests/vnext-shadow-runner.test.ts
```

## 14. Candidate acceptance tests

Before any Gate 17 freeze:

```text
G17-01 corpus is exactly 33 unique members
G17-02 all baseline snapshots are hash-pinned
G17-03 all source RUN_IDs are pinned
G17-04 VNext executor receives no V2 state/payload/hash/snapshot id
G17-05 VNext executes before V2 baseline load
G17-06 baseline snapshot mismatch fails member
G17-07 baseline hash mismatch fails member
G17-08 all 33 members are attempted despite isolated failure
G17-09 exact divergence field set is enforced
G17-10 duplicate CHANGE_ID fails closed
G17-11 zero divergences is valid
G17-12 33/33 deterministic execution passes runner condition
G17-13 production baseline fingerprint is pinned
G17-14 production fingerprint mutation fails Gate 17
G17-15 no production write surface exists in runner
G17-16 no Supabase/network/env/provider dependency exists in runner
G17-17 publication remains disabled
G17-18 physical-model calibration remains outside Gate 17
```

## 15. Freeze boundary

This document is an implementation candidate.

```text
GATE 17
!= FROZEN
```

until:

```text
implementation merged
verify-vnext = SUCCESS
verify-screener = SUCCESS
33/33 acceptance execution = PASS
production before/after fingerprint = IDENTICAL
separate Gate 17 freeze artifact = merged
```
