# OROTITAN_VNEXT_RECOVERY_ENGINE_V0.1

**Project:** OroTitan Equity Research  
**Status:** FROZEN — GATE 11 PASS  
**Methodology change:** NO  
**Depends on:** Gates 6 through 10  
**AI dependency:** NONE  
**Production mutation:** NONE  
**Shadow DB mutation:** NONE

## 0. Purpose

Gate 11 classifies and executes only recovery actions whose safety can be proven deterministically.

The fixed recovery classes are:

```text
AUTO_RETRY
DETERMINISTIC_AUTO_REPAIR
VERIFIABLE_RECOVERY
HUMAN_REQUIRED
```

Core rule:

```text
AUTOMATE ONLY WHAT IS PROVEN
AMBIGUITY OR SEMANTIC AUTHORITY RISK
=> HUMAN_REQUIRED
```

## 1. Historical immutability

Recovery is not history rewriting.

Before an automatic action, Gate 11 snapshots the existing run-history entries.

After the action, every pre-existing entry must remain byte/fingerprint-identical as an ordered prefix.

Allowed recovery may append new recovery evidence/events or repair current routing state within a frozen safe scope.

It may never:

```text
rewrite a sealed historical artifact
rewrite a historical event
silently change analytical meaning
silently change DATA_CUTOFF
silently resolve contract drift
make an ambiguous identity choice
substitute machine action for required analyst judgment
```

Any such condition forces `HUMAN_REQUIRED`.

## 2. AUTO_RETRY

Typical use: a transient execution failure such as the ASM regression case in the VNext roadmap.

Automatic retry requires all of:

```text
same operation identity available
same request fingerprint proven
operation is idempotent
retry budget remains
attempt < max_attempts
no committed mutation occurred
expected state_version is still fresh
no semantic/historical-risk flag
```

Action:

```text
RETRY_SAME_REQUEST
```

Mutation scope:

```text
EXECUTION_RETRY_ONLY
```

A transient label by itself is never sufficient.

## 3. DETERMINISTIC_AUTO_REPAIR

Typical use: a manifest-routing mismatch such as the STMicro regression case, but only when every required fact is proven.

Automatic repair requires:

```text
repair target is unique
authoritative target resolves exactly
target is SEALED
target is AVAILABLE
target bytes hash-verify
target contract pins match
expected state_version is fresh
no analytical meaning changes
no historical record is rewritten
```

Action:

```text
REBIND_PROVEN_MANIFEST_POINTER
```

Mutation scope:

```text
CURRENT_ROUTING_METADATA_ONLY
```

If there are two plausible targets, a hash failure, contract drift, or semantic ambiguity, the classification becomes `HUMAN_REQUIRED`.

## 4. VERIFIABLE_RECOVERY

Typical use: an unresolved durable locator such as the Topicus regression case.

Automatic verifiable recovery requires:

```text
expected content SHA-256 is known
candidate locator is durable
candidate bytes resolve
candidate bytes SHA-256 == expected SHA-256
artifact identity is exact
expected state_version is fresh
no semantic or historical-risk flag
```

Action:

```text
RESOLVE_AND_VERIFY_LOCATOR
```

Mutation scope:

```text
VERIFIED_LOCATOR_BINDING_ONLY
```

The recovery is accepted because content identity is independently proved, not because the candidate location looks plausible.

## 5. HUMAN_REQUIRED

Gate 11 fails closed to `HUMAN_REQUIRED` for:

```text
analytical ambiguity
identity ambiguity
unresolved contract drift
required DATA_CUTOFF change
unproven hash
unproven locator
unproven manifest target
retry fingerprint mismatch
stale state_version
retry budget exhaustion
non-idempotent operation
possible committed partial mutation
any historical rewrite requirement
any analytical-meaning change
unknown incident without a frozen automatic rule
```

The engine returns a structured escalation plan. It does not improvise a repair.

## 6. Execution protocol

Implementation:

```text
runtime/vnext/recovery-engine.ts
```

Protocol:

```text
CLASSIFY
-> READ HISTORY SNAPSHOT
-> if HUMAN_REQUIRED: STOP WITHOUT AUTOMATIC MUTATION
-> otherwise APPLY EXACT ALLOWED ACTION
-> REREAD HISTORY
-> VERIFY OLD HISTORY IS UNCHANGED PREFIX
-> return receipt
```

Every automatic action requires a fresh reread.

## 7. Regression fixtures

Implementation tests:

```text
tests/vnext-recovery-engine.test.ts
```

The three roadmap incidents are encoded only at the abstraction level established by the roadmap:

```text
ASM
transient + idempotent + same request + no committed mutation
=> AUTO_RETRY

STMicro
manifest pointer mismatch + one fully proven authoritative target
=> DETERMINISTIC_AUTO_REPAIR

Topicus
locator unresolved + exact identity + candidate bytes hash-verified
=> VERIFIABLE_RECOVERY
```

The tests deliberately remove one proof from STMicro and Topicus fixtures and verify that automatic recovery is then refused.

They do not invent additional historical facts about those incidents.

## 8. Negative assurance

Gate 11 tests also prove:

```text
fingerprint mismatch                  -> HUMAN_REQUIRED
stale state_version                   -> HUMAN_REQUIRED
retry budget exhausted                -> HUMAN_REQUIRED
analyst judgment required             -> HUMAN_REQUIRED
historical artifact rewrite requested -> HUMAN_REQUIRED
contract drift unresolved             -> HUMAN_REQUIRED
DATA_CUTOFF change required           -> HUMAN_REQUIRED
unknown incident                      -> HUMAN_REQUIRED
HUMAN_REQUIRED                        -> no automatic mutation
old history changed during recovery   -> fatal detection
```

## 9. Gate 11 acceptance matrix

```text
G11-01 fixed four-class taxonomy                    PASS
G11-02 AUTO_RETRY proof conditions                  PASS
G11-03 DETERMINISTIC_AUTO_REPAIR proof conditions   PASS
G11-04 VERIFIABLE_RECOVERY proof conditions         PASS
G11-05 ambiguous cases HUMAN_REQUIRED               PASS
G11-06 semantic-change firewall                     PASS
G11-07 historical-artifact rewrite firewall         PASS
G11-08 historical-event rewrite firewall            PASS
G11-09 state-version freshness                      PASS
G11-10 retry idempotency/fingerprint                 PASS
G11-11 retry-budget control                          PASS
G11-12 manifest repair requires unique target        PASS
G11-13 manifest target hash verification             PASS
G11-14 locator recovery requires byte hash proof     PASS
G11-15 ASM regression fixture                        PASS
G11-16 STMicro regression fixture                    PASS
G11-17 Topicus regression fixture                    PASS
G11-18 prior history prefix immutability             PASS
G11-19 HUMAN_REQUIRED causes no automatic mutation   PASS
G11-20 deterministic tests                           PASS
G11-21 no ChatGPT / model dependency                 PASS
G11-22 production mutation                           NONE
G11-23 shadow DB mutation                            NONE
G11-24 VNext CI                                      PASS
```

## 10. Freeze record

```text
GATE                              = 11
RESULT                            = PASS
IMPLEMENTATION                    = runtime/vnext/recovery-engine.ts
TESTS                             = tests/vnext-recovery-engine.test.ts
VALIDATED_IMPLEMENTATION_HEAD     = 9842b1c40d7bba0b6ddbf1826c8686acc064cd72
CI_RUN                            = 46
CI_RESULT                         = SUCCESS
ANALYTICAL_METHODOLOGY_CHANGE     = NONE
PRODUCTION_MUTATION               = NONE
SHADOW_DB_MUTATION                = NONE
```

V0.1 is frozen as the Gate 11 recovery-classification authority.

Any new automatic recovery case requires an explicit rule and regression fixture. Unknown cases remain `HUMAN_REQUIRED`.

## 11. Out of scope

Gate 11 does not:

- invent a new analytical conclusion;
- change scoring or valuation methodology;
- authorize publication;
- replace the Stage Preflight;
- replace Post-Stage Certification;
- bind the recovery actions to production Supabase;
- invoke ChatGPT or an LLM.
