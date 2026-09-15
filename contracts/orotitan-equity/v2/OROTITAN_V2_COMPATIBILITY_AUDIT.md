# OROTITAN V2 — COMPATIBILITY AUDIT

**Status:** ACTIVATION CANDIDATE AUDIT  
**Target:** OroTitan Execution Process V2.0  
**Rule:** fail closed on any unresolved semantic or version conflict.

## B01 — Analysis Standard

**Result: PASS**

The frozen Analysis Standard already makes scoring conditional on certification and assigns publication authority to `SCORE_PERMISSION`. Its rules include:

```text
BUSINESS_RESEARCH_STATUS = NOT_CERTIFIED -> NO OQS
SCORE_PERMISSION = SUSPENDED -> NO FINAL SCORE
VALUATION NOT CERTIFIABLE -> OQS may exist, OVS / INVESTMENT_SCORE prohibited
```

V2 changes execution sequencing only: Research and Fundamentals withhold final scores, Certification establishes certification state and score permission, then deterministic scoring is exposed. No analytical definition, weight, cap, threshold or semantic state is changed.

## B02 — Master Prompt

**Result: PASS**

The frozen Master Prompt prohibits discretionary score override and requires a changed dimension judgment to update rationale/evidence/version before deterministic recomputation. It also explicitly permits price alone to change OVS and Investment Score without changing OQS and forbids deriving terminal OroTitan status from score bands.

V2 limited reopen and targeted refresh preserve these rules. V2 introduces no analyst score override.

## B03 — Investment Policy

**Result: PASS — REUSED UNCHANGED**

`OROTITAN_INVESTMENT_POLICY_V1.0.0` remains the investment-policy authority. V2 introduces no new target return, margin-of-safety threshold, portfolio rule, valuation threshold, score weight or discretionary exception.

## B04 — Execution Patch

**Result: PASS — REUSED UNCHANGED**

`OROTITAN_EXECUTION_CONTRACT_PATCH_V1.0.1` remains higher-authority compatibility input. V2 does not weaken identity locks, persistence-before-handoff rules, point-in-time discipline, fail-closed behavior or publication authorization separation.

## B05 — Integration Spec

**Result: PASS — V2 SUCCESSOR**

V2 uses `04_INTEGRATION_SPEC_V2`. It preserves V1 analytical mapping and deterministic computation and adds only the V2 product projection plus a separate V2 persistence path. Integration remains non-analytical and cannot reconstruct or improve upstream analysis.

## B06 — Canonical Screener Schema

**Result: PASS — V2 SUCCESSOR**

V2 uses `04_SCREENER_SCHEMA_V2` as a composed overlay:

```text
unchanged V1 core schema
+
strict V2 product extension
```

The V2 validator removes only `v2_product` before applying the exact V1 core validator, then validates the V2 extension independently. Historical V1 snapshots remain valid and are not backfilled.

## B07 — I2

**Result: PASS — REUSED UNCHANGED**

I2 deterministic computation is unchanged. V2 Certification calls the same deterministic calculation authority after certification/score-permission state is established. V2 product taxonomy has zero scoring authority.

## B08 — I3-B

**Result: PASS — SEMANTICS PRESERVED, ADDITIVE V2 ADAPTER**

The V1 I3-B writer remains installed and V1-only. V2 adds a separate `persist_orotitan_research_snapshot_v2` function that preserves:

```text
validated payload before write
immutable snapshot rows
CAS dossier-pointer update
idempotent replay semantics
server-only SECURITY DEFINER boundary
service_role-only EXECUTE
direct table writes forbidden
```

The database accepts only the exact version pairs:

```text
04_SCREENER_SCHEMA_V1 / 1.0.0
04_SCREENER_SCHEMA_V2 / 2.0.0
```

A payload-version firewall rejects V2 product payloads on the V1 storage pair and requires a V2 product object on the V2 pair. No historical V1 snapshot is rewritten.

## B09 — Registry / Stage Manifest

**Result: PASS — REUSED MODEL**

Registry stage codes remain exactly:

```text
RESEARCH
DEEP_DIVE
INTEGRATION
```

Fundamentals and Valuation are internal Deep Dive phases and use `CHECKPOINT` Stage Manifests. A CHECKPOINT cannot admit Integration. Certification alone may produce the FINAL Deep Dive manifest. No new Registry table or stage code is introduced solely for V2 phase splitting.

The 13 logical contract-pin keys remain unchanged. V2 changes versions/locators for V2 authorities while reusing unchanged higher authorities under immutable V1 locators.

## Version firewall conclusion

```text
V1 SNAPSHOTS / RUNS           GRANDFATHERED
V1 WRITER                     PRESERVED
V2 NEW RUNS                   V2 PINS ONLY AFTER ACTIVATION
V1/V2 SILENT MIXING           PROHIBITED
ANALYTICAL METHOD             UNCHANGED
SCORING FORMULAS              UNCHANGED
VALUATION POLICY              UNCHANGED
I2 DETERMINISM                UNCHANGED
CANONICAL PUBLICATION GATE    GO PUBLISH REMAINS REQUIRED
```

Any contradiction found during Contract Pin Pack construction, CI, Vercel preview, production preflight or postflight invalidates this PASS and blocks activation.
