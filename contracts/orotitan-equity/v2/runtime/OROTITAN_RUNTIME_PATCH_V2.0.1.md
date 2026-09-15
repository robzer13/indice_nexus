# OROTITAN_RUNTIME_PATCH_V2.0.1

**Status:** IMPLEMENTATION PATCH — BLOCKING RUNTIME DEFECT  
**Applies to:** OroTitan V2 new runs  
**Does not alter:** analytical methodology, scoring, stage contracts, Contract Pin Pack V2, I2, I3-B, publication authorization

## 1. Defect corrected

A fresh Pilotage conversation could reconstruct the wrong operational environment or infer run type conversationally before persistent run creation. This could produce false blockers such as querying a non-authoritative Supabase project or selecting `INITIAL` when a canonical baseline already exists.

This is a blocking persistence / Registry execution defect under the V2.0 emergency-patch rule. It is not a methodology redesign.

## 2. Single runtime bootstrap

Before any Research work, Pilotage MUST load:

```text
contracts/orotitan-equity/v2/runtime/OROTITAN_RUNTIME_BOOTSTRAP_V2.0.1.json
```

The bootstrap is operational configuration only. Frozen contracts and the 13-pin Contract Pin Pack remain higher authority.

## 3. Environment firewall

The production Supabase environment is exact-allowlist only:

```text
PROJECT_REF  = cugpgtzygqqlxetyeven
PROJECT_NAME = orotitan-screener
REGION       = eu-west-3
STATUS       = ACTIVE_HEALTHY
```

Any mismatch is:

```text
WRONG_ENVIRONMENT
```

and Pilotage MUST stop before run creation or Research.

No alternative project may be treated as a fallback, legacy substitute or evidence that the Registry is missing.

## 4. Deterministic run routing

Run type is derived only from the current canonical dossier pointer:

```text
current_snapshot_id = NULL
=> RUN_TYPE = INITIAL
=> baseline_snapshot_id = NULL

current_snapshot_id != NULL
=> RUN_TYPE = REFRESH
=> baseline_snapshot_id = current_snapshot_id
```

Chat history, prior narrative and legacy tables have zero routing authority.

A V1 canonical snapshot is a valid REFRESH baseline for a V2 successor run. Existing V1 snapshots remain grandfathered and immutable.

## 5. Mandatory Pilotage order

```text
LOAD_RUNTIME_BOOTSTRAP
VERIFY_PRODUCTION_ENVIRONMENT
VERIFY_CONTRACT_PIN_PACK
RESOLVE_ISSUER_SECURITY_DOSSIER
RESOLVE_CURRENT_CANONICAL_SNAPSHOT
DETERMINE_RUN_TYPE
CREATE_PERSISTENT_RUN
BIND_IDENTITY_IF_REQUIRED
REQUERY_PERSISTED_RUN
BUILD_RUN_CONTEXT_V2
START_RESEARCH
```

Research MUST NOT start before a persistent `RUN_ID` has been created and re-read successfully.

## 6. Persistent run creation

Pilotage uses only the controlled Registry RPC:

```text
public.create_orotitan_run(...)
```

with the exact active Contract Pin Pack V2 and its deterministic `contract_set_sha256`.

For `INITIAL`, Pilotage binds the already-resolved security/dossier through:

```text
public.bind_orotitan_run_identity(...)
```

For `REFRESH`, security/dossier identity is derived by the Registry from the canonical baseline snapshot and must reconcile to the resolved identity.

Direct Registry writes remain forbidden.

## 7. RUN_CONTEXT_V2

No downstream stage may rely on conversational memory for run identity. After persistence and re-query, Pilotage constructs:

```text
OROTITAN_RUN_CONTEXT_V2
```

which locks at minimum:

```text
runtime bootstrap version/hash
Contract Pin Pack contract_set_sha256
production project ref
company
issuer/security/dossier identity
RUN_ID
RUN_TYPE
CANONICAL_MODE
ENTRY_PATH
DATA_CUTOFF
BASELINE_SNAPSHOT_ID
current Registry stage / execution phase
PUBLICATION_AUTHORIZED = false
```

Any mismatch on reconstruction is fail-closed.

## 8. Handoff envelope

Every V2 Research / Fundamentals / Valuation / Certification / Integration handoff carries:

```text
OROTITAN_VERSION
RUNTIME_BOOTSTRAP_VERSION
RUNTIME_BOOTSTRAP_SHA256
CONTRACT_SET_SHA256
PRODUCTION_PROJECT_REF
RUN_ID
REGISTRY_STAGE
EXECUTION_PHASE
DATA_CUTOFF
BASELINE_SNAPSHOT_ID
```

plus the exact stage-specific manifest / lock references.

The receiving conversation verifies the persistent Registry state before doing analytical work.

## 9. User-facing preflight

After successful run persistence, the normal Pilotage preflight is deliberately compact:

```text
OROTITAN V2 PREFLIGHT

AUTHORITY = PASS
PRODUCTION DB = PASS
IDENTITY = PASS
BASELINE = PASS
RUN TYPE = INITIAL | REFRESH
RUN PERSISTENCE = PASS

RUN_ID = <persistent UUID>
NEXT = RESEARCH
```

The exact Research bootstrap follows. On failure, Pilotage emits only the blocker and resolution route; it does not improvise an alternate environment.

## 10. Publication boundary

`GO <COMPANY>` never authorizes canonical publication.

Publication still requires the separate explicit command:

```text
GO PUBLISH <COMPANY>
```

Nothing in this patch modifies that boundary.

## 11. Contract authority boundary

This patch does not replace or mutate any frozen V2 contract and does not change the 13-pin Contract Pin Pack V2. It is a deterministic runtime implementation of already-frozen requirements: persistent identity before handoff, fail-closed environment validation, immutable baseline/history, deterministic routing and separate publication authorization.
