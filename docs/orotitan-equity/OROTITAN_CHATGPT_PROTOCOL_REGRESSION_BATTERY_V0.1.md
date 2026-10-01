# OROTITAN CHATGPT PROTOCOL — REGRESSION BATTERY V0.1

Status: EXECUTED BASELINE  
Date: 2026-10-01  
Protocol: `OROTITAN_CHATGPT_OPERATING_PROTOCOL_V1_FREEZE_V1.0`  
Production mutation: NO

## 1. Purpose

This battery verifies that the frozen ChatGPT operating protocol is supported by the current OroTitan infrastructure before Data Contracts V2 and the Process Engine are designed.

It does not claim that the full V2 workflow is already implemented.

The battery distinguishes:

- protections already proven live;
- protections already covered by code / contracts;
- analytical rules frozen at protocol level;
- implementation gaps that must be closed before a real vertical slice.

## 2. Baseline CI

PR #323 was merged only after:

```text
VNext CI   = PASS
Screener CI = PASS
```

Merge baseline:

`e4dcec16e9790e5dff88a032db64a742bcaca3cf`

## 3. Live Supabase preflight — read only

The live project was inspected without analytical or production mutation.

### Registry RLS

RLS is enabled on:

- `orotitan_runs`
- `orotitan_run_stages`
- `orotitan_artifacts`
- `orotitan_artifact_edges`
- `orotitan_run_events`
- `research_snapshots`

### Guarded RPC permissions

Nine critical OroTitan RPCs were checked.

Observed:

```text
anon EXECUTE           = 0 / 9
authenticated EXECUTE  = 0 / 9
service_role EXECUTE   = 9 / 9
SECURITY DEFINER       = 9 / 9
```

This supports the intended bounded write architecture.

### Concurrency / idempotency

Observed function definitions confirm that:

- checkpoint checks expected run + stage state versions;
- finalize checks expected run + stage state versions;
- reopen checks expected run + stage state versions;
- checkpoint/finalize/reopen include idempotency logic;
- publish authorization checks expected run state version;
- publish authorization is tied to `READY_TO_PUBLISH`;
- publish result is separated from authorization.

### Artifact authority

`resolve_orotitan_artifact` checks:

- expected SHA-256;
- required authority class.

This supports exact artifact resolution rather than conversational memory.

## 4. Security advisor findings

The Supabase security advisor reports:

### RLS enabled without policy — INFO

14 tables have RLS enabled and no explicit policy.

This is compatible with a service-only posture, but the posture must be made explicit during bridge / UI design so that no future client-side flow accidentally assumes authenticated direct table access.

Disposition:

`REVIEW_AND_DOCUMENT_SERVICE_ONLY_POSTURE`

### Mutable function search_path — WARN

Three functions are flagged:

- `set_companies_updated_at`
- `reject_market_sync_run_mutation`
- `reject_snapshot_mutation`

None of the critical run-registry RPCs checked by this battery is in that warning set.

Disposition:

`TECHNICAL_DEBT_TO_HARDEN`

This is not a blocker for Data Contracts V2 but should be cleared before the final production hardening pass.

## 5. Frozen 25-scenario matrix

Current classification:

| Class | Count |
|---|---:|
| PASS_LIVE | 5 |
| PASS_INFRASTRUCTURE | 1 |
| PASS_PROTOCOL | 4 |
| PARTIAL_INFRASTRUCTURE | 4 |
| PARTIAL_CODE | 1 |
| GAP_IMPLEMENTATION | 10 |
| Total | 25 |

Fifteen scenarios require additional implementation before the vertical slice.

## 6. Main gaps before vertical slice

The remaining gaps cluster into three design packages.

### Data Contracts V2

Must provide:

- source / evidence cutoff validation;
- V2 Evidence + Conflict structures;
- analytical Evidence-ID validation;
- capital-allocation / serial-acquirer structures;
- exact cross-block references needed for revalidation.

### Process Engine V2

Must provide:

- material-change revalidation gate;
- sector overlay validation;
- block-level gap / NOT_ASSESSABLE state;
- block dependency reopening;
- PRICE_ONLY_DELTA routing;
- ROUTINE_FUNDAMENTAL_DELTA routing;
- FULL_REFRESH_REQUIRED routing.

### ChatGPT ↔ Supabase bridge

Must provide:

- deterministic LOAD resolver;
- layered context assembler;
- connector-failure fail-closed behavior;
- SAVE orchestration;
- finalization eligibility detection;
- bounded invocation of guarded RPCs.

## 7. Result

```text
PROTOCOL DESIGN          = PASS
BASELINE CI              = PASS
LIVE REGISTRY PREFLIGHT  = PASS
MUTATION FIREWALL        = SUPPORTED
CONCURRENCY GUARDS       = SUPPORTED
PUBLICATION SEPARATION   = SUPPORTED
ARTIFACT AUTHORITY       = SUPPORTED

VERTICAL SLICE READY     = NO
```

Conclusion:

`FOUNDATION_PASS_WITH_IMPLEMENTATION_GAPS`

This is the expected and healthy result at this stage.

## 8. Exact next action

`DESIGN_ANALYTICAL_ENGINE_V2_DATA_CONTRACTS`

The battery therefore validates the planned sequence:

```text
TEST FOUNDATION
→ DATA CONTRACTS V2
→ PROCESS ENGINE V2
→ CHATGPT/SUPABASE BRIDGE
→ VERTICAL SLICE
→ FINAL FRENCH-FIRST UI IMPLEMENTATION
```
