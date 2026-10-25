# OroTitan Registry — Issue #427 / controlled run-cancellation execution card

**State:** proposed code + regression tests; **not deployed, not invoked, not merged**.
**Date:** 2026-10-25.
**Production project:** `cugpgtzygqqlxetyeven` (do not use the shadow project).

## Mission and authorization boundary

The user authorized cleaning obsolete, erroneous and redundant runs. The immutable Registry and existing canonical snapshots remain the authorities; the user did **not** authorize falsifying a gate, changing analytical results, rebinding historical Contract Sets, or deleting audit history.

Prior ad-hoc SQL cleanup was blocked by tool security controls; SELECT verification proved **zero closures**. Do not bypass those controls. This proposal introduces a service-role-only administrative RPC through normal reviewed schema-change workflow.

## Frozen candidates (allowlist)

| Candidate to cancel | Expected version | Replacement to preserve | Reason |
|---|---:|---|---|
| `d5cd803b-f959-4e41-ba55-5fbfc2a7d245` (Qualys V1 REFRESH) | 3 | `e41a8fe0-29cb-45e3-8bca-b1f19ac67d12` (PUBLISHED V2) | `REDUNDANT_HISTORICAL_REFRESH_ALREADY_PUBLISHED_V2` |
| `d56a0bc9-4e80-48b3-b326-74d1fea15e63` (Veolia V3.1) | 2 | `a4cf9002-b52d-4440-8902-adc70bd777dd` (BLOCKED Deep Dive with artifacts) | `EMPTY_PARALLEL_METHOD_V3_1_SUCCESSOR_REDUNDANT` |

Qualys replacement shares exact dossier, issuer/security, data cutoff and baseline; V2 published snapshot remains `e42a1396-357a-42f5-ae0c-5e750f321180`. The old V1 refresh retains eight historical artifacts.

Veolia is an empty sibling of a genuine same-parent, same-contract and same-cutoff V3.1 run with Deep Dive artifacts. Its historical parent `76ee7eb1-8658-4038-83bf-e2c951db731a` remains untouched. The BLOCKED valuation denominator is **not resolved by this cancellation**.

## Migration

`migrations/20261025_orotitan_registry_controlled_run_cancellation_v1.sql`

A single `SECURITY DEFINER` RPC `public.cancel_orotitan_redundant_run_v1(uuid,bigint,uuid,text,text,text)`:
- accepts only the two exact run IDs and exact replacement/reason/version tuples;
- locks target, replacement, stage and dossier; performs version CAS;
- refuses if there are child runs, cross-run artifact dependencies, Pilotage attempts, publication events, or any target state mismatch;
- enforces the replacement/canonical pointer proof specific to each pair;
- atomically changes `run_status` to `CANCELLED`, stamps `cancelled_at`, increments state version and appends `RUN_CANCELLED` to the immutable event ledger;
- returns an idempotent success only for the exact prior idempotency key/fingerprint/event;
- does **not** write to `research_snapshots`, `research_dossiers`, any artifact or any edge;
- revokes `EXECUTE` from `PUBLIC`, `anon`, `authenticated` and grants only `service_role`.

### Required pre-production proof

1. Review SQL for least privilege, RLS, fail-closed semantics, deadlocks and historical immutability.
2. Run the PostgreSQL 17 test runner `tests/postgres/run-registry-v1-migration-tests.sh` on a **throwaway local database** only. It includes `tests/postgres/registry-controlled-run-cancellation-v1-verify.sql`, protected by an isolated-database guard. This test uses synthetic UUID fixtures and `ROLLBACK`.
3. Verify that the RPC's allowlist remains exactly the two audited candidates and that no extra `EXECUTE` grants exist.
4. Approve migration/deployment via standard schema-change pipeline. Migration creates only the RPC and grants; **deploying alone does not cancel a run**.
5. Re-read live candidate rows, related event counts, snapshot pointer, descendants and edges before each invocation. Abort on any change.

### Invocation requirements after authorized deployment

Only a trusted server-side `service_role` caller may invoke the RPC, with the exact tuple above, a unique stable idempotency key and its SHA-256 request fingerprint. **Never place a service-role credential in a browser or a public issue.** The two cancellations are separate transactions, not an uncontrolled batch.

Confirm each RPC response, then independently query:
- `orotitan_runs.run_status='CANCELLED'`, `cancelled_at IS NOT NULL`, expected new `state_version`;
- one matching `RUN_CANCELLED` event with exact replacement and reason;
- all prior artifacts still present, no removed event;
- Qualys and Prosus canonical snapshot pointers unchanged, no `research_snapshots` mutations;
- sibling runs and historical parents unchanged.

### Expected outcome if nothing else changes

`58 runs = 37 PUBLISHED + 12 ACTIVE + 7 BLOCKED + 2 CANCELLED`; `36/50` published dossiers.

**Do not report this outcome as achieved before the migration is approved, deployed, both RPCs are invoked successfully, and all postconditions pass.**

## Explicit exclusions

- Do not cancel Prosus V3 automatically: it has later data cutoff than the published Prosus V2 run.
- Do not cancel ASML, Tesla, L'Oréal, LVMH or other Veolia lineages merely on matching issuer; these include distinct cutoffs and/or governed methodology successors.
- Do not rebind old runs to V3.1, delete Registry rows, remove artifact history, or move a published canonical pointer.

Ticket: [#427](https://github.com/robzer13/indice_nexus/issues/427).
