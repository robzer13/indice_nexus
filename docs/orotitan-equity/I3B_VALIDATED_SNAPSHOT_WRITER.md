# I3-B Validated Canonical Snapshot Writer

## Authorities and boundaries

The **MASTER PROMPT / CERTIFIED DOSSIER** remains analytical truth. The vendorized Phase-4 JSON Schema is the structural projection contract. I2 is the deterministic calculation authority. `OROTITAN_INVESTMENT_POLICY_V1.0.0` is the versioned project investment-policy authority. `OROTITAN_EXECUTION_CONTRACT_PATCH_V1.0.1` resolves only the N-basis selector. The I3-B validator is the canonical admission gate, and the I3-B PostgreSQL RPC is the atomic persistence boundary.

`research_snapshots` is immutable history. The nullable `research_dossiers.current_snapshot_id` is the latest accepted projection pointer.

The normal application path is:

1. Validate `researchSnapshot` against the frozen Draft 2020-12 schema.
2. Apply persistence-boundary checks and reject reversed ranges.
3. Verify the exact V1 investment-policy values carried by the Price Ladder.
4. Use the canonical I2 N-basis selector, adapt analysis-capable fields to the existing I2 contract, and reject any deterministic mismatch.
5. Call `persist_orotitan_research_snapshot` from the server-only writer.

No stage repairs, normalizes, fills, or silently regenerates the submitted analytical payload.

## Investment-policy admission rule

Analysis-capable snapshots must carry exactly:

```text
POLICY_VERSION
= OROTITAN_INVESTMENT_POLICY_V1.0.0

price_ladder.required_return_h
= 10.0

price_ladder.strong_return_threshold
= 12.5

price_ladder.exceptional_return_threshold
= 15.0
```

The Phase-4 shape already contains these fields, so no payload-shape change is required. The validator checks the values; it does not infer or optimize them. `DISCOVER` snapshots remain exempt from analysis-only valuation requirements exactly as before.

Expected returns are converted to I2 deltas using the locked `price_ladder.required_return_h` hurdle before scoring.

## Canonical N-basis admission rule

I3-B no longer rejects a snapshot merely because both `no_multiple_expansion_return` and `mature_normalization_return` are valid numeric values. It delegates selection to the canonical I2 selector under `OROTITAN_EXECUTION_CONTRACT_PATCH_V1.0.1`:

```text
IF valid numeric MATURE_NORMALIZATION_RETURN exists
→ select Mature Normalization as N basis

ELSE IF Mature Normalization is legitimately
NOT_ASSESSABLE / NOT_AVAILABLE
AND valid numeric NO_MULTIPLE_EXPANSION_RETURN exists
→ select Same-Multiple as fallback N basis

ELSE
→ preserve the applicable Mature-Normalization unavailable semantic state
→ numeric OVS prohibited

INVALID / UNRECONCILED MATURE_NORMALIZATION_RETURN
→ FAIL CLOSED
→ no fallback
```

When both values are valid and numeric, Mature Normalization is selected regardless of whether it is greater than or less than Same-Multiple. I3-B is therefore not a `MIN`/`MAX` selector and does not exercise analyst discretion.

A structurally invalid Mature value is rejected by schema/adaptation validation. A numerically valid Mature value with persisted deterministic outputs that reconcile only to Same-Multiple is rejected at I2 reconciliation. That mismatch is an invalid/unreconciled Mature case and cannot be bypassed by fallback.

Semantic return states remain preserved. When Mature is legitimately unavailable and Same-Multiple is also nonnumeric, the applicable Mature semantic state propagates into the I2 normalized-return branch and numeric OVS is prohibited.

I3-B remains an admission gate, not an economic authority. The schema validation, persistence boundary checks, I2 reconciliation, and atomic persistence semantics are otherwise unchanged.

## Physical persistence narrowing

The Phase-4 `id` definition is a non-empty string up to 256 characters. I3-A physically stores `snapshot_id`, `issuer_id`, `security_id`, and `dossier_id` as PostgreSQL UUIDs. I3-B validates UUID compatibility at the persistence boundary. This is **PHYSICAL PERSISTENCE NARROWING**, not a modification of the canonical Phase-4 schema.

## Atomic write and concurrency

The RPC locks the target dossier row, verifies the payload issuer, and compares the current pointer with the nullable expected pointer using compare-and-swap semantics. `NULL` means that no canonical snapshot is expected; a UUID means that exact snapshot must still be current. It then inserts the immutable row and advances the pointer in one transaction.

A failed insert or pointer transition rolls back the whole call. A stale request cannot overwrite or move the pointer backwards.

## Retry behavior

An identical retry succeeds idempotently only when the snapshot identity, relational lock values, JSONB payload, dossier, and current pointer all match. A reused `snapshot_id` with different content is rejected. A replay of an old snapshot after the pointer has advanced is rejected; ordering is never inferred from `data_cutoff`.

## Security boundary

The RPC is `SECURITY DEFINER` with fixed `search_path = pg_catalog, public`. Execution is revoked from `PUBLIC`, `anon`, and `authenticated`, and granted only to `service_role`. Browser roles have no write path and no RPC access.

`service_role` retains no direct `INSERT`, `UPDATE`, or `DELETE` privilege on `research_snapshots`. This is intentional: direct table writes would bypass schema validation, I2 reconciliation, the dossier lock, optimistic concurrency, and retry protection. The service role can write only through the validated server-only boundary.

## Non-regression boundary

This execution-contract patch does not change OQS weights, weak-link behavior, OVS anchors, OVS interpolation, the `N + 15` cap, MOS caps, valuation-reliability caps, the Investment Score formula, certification states, or the OroTitan terminal gate. The Phase-4 JSON payload shape remains unchanged.

The implementation mission itself performs no production Supabase write, migration, backfill, or data mutation. Existing persistence code is tested but production persistence is not invoked.
