# I3-B Validated Canonical Snapshot Writer

## Authorities and boundaries

The **MASTER PROMPT / CERTIFIED DOSSIER** remains analytical truth. The
vendorized Phase-4 JSON Schema is the structural projection contract. I2 is
the deterministic calculation authority. The I3-B validator is the canonical
admission gate, and the I3-B PostgreSQL RPC is the atomic persistence boundary.

`research_snapshots` is immutable history. The nullable
`research_dossiers.current_snapshot_id` is the latest accepted projection
pointer.

The normal application path is:

1. Validate `researchSnapshot` against the frozen Draft 2020-12 schema.
2. Apply persistence-boundary checks and reject reversed ranges.
3. Adapt analysis-capable fields to the existing I2 contract and reject any
   deterministic mismatch.
4. Call `persist_orotitan_research_snapshot` from the server-only writer.

No stage repairs, normalizes, fills, or silently regenerates the submitted
analytical payload.

## Physical persistence narrowing

The Phase-4 `id` definition is a non-empty string up to 256 characters. I3-A
physically stores `snapshot_id`, `issuer_id`, `security_id`, and `dossier_id`
as PostgreSQL UUIDs. I3-B validates UUID compatibility at the persistence
boundary. This is **PHYSICAL PERSISTENCE NARROWING**, not a modification of the
canonical Phase-4 schema.

## Atomic write and concurrency

The RPC locks the target dossier row, verifies the payload issuer, and compares
the current pointer with the nullable expected pointer using compare-and-swap
semantics. `NULL` means that no canonical snapshot is expected; a UUID means
that exact snapshot must still be current. It then inserts the immutable row
and advances the pointer in one transaction.

A failed insert or pointer transition rolls back the whole call. A stale
request cannot overwrite or move the pointer backwards.

## Retry behavior

An identical retry succeeds idempotently only when the snapshot identity,
relational lock values, JSONB payload, dossier, and current pointer all match.
A reused `snapshot_id` with different content is rejected. A replay of an old
snapshot after the pointer has advanced is rejected; ordering is never inferred
from `data_cutoff`.

## Security boundary

The RPC is `SECURITY DEFINER` with fixed `search_path = pg_catalog, public`.
Execution is revoked from `PUBLIC`, `anon`, and `authenticated`, and granted
only to `service_role`. Browser roles have no write path and no RPC access.

`service_role` retains no direct `INSERT`, `UPDATE`, or `DELETE` privilege on
`research_snapshots`. This is intentional: direct table writes would bypass
schema validation, I2 reconciliation, the dossier lock, optimistic
concurrency, and retry protection. The service role can write only through the
validated server-only boundary.

The migration does not touch production Supabase, legacy analytical data,
frontend code, or I2 formulas. DISCOVER snapshots remain valid without
analysis-only blocks; analysis modes must satisfy the frozen conditional
requirements.