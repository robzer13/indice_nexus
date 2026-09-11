# I3-A Canonical Snapshot Persistence

## Scope

I3-A creates the PostgreSQL storage boundary for OROTitan Equity Research V1
canonical snapshots. It does not create an application writer, a persistence
RPC, a read projection, or a production migration workflow.

**MASTER PROMPT / CERTIFIED DOSSIER = analytical source of truth**

**`research_snapshots` = immutable structured projection/history**

The database is **not** analytical truth.

## Storage model

`research_snapshots` uses a hybrid relational and JSONB representation. Relational
columns provide queryable identity, dates, versions, and access boundaries.
`canonical_payload` stores only the frozen Phase-4 `researchSnapshot` object.
PostgreSQL checks that the payload is an object and that every required lock
field exists with the expected primitive type and equals its relational value.
Full canonical schema validation remains outside this migration.

Snapshots are append-only. An immutability trigger rejects updates and deletes,
and `created_at` gives the history a stable PIT ordering without an
`updated_at` column.

## Identity and current pointer

Composite foreign keys ensure that a dossier and security both belong to the
snapshot issuer. The transitional `research_dossiers.current_snapshot_id`
pointer remains nullable and can reference only an existing snapshot from the
same dossier. I3-A does not backfill or advance this pointer.

## Access boundary

RLS is enabled with no browser policies. `anon` and `authenticated` have no
table access. `service_role` has `SELECT` only. Direct writes are intentionally
unavailable because I3-B must provide the validated, atomic write boundary.

## I3-B responsibilities

I3-B will validate the complete canonical contract, perform the atomic
application write, and manage the approved current pointer transition. It must
not weaken the immutable storage, identity locks, RLS, or privilege boundary.

## Non-goals

This migration does not implement I3-B, I4 legacy backfill, I5 market-price
cutover, I6 activation events, I7 read projections, I8 admin writes, I9
frontend work, I10 cutover, autonomous Deep Dive, portfolio management, new
scoring, or methodology changes. Legacy tables and legacy scoring remain
untouched.