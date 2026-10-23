# D29b — RESEARCH Evidence Intake (Besi pilot) — REVIEW ONLY

**Branch**: `review/orotitan-d29b-research-evidence-intake-20261023` (based exclusively on D29a OAuth pilot).
**Status**: code-only candidate; **NOT deployed, enabled, or an authorization to upload**.

## Why this separate slice exists

D29a `get_orotitan_blocker_worklist` can return Research gaps, but cannot
ingest or admit the subsequent evidence. The existing manual-intake helper is
not an OAuth delegated write path: `mcpWriteEnabled` is hardcoded `false`,
and `list_orotitan_submissions` remains disabled to delegated OAuth.
Never make these older functions writable by toggling an environment variable.

The draft D29b PRs #390–#397 concern signed D2 handoff receipts and a private
signer/writer chain. Those are **not a replacement** for acquisition and
verification of Research source evidence. This PR is a separate intake
**candidate** using an injected, privileged `ResearchIntakePort` only.

## Implemented in this PR

- `research-evidence-intake-candidate-v1.ts`: typed submission and verifier
  metadata boundary. It compares an authenticated actor with a server-resolved
  short-lived grant bound to exact `runId`, issuer, `RESEARCH`, both CAS
  versions, OAuth `sub/client_id/resource`, data cutoff and contract hash.
- Submission rejects expired/overlong grants, mismatch, post-cutoff source
  dates, unsupported file extensions/MIME, paths, invalid UTF-8, malformed
  JSON objects, files above 1 MB and incorrectly declared SHA-256.
- The prepared draft defensively copies the exact bytes after hashing.
  The storage adapter contract requires immutable private storage, a
  content-addressed object, an atomic version-scoped receipt and actor audit,
  and server-side SHA-256 verification. **No production adapter exists here.**
- The only permitted receipt is `RECEIVED_NOT_VERIFIED` with
  `QUARANTINE_ONLY`, `admitted=false`, `mayAdvance=false`,
  `mayPublish=false`; duplicate receipts must be idempotent.
- A distinct authorized D2 reviewer may retrieve stage- and version-scoped
  **metadata**, not source bytes, after independent permission checks.
- Synthetic adversarial tests demonstrate the pure port on the Besi run.

### Scope of the demonstration

The tests use a deliberately fake in-memory port. They show the contract for
"submit -> quarantine receipt -> independently scoped metadata read" and the
negative cases; they **do not** prove transactional guarantees, remote storage
integrity, OAuth bearer validation, RLS, permission revocation or deployed
access. These must be tested with the real adapter under an authorized pilot.

## Besi fixture and provenance constraints

`runId: 48b9426b-4d0d-4858-9d87-efcd13ac415d`
`stage: RESEARCH`
`dataCutoff: 2026-10-12`
`runStateVersion: 6`
`stageStateVersion: 4`
`contractStatusCode: BLOCKED_INSUFFICIENT_INPUT`
`handoffGate: NO`

The supplied `00_BESI_RESEARCH_SYNTHESIS` is an unadmitted local workpack.
Its identifiers `E001–E026` and `S001–S023` are **local** and are not
Registry `ARTIFACT_ID`s. The full ZIP, original ESEF files and corresponding
hashes have **not** been received by the present change. Do not commit
private licensed documents, source ZIPs, evidence ledgers with confidential
material or Supabase credentials to this publicly readable repository.

The six gaps C003, G003/G009, G004, G005, G006/G016 and G012 remain
`OPEN_UNVERIFIED`. Research does not become admissible when a draft is
received. The issuer's FY2025 18-vs-15 adoption count stays unresolved,
and preliminary cash-flow residuals are not normalized Owner Earnings.

## Production blockers — each requires its own review/authorization

1. Implement independent OAuth consent and a short-lived scoped capability
   issuer. Do not use user-editable JWT `user_metadata`; do not treat generic
   OIDC `email/profile` as a write scope. Bind `sub`, `client_id`, exact
   resource, run, issuer, stage, both CAS revisions and expiry.
2. Implement a **private** storage port with immutable objects, no upsert,
   checksum reread and no bucket-wide list. Never place privileged service-role
   credentials or D2 source bytes on the public MCP path.
3. Design a single transactional receipt+audit CAS operation in a private
   server-only writer with revocation and idempotency; prove race handling
   and orphan cleanup for non-transactional blob storage. Neither a stored hash
   nor a receipt authenticates the root source.
4. Test real JWT/RLS policies, wrong subject/client/run/issuer, revoked
   grants, stale run and stage versions, hash collisions, oversized files,
   forged receipts, malformed media, cross-run reads and atomic audit.
5. Build the **independent evidence verifier** for actual source bytes,
   root-source independence, historical versions and the Research Evidence/
   Conflict/Calculation Ledgers. Existing D2 signed-handoff prototypes
   must not be repurposed as substitute Research evidence.
6. Produce new immutable CHECKPOINT artefacts via authorized Registry
   authorizer only after verification, preserving the frozen contracts.
   Research `FINAL`, Deep Dive admission and any publication remain separate,
   gated operations requiring explicit authorization.

## Non-goals and release guard

This change adds **no MCP tool registration**, no executable public upload
route, no database migration, no Supabase GRANT/RLS, no bucket policy,
no secret, no cloud resource, no Vercel configuration, no Registry writes,
no stage transition and no deployment. The D29a OAuth pilot remains
read-only. The draft PR should target only the D29a pilot branch, not `main`.

Passing CI is necessary but not sufficient. An independent security review,
proof of backend-enforced authorization and CAS on a safe isolated environment,
and a separate explicit approval are required before live operations.
