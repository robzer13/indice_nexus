# D33 — D31 signed OAuth → D32 private PG17 confirmation end-to-end
**Date:** 2026-10-23. **Status:** REVIEW ONLY / DRAFT, stacked on #412.
No Supabase Pro, preview branch, hosted migration, Registry modification, MCP
tool activation or Vercel deployment.

## Implemented

- New server-only candidate bridge:
  `lib/orotitan-equity/v3/mobile-command-private-pg-bridge-candidate-v1.ts`
  composes `issueServerPrivateCommandCapability` and
  `confirmWithServerPrivateCapability` with D32's two narrow procedures.
  It re-verifies signed OAuth at approval registration, binds verified
  `subject`, signed allowlisted `client_id` and fixed MCP `resource` to
  the exact D30 preview and independent D31 policy, and registers only
  SHA-256 of the sealed **server-only** HMAC capability.
- The confirmation side captures the sealed capability within a private
  server closure; the D32 submitter receives only
  `(previewId, sealedCapability)`. Neither the model nor browser chooses
  actor, kind, fingerprint, run ID or Research authority.
- The durable SQL receipt remains
  `REQUESTED_NOT_EXECUTED`, `companyInserted=false`,
  `researchStarted=false`, `registryMutationPerformed=false`.
- The test in `tests/postgres/d33-mobile-oauth-pg-e2e.ts` uses real ES256
  keypair and JOSE JWT verification plus the actual disposable PG17 D30/D32
  functions and `SET ROLE` for the distinct NOLOGIN approval and submitter
  roles. No hosted Supabase credentials, database URLs or real customer data.
- It covers missing approval, independent policy denial, another client
  (valid OAuth, wrong preview actor), simultaneous duplicate confirmations,
  correct Research request with **no** run creation, and revoked approval.
  Existing D32 two-session SQL race is retained as a separate gate.

## Deliberate test-fixture limits

- Only the D32 **approval and confirmation** procedures execute as narrow
  database roles. Test preview seeding and readback uses the disposable
  PostgreSQL fixture superuser; no production-safe preview creation/read
  principal has yet been built. The local company resolver is synthetic.
- Test fixtures model identity *shapes*, not verified real ISIN ownership or
  ISO 6166 check digits. The real issuer/security resolver with provenance,
  ambiguity handling and normalized identifier mapping is a release gate.
- The capability is held in memory for the test; production requires an
  encrypted server-only store with TTL, expiration/revocation, key rotation,
  session-level access control, audit and rate limiting. The SQL verifies a
  registered digest, not the capability HMAC signature; D31 verifies HMAC.
- The D33 bridge is deliberately **not imported** by the live MCP route,
  and cannot by itself make ChatGPT commands available on phones.
- Database identity access permissions, PostgreSQL RLS, cross-run isolation
  and SQL's security-definer provenance require independent audit in a
  representative Supabase-compatible environment before any migration.

## Next stage gating

1. Audit the entire D30–D33 stack, including authorization impersonation,
   privilege/role grant matrix, race/replay and failure recoverability.
2. Build least-privileged preview creation/readback + durable server-only
   capability vault in an isolated no-cost environment.
3. Verify actual ChatGPT mobile client supports the intended OAuth connector.
4. Only with separately requested user consent consider a hosted, quota-audited
   Supabase Free migration. Never automatically merge or deploy this PR.

Besi Registry remains RESEARCH, with original six evidence blockers and
Deep Dive gate NO.
