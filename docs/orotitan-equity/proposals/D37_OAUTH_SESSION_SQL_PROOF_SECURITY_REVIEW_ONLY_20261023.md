# D37 — Verified OAuth session proof → SQL actor binding (review only)
**2026-10-23. Stacked DRAFT on D36 / PR #416. NOT DEPLOYED.**

## Purpose and trust boundary

D35 found that early D30–D36 private PostgreSQL entrypoints accept actor
subject/client/resource as direct function arguments, which a mistakenly
privileged or compromised caller could forge. D37 tests a safer, narrower
**replacement at the Research-scope and preview-read boundary**:

1. The trusted Node server cryptographically verifies a real ES256/RS256
   OAuth access token with fixed issuer, token audience, allowlisted subject
   and client and signed MCP resource. It additionally **requires** a signed
   UUID `session_id` claim for D37 (D31 remains read-only without it).
2. An independent, mandatory server policy verifies that the session is
   currently active and that this subject/client/session is permitted for
   the exact preview + operation + issuer + contract hash + cutoff.
3. The server, and **only** that server, HMAC-SHA256 signs a 30-second,
   randomly nonced attestation. All identity and scope fields come from
   verified server context. No HMAC secret, write grant, arbitrary actor
   field or attestation is an MCP model-visible tool response.
4. In disposable PG17, the D37 private function checks 13 **exact**
   attestation fields, permitted action, fixed MCP resource, TTL, HMAC
   against a local-only private signing key table, and inserts the
   unique nonce in the same transaction as reading the preview or
   registering the immutable Research scope. It derives subject/client
   and scope exclusively from the validated proof.
5. The experimental D37 `NOLOGIN/NOINHERIT/NOBYPASSRLS`
   `orotitan_d37_session_gateway` gets EXECUTE-only on
   `read_oauth_bound_preview_v5` and
   `issue_oauth_bound_scope_v5`. Direct SQL grants to the former
   D34 preview reader and D36 scope issuer are revoked **in the local test
   environment**, preventing those roles from bypassing D37 for these
   two operations.

## Test matrix

The D37 PG17 synthetic E2E uses a real ES256-signed OAuth JWT and signed
`session_id`, a verified server-side active-session policy, and actual
private SQL HMAC verification. It checks valid preview read, same-proof
replay denial, changed subject/client/session/contract denied, wrong action
denied, an expired attestation, another legitimately signed actor, a
revoked/missing session, old-function privileges denied, and successful
scoped Research request registration without any Research execution.
A synthetic fixture key is fixed at 0x37 (32 bytes) solely for GitHub
Actions; it MUST NEVER be reused in a real environment.

## Unresolved (production NO-GO remains mandatory)

- D37 attestation is **not** independently verified by Supabase Auth.
  The server policy interface simulates active-session checks; production
  would require actual revocation-aware session introspection and
  operational proof of OAuth token binding. A signed, non-revoked JWT by
  itself is not proof the session is still active.
- The HMAC signing key exists in Node and a private PostgreSQL table in
  the local test; real key provisioning/rotation/KMS and database-owner
  compromise models require independent security review. SQL bytea
  comparison is not guaranteed constant-time. A compromised high-trust
  signer could still fabricate a valid attestation.
- **Partial migration only:** D37 secures two entrypoints, not D34
  `create_preview_v3`, the capability vault issuer/reader, D32 final
  confirmation, the D36 lifecycle closer or D30 legacy add-company
  approvals. Those actor-supplied APIs remain a deployment blocker and
  must undergo equivalent review before any production activation.
- `session_id` presence and format in real OAuth tokens must be
  independently confirmed: tests use a synthetic signed claim and do
  not guarantee availability in the actual user/mobile connector.
- No tenant/org authorization beyond allowlisted user subjects; no
  production-grade quota, data retention, audit, nonce cleanup, rate
  limiting, live Supabase RLS validation, contract-pack provenance,
  ISIN attribution or real mobile acceptance.
- Do not deploy D37 SQL as a migration or expose any private schema
  through Supabase Data API. No Pro/branch purchase, Supabase Free
  writes, Vercel deployment, main merge, active Research worker,
  company creation, Registry write or Besi promotion is authorized.

**Decision: still NO-GO for live write operations.** These synthetic
tests can verify this trust boundary only; they do not provide a
production security sign-off.
