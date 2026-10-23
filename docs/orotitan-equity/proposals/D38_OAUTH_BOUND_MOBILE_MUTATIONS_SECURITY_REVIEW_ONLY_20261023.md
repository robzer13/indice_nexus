# D38 — session-bound private preview, encrypted vault and queue confirmation
**2026-10-23 / DRAFT ONLY / disposable PostgreSQL 17.** Stacked on D37/#417.

D38 extends the existing D37 session-attribution experiment to four
operation classes previously accessible through legacy actor-supplied SQL
parameters: (1) creating a preview, (2) approving/storing encrypted
capability material, (3) reading a private encrypted capability, and (4)
confirming an add-company or scoped Research request.

## Design in the LOCAL test fixture

An ES256-signed, resource-bound OAuth access token is independently checked
for issuer/audience, allowlisted subject/client and signed UUID session ID.
A private server policy checks that session against a synthetic active-session
authority and authorizes the exact preview/operation/argument digest. Only
then is a 30-second, nonced HMAC-SHA256 proof minted by the trusted server.

PostgreSQL validates all operation/actor/session/preview fields, HMAC, TTL,
and nonce. Its D38 private test session mirror must still indicate ACTIVE
when the proof is consumed. The proof carries a SHA-256 of the specific
arguments:
- `create_preview`: kind, exact canonical D30 fingerprint, expiry in ms.
  The D34 private function also recomputes the fingerprint from the JSON.
- `approve_vault`: kind, fingerprint, D31 sealed-token digest, key version,
  and raw nonce/ciphertext/auth-tag bytes (all envelope material authenticated).
- `read_vault`: zero binding hash + actor-specific preview ID; output
  contains AES-GCM ciphertext, never decryption key or plaintext.
- `confirm_company`/`confirm_research`: exact digest of the sealed
  server-held capability, plus verified OAuth actor/client in the proof.

All private wrappers run as the D32 NOLOGIN RLS-aware function owner under
a restricted `search_path`. A new `orotitan_d38_mutation_gateway` role
gets EXECUTE-only on these new functions, no private table access.
D38 revokes corresponding legacy call-only grants **inside the disposable
fixture**, after the previous D30–D37 regression suites complete.
All nonce records are transactionally consumed or rolled back on failure.

The synthetic end-to-end test checks a valid ES256 session, private preview,
AES-256-GCM envelope and D31 token, D32 company queue, D36 Research queue,
changed ciphertext/capability, nonce replay, and proof issued before
revocation but consumed AFTER a simulated DB session revocation. It
asserts that no company or Research run is actually executed.

## Important unresolved release blockers

**The DB session mirror is not Supabase Auth.** A real production path must
verify session revocation against the authoritative identity provider, with
bounded freshness and race-handling, rather than trusting a manually seeded
`d38_test_active_sessions` table. D38 does not establish real Supabase
session introspection or the exact JWT claims emitted by a mobile connector.

The D38 TypeScript signature issuer is review-only, with no deployed HTTP
endpoint or connected ChatGPT tool. External callers must never supply
signing keys, choose arbitrary proof claims, see sealed capabilities or
receive privileged DB credentials. The test passes sealed token values
through local `psql` process arguments; that unsafe transport is
specifically **not an approved production architecture**.

The D38 SQL uses a synthetic signing key that also exists in the local
D37 fixture. Production needs independent secrets management/KMS, key
versioning/rotation, HMAC timing review, nonce garbage collection, rate
limiting, audit and independent credential boundary tests. D38 SQL's
bytea comparison is not a guaranteed constant-time implementation.

**Partial migration still:** the older D30/D32 approval and revocation
functions, D36 closure, and D37 scope issuance are not completely replaced
by D38. Other trusted SQL roles may retain restricted grants for those
operations, so D38 is not comprehensive protection of the entire command
system. The D30 global uniqueness remains; D36 does not prove the contract
pack was genuinely frozen, or that ISIN attribution is reliable. Preview
creation uses synthetic identities and not a real identity-resolution feed.

D38 demonstrates specific local security invariants only. It does not
enable actual ChatGPT mobile add-company/Research commands, write to
Supabase or authorize main merges/deployment. No Supabase Pro fee, paid
branch, hosted migration, Vercel publish, Registry mutation, Besi evidence
admission or Deep Dive progression is authorized.

**Release decision: NO-GO.**
