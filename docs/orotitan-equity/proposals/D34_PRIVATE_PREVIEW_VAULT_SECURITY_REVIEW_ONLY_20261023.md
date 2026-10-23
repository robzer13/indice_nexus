# D34 — least-privileged private previews and AES-GCM capability vault
**2026-10-23 — DRAFT / LOCAL POSTGRESQL 17 ONLY.** Stacked on #413.
No hosted Supabase migration, Pro plan, branching fee, production MCP write
tools, key deployment, Registry write, merger or Vercel deployment.

## Implemented

- SQL **review proposal** `D34_PRIVATE_PREVIEW_ENCRYPTED_CAPABILITY_SQL_REVIEW_ONLY.sql`
  installs four purpose-bound, non-login, non-inheriting, non-bypass-RLS
  principals on a disposable PG17 database:
  `d34_preview_issuer`, `d34_preview_reader`, `d34_vault_issuer` and
  `d34_vault_reader` (all prefixed `orotitan_`). The D32 function owner
  holds narrow INSERT/SELECT on previews and encrypted vault records, and is
  used only through `SECURITY DEFINER` routines with locked `search_path`.
  The four callers have **EXECUTE-only**, no direct table rights.
- A database preview creation routine enforces expiry, actor/client/resource,
  string-only payload shape, and independently re-derives the D30 SHA-256
  canonical fingerprint (kind plus lexicographically sorted key-value pairs).
  It refuses a falsified hash and duplicate preview UUID.
- A preview read routine returns only the exact actor/client/resource preview,
  without exposing other private records.
- A D34 vault issuer atomically invokes D32 approval and stores **only a
  ciphertext/96-bit nonce/128-bit authentication tag and key-version** in
  PostgreSQL. The database also stores D32's digest of the sealed capability.
  It never receives a plaintext sealed capability or the AES key.
- The TypeScript module uses random 96-bit AES-GCM nonces and AES-256-GCM
  encryption with associated data binding the preview, actor, client, resource,
  command kind and fingerprint. Decryption verifies GCM authentication,
  canonical Base64, key version and a constant-time SHA-256 comparison.
- The vault reader returns only ciphertext for the precise approved,
  unrevoked, unexpired and not-yet-confirmed preview. Its database role cannot
  directly SELECT the vault table.
- `npm test` exercises cryptographic alteration, wrong user/client/key/
  action/preview, tag corruption and key-version mismatch.
- The disposable PG17 D34 integration uses real ES256-signed JWTs, D30
  preview creation via the limited SQL issuer, private preview reader, D31
  HMAC capability, D34 vault issuer/reader and D32 EXECUTE-only submitter.
  The Registry is not created; outputs remain requests only.

## Explicit limitations, not production authorization

This is **not** a production credential/role topology. Local scripts use
the PostgreSQL fixture superuser to issue `SET ROLE`, and the test-only
identity resolver does not independently verify ISIN checksum or provenance.
A server that possesses the SQL issuer role must still verify actual OAuth
bearer, allowlisted signed client and independent grant; SQL procedures have
caller-supplied actor parameters. No `anon`, `authenticated` or PUBLIC
role may invoke them. A compromised trusted issuer/reader role could
impersonate a subject by supplying its attributes. A production connector
must use isolated credentials, server-side actor binding and an independent
privilege/security assessment; the toy model is **not** a substitute.

The AES key is randomly generated for local tests only. Production needs an
audited key-management service, envelope key rotation, encrypted backups,
strict access controls, one-time read semantics, rate limits, privacy-safe
audit logs, and failure recovery before hosting any secret. The database
does not and cannot verify the D31 HMAC signature; that verification
happens on the trusted application server. D32's digest check is an
additional binding, not an authentication substitute.

TTL is limited to two minutes; no real mobile app/OAuth write connector,
hosted queue, Research worker or production activation is included. The
D29b Besi Registry remains at its existing RESEARCH gate, with six
unverified blockers. Changes must not be merged or deployed until separately
authorized and independently reviewed.
