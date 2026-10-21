# D29b — Admin D2 issuer, OAuth RLS reader & revocation integration

**Status:** DRAFT / REVIEW ONLY / NOT RELEASED  
**Base:** PR #390 D29b cryptographic verifier, commit `9c6996b756c8425763bb772ff75dbdb8bbafcff4`  
**Production pilot:** D29a remains live. **Main Screener / Supabase / Registry:** unchanged.

## Security architecture

- **Privileged issuer** (`mcp-d29b-d2-issuer-server.ts`) requires `requireAdminSession()` and independently calls the existing D2 `loadTrustedPilotageDecision` over pinned private immutable GitHub bytes and authoritative Registry state. Exact run, subject, client, resource are limited by four explicit issuer environment allowlists. The issuer cannot sign `BLOCKED`, `DONE` or `AWAIT_EXPLICIT_GO_PUBLISH` decisions. It confirms exact run/stage CAS and manifest hash after verification and again *after remote signing*. It does not create a route, store receipts or mutate Registry.
- **Signing key custody:** a required `sign(message)` adapter is invoked only after D2 admission. Private key bytes are never passed to the OAuth MCP bridge. Production must use a separately reviewed authenticated KMS/HSM adapter with Ed25519 and immutable audit logs. This PR supplies **no key generation, signer credential, or DB writer**.
- **OAuth reader** (`mcp-d29b-delegated-reader-v1.ts`): receives identity claims from the signature-validated OAuth JWT and fetches signed receipts with **that JWT + Supabase publishable key** under RLS. It reads run/stage state twice, receipt twice, checks expiration and revocation and invokes the standalone Ed25519 verifier against a server-pinned public-key ring. Errors, races, revoked or missing receipts return no prompt.
- **MCP route:** passes request-local **verified** OAuth `subject`, `clientId`, and `resource` into the delegated reader. Feature is **disabled by default**: `OROTITAN_MCP_D29B_READ_ENABLED !== "true"` preserves D29a's observation and `prompt:null` behavior.
- **Proposed SQL** (`D29B_SIGNED_RECEIPT_RLS_SQL_REVIEW_ONLY.sql`): adds nullable `revoked_at` to the review-only receipt table and denies selection of revoked/expired receipts under RLS. No live Supabase DDL is applied by this PR. Do not grant insert/update/delete to OAuth.
- **Visibility**: receipts contain a manual execution prompt only, not source artifact bytes, private GitHub paths, signing keys or a publication command. Signed proof is bound to exact OAuth subject/client/resource/run, cutoff, D1/D2 fingerprint, pinned contract pack and live run/stage CAS.
- **Trust naming**: a positive result means `SIGNED_D2_ATTESTATION_WITH_LIVE_REGISTRY_CAS`, not local re-verification of every private immutable artifact.

## Tests

- Prior PR #390 adversarial Ed25519 signature, contract, scope, expiry, CAS and publication denial tests.
- New double-checked RLS reader tests for positive receipt, missing receipt, revocation between reads, receipt rotation, CAS drift, database outage, malformed/expired receipt and untrusted verifier keys.
- New local PostgreSQL 17 test `bash tests/postgres/run-mcp-d29b-signed-receipt-rls-tests.sh`, using synthetic identities **on 127.0.0.1 only**, verifying exact subject/client/resource/Besi run filter and immediate read denial upon `revoked_at`.
- Static trust-path regression: OAuth never calls the privileged D2 reader/issuer or exposes a signing private key. D29a behavior remains default.

## Residual risks & release blockers

**DO NOT DEPLOY OR ENABLE YET.** This PR is an integration proposal, not a production authorization. Remaining steps:

1. Independent security review of issuer's D2 derivation and full artifact lineage; ensure `CONTINUE` decisions only sign when D2 has actually admitted the exact checkpoint and verified all needed proof bytes.
2. Review and explicitly authorize receipt-table migration and role grants; test live RLS with exact authorized JWT and negative caller without opening other runs or tables.
3. Implement a narrowly authorized **writer** that stores exactly the independently issued receipt, without exposing a write endpoint to ChatGPT. Define audit event, signer role, revocation/rotation and cleanup transaction policies.
4. Configure a trusted KMS/HSM signer in a separate privileged runtime and distribute *public key only* to the pilot; explicit separation of issuer credentials from the Vercel pilot project.
5. Validate revocation/expiry during operational tests and the residual PostgREST multi-query time-of-check/time-of-use window. The dual-read CAS checks are not atomic DB locks; never interpret them as transactionally guaranteeing no subsequent state changes.
6. Add protected end-to-end tests with audited **synthetic** D2 attestations and negative tests for stale/unauthorized receipts.
7. Only after all gates, obtain **separate** user authorization to deploy the exact reviewed commit to `orotitan-chatgpt-mcp-pilot` and independently authorize the SQL if needed.
8. Keep Besi `RESEARCH` blocked until its six evidence gaps are independently resolved. No D29b feature permits bypassing `READY_FOR_DEEP_DIVE = NO`.

The public-facing write tool remains disabled. No `service_role` credentials are passed via any OAuth action. `GO PUBLISH` remains a distinct authority.
