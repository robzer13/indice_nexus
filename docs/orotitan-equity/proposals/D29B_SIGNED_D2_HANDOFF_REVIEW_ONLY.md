# D29b — Signed D2 handoff proof boundary (REVIEW ONLY)

**Status:** PROPOSAL / NOT DEPLOYED / NOT ENABLED IN THE MCP ROUTE.
**Baseline:** D29a tested pilot SHA `5740dc40022993f414da1f8397de0d7725533095`.
**Contracts:** frozen V3.1 D1/D2 Pilotage and immutable Registry/Manifest hierarchy; no methodology changes.
**Allowed scope:** Pilot Besi only. No Supabase, Vercel, main Screener, RLS, storage or Registry mutation from this PR.

## The trust-boundary problem

D29a's OAuth `get_orotitan_next_prompt` correctly returns a Registry observation with `prompt:null`. The privileged D2 reader in `pilotage-trusted-loader-server.ts` fetches immutable bytes from allowlisted private GitHub repositories and reads wider Registry provenance using protected infrastructure credentials. Directly exposing that reader or `service_role` to the OAuth MCP endpoint would violate the trust boundary. D28 intentionally grants the OAuth role only four tables' selected columns and only Besi rows.

An apparently `FINAL` or `YES` Registry value alone is not proof that the artifacts and contracts were verified. The Besi run currently has `RESEARCH` blocker state; no positive handoff claim is made here.

## Proposed D29b design

### Independent issuer (NOT implemented or deployed here)

An administrative, independently authorized D2 issuer MUST:

1. Authenticate independently using the existing private admin control. This is not the ChatGPT OAuth token.
2. Use D2's `loadTrustedPilotageDecision` with the server-owned private GitHub readers. Validate full immutable bytes, SHA-256, manifest, version, lineage, exact pinned contracts, data cutoff, admission gates and current run/stage CAS.
3. Abort unless D2 outcome is `CONTINUE` or `HANDOFF` and `exactPrompt` is nonnull. `BLOCKED`, `DONE`, `AWAIT_EXPLICIT_GO_PUBLISH`, `CHECKPOINT` without independently admitted continuation, or contract drift MUST NOT be signed.
4. Re-observe current Registry state at signing; bind explicit `sub`, exact OAuth `client_id`, fixed MCP `resource`, Besi `run_id`, current stage and state versions, cutoff and exact contract-set SHA.
5. Produce a **canonical JSON attestation** with the D2 proof marker, verified manifest digest, evidence count, D1 state fingerprint, D1 exact prompt and its SHA-256, issuance and expiration <=5 minutes.
6. Sign ONLY in this privileged subsystem with an isolated Ed25519 private key, ideally KMS/HSM controlled; publish only the public key under a short, trusted `keyId`. NEVER install signing key in the Vercel public MCP project or in the public repository.
7. Store an immutable/overwritten short-lived signed receipt (no source artifacts) in an independently reviewed and RLS-scoped Supabase table. Signing/writing requires its own authorization and evidence audit trail; no auto-admission or publication.

### Delegated verifier (implemented as **pure, unconnected review code**)

`lib/orotitan-equity/v3/mcp-d29b-signed-handoff-gate.ts` validates:
- receipt format and canonical JSON; Ed25519 signature with **server-configured pinned verifier public keys**, never key material supplied by a caller;
- domain-separated signed message; exact schema, SHA-256 of the prompt, pinned contract-set digest and signed immutable-evidence provenance claims;
- <=300 second age/expiry and rejection of stale or future-dated proof;
- exact previously validated OAuth subject/client/resource and RLS-reobserved run identity;
- exact live run and current-stage CAS versions, current-stage code/lifecycle/gate/manifest-kind, cutoff and process versions;
- no currently observed blocker and no `BLOCKED_*` contract code;
- `HANDOFF` only for independently attested `COMPLETE / FINAL / YES`; `CONTINUE` only for `IN_PROGRESS` or `PAUSED`.

On any mismatch: `prompt:null` and no authorization. On success, **only a manual execution prompt** may be returned; `publicationAuthorized:false` and `mutationPerformed:false` are fixed. A signed D2 attestation is not the same as re-verifying all private bytes inside the OAuth process, so its trust label is `SIGNED_D2_ATTESTATION_WITH_LIVE_REGISTRY_CAS`, not `VERIFIED_FROM_IMMUTABLE_BYTES`.

### Storage proposal (NOT APPLIED)

`docs/orotitan-equity/proposals/D29B_SIGNED_RECEIPT_RLS_SQL_REVIEW_ONLY.sql` proposes a separate signed receipt table with immutable read-only OAuth access under RLS, bound to the exact approved run, subject, client and MCP resource. All identifiers remain placeholders in GitHub. No broad Registry grants and no `anon` read access.

**Important:** the table, the privileged signing issuer, private key custody, record rotation/revocation, REST adapter, race-safe CAS re-observation and end-to-end integration remain **unimplemented**. Do not connect the pure verifier to the live MCP tool or grant a signer write role without a separately approved rollout. An unsigned or self-signed receipt is not authoritative.

## Tests and limits

`tests/equity-v3-mcp-d29b-signed-handoff.test.ts` exercises synthetic Ed25519 fixtures and:
- valid signer plus exact scope + pinned pack + live CAS;
- wrong key, tampering, noncanonical payload, unauthenticated input, malformed signatures;
- wrong subject/client/resource/run, another run, stale versions and cutoff;
- observed blockers, expired/long-lived/future-dated receipts;
- wrong contract hash, payload prompt tampering, unadmitted publication;
- apparent `FINAL/YES` without the signed matching D2 receipt.

**These synthetic signatures prove the pure gate's behavior, NOT that any real Besi artifact is currently verified or that a D2 authorized issuer has been installed.**

## Remaining release checklist

1. Security review and implementation of trusted D2 issuer/signing custody (no issuance from ChatGPT/OAuth).
2. Dedicated RLS migration for the receipt table, negative access matrix (anon, unrelated OAuth app/subject/resource/run), and independent internal signer permissions.
3. Implement a caller-JWT RLS-scoped receipt loader with no private artifact access, and stable read-before/read-after state/CAS matching.
4. Bind the public verifier keys to trusted deployment configuration; validate key rotation, replay/revocation and signer availability.
5. Wire the `get_orotitan_next_prompt` OAuth route, retaining D29a fallback on proof unavailable. Validate end-to-end using **only synthetic or independently verified non-sensitive** D2 receipts.
6. Only after separate approval, deploy to `orotitan-chatgpt-mcp-pilot`; never automatically merge unrelated PRs or change the production Screener.
7. Independently inspect evidence state for Besi. Its current `RESEARCH` blockers prohibit claiming a completed Research handoff.

**This PR is a crypto trust-boundary prototype, not an executable D29b release.**
