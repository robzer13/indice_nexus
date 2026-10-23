# D31 — Independent ChatGPT mobile command authorization (review only)

Status: **DRAFT, NOT DEPLOYED**. Stacked on D30 draft PR #410. No Supabase Pro subscription, preview branch, hosted migration, production OAuth write tool, additional expense or Registry mutation.

## Verified upstream limitation (2026-10-23)

Supabase OAuth Server access tokens expose a signed `client_id` claim but currently do **not** support custom OAuth scopes such as `orotitan.write`. Standard `email/profile/openid/phone` scopes govern OIDC fields and must **not** be confused with write permission to OroTitan. In the live MCP resource server, `mcpWriteEnabled()` remains hardcoded `false`; `verifyMcpOAuthBearer` still returns **READ** authority only.

Sources (official Supabase documentation):
- https://supabase.com/docs/guides/auth/oauth-server/oauth-flows
- https://supabase.com/docs/guides/auth/oauth-server/token-security

## Increment implemented and tested

1. `verifyMcpOAuthBearer` now returns `clientId` **extracted from the validated signed JWT**, after the existing signature, issuer, audience, resource, authorized subject and allowed-client checks. It never trusts caller-supplied client fields. The live read-only tool behavior is unchanged.
2. The review-only `mobile-command-oauth-capabilities-v1.ts` verifies a signed bearer again at BOTH issuance and confirmation, constructs the actor using the verified `subject`, `clientId`, and fixed resource, and checks an **independent private policy** for a specific D30 command kind/preview/fingerprint.
3. The server-only capability is HMAC-SHA256 sealed with a **separate high-entropy secret key** (minimum 32 bytes), versioned purpose, a random 128-bit nonce, and an expiry no later than the original 2-minute preview. MAC is compared in constant time.
4. Confirmation checks sealed capability, JWT actor, exact action and fingerprint, private preview existence, matching actor, expiry, and independent policy **again** before calling the D30 private queue interface. Its receipt must be `REQUESTED_NOT_EXECUTED` with `companyInserted=false`, `researchStarted=false`, `registryMutationPerformed=false`.
5. Offline tests use **real ES256-signed OAuth fixture JWTs** and a local JWK set, without external internet or Supabase, covering wrong client/subject/resource, changed preview, expired/revoked authorization, fake signing key, corrupted MAC, idempotency and zero Research execution. Existing D30 PostgreSQL 17 tests remain in CI.

## Security boundary still NOT implemented

This is a **server-side capability prototype**, not a complete working mobile command. The sealed capability must never be returned in ChatGPT tool output, browser state, model context, logs, telemetry or client-side `NEXT_PUBLIC` variables. In production it must be stored in an **encrypted private store** bound to the preview, and confirmation must retrieve it server-side after verifying a fresh OAuth bearer. The signing key needs an independently managed secret, rotation, strict environment scoping and audit.

The D30 PostgreSQL SQL proposal is superuser-driven in a disposable fixture. It is **not a safe production writer API** yet: `p_subject`, `p_client`, and `p_resource` are function arguments which a future careless GRANT could forge. Do not expose the function through Supabase Data API or to `anon/authenticated`. A further privileged write issuer, private database principal, per-user grant lifecycle, rate limiting, atomic one-use confirmation and audit must be independently reviewed and tested before any hosted migration.

An official listing/issuer resolver with ISIN checksum, provenance, duplicate reconciliation and securities-vs-issuer logic is also not yet implemented. A durable, independently authorized worker/orchestrator must consume requests: *confirmation of a command is not creation of a company or launch of Research*.

## Acceptance / non-goals

- **NOT** importing the D31 module into either live `app/api/orotitan/mcp/route.ts` or the public gateway.
- **NOT** changing `mcpWriteEnabled()`, `OROTITAN_MCP_*` production configuration, OAuth consent scopes, Supabase permissions, migrations, or deploying code.
- **NOT** writing `companies`, `issuers`, `securities`, `orotitan_runs`, `orotitan_run_stages` or advancing Besi.
- **NOT** claiming an isolated local preview is a hosted queue; D30's queue persists only within disposable PostgreSQL 17 tests.
- **DO** require independent security review of grant issuance, exposed interfaces, token lifecycle, and privacy before requesting explicit production-write authorization.

## Next release gate

Build a **separate local PostgreSQL test for narrow private write role + sealed one-use approvals**, and an OAuth integration test exercising exact client binding through the MCP transport. Only after that and explicit production migration authorization can one decide how to host the durable queue on the existing Free-plan Supabase project, with a quota/billing impact check.
