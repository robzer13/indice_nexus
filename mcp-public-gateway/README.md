# OroTitan public MCP connectivity test

This is a **separate Next.js deployment root**, not the OroTitan Screener.
It exposes no Supabase credentials, Registry reads, artifacts or admin endpoints.

Current deliberately non-operational behavior:
- `GET /health` -> 200 with public, non-sensitive connectivity flags.
- `GET|POST|DELETE /api/orotitan/mcp` -> 503, no tool execution.
- `GET /.well-known/oauth-protected-resource` -> 404 pending tested issuer.
- No OAuth consent screen, login, write operation, or Supabase integration.
- No auto-deploy from the original production project.

**Do not tell a user to register this endpoint in ChatGPT yet.** It is a
public connectivity probe only. An audited, issuer-bound OAuth MCP
implementation, proper access control, approved Supabase Auth configuration
and read-only Besi test must be delivered before activation. Never proxy
original Admin endpoints or copy a production service-role key into this
gateway.

## Deployment status — 2026-10-20

Created an independent **unlinked** Vercel project under team OROTITAN:
`orotitan-mcp-public-test`, ID `prj_0auQ2tqRQPKnT6K03N9lxZDcIfqA`.
Its Vercel SSO protection is intentionally disabled **only for this project**.
The app sources are committed under this dedicated directory on PR #386.
No database credentials or production environment variables were copied.

Attempting to deploy the exact contents of this folder to the separate
project was rejected by the Vercel API with HTTP 402,
`api-deployments-free-per-day` (more than 100 daily API deployments,
retry-after 86400 seconds). Vercel CLI fallback is not available in
the active environment (no CLI binary or configured token).
**No deployment was created**, so no public endpoint is live yet.
Do not claim the existence of the public MCP endpoint or give users a
placeholder URL to install. Wait for quota reset or use an owner-authorized
alternative release mechanism. Do not upgrade billing automatically.

After the quota resets:
1. Deploy this directory **only** to the new separate project (not to
   `orotitan-screener`), retaining fail-closed endpoints.
2. Confirm publicly accessible `GET /health` returns 200;
   `/api/orotitan/mcp` returns 503 and the metadata path returns 404.
3. Only then design audited OAuth/RFC9728 discovery and the
   independently validated **read-only** Besi MCP tools. A public
   health endpoint is not equivalent to authenticated Registry access.
