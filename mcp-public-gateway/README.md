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
