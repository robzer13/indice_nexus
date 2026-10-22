/**
 * Independent public OAuth/MCP pilot surface. This file deliberately has no
 * Supabase, admin-session or Registry imports.
 *
 * The pilot shares source with the main Screener, but never serves the
 * Screener/admin routes to unauthenticated public clients.
 */
export const PILOT_ORIGIN = "https://orotitan-chatgpt-mcp-pilot.vercel.app";

export function isIsolatedMcpPilot(env: NodeJS.ProcessEnv): boolean {
  return env.OROTITAN_MCP_PUBLIC_ORIGIN === PILOT_ORIGIN &&
    env.OROTITAN_OAUTH_SITE_ORIGIN === PILOT_ORIGIN &&
    env.OROTITAN_MCP_AUTH_MODE === "OAUTH";
}

/** Explicit allowlist; never match path prefixes or accept traversal. */
export function isPilotPublicPath(pathname: string): boolean {
  return pathname === "/" ||
    pathname === "/oauth/consent" ||
    pathname === "/api/orotitan/oauth/eligibility" ||
    pathname === "/api/orotitan/mcp" ||
    pathname === "/.well-known/oauth-protected-resource" ||
    pathname === "/favicon.ico" ||
    pathname.startsWith("/_next/static/") ||
    pathname.startsWith("/_next/image/");
}
