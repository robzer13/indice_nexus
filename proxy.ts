import { NextRequest, NextResponse } from "next/server";

/**
 * Pure, self-contained route gate for the standalone public MCP pilot.
 *
 * IMPORTANT: also compiled when the repository's independent
 * mcp-public-gateway fixture is built. Do NOT import from "@/lib" or a parent
 * Next.js app here: the fixture must remain compilable in isolation.
 */
const PILOT_ORIGIN = "https://orotitan-chatgpt-mcp-pilot.vercel.app";
const pilotPaths = new Set([
  "/",
  "/oauth/consent",
  "/api/orotitan/oauth/eligibility",
  "/api/orotitan/mcp",
  "/.well-known/oauth-protected-resource",
  "/favicon.ico",
]);
function isPilotPublicPath(pathname: string): boolean {
  return pilotPaths.has(pathname) ||
    pathname.startsWith("/_next/static/") ||
    pathname.startsWith("/_next/image/");
}

export function proxy(request: NextRequest) {
  const isPilot = process.env.OROTITAN_MCP_PUBLIC_ORIGIN === PILOT_ORIGIN &&
    process.env.OROTITAN_OAUTH_SITE_ORIGIN === PILOT_ORIGIN &&
    process.env.OROTITAN_MCP_AUTH_MODE === "OAUTH";
  if (!isPilot || isPilotPublicPath(request.nextUrl.pathname)) {
    return NextResponse.next();
  }
  return new NextResponse("Not found", {
    status: 404,
    headers: {
      "cache-control": "no-store",
      "x-content-type-options": "nosniff",
    },
  });
}

export const config = { matcher: "/:path*" };
