// No OAuth issuer is advertised until the public gateway implements strict
// signed-token verification and a real, scoped MCP read capability.
export const runtime = "nodejs";
export const dynamic = "force-dynamic";
export function GET() {
  return new Response("OAuth configuration pending", {
    status: 404, headers: { "cache-control": "no-store" },
  });
}
