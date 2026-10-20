// FAIL CLOSED: no MCP session, token validation or Registry access is
// installed in this minimal public gateway. Never imply an active connector.
export const runtime = "nodejs";
export const dynamic = "force-dynamic";
const unavailable = () => new Response("MCP connection is not activated", {
  status: 503,
  headers: { "cache-control": "no-store", "x-content-type-options": "nosniff" },
});
export const GET = unavailable;
export const POST = unavailable;
export const DELETE = unavailable;
