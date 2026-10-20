export const runtime = "nodejs";
export const dynamic = "force-dynamic";
export function GET() {
  return Response.json({ service: "orotitan-mcp-test", stage: "connection-preparation",
    mcpEnabled: false, registryAccessible: false, writesEnabled: false }, {
    headers: { "cache-control": "no-store", "x-content-type-options": "nosniff" },
  });
}
