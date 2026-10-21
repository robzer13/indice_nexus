import { NextRequest, NextResponse } from "next/server";
import { isIsolatedMcpPilot, isPilotPublicPath } from "@/lib/orotitan-equity/v3/mcp-pilot-boundary-v1";

/**
 * Route isolation only on the separately configured Vercel OAuth test pilot.
 * Main OroTitan routes remain unchanged on every other deployment.
 */
export function proxy(request: NextRequest) {
  if (!isIsolatedMcpPilot(process.env)) return NextResponse.next();
  if (isPilotPublicPath(request.nextUrl.pathname)) return NextResponse.next();
  return new NextResponse("Not found", {
    status: 404,
    headers: {
      "cache-control": "no-store",
      "x-content-type-options": "nosniff",
    },
  });
}

export const config = { matcher: "/:path*" };
