/**
 * RFC 9728 MCP protected-resource discovery. This endpoint advertises only an
 * external, explicitly configured OAuth AS; it never issues OAuth tokens.
 */
import {
  mcpOAuthConfig, oauthResourceMetadata,
} from "@/lib/orotitan-equity/v3/mcp-oauth-v1";

export const dynamic = "force-dynamic";

export function GET(request:Request):Response {
  const config=mcpOAuthConfig(process.env);
  if(!config)return new Response("Not found",{status:404});
  if(new URL(request.url).origin !== process.env.OROTITAN_MCP_PUBLIC_ORIGIN) {
    return new Response("Unavailable",{status:503});
  }
  return Response.json(oauthResourceMetadata(config),{
    headers:{
      "cache-control":"no-store",
      "x-content-type-options":"nosniff",
      "access-control-allow-origin":"*",
    },
  });
}
