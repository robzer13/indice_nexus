/**
 * Explicitly enrolled Supabase Auth user check for the ChatGPT consent pilot.
 * A browser admin cookie is NOT accepted as a substitute for a Supabase login.
 */
import { createClient } from "@supabase/supabase-js";
import { oauthConsentConfig } from "@/lib/orotitan-equity/v3/mcp-consent-policy-v1";

export const runtime="nodejs";
export const dynamic="force-dynamic";

export async function POST(request: Request): Promise<Response> {
  const deny=(status:number)=>Response.json({allowed:false},{status,
    headers:{"cache-control":"no-store","x-content-type-options":"nosniff"}});
  const cfg=oauthConsentConfig(process.env);
  if(!cfg)return deny(404);
  if(new URL(request.url).origin!==cfg.siteOrigin ||
    request.headers.get("origin")!==cfg.siteOrigin)return deny(403);
  const authorization=request.headers.get("authorization");
  if(!authorization || !/^Bearer [A-Za-z0-9_\-.]{30,9000}$/.test(authorization))return deny(401);
  const supabase=createClient(cfg.supabaseUrl,cfg.publishableKey,{
    auth:{persistSession:false,autoRefreshToken:false,detectSessionInUrl:false},
  });
  const {data,error}=await supabase.auth.getUser(authorization.slice(7));
  if(error || !data.user || !cfg.allowedUserIds.includes(data.user.id.toLowerCase()) ||
    data.user.is_anonymous === true)return deny(403);
  return Response.json({allowed:true},{headers:{"cache-control":"no-store"}});
}
