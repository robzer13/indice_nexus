/**
 * ChatGPT OAuth consent: isolated pre-deployment policy.
 * The existing OroTitan HMAC admin cookie is NEVER an OAuth identity.
 * Supabase Auth must authenticate an explicitly enrolled human user.
 */
export const CHATGPT_REDIRECT_STABLE = "https://chatgpt.com/connector_platform_oauth_redirect";
const CHATGPT_CALLBACK_ID = /^https:\/\/chatgpt\.com\/connector\/oauth\/[a-zA-Z0-9_-]{1,160}$/;
const UUID = /^[0-9a-f]{8}-[0-9a-f]{4}-[1-8][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i;

export function isAllowedChatGptRedirect(value: unknown): value is string {
  return typeof value === "string" &&
    (value === CHATGPT_REDIRECT_STABLE || CHATGPT_CALLBACK_ID.test(value));
}

export function isAllowedChatGptReturnUrl(value:unknown):value is string {
  if(typeof value!=="string" || value.length>2500)return false;
  try {
    const url=new URL(value);
    return url.protocol==="https:" && url.origin==="https://chatgpt.com" &&
      !url.username && !url.password && url.hash==="" &&
      isAllowedChatGptRedirect(url.origin+url.pathname);
  }catch{return false;}
}

/** Reject absent or inconsistent OAuth client IDs across SDK response shapes. */
export function consentClientId(details: {
  client?: { id?: unknown; client_id?: unknown } | null;
  client_id?: unknown;
}): string | null {
  const ids=[details.client?.id,details.client?.client_id,details.client_id]
    .filter((x):x is string=>typeof x==="string" && x.length>0);
  return ids.length>0 && new Set(ids).size===1 ? ids[0] : null;
}

export function isAuthorizationId(value: unknown): value is string {
  return typeof value === "string" && UUID.test(value);
}

export function oauthConsentConfig(env: NodeJS.ProcessEnv) {
  if (env.OROTITAN_OAUTH_CONSENT_ENABLED !== "true") return null;
  const base = env.OROTITAN_OAUTH_SITE_ORIGIN;
  const authUrl = env.NEXT_PUBLIC_SUPABASE_URL;
  const publishableKey = env.SUPABASE_PUBLISHABLE_KEY;
  const ids = env.OROTITAN_OAUTH_ALLOWED_USER_IDS?.split(",").map(x=>x.trim().toLowerCase()) ?? [];
  const clients = env.OROTITAN_MCP_OAUTH_ALLOWED_CLIENT_IDS?.split(",").map(x=>x.trim()) ?? [];
  if (!base || !authUrl || !publishableKey || publishableKey.length < 12 ||
    ids.length < 1 || ids.length > 10 || ids.some(x=>!UUID.test(x)) ||
    new Set(ids).size !== ids.length ||
    clients.length < 1 || clients.length > 10 ||
    clients.some(x=>x.length<6 || x.length>160 || !/^[a-zA-Z0-9_-]+$/.test(x)) ||
    new Set(clients).size!==clients.length) return null;
  try {
    const origin = new URL(base);
    const supabase = new URL(authUrl);
    if (origin.protocol !== "https:" || origin.origin !== base ||
      origin.username || origin.password || origin.search || origin.hash ||
      supabase.protocol !== "https:" || supabase.pathname !== "/" ||
      supabase.search || supabase.hash ||
      supabase.hostname !== "cugpgtzygqqlxetyeven.supabase.co") return null;
  } catch { return null; }
  return {
    supabaseUrl:authUrl,
    publishableKey,
    siteOrigin:base,
    allowedUserIds:ids,
    allowedClientIds:clients,
  };
}
