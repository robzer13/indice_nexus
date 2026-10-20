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

export function isAuthorizationId(value: unknown): value is string {
  return typeof value === "string" && UUID.test(value);
}

export function oauthConsentConfig(env: NodeJS.ProcessEnv) {
  if (env.OROTITAN_OAUTH_CONSENT_ENABLED !== "true") return null;
  const base = env.OROTITAN_OAUTH_SITE_ORIGIN;
  const authUrl = env.NEXT_PUBLIC_SUPABASE_URL;
  const publishableKey = env.SUPABASE_PUBLISHABLE_KEY;
  const ids = env.OROTITAN_OAUTH_ALLOWED_USER_IDS?.split(",").map(x=>x.trim().toLowerCase()) ?? [];
  if (!base || !authUrl || !publishableKey || publishableKey.length < 12 ||
    ids.length < 1 || ids.length > 10 || ids.some(x=>!UUID.test(x)) ||
    new Set(ids).size !== ids.length) return null;
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
  };
}
