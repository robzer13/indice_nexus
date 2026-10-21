import assert from "node:assert/strict";
import test from "node:test";
import {readFileSync} from "node:fs";
import {
  isAllowedChatGptRedirect,isAllowedChatGptReturnUrl,consentClientId,
  isAuthorizationId,oauthConsentConfig,
} from "../lib/orotitan-equity/v3/mcp-consent-policy-v1";

const base:NodeJS.ProcessEnv={
  NODE_ENV:"test",
  OROTITAN_OAUTH_CONSENT_ENABLED:"true",
  OROTITAN_OAUTH_SITE_ORIGIN:"https://orotitan-screener.vercel.app",
  NEXT_PUBLIC_SUPABASE_URL:"https://cugpgtzygqqlxetyeven.supabase.co",
  SUPABASE_PUBLISHABLE_KEY:"sb_publishable_xxxxxxxxxxxxxxxxxxxxxxxxx",
  OROTITAN_OAUTH_ALLOWED_USER_IDS:"48b9426b-4d0d-4858-9d87-efcd13ac415d",
  OROTITAN_MCP_OAUTH_ALLOWED_CLIENT_IDS:"approved-client-123",
};

test("OAuth consent is dark until user enrollment + site origin + public key provided",()=>{
  assert.equal(oauthConsentConfig({NODE_ENV:"test"}),null);
  assert.equal(oauthConsentConfig({...base,OROTITAN_OAUTH_CONSENT_ENABLED:"false"}),null);
  assert.equal(oauthConsentConfig({...base,OROTITAN_OAUTH_ALLOWED_USER_IDS:""}),null);
  assert.equal(oauthConsentConfig({...base,OROTITAN_OAUTH_ALLOWED_USER_IDS:"*"}),null);
  assert.equal(oauthConsentConfig({...base,OROTITAN_OAUTH_ALLOWED_USER_IDS:
    base.OROTITAN_OAUTH_ALLOWED_USER_IDS+","+base.OROTITAN_OAUTH_ALLOWED_USER_IDS}),null);
  assert.equal(oauthConsentConfig({...base,OROTITAN_OAUTH_SITE_ORIGIN:"http://localhost:3000"}),null);
  assert.equal(oauthConsentConfig({...base,OROTITAN_OAUTH_SITE_ORIGIN:"https://fake.example.com/redirect"}),null);
  assert.equal(oauthConsentConfig({...base,NEXT_PUBLIC_SUPABASE_URL:"https://evil.supabase.co"}),null);
  assert.equal(oauthConsentConfig({...base,SUPABASE_PUBLISHABLE_KEY:""}),null);
  assert.equal(oauthConsentConfig({...base,OROTITAN_MCP_OAUTH_ALLOWED_CLIENT_IDS:""}),null);
  assert.equal(oauthConsentConfig({...base,OROTITAN_MCP_OAUTH_ALLOWED_CLIENT_IDS:"*"}),null);
  assert.equal(oauthConsentConfig({...base,OROTITAN_MCP_OAUTH_ALLOWED_CLIENT_IDS:"approved-client-123,approved-client-123"}),null);
  const config=oauthConsentConfig(base);
  assert.equal(config?.siteOrigin,"https://orotitan-screener.vercel.app");
  assert.equal(config?.allowedUserIds.length,1);
  assert.deepEqual(config?.allowedClientIds,["approved-client-123"]);
});

test("OAuth consent only accepts the exact user-provided ChatGPT callback and return URLs",()=>{
  const expected="https://chatgpt.com/connector/oauth/tru5_CiXfOSM";
  assert.equal(isAllowedChatGptRedirect(expected),true);
  assert.equal(isAllowedChatGptReturnUrl(expected+"?code=123&state=abc"),true);
  assert.equal(isAllowedChatGptReturnUrl(expected+"?error=access_denied"),true);
  for(const url of [
    "https://chatgpt.com/connector_platform_oauth_redirect",
    "https://chatgpt.com/connector/oauth/another_client",
    "https://chatgpt.com/connector/oauth/tru5_CiXfOSM/",
    "https://evil.example.com/connector/oauth/tru5_CiXfOSM",
    "https://chatgpt.com.evil.example/connector/oauth/tru5_CiXfOSM",
    "https://chatgpt.com/other",
    "javascript:alert(1)",
    "//chatgpt.com/connector/oauth/tru5_CiXfOSM",
    "https://user@chatgpt.com/connector/oauth/tru5_CiXfOSM",
    expected+"#hash",
  ]){
    assert.equal(isAllowedChatGptRedirect(url),false,url);
    assert.equal(isAllowedChatGptReturnUrl(url),false,url);
  }
  // Supabase currently returns opaque URL-safe authorization IDs (e.g. 32 chars).
  // This check must never require a UUID; the SDK validates authorization existence.
  assert.equal(isAuthorizationId("abcdefghijklmnop0123456789ABCDEF"),true);
  assert.equal(isAuthorizationId("48b9426b-4d0d-4858-9d87-efcd13ac415d"),true);
  assert.equal(isAuthorizationId("abc_DEF-0123456789ghi"),true);
  for (const invalid of [
    null,undefined,123,["abcdefghijklmnop0123456789ABCDEF"],
    "../bad","too-short", "a".repeat(161), "a".repeat(15),
    "abcdefghijklmnop0123456789ABCDEF?next=x",
    "abcdefghijklmnop0123456789ABCDEF#fragment",
    "abcdefghijklmnop0123456789ABCDEF/../",
    "abcdefghijklmnop0123456789ABCDEF ",
    "abcdefghijklmnop0123456789ABCDEF%20",
    "abcdefghijklmnop0123456789ABCDEF\\n",
  ]) assert.equal(isAuthorizationId(invalid),false,String(invalid));
});

test("Consent client ID must be stable, not merely a display name",()=>{
  assert.equal(consentClientId({}),null);
  assert.equal(consentClientId({client:{}}),null);
  assert.equal(consentClientId({client:{id:"approved-client-123"}}),"approved-client-123");
  assert.equal(consentClientId({client_id:"approved-client-123"}),"approved-client-123");
  assert.equal(consentClientId({client:{id:"approved-client-123",client_id:"attacker"}}),null);
});

test("Consent UI is separate from OroTitan admin and does not expose service role or write Registry",()=>{
  const page=readFileSync(new URL("../app/oauth/consent/page.tsx",import.meta.url),"utf8");
  const client=readFileSync(new URL("../components/admin/orotitan-oauth-consent.tsx",import.meta.url),"utf8");
  const eligibility=readFileSync(new URL("../app/api/orotitan/oauth/eligibility/route.ts",import.meta.url),"utf8");
  assert.match(page,/if\(!cfg\)notFound\(\)/);
  assert.match(page,/assertProductionUrl/);
  assert.match(eligibility,/\.auth\.getUser\(authorization\.slice\(7\)\)/);
  assert.match(eligibility,/allowedUserIds\.includes/);
  assert.match(client,/\.auth\.oauth\.getAuthorizationDetails/);
  assert.match(client,/\.auth\.oauth\.approveAuthorization/);
  assert.match(client,/\.auth\.oauth\.denyAuthorization/);
  assert.match(client,/\.auth\.signInWithPassword/);
  assert.match(client,/\.auth\.getUser\(\)/);
  assert.match(page,/allowedClientIds=\{cfg\.allowedClientIds\}/);
  assert.match(client,/allowedClientIds\.includes\(consentClientId\(next\)/);
  assert.match(client,/consentClientId\(latest\)!==consentClientId\(details\)/);
  assert.doesNotMatch(client,/ADMIN_PASSWORD|ADMIN_SESSION_SECRET|SUPABASE_SERVICE_ROLE_KEY/);
  assert.doesNotMatch(client,/\.rpc\s*\(|\.insert\s*\(|\.update\s*\(|\.delete\s*\(/);
});
