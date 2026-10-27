/**
 * Explicit opt-in test ONLY: two existing Supabase Auth accounts, in any order.
 * Identifies Writer and Attestor from fresh server-verified Auth app_metadata.
 * Credentials are provided in a private operator environment, NEVER committed.
 * Uploads only a synthetic PDF via the existing shadow-only smoke module.
 */
import { createClient } from "@supabase/supabase-js";
import { generateKeyPairSync } from "node:crypto";
import { fileURLToPath } from "node:url";
import { resolve } from "node:path";

type PrincipalRole = "writer_v1" | "attestor_v1";
export interface ShadowLocalConfig {
  url: string;
  testRef: string;
  productionRef: string;
  publishableKey: string;
  accountAEmail: string;
  accountAPassword: string;
  accountBEmail: string;
  accountBPassword: string;
}
function required(env: NodeJS.ProcessEnv, key: string): string {
  const value = env[key];
  if (!value || !value.trim()) throw new Error("SHADOW_CONFIG_REQUIRED_" + key);
  return value;
}
export function parseShadowLocalConfig(env: NodeJS.ProcessEnv): ShadowLocalConfig {
  if (env.PREFLIGHT_NONPROD_APPROVED !== "YES")
    throw new Error("NONPRODUCTION_APPROVAL_REQUIRED");
  const testRef = required(env, "SUPABASE_TEST_REF");
  const productionRef = required(env, "SUPABASE_PRODUCTION_REF");
  const url = required(env, "SUPABASE_TEST_URL");
  if (!/^[a-z0-9]{20}$/.test(testRef) || !/^[a-z0-9]{20}$/.test(productionRef) ||
      testRef === productionRef || url !== "https://" + testRef + ".supabase.co")
    throw new Error("PRODUCTION_OR_AMBIGUOUS_PROJECT_DENIED");
  const publishableKey = required(env, "SUPABASE_TEST_PUBLISHABLE_KEY");
  if (!publishableKey.startsWith("sb_publishable_"))
    throw new Error("PUBLISHABLE_KEY_REQUIRED_NO_SECRET_KEY");
  const accountAEmail = required(env, "SHADOW_ACCOUNT_A_EMAIL");
  const accountAPassword = required(env, "SHADOW_ACCOUNT_A_PASSWORD");
  const accountBEmail = required(env, "SHADOW_ACCOUNT_B_EMAIL");
  const accountBPassword = required(env, "SHADOW_ACCOUNT_B_PASSWORD");
  if (!accountAEmail.includes("@") || !accountBEmail.includes("@") ||
      accountAEmail.trim().toLowerCase() === accountBEmail.trim().toLowerCase())
    throw new Error("DISTINCT_ACCOUNT_EMAILS_REQUIRED");
  return { url, testRef, productionRef, publishableKey,
    accountAEmail, accountAPassword, accountBEmail, accountBPassword };
}
interface VerifiedAccount {
  role: PrincipalRole;
  id: string;
  jwt: string;
}
async function authenticate(
  cfg: ShadowLocalConfig, email: string, password: string
): Promise<VerifiedAccount> {
  const client = createClient(cfg.url, cfg.publishableKey, {
    auth: { autoRefreshToken: false, persistSession: false, detectSessionInUrl: false },
  });
  const { data, error } = await client.auth.signInWithPassword({ email, password });
  if (error || !data?.session?.access_token || !data.user)
    throw new Error("SHADOW_ACCOUNT_LOGIN_FAILED");
  const jwt = data.session.access_token;
  const { data: checked, error: checkError } = await client.auth.getUser(jwt);
  if (checkError || !checked?.user || checked.user.id !== data.user.id ||
      checked.user.email?.toLowerCase() !== email.trim().toLowerCase())
    throw new Error("SHADOW_SERVER_IDENTITY_CHECK_FAILED");
  const role = checked.user.app_metadata?.orotitan_shadow_source_role;
  if (role !== "writer_v1" && role !== "attestor_v1")
    throw new Error("SHADOW_UNEXPECTED_ACCOUNT_ROLE");
  const pieces = jwt.split(".");
  if (pieces.length !== 3) throw new Error("SHADOW_JWT_SHAPE_INVALID");
  let claims: { sub?: string; app_metadata?: { orotitan_shadow_source_role?: string } };
  try {
    claims = JSON.parse(Buffer.from(pieces[1], "base64url").toString("utf8"));
  } catch { throw new Error("SHADOW_JWT_CLAIMS_INVALID"); }
  if (claims.sub !== checked.user.id ||
      claims.app_metadata?.orotitan_shadow_source_role !== role)
    throw new Error("SHADOW_STALE_JWT_ROLE_CLAIM");
  if (typeof data.session.expires_at !== "number" ||
      data.session.expires_at * 1000 - Date.now() < 180000)
    throw new Error("SHADOW_SESSION_TOO_OLD");
  return { role, id: checked.user.id, jwt };
}
export async function runAuthenticatedShadowSmoke(env: NodeJS.ProcessEnv): Promise<void> {
  const cfg = parseShadowLocalConfig(env);
  const a = await authenticate(cfg, cfg.accountAEmail, cfg.accountAPassword);
  const b = await authenticate(cfg, cfg.accountBEmail, cfg.accountBPassword);
  if (a.id === b.id || a.role === b.role) throw new Error("SHADOW_TWO_DISTINCT_ROLES_REQUIRED");
  const writer = a.role === "writer_v1" ? a : b;
  const attestor = a.role === "attestor_v1" ? a : b;
  const keypair = generateKeyPairSync("ed25519");
  env.SUPABASE_TEST_WRITER_JWT = writer.jwt;
  env.SUPABASE_TEST_READER_JWT = attestor.jwt;
  env.ATTESTOR_TEST_PRIVATE_KEY_PEM_BASE64 =
    Buffer.from(keypair.privateKey.export({ type: "pkcs8", format: "pem" })).toString("base64");
  env.ATTESTOR_TEST_PINNED_PUBLIC_KEY_PEM_BASE64 =
    Buffer.from(keypair.publicKey.export({ type: "spki", format: "pem" })).toString("base64");
  delete env.SHADOW_ACCOUNT_A_PASSWORD;
  delete env.SHADOW_ACCOUNT_B_PASSWORD;
  process.stdout.write("Two distinct roles validated by Supabase Auth; running synthetic Storage test.\n");
  await import("./shadow-private-source-smoke");
}
if (process.argv[1] && resolve(process.argv[1]) === resolve(fileURLToPath(import.meta.url))) {
  runAuthenticatedShadowSmoke(process.env).catch(error => {
    console.error("AUTHENTICATED_SHADOW_SMOKE_NOT_COMPLETED: " +
      (error instanceof Error ? error.message : "UNKNOWN_FAILURE"));
    process.exitCode = 1;
  });
}
