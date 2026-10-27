/**
 * Opt-in LOCAL TEST RUNNER. Signs in with two existing Auth users using
 * supplied local environment only, then invokes synthetic source-byte smoke.
 * NEVER store passwords in source files, GitHub variables, logs or this chat.
 * NOT an authority/Registry writer. Does not run without opt-in and two accounts.
 */
import { createClient } from "@supabase/supabase-js";
import { generateKeyPairSync } from "node:crypto";
import { fileURLToPath } from "node:url";
import { resolve } from "node:path";

export interface ShadowLocalConfig {
  url: string;
  testRef: string;
  productionRef: string;
  publishableKey: string;
  writerEmail: string;
  writerPassword: string;
  attestorEmail: string;
  attestorPassword: string;
}
function required(env: NodeJS.ProcessEnv, key: string): string {
  const value = env[key];
  if (!value || !value.trim()) throw new Error("SHADOW_CONFIG_REQUIRED_" + key);
  return value;
}
export function parseShadowLocalConfig(env: NodeJS.ProcessEnv): ShadowLocalConfig {
  if (env.PREFLIGHT_NONPROD_APPROVED !== "YES") throw new Error("NONPRODUCTION_APPROVAL_REQUIRED");
  const testRef = required(env, "SUPABASE_TEST_REF");
  const productionRef = required(env, "SUPABASE_PRODUCTION_REF");
  const url = required(env, "SUPABASE_TEST_URL");
  if (!/^[a-z0-9]{20}$/.test(testRef) || !/^[a-z0-9]{20}$/.test(productionRef) ||
      testRef === productionRef || url !== "https://" + testRef + ".supabase.co")
    throw new Error("PRODUCTION_OR_AMBIGUOUS_PROJECT_DENIED");
  const publishableKey = required(env, "SUPABASE_TEST_PUBLISHABLE_KEY");
  if (!publishableKey.startsWith("sb_publishable_") || publishableKey.startsWith("sb_secret_"))
    throw new Error("PUBLISHABLE_KEY_REQUIRED_NO_SECRET_KEY");
  const writerEmail = required(env, "SHADOW_WRITER_EMAIL");
  const writerPassword = required(env, "SHADOW_WRITER_PASSWORD");
  const attestorEmail = required(env, "SHADOW_ATTESTOR_EMAIL");
  const attestorPassword = required(env, "SHADOW_ATTESTOR_PASSWORD");
  if (!writerEmail.includes("@") || !attestorEmail.includes("@") ||
      writerEmail.toLowerCase() === attestorEmail.toLowerCase())
    throw new Error("DISTINCT_ACCOUNT_EMAILS_REQUIRED");
  return { url, testRef, productionRef, publishableKey,
    writerEmail, writerPassword, attestorEmail, attestorPassword };
}
async function signInScoped(
  config: ShadowLocalConfig, email: string, password: string,
  expectedRole: "writer_v1" | "attestor_v1",
): Promise<{ accessToken: string; userId: string }> {
  const client = createClient(config.url, config.publishableKey, {
    auth: { autoRefreshToken: false, persistSession: false, detectSessionInUrl: false },
  });
  const { data, error } = await client.auth.signInWithPassword({ email, password });
  if (error || !data?.session?.access_token)
    throw new Error("SHADOW_AUTHENTICATION_FAILED_" + expectedRole);
  const { data: verified, error: verifyError } = await client.auth.getUser(data.session.access_token);
  if (verifyError || !verified?.user || verified.user.id !== data.user?.id ||
      verified.user.app_metadata?.orotitan_shadow_source_role !== expectedRole ||
      verified.user.email?.toLowerCase() !== email.toLowerCase())
    throw new Error("SHADOW_SERVER_VERIFIED_ROLE_MISMATCH_" + expectedRole);
  if (typeof data.session.expires_at !== "number" ||
      data.session.expires_at * 1000 - Date.now() < 180000)
    throw new Error("SHADOW_JWT_TOO_CLOSE_TO_EXPIRY");
  return { accessToken: data.session.access_token, userId: verified.user.id };
}
export async function runAuthenticatedShadowSmoke(env: NodeJS.ProcessEnv): Promise<void> {
  const cfg = parseShadowLocalConfig(env);
  const writer = await signInScoped(cfg, cfg.writerEmail, cfg.writerPassword, "writer_v1");
  const attestor = await signInScoped(cfg, cfg.attestorEmail, cfg.attestorPassword, "attestor_v1");
  if (writer.userId === attestor.userId) throw new Error("SHADOW_PRINCIPALS_NOT_DISTINCT");
  // Ephemeral test-only key proves transport signature mechanics, NOT distinct
  // service processes / independent custody of a production signing key.
  const keys = generateKeyPairSync("ed25519");
  env.SUPABASE_TEST_WRITER_JWT = writer.accessToken;
  env.SUPABASE_TEST_READER_JWT = attestor.accessToken;
  env.ATTESTOR_TEST_PRIVATE_KEY_PEM_BASE64 =
    Buffer.from(keys.privateKey.export({ type: "pkcs8", format: "pem" })).toString("base64");
  env.ATTESTOR_TEST_PINNED_PUBLIC_KEY_PEM_BASE64 =
    Buffer.from(keys.publicKey.export({ type: "spki", format: "pem" })).toString("base64");
  // Avoid holding the passwords in process.env after successful sign-in.
  delete env.SHADOW_WRITER_PASSWORD;
  delete env.SHADOW_ATTESTOR_PASSWORD;
  process.stdout.write("Two shadow users authenticated and role-verified. Starting synthetic HTTP storage smoke.\n");
  await import("./shadow-private-source-smoke");
}
if (process.argv[1] &&
    resolve(process.argv[1]) === resolve(fileURLToPath(import.meta.url))) {
  runAuthenticatedShadowSmoke(process.env).catch(error => {
    const safeMessage = error instanceof Error ? error.message : "UNKNOWN_FAILURE";
    console.error("AUTHENTICATED_SHADOW_SMOKE_NOT_COMPLETED: " + safeMessage);
    process.exitCode = 1;
  });
}
