/**
 * ATTTESTOR-ONLY local login wrapper. Separate operator and signing environment
 * are REQUIRED for any real independence claim; this tool cannot certify custody.
 * Never enter credentials on the Writer machine for an independence assessment.
 */
import { createClient } from "@supabase/supabase-js";
import { resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { parseDetachedAttestorConfig } from "../../lib/orotitan-equity/v3/shadow-detached-attestor-boundary";
import { runDetachedAttestor } from "./shadow-detached-attestor";

export async function runDetachedAttestorWithLogin(env: NodeJS.ProcessEnv): Promise<void> {
  // Check all offline guards before doing any network login.
  const checkedEnv: NodeJS.ProcessEnv = {
    ...env, SHADOW_DETACHED_ATTESTOR_JWT: "preflight-placeholder-not-a-token",
  };
  const checked = parseDetachedAttestorConfig(checkedEnv);
  if (env.SHADOW_DETACHED_ATTESTOR_JWT)
    throw new Error("DETACHED_NO_MIXED_LOGIN_AND_TOKEN");
  const email = env.SHADOW_DETACHED_ATTESTOR_EMAIL;
  const password = env.SHADOW_DETACHED_ATTESTOR_PASSWORD;
  if (!email || !email.includes("@") || !password)
    throw new Error("DETACHED_ATTESTOR_LOCAL_LOGIN_REQUIRED");
  const client = createClient(checked.url, checked.publishableKey, {
    auth: { autoRefreshToken: false, persistSession: false, detectSessionInUrl: false },
  });
  const {data, error} = await client.auth.signInWithPassword({email, password});
  if (error || !data.session || !data.user)
    throw new Error("DETACHED_ATTESTOR_LOGIN_FAILED");
  if (data.user.app_metadata?.orotitan_shadow_source_role !== "attestor_v1")
    throw new Error("DETACHED_ATTESTOR_ROLE_REQUIRED");
  if (!data.session.expires_at ||
    data.session.expires_at * 1000 - Date.now() < 180000)
    throw new Error("DETACHED_SESSION_NOT_FRESH");
  const temporary: NodeJS.ProcessEnv = {
    ...env, SHADOW_DETACHED_ATTESTOR_JWT: data.session.access_token,
  };
  delete temporary.SHADOW_DETACHED_ATTESTOR_PASSWORD;
  delete temporary.SHADOW_DETACHED_ATTESTOR_EMAIL;
  // Clear credentials from the supplied process environment too (not disk).
  delete env.SHADOW_DETACHED_ATTESTOR_PASSWORD;
  await runDetachedAttestor(temporary);
}
if (process.argv[1] && resolve(process.argv[1]) === resolve(fileURLToPath(import.meta.url))) {
  runDetachedAttestorWithLogin(process.env).catch((error: unknown) => {
    console.error("DETACHED_ATTESTOR_LOGIN_FAILED: " +
      (error instanceof Error ? error.message : "UNKNOWN_ERROR"));
    process.exitCode = 1;
  });
}
