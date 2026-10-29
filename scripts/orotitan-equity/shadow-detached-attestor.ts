/**
 * Detached nonproduction attestor. IMPORTANT: run this on the Attestor-controlled
 * execution host. NO Writer JWT, password, or private key generation in this code.
 * Never treat its signed output as a Registry persistence attestation.
 */
import { readFileSync, writeFileSync } from "node:fs";
import { resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { createClient } from "@supabase/supabase-js";
import {
  parseDetachedAttestorConfig, parseDetachedHandoff,
} from "../../lib/orotitan-equity/v3/shadow-detached-attestor-boundary";
import { signRereadFromPrivateStorage } from "../../lib/orotitan-equity/v3/private-source-signed-reread";
import { type SourceStorageClient } from "../../lib/orotitan-equity/v3/private-source-bytes-preflight";

function fail(ok: unknown, code: string): asserts ok {
  if (!ok) throw new Error(code);
}
function jwtClaims(token: string): Record<string, unknown> {
  const parts = token.split(".");
  fail(parts.length === 3, "DETACHED_JWT_SHAPE_INVALID");
  try {
    const v: unknown = JSON.parse(Buffer.from(parts[1], "base64url").toString("utf8"));
    fail(!!v && typeof v === "object" && !Array.isArray(v), "DETACHED_JWT_CLAIMS_INVALID");
    return v as Record<string, unknown>;
  } catch {
    throw new Error("DETACHED_JWT_CLAIMS_INVALID");
  }
}
export async function runDetachedAttestor(env: NodeJS.ProcessEnv): Promise<void> {
  const config = parseDetachedAttestorConfig(env);
  const handoff = parseDetachedHandoff(JSON.parse(readFileSync(config.handoffFile, "utf8")) as unknown);
  fail(resolve(config.handoffFile) !== resolve(config.receiptFile) &&
    resolve(config.privateKeyFile) !== resolve(config.receiptFile) &&
    resolve(config.privateKeyFile) !== resolve(config.handoffFile),
    "DETACHED_FILE_BOUNDARY_INVALID");
  const claims = jwtClaims(config.attestorJwt);
  const app: unknown = claims.app_metadata;
  fail(claims.role === "authenticated" && typeof claims.sub === "string" &&
    !!app && typeof app === "object" &&
    (app as Record<string, unknown>).orotitan_shadow_source_role === "attestor_v1",
    "DETACHED_ATTESTOR_JWT_ROLE_REQUIRED");

  const client = createClient(config.url, config.publishableKey, {
    auth: { autoRefreshToken: false, persistSession: false, detectSessionInUrl: false },
    global: { headers: { Authorization: "Bearer " + config.attestorJwt } },
  });
  const checked = await client.auth.getUser(config.attestorJwt);
  fail(!checked.error && !!checked.data.user &&
    checked.data.user.id === claims.sub &&
    checked.data.user.app_metadata?.orotitan_shadow_source_role === "attestor_v1",
    "DETACHED_ATTESTOR_SERVER_ROLE_UNVERIFIED");
  const privateKey = readFileSync(config.privateKeyFile, "utf8");
  const envelope = await signRereadFromPrivateStorage(
    client as unknown as SourceStorageClient, handoff.locator,
    {
      attestorId: config.attestorId, keyId: config.keyId,
      challengeNonce: handoff.challengeNonce, privateKey,
    },
  );
  // Create exclusively; do not silently replace a signed receipt.
  writeFileSync(config.receiptFile, JSON.stringify({
    schema: "OROTITAN_SHADOW_DETACHED_RECEIPT_V1_NONAUTHORITATIVE",
    envelope,
    transport: "EXACT_BYTES_REREAD",
    sourcePersistenceCertified: false,
    independentExecutionAndCustodyCertified: false,
    registryAdmitted: false,
  }, null, 2) + "\n", { flag: "wx", mode: 0o600 });
  console.log(JSON.stringify({
    schema: "OROTITAN_SHADOW_DETACHED_ATTESTOR_RESULT_V1",
    signedReread: true,
    sha256: envelope.statement.sha256,
    sizeBytes: envelope.statement.sizeBytes,
    sourcePersistenceCertified: false,
    independentExecutionAndCustodyCertified: false,
    registryAdmitted: false,
  }));
}
if (process.argv[1] && resolve(process.argv[1]) === resolve(fileURLToPath(import.meta.url))) {
  runDetachedAttestor(process.env).catch((error: unknown) => {
    console.error("DETACHED_ATTESTOR_FAILED: " +
      (error instanceof Error ? error.message : "UNKNOWN_ERROR"));
    process.exitCode = 1;
  });
}
