/** Gate-side, offline, synthetic-only challenge for an ALREADY stored shadow object. */
import { randomUUID } from "node:crypto";
import { writeFileSync } from "node:fs";
import { resolve } from "node:path";
import { fileURLToPath } from "node:url";
import {
  parseDetachedHandoff, SHADOW_REF, PRODUCTION_REF, SYNTHETIC_SHA, SYNTHETIC_URL,
} from "../../lib/orotitan-equity/v3/shadow-detached-attestor-boundary";

export function runGateChallenge(env: NodeJS.ProcessEnv): void {
  if (env.PREFLIGHT_NONPROD_APPROVED !== "YES" ||
    env.SUPABASE_TEST_REF !== SHADOW_REF ||
    env.SUPABASE_PRODUCTION_REF !== PRODUCTION_REF)
    throw new Error("DETACHED_GATE_SHADOW_ONLY_OPT_IN_REQUIRED");
  const uri = env.SHADOW_GATE_EXISTING_SYNTHETIC_URI;
  const output = env.SHADOW_GATE_HANDOFF_FILE;
  if (!uri || !output) throw new Error("DETACHED_GATE_URI_AND_OUTPUT_REQUIRED");
  const match = /^supabase:\/\/orotitan-source-files-v1\/(preadmission\/([0-9a-f-]{36})\/([0-9a-f-]{36})\/v0001\/f9cda65b308c24e62fd9a5070df43202c2c0d4560a8aba8f6e697a8a9f467b9b\.pdf)$/.exec(uri);
  if (!match) throw new Error("DETACHED_GATE_SYNTHETIC_URI_REQUIRED");
  const path = match[1], runId = match[2], artifactId = match[3];
  const handoff = parseDetachedHandoff({
    schema: "OROTITAN_SHADOW_DETACHED_HANDOFF_V1",
    challengeNonce: randomUUID(),
    locator: {
      runId, artifactId, version: 1, bucket: "orotitan-source-files-v1",
      path, uri, backend: "SUPABASE_STORAGE", mediaType: "application/pdf",
      sizeBytes: 715, sha256: SYNTHETIC_SHA, sourceUrl: SYNTHETIC_URL,
      publishedOn: "2026-01-01", dataCutoff: "2026-09-21",
      dateAuthority: "DECLARED_NOT_INDEPENDENTLY_VERIFIED",
    },
  });
  writeFileSync(output, JSON.stringify(handoff, null, 2) + "\n",
    {flag: "wx", mode: 0o600});
  console.log(JSON.stringify({
    schema: "OROTITAN_SHADOW_GATE_CHALLENGE_V1",
    challengeNonce: handoff.challengeNonce, fileCreated: true,
    provenance: "SYNTHETIC_ONLY_DECLARED_DATE",
    registryAdmitted: false,
  }));
}
if (process.argv[1] && resolve(process.argv[1]) === resolve(fileURLToPath(import.meta.url))) {
  try { runGateChallenge(process.env); }
  catch (error: unknown) {
    console.error("DETACHED_GATE_CHALLENGE_FAILED: " +
      (error instanceof Error ? error.message : "UNKNOWN_ERROR"));
    process.exitCode = 1;
  }
}
