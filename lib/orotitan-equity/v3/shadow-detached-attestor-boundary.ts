import { type PreAdmissionLocator, PrivateSourceError } from "./private-source-bytes-preflight";

/**
 * SHADOW TEST CONTROL ONLY. Fail closed before any authenticated HTTP request.
 * The returned handoff is NON_AUTHORITATIVE and never causes Registry writes.
 */
export const SHADOW_REF = "awgsurdyvsyolcgpnygh";
export const PRODUCTION_REF = "cugpgtzygqqlxetyeven";
export const SYNTHETIC_SHA = "f9cda65b308c24e62fd9a5070df43202c2c0d4560a8aba8f6e697a8a9f467b9b";
export const SYNTHETIC_URL = "https://ourbrand.asml.com/m/synthetic/original/shadow-only.pdf";
const noncePattern = /^[0-9a-f]{8}-[0-9a-f]{4}-[1-8][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i;
const uuidPattern = noncePattern;
const idPattern = /^[a-z][a-z0-9-]{2,63}$/;

function requireGuard(condition: unknown, code: string): asserts condition {
  if (!condition) throw new PrivateSourceError(code);
}
function record(value: unknown): value is Record<string, unknown> {
  return !!value && typeof value === "object" && !Array.isArray(value);
}
function exact(value: Record<string, unknown>, fields: readonly string[], code: string): void {
  requireGuard(Object.keys(value).length === fields.length &&
    fields.every(field => Object.prototype.hasOwnProperty.call(value, field)), code);
}

export interface DetachedHandoff {
  schema: "OROTITAN_SHADOW_DETACHED_HANDOFF_V1";
  challengeNonce: string;
  locator: PreAdmissionLocator;
}
const locatorFields = [
  "runId", "artifactId", "version", "bucket", "path", "uri", "backend", "mediaType",
  "sizeBytes", "sha256", "sourceUrl", "publishedOn", "dataCutoff", "dateAuthority",
] as const;

export function parseDetachedHandoff(input: unknown): DetachedHandoff {
  requireGuard(record(input), "DETACHED_HANDOFF_INVALID");
  exact(input, ["schema", "challengeNonce", "locator"], "DETACHED_HANDOFF_SHAPE_INVALID");
  requireGuard(input.schema === "OROTITAN_SHADOW_DETACHED_HANDOFF_V1" &&
    typeof input.challengeNonce === "string" && noncePattern.test(input.challengeNonce),
    "DETACHED_HANDOFF_NONCE_INVALID");
  const loc = input.locator;
  requireGuard(record(loc), "DETACHED_LOCATOR_INVALID");
  exact(loc, locatorFields, "DETACHED_LOCATOR_SHAPE_INVALID");
  requireGuard(
    typeof loc.runId === "string" && uuidPattern.test(loc.runId) &&
    typeof loc.artifactId === "string" && uuidPattern.test(loc.artifactId) &&
    loc.version === 1 && loc.bucket === "orotitan-source-files-v1" &&
    loc.backend === "SUPABASE_STORAGE" &&
    loc.mediaType === "application/pdf" && loc.sizeBytes === 715 &&
    loc.sha256 === SYNTHETIC_SHA && loc.sourceUrl === SYNTHETIC_URL &&
    loc.publishedOn === "2026-01-01" && loc.dataCutoff === "2026-09-21" &&
    loc.dateAuthority === "DECLARED_NOT_INDEPENDENTLY_VERIFIED",
    "DETACHED_NON_SYNTHETIC_INPUT_DENIED",
  );
  const canonical = "preadmission/" + loc.runId.toLowerCase() + "/" +
    loc.artifactId.toLowerCase() + "/v0001/" + SYNTHETIC_SHA + ".pdf";
  requireGuard(loc.path === canonical &&
    loc.uri === "supabase://orotitan-source-files-v1/" + canonical,
    "DETACHED_LOCATOR_NOT_CONTENT_ADDRESSED");
  return input as unknown as DetachedHandoff;
}

function required(env: NodeJS.ProcessEnv, key: string): string {
  const value = env[key];
  requireGuard(typeof value === "string" && value.trim().length > 0,
    "DETACHED_ENV_REQUIRED_" + key);
  return value;
}
function safeProject(env: NodeJS.ProcessEnv): { url: string; publishableKey: string } {
  requireGuard(env.PREFLIGHT_NONPROD_APPROVED === "YES", "DETACHED_SHADOW_OPT_IN_REQUIRED");
  requireGuard(env.SUPABASE_TEST_REF === SHADOW_REF &&
    env.SUPABASE_PRODUCTION_REF === PRODUCTION_REF &&
    env.SUPABASE_TEST_URL === "https://" + SHADOW_REF + ".supabase.co",
    "DETACHED_PRODUCTION_OR_UNPINNED_PROJECT");
  const publishableKey = required(env, "SUPABASE_TEST_PUBLISHABLE_KEY");
  requireGuard(publishableKey.startsWith("sb_publishable_") &&
    !publishableKey.includes("\n"), "DETACHED_PUBLISHABLE_KEY_REQUIRED");
  return { url: env.SUPABASE_TEST_URL, publishableKey };
}

export interface DetachedAttestorConfig {
  url: string;
  publishableKey: string;
  attestorJwt: string;
  privateKeyFile: string;
  handoffFile: string;
  receiptFile: string;
  attestorId: string;
  keyId: string;
}

export function parseDetachedAttestorConfig(env: NodeJS.ProcessEnv): DetachedAttestorConfig {
  const shared = safeProject(env);
  for (const forbidden of [
    "SUPABASE_TEST_WRITER_JWT", "SHADOW_ACCOUNT_A_PASSWORD", "SHADOW_ACCOUNT_B_PASSWORD",
    "SHADOW_WRITER_PASSWORD", "ATTESTOR_TEST_PRIVATE_KEY_PEM_BASE64",
    "SUPABASE_SERVICE_ROLE_KEY", "SUPABASE_SECRET_KEY",
  ]) {
    requireGuard(!env[forbidden], "DETACHED_FORBIDDEN_SHARED_CREDENTIAL_" + forbidden);
  }
  const attestorId = required(env, "SHADOW_DETACHED_ATTESTOR_ID");
  const keyId = required(env, "SHADOW_DETACHED_KEY_ID");
  requireGuard(idPattern.test(attestorId) && idPattern.test(keyId),
    "DETACHED_ATTESTOR_ID_INVALID");
  return {
    ...shared, attestorId, keyId,
    attestorJwt: required(env, "SHADOW_DETACHED_ATTESTOR_JWT"),
    privateKeyFile: required(env, "SHADOW_DETACHED_PRIVATE_KEY_FILE"),
    handoffFile: required(env, "SHADOW_DETACHED_HANDOFF_FILE"),
    receiptFile: required(env, "SHADOW_DETACHED_RECEIPT_FILE"),
  };
}

export interface DetachedVerifierConfig {
  handoffFile: string;
  receiptFile: string;
  publicKeyFile: string;
  publicKeySha256: string;
  challengeNonce: string;
  attestorId: string;
  keyId: string;
}
export function parseDetachedVerifierConfig(env: NodeJS.ProcessEnv): DetachedVerifierConfig {
  requireGuard(env.PREFLIGHT_NONPROD_APPROVED === "YES", "DETACHED_SHADOW_OPT_IN_REQUIRED");
  requireGuard(env.SUPABASE_TEST_REF === SHADOW_REF &&
    env.SUPABASE_PRODUCTION_REF === PRODUCTION_REF,
    "DETACHED_PRODUCTION_OR_UNPINNED_PROJECT");
  const attestorId = required(env, "SHADOW_GATE_TRUSTED_ATTESTOR_ID");
  const keyId = required(env, "SHADOW_GATE_TRUSTED_KEY_ID");
  requireGuard(idPattern.test(attestorId) && idPattern.test(keyId),
    "DETACHED_VERIFIER_ID_INVALID");
  const publicKeySha256 = required(env, "SHADOW_GATE_TRUSTED_PUBLIC_PEM_SHA256");
  requireGuard(/^[0-9a-f]{64}$/.test(publicKeySha256), "DETACHED_PUBLIC_KEY_PIN_INVALID");
  const challengeNonce = required(env, "SHADOW_GATE_PINNED_CHALLENGE_NONCE");
  requireGuard(noncePattern.test(challengeNonce), "DETACHED_VERIFIER_NONCE_INVALID");
  return {
    attestorId, keyId, challengeNonce, publicKeySha256,
    publicKeyFile: required(env, "SHADOW_GATE_TRUSTED_PUBLIC_PEM_FILE"),
    handoffFile: required(env, "SHADOW_GATE_HANDOFF_FILE"),
    receiptFile: required(env, "SHADOW_GATE_RECEIPT_FILE"),
  };
}
