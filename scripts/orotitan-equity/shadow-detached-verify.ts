/**
 * Offline trust gate, separately operated from the Attestor. It reads only a
 * trusted locally pinned public key, the gate-pinned challenge and the receipt.
 * NO Supabase token, private signing key or Registry credentials required.
 */
import { readFileSync } from "node:fs";
import { createHash } from "node:crypto";
import { resolve } from "node:path";
import { fileURLToPath } from "node:url";
import {
  parseDetachedHandoff, parseDetachedVerifierConfig,
} from "../../lib/orotitan-equity/v3/shadow-detached-attestor-boundary";
import {
  verifySignedReread, type SignedRereadEnvelope,
} from "../../lib/orotitan-equity/v3/private-source-signed-reread";

function assertThat(value: unknown, code: string): asserts value {
  if (!value) throw new Error(code);
}
function asObject(value: unknown): Record<string, unknown> {
  assertThat(!!value && typeof value === "object" && !Array.isArray(value),
    "DETACHED_RECEIPT_INVALID");
  return value as Record<string, unknown>;
}
export function runDetachedVerifier(env: NodeJS.ProcessEnv): void {
  const c = parseDetachedVerifierConfig(env);
  assertThat(resolve(c.handoffFile) !== resolve(c.receiptFile) &&
    resolve(c.publicKeyFile) !== resolve(c.receiptFile) &&
    resolve(c.publicKeyFile) !== resolve(c.handoffFile),
    "DETACHED_VERIFIER_FILE_BOUNDARY_INVALID");
  const handoff = parseDetachedHandoff(JSON.parse(readFileSync(c.handoffFile, "utf8")) as unknown);
  assertThat(handoff.challengeNonce === c.challengeNonce,
    "DETACHED_GATE_NONCE_NOT_PINNED");
  const result = asObject(JSON.parse(readFileSync(c.receiptFile, "utf8")) as unknown);
  assertThat(result.schema === "OROTITAN_SHADOW_DETACHED_RECEIPT_V1_NONAUTHORITATIVE" &&
    result.transport === "EXACT_BYTES_REREAD" &&
    result.sourcePersistenceCertified === false &&
    result.independentExecutionAndCustodyCertified === false &&
    result.registryAdmitted === false &&
    Object.keys(result).length === 6,
    "DETACHED_RECEIPT_AUTHORITY_ESCALATION_DENIED");
  // Import the public key from separate local configuration, never from the
  // untrusted receipt or source handoff.
  const publicKey = readFileSync(c.publicKeyFile, "utf8");
  const keyFingerprint = createHash("sha256").update(Buffer.from(publicKey, "utf8")).digest("hex");
  assertThat(keyFingerprint === c.publicKeySha256, "DETACHED_PUBLIC_KEY_FINGERPRINT_MISMATCH");
  const checked = verifySignedReread(result.envelope as SignedRereadEnvelope, handoff.locator, {
    trustedAttestorId: c.attestorId, trustedKeyId: c.keyId,
    trustedPublicKey: publicKey, challengeNonce: c.challengeNonce,
    nowUtc: new Date(), maxAgeMs: 600000,
  });
  assertThat(checked.signature === "VALID_PINNED_ED25519" &&
    checked.registryAdmission === false && checked.registryAttestation === false &&
    checked.independentPrincipalProven === false,
    "DETACHED_REGISTRY_ESCALATION_DENIED");
  console.log(JSON.stringify({
    schema: "OROTITAN_SHADOW_DETACHED_GATE_V1_NONAUTHORITATIVE",
    signature: checked.signature,
    exactByteRereadClaim: checked.bytesRereadClaim,
    independentExecutionAndCustodyCertified: false,
    immutableStorageCertified: false,
    historicCutoffBytesCertified: false,
    registryAdmitted: false,
  }, null, 2));
}
if (process.argv[1] && resolve(process.argv[1]) === resolve(fileURLToPath(import.meta.url))) {
  try { runDetachedVerifier(process.env); }
  catch (error: unknown) {
    console.error("DETACHED_GATE_FAILED: " +
      (error instanceof Error ? error.message : "UNKNOWN_ERROR"));
    process.exitCode = 1;
  }
}
