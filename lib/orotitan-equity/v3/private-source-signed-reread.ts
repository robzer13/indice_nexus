import {
  createPrivateKey, createPublicKey, sign, verify, randomUUID,
  KeyObject, type KeyLike,
} from "node:crypto";
import {
  rereadPrivateCandidate,
  PrivateSourceError,
  type PreAdmissionLocator,
  type SourceStorageClient,
} from "./private-source-bytes-preflight";

/**
 * Cryptographic transport receipt. NOT a Registry PERSISTENCE_ATTESTED event.
 * Run only inside the separately credentialed reread service; never in
 * browser code or the uploader's execution boundary.
 */
export interface SignedRereadStatement {
  schema: "OROTITAN_SIGNED_PRIVATE_REREAD_V1_UNATTESTED";
  attestorId: string;
  keyId: string;
  challengeNonce: string;
  signedAtUtc: string;
  runId: string;
  artifactId: string;
  version: number;
  bucket: string;
  objectPath: string;
  storageUri: string;
  sizeBytes: number;
  sha256: string;
  mediaType: string;
  sourceUrl: string;
  publishedOn: string;
  dataCutoff: string;
  verification: "PRIVATE_STORAGE_EXACT_BYTES_REREAD";
  authority: "TRANSPORT_ONLY_NOT_REGISTRY_ADMITTED";
}
export interface SignedRereadEnvelope {
  statement: SignedRereadStatement;
  signatureBase64Url: string;
}
export interface AttestorContext {
  attestorId: string;
  keyId: string;
  privateKey: KeyLike;
  challengeNonce: string;
  verifiedAt?: Date;
}
export interface GateTrustPolicy {
  trustedAttestorId: string;
  trustedKeyId: string;
  trustedPublicKey: KeyLike;
  challengeNonce: string;
  nowUtc: Date;
  maxAgeMs?: number;
}
function fail(ok: unknown, code: string): asserts ok {
  if (!ok) throw new PrivateSourceError(code);
}
const UUID = /^[0-9a-f]{8}-[0-9a-f]{4}-[1-8][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i;
const SHA256 = /^[0-9a-f]{64}$/;
const IDENT = /^[a-z][a-z0-9-]{2,63}$/;
const fields = [
  "schema", "attestorId", "keyId", "challengeNonce", "signedAtUtc",
  "runId", "artifactId", "version", "bucket", "objectPath", "storageUri",
  "sizeBytes", "sha256", "mediaType", "sourceUrl", "publishedOn",
  "dataCutoff", "verification", "authority",
] as const;
/** Length-prefixed canonical values avoid property ordering dependence. */
function wire(statement: SignedRereadStatement): Buffer {
  fail(Object.keys(statement).length === fields.length &&
    fields.every(k => Object.prototype.hasOwnProperty.call(statement, k)),
    "SIGNED_RECEIPT_SHAPE_INVALID");
  const values = fields.map(k => {
    const v = statement[k];
    fail(typeof v === "string" || (typeof v === "number" && Number.isSafeInteger(v)),
      "SIGNED_RECEIPT_VALUE_INVALID");
    return String(v);
  });
  return Buffer.from(values.map(v => String(Buffer.byteLength(v)) + ":" + v).join("|"), "utf8");
}
function validateAttestorId(attestorId: string, keyId: string, nonce: string): void {
  fail(IDENT.test(attestorId) && IDENT.test(keyId), "UNTRUSTED_ATTESTOR_ID_FORMAT");
  fail(UUID.test(nonce), "CHALLENGE_NONCE_INVALID");
}
function asPrivateKey(key: KeyLike): KeyObject {
  const obj = key instanceof KeyObject ? key : createPrivateKey(key);
  fail(obj.type === "private" && obj.asymmetricKeyType === "ed25519", "ATTESTOR_KEY_MUST_BE_ED25519");
  return obj;
}
function asPublicKey(key: KeyLike): KeyObject {
  const obj = key instanceof KeyObject ? key : createPublicKey(key);
  fail(obj.type === "public" && obj.asymmetricKeyType === "ed25519", "VERIFIER_KEY_MUST_BE_ED25519");
  return obj;
}
/**
 * Binds the signature to a *real* storage reread performed in this call.
 * It cannot prove that the holder of the private signing key is an independent
 * human/role/credential. That is an external deployment and audit control.
 */
export async function signRereadFromPrivateStorage(
  reader: SourceStorageClient,
  locator: PreAdmissionLocator,
  ctx: AttestorContext,
): Promise<SignedRereadEnvelope> {
  validateAttestorId(ctx.attestorId, ctx.keyId, ctx.challengeNonce);
  const time = ctx.verifiedAt ?? new Date();
  fail(time instanceof Date && Number.isFinite(time.getTime()), "ATTESTATION_TIME_INVALID");
  const reread = await rereadPrivateCandidate(reader, locator);
  fail(reread.bytesReread === "EXACT_MATCH" && reread.registryAdmitted === false,
    "BYTES_REREAD_NOT_VERIFIED");
  const statement: SignedRereadStatement = {
    schema: "OROTITAN_SIGNED_PRIVATE_REREAD_V1_UNATTESTED",
    attestorId: ctx.attestorId, keyId: ctx.keyId,
    challengeNonce: ctx.challengeNonce, signedAtUtc: time.toISOString(),
    runId: reread.runId, artifactId: reread.artifactId,
    version: reread.version, bucket: reread.bucket,
    objectPath: reread.path, storageUri: reread.uri,
    sizeBytes: reread.sizeBytes, sha256: reread.sha256,
    mediaType: reread.mediaType, sourceUrl: reread.sourceUrl,
    publishedOn: reread.publishedOn, dataCutoff: reread.dataCutoff,
    verification: "PRIVATE_STORAGE_EXACT_BYTES_REREAD",
    authority: "TRANSPORT_ONLY_NOT_REGISTRY_ADMITTED",
  };
  const signature = sign(null, wire(statement), asPrivateKey(ctx.privateKey));
  return { statement, signatureBase64Url: signature.toString("base64url") };
}
export interface CheckedSignedReread {
  signature: "VALID_PINNED_ED25519";
  bytesRereadClaim: "SIGNED_EXACT_BYTES";
  registryAttestation: false;
  registryAdmission: false;
  independentPrincipalProven: false;
  publicationDateIndependentlyVerified: false;
  historicalCutoffBytesProven: false;
}
/** A trust policy must be installed OUTSIDE the received signed document. */
export function verifySignedReread(
  envelope: SignedRereadEnvelope,
  locator: PreAdmissionLocator,
  policy: GateTrustPolicy,
): CheckedSignedReread {
  fail(!!envelope && !!envelope.statement &&
    typeof envelope.signatureBase64Url === "string" &&
    /^[A-Za-z0-9_-]{86}$/.test(envelope.signatureBase64Url),
    "SIGNED_RECEIPT_SIGNATURE_INVALID");
  const s = envelope.statement;
  validateAttestorId(policy.trustedAttestorId, policy.trustedKeyId, policy.challengeNonce);
  validateAttestorId(s.attestorId, s.keyId, s.challengeNonce);
  fail(s.schema === "OROTITAN_SIGNED_PRIVATE_REREAD_V1_UNATTESTED" &&
    s.verification === "PRIVATE_STORAGE_EXACT_BYTES_REREAD" &&
    s.authority === "TRANSPORT_ONLY_NOT_REGISTRY_ADMITTED",
    "SIGNED_RECEIPT_AUTHORITY_INVALID");
  fail(s.attestorId === policy.trustedAttestorId &&
    s.keyId === policy.trustedKeyId &&
    s.challengeNonce === policy.challengeNonce, "SIGNED_RECEIPT_TRUST_BINDING_MISMATCH");
  fail(s.runId === locator.runId && s.artifactId === locator.artifactId &&
    s.version === locator.version && s.bucket === locator.bucket &&
    s.objectPath === locator.path && s.storageUri === locator.uri &&
    s.sizeBytes === locator.sizeBytes && s.sha256 === locator.sha256 &&
    s.mediaType === locator.mediaType && s.sourceUrl === locator.sourceUrl &&
    s.publishedOn === locator.publishedOn && s.dataCutoff === locator.dataCutoff,
    "SIGNED_RECEIPT_LOCATOR_MISMATCH");
  fail(SHA256.test(s.sha256) && Number.isSafeInteger(s.sizeBytes) && s.sizeBytes > 0,
    "SIGNED_RECEIPT_BYTES_INVALID");
  fail(policy.nowUtc instanceof Date && Number.isFinite(policy.nowUtc.getTime()), "VERIFIER_TIME_INVALID");
  const at = new Date(s.signedAtUtc);
  fail(!Number.isNaN(at.getTime()) && at.toISOString() === s.signedAtUtc,
    "SIGNED_RECEIPT_TIME_INVALID");
  const age = policy.nowUtc.getTime() - at.getTime();
  fail(Number.isSafeInteger(policy.maxAgeMs ?? 600000) &&
    (policy.maxAgeMs ?? 600000) > 0 && (policy.maxAgeMs ?? 600000) <= 3600000 &&
    age >= -30000 && age <= (policy.maxAgeMs ?? 600000),
    "SIGNED_RECEIPT_EXPIRED_OR_FUTURE");
  const signature = Buffer.from(envelope.signatureBase64Url, "base64url");
  fail(signature.length === 64 &&
    verify(null, wire(s), asPublicKey(policy.trustedPublicKey), signature),
    "SIGNED_RECEIPT_SIGNATURE_MISMATCH");
  return {
    signature: "VALID_PINNED_ED25519",
    bytesRereadClaim: "SIGNED_EXACT_BYTES",
    registryAttestation: false, registryAdmission: false,
    independentPrincipalProven: false,
    publicationDateIndependentlyVerified: false,
    historicalCutoffBytesProven: false,
  };
}
export function newRereadChallenge(): string { return randomUUID(); }
