/** D29b private Cloud Run signer — pure request-policy checks.
 * IAM (roles/run.invoker, exact service) is the mandatory ingress auth layer.
 * This module cannot verify D2 immutable bytes and MUST NOT be exposed as a
 * public/self-service signing endpoint.
 */
import { createHash } from "node:crypto";

export const DOMAIN = "OROTITAN:D29B:D2_HANDOFF:V1\n";
export const KEY_VERSION =
  "projects/orotitan-d29b-signing/locations/europe-west1/keyRings/d29b-signing/cryptoKeys/receipt-sign/cryptoKeyVersions/1";
const UUID = /^[0-9a-f]{8}-[0-9a-f]{4}-[1-8][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i;
const B64URL = /^[A-Za-z0-9_-]+$/;
const SHA = /^[0-9a-f]{64}$/;
const FIELDS = [
  "purpose","verifiedBy","runId","subject","clientId","resource",
  "runStatus","stage","stageLifecycle","handoffGateState",
  "activeManifestKind","processVersion","pilotageVersion","dataCutoff",
  "runStateVersion","stageStateVersion","outcome","phase",
  "contractSetSha256","manifestSha256","stateFingerprintSha256",
  "verifiedArtifactsCount","prompt","promptSha256","issuedAt","expiresAt",
];
export class SignerPolicyError extends Error {
  constructor(code) {super(code);this.code=code;}
}
const reject=(code)=>{throw new SignerPolicyError(code);};

export function loadSignerPolicy(env) {
  const policy = {
    runId:env.D29B_RUN_ID,subject:env.D29B_SUBJECT_ID,
    clientId:env.D29B_OAUTH_CLIENT_ID,
    resource:env.D29B_RESOURCE,
    contractSetSha256:env.D29B_CONTRACT_SET_SHA256,
    keyId:env.D29B_KEY_ID,
    keyVersion:env.D29B_KMS_KEY_VERSION,
    publicKeyPem:env.D29B_PUBLIC_KEY_SPKI_PEM,
  };
  if(!UUID.test(policy.runId??"") ||
    !UUID.test(policy.subject??"") ||
    !/^[A-Za-z0-9_-]{6,160}$/.test(policy.clientId??"") ||
    policy.resource!=="https://orotitan-chatgpt-mcp-pilot.vercel.app/api/orotitan/mcp" ||
    !SHA.test(policy.contractSetSha256??"") ||
    !/^[A-Za-z0-9_-]{1,64}$/.test(policy.keyId??"") ||
    policy.keyVersion!==KEY_VERSION ||
    typeof policy.publicKeyPem!=="string" ||
    !policy.publicKeyPem.startsWith("-----BEGIN PUBLIC KEY-----") ||
    policy.publicKeyPem.length>1600)
    reject("SIGNER_CONFIG_NOT_PINNED");
  return Object.freeze(policy);
}
export function decodeSignablePayload(input,policy,nowSeconds) {
  if(!input || typeof input!=="object" || Array.isArray(input) ||
    Object.keys(input).length!==1 ||
    typeof input.payloadBase64Url!=="string") reject("REQUEST_SHAPE");
  const b64=input.payloadBase64Url;
  if(b64.length<100 || b64.length>24000 || !B64URL.test(b64))
    reject("REQUEST_SIZE_OR_ENCODING");
  const raw=Buffer.from(b64,"base64url");
  if(raw.length>16000 || raw.toString("base64url")!==b64)
    reject("REQUEST_ENCODING");
  let p;
  try {p=JSON.parse(raw.toString("utf8"));}
  catch {reject("REQUEST_JSON");}
  if(!p || typeof p!=="object" || Array.isArray(p) ||
    Object.keys(p).length!==FIELDS.length ||
    !FIELDS.every(k=>Object.hasOwn(p,k)) ||
    JSON.stringify(p)!==raw.toString("utf8")) reject("ATTESTATION_NOT_CANONICAL");
  if(p.purpose!=="OROTITAN_D29B_D2_HANDOFF_V1" ||
    p.verifiedBy!=="D2_IMMUTABLE_BYTES_AND_CONTRACTS_V3_1" ||
    p.runId!==policy.runId || p.subject!==policy.subject ||
    p.clientId!==policy.clientId || p.resource!==policy.resource ||
    p.contractSetSha256!==policy.contractSetSha256 ||
    p.runStatus!=="ACTIVE" || p.processVersion!=="3.1" ||
    p.pilotageVersion!=="3.1" ||
    !["RESEARCH","DEEP_DIVE","INTEGRATION"].includes(p.stage) ||
    !["RESEARCH","FUNDAMENTALS","VALUATION","CERTIFICATION_RECONCILIATION","INTEGRATION"].includes(p.phase) ||
    !["CHECKPOINT","FINAL"].includes(p.activeManifestKind) ||
    !["CONTINUE","HANDOFF"].includes(p.outcome) ||
    !["IN_PROGRESS","PAUSED","COMPLETE"].includes(p.stageLifecycle) ||
    !["YES","NO","NOT_EVALUATED"].includes(p.handoffGateState) ||
    !SHA.test(p.manifestSha256??"") ||
    !SHA.test(p.stateFingerprintSha256??"") ||
    !SHA.test(p.promptSha256??"") ||
    typeof p.dataCutoff!=="string" || !/^\d{4}-\d{2}-\d{2}$/.test(p.dataCutoff) ||
    !Number.isSafeInteger(p.runStateVersion) || p.runStateVersion<1 ||
    !Number.isSafeInteger(p.stageStateVersion) || p.stageStateVersion<1 ||
    !Number.isSafeInteger(p.verifiedArtifactsCount) ||
    p.verifiedArtifactsCount<1 || p.verifiedArtifactsCount>3000)
    reject("ATTESTATION_SCOPE_INVALID");
  if(p.outcome==="HANDOFF" &&
    (p.stageLifecycle!=="COMPLETE" || p.handoffGateState!=="YES" ||
     p.activeManifestKind!=="FINAL")) reject("HANDOFF_NOT_ADMITTED");
  if(p.outcome==="CONTINUE" &&
    !["IN_PROGRESS","PAUSED"].includes(p.stageLifecycle))
    reject("CONTINUE_NOT_ADMITTED");
  if(typeof p.prompt!=="string" || p.prompt.length<50 || p.prompt.length>12000 ||
    !p.prompt.includes("RUN_ID = "+p.runId) ||
    !p.prompt.includes("PUBLICATION_AUTHORIZED = NO") ||
    !p.prompt.includes("MANUAL_EXECUTION = YES") ||
    createHash("sha256").update(p.prompt,"utf8").digest("hex")!==p.promptSha256)
    reject("PROMPT_HASH_OR_BOUNDARY_INVALID");
  if(!Number.isSafeInteger(nowSeconds) ||
    !Number.isSafeInteger(p.issuedAt) || !Number.isSafeInteger(p.expiresAt) ||
    p.issuedAt>nowSeconds+5 || p.issuedAt<nowSeconds-60 ||
    p.expiresAt<=nowSeconds || p.expiresAt-p.issuedAt>120 ||
    p.expiresAt<=p.issuedAt)
    reject("ATTESTATION_EXPIRED_OR_UNBOUNDED");

  // The signing service trusts only a separately authenticated D2 issuer's
  // claim of verification; it cannot itself certify the private artifact bytes.
  return Buffer.from(DOMAIN+b64,"utf8");
}
