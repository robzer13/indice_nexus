/**
 * D29b REVIEW-ONLY: offline-verifiable D2 attestation gate.
 *
 * A server-side OAuth resource must NEVER use this signature check as a
 * replacement for immutable byte verification by the trusted D2 issuer.
 * Only a separately approved, independently authenticated D2 issuer may sign.
 * No private signing key, artifact store token, or service_role belongs here.
 *
 * Not connected to the live MCP route pending the scoped receipt-store review.
 */
import { createHash, createPublicKey, verify } from "node:crypto";
import pins from "../../../contracts/orotitan-equity/v3/contract-pin-pack-v3/OROTITAN_CONTRACT_PIN_PACK_V3_0_2.json";

export const HANDOFF_PURPOSE = "OROTITAN_D29B_D2_HANDOFF_V1";
export const HANDOFF_SIGNATURE_DOMAIN = "OROTITAN:D29B:D2_HANDOFF:V1\n";
export const MAX_SIGNED_AGE_SECONDS = 300;
export type HandoffStage = "RESEARCH" | "DEEP_DIVE" | "INTEGRATION";
export type HandoffPhase = "RESEARCH" | "FUNDAMENTALS" | "VALUATION" |
  "CERTIFICATION_RECONCILIATION" | "INTEGRATION";

export type CallerBoundState = {
  subject: string; // from already verified, resource-bound Supabase OAuth JWT
  clientId: string; // from the same verified JWT, not a tool argument
  resource: string; // pinned MCP resource
  runId: string; // re-observed with caller's JWT under Besi-only RLS
  runStatus: string;
  stage: string | null;
  stageLifecycle: string;
  handoffGateState: string;
  activeManifestKind: string | null;
  contractStatusCode: string | null;
  observedBlockerCount: number;
  processVersion: string;
  pilotageVersion: string;
  dataCutoff: string;
  runStateVersion: number;
  stageStateVersion: number | null;
};

export type SignedHandoffEnvelope = {
  version: 1;
  keyId: string;
  payloadBase64Url: string;
  signatureBase64Url: string;
};

export type HandoffAttestationV1 = {
  purpose: typeof HANDOFF_PURPOSE;
  verifiedBy: "D2_IMMUTABLE_BYTES_AND_CONTRACTS_V3_1";
  runId: string;
  subject: string;
  clientId: string;
  resource: string;
  runStatus: "ACTIVE";
  stage: HandoffStage;
  stageLifecycle: "IN_PROGRESS" | "PAUSED" | "COMPLETE";
  handoffGateState: "YES" | "NO" | "NOT_EVALUATED";
  activeManifestKind: "CHECKPOINT" | "FINAL";
  processVersion: "3.1";
  pilotageVersion: "3.1";
  dataCutoff: string;
  runStateVersion: number;
  stageStateVersion: number;
  outcome: "CONTINUE" | "HANDOFF";
  phase: HandoffPhase;
  contractSetSha256: string;
  manifestSha256: string;
  stateFingerprintSha256: string;
  verifiedArtifactsCount: number;
  prompt: string;
  promptSha256: string;
  issuedAt: number; // Unix seconds
  expiresAt: number; // Unix seconds
};

export type HandoffGateResult = {
  status: "SIGNED_D2_HANDOFF_VERIFIED" | "SIGNED_D2_HANDOFF_DENIED";
  reasonCode: string;
  prompt: string | null;
  trust: "SIGNED_D2_ATTESTATION_WITH_LIVE_REGISTRY_CAS" |
    "REGISTRY_OBSERVATION_NOT_ARTIFACT_VERIFICATION";
  nextStepAuthorized: boolean;
  publicationAuthorized: false;
  mutationPerformed: false;
};

/** Explicit key-ring from trusted server configuration, never the envelope. */
export type HandoffVerifyKey = { keyId: string; publicKeySpkiPem: string };

const SHA256 = /^[0-9a-f]{64}$/;
const B64URL = /^[A-Za-z0-9_-]+$/;
const UUID = /^[0-9a-f]{8}-[0-9a-f]{4}-[1-8][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i;
const HEX_SHA = (v:unknown):v is string => typeof v==="string" && SHA256.test(v);
const DENIED:Omit<HandoffGateResult,"reasonCode"> = {
  status:"SIGNED_D2_HANDOFF_DENIED",
  prompt:null,
  trust:"REGISTRY_OBSERVATION_NOT_ARTIFACT_VERIFICATION",
  nextStepAuthorized:false,
  publicationAuthorized:false,
  mutationPerformed:false,
};
function denied(reasonCode:string):HandoffGateResult {
  return {...DENIED,reasonCode};
}
function base64UrlDecodeCanonical(value:string,maxLength:number):Buffer|null {
  if(value.length===0 || value.length>maxLength || !B64URL.test(value)) return null;
  const decoded=Buffer.from(value,"base64url");
  return decoded.toString("base64url")===value ? decoded : null;
}
function safeObject(value:unknown):value is Record<string,unknown> {
  return value!==null && typeof value==="object" && !Array.isArray(value);
}
function exactShape(value:Record<string,unknown>):boolean {
  const fields=[
    "purpose","verifiedBy","runId","subject","clientId","resource",
    "runStatus","stage","stageLifecycle","handoffGateState",
    "activeManifestKind","processVersion","pilotageVersion","dataCutoff",
    "runStateVersion","stageStateVersion","outcome","phase","contractSetSha256",
    "manifestSha256","stateFingerprintSha256","verifiedArtifactsCount",
    "prompt","promptSha256","issuedAt","expiresAt",
  ];
  return Object.keys(value).length===fields.length && fields.every(x=>Object.hasOwn(value,x));
}
const stringIs=(v:unknown):v is string=>typeof v==="string";

export function verifyDelegatedD2Handoff(
  receipt:unknown,
  live:CallerBoundState,
  keyRing:readonly HandoffVerifyKey[],
  nowSeconds:number,
):HandoffGateResult {
  // All denials are coarse; never echo private proof bytes or exceptions.
  if(!safeObject(receipt) || Object.keys(receipt).length!==4 ||
    receipt.version!==1 || !stringIs(receipt.keyId) ||
    !stringIs(receipt.payloadBase64Url) ||
    !stringIs(receipt.signatureBase64Url) ||
    !["version","keyId","payloadBase64Url","signatureBase64Url"].every(
      k=>Object.hasOwn(receipt,k))) return denied("RECEIPT_SHAPE_INVALID");
  if(!Number.isSafeInteger(nowSeconds) || keyRing.length===0 || keyRing.length>3)
    return denied("VERIFIER_CONFIGURATION_INVALID");
  if(keyRing.some(k=>!k.keyId || k.keyId.length>64 ||
    !/^[A-Za-z0-9_-]+$/.test(k.keyId)) ||
    new Set(keyRing.map(k=>k.keyId)).size!==keyRing.length)
    return denied("VERIFIER_KEY_RING_INVALID");
  const key=keyRing.find(k=>k.keyId===receipt.keyId);
  if(!key) return denied("SIGNER_NOT_TRUSTED");
  const raw=base64UrlDecodeCanonical(receipt.payloadBase64Url,24000);
  const signature=base64UrlDecodeCanonical(receipt.signatureBase64Url,100);
  if(!raw || !signature || raw.length>16000 || signature.length!==64)
    return denied("RECEIPT_BYTES_INVALID");
  try {
    const publicKey=createPublicKey(key.publicKeySpkiPem);
    if(publicKey.asymmetricKeyType!=="ed25519" || !verify(null,
      Buffer.from(HANDOFF_SIGNATURE_DOMAIN+receipt.payloadBase64Url,"utf8"),
      publicKey,signature)) return denied("SIGNATURE_INVALID");
  } catch { return denied("SIGNATURE_INVALID"); }

  let parsed:unknown;
  try { parsed=JSON.parse(raw.toString("utf8")); }
  catch { return denied("PAYLOAD_INVALID"); }
  if(!safeObject(parsed) || !exactShape(parsed)) return denied("PAYLOAD_SHAPE_INVALID");
  const p=parsed as HandoffAttestationV1;
  if(p.purpose!==HANDOFF_PURPOSE ||
    p.verifiedBy!=="D2_IMMUTABLE_BYTES_AND_CONTRACTS_V3_1" ||
    p.processVersion!=="3.1" || p.pilotageVersion!=="3.1" ||
    p.contractSetSha256!==pins.contract_set_sha256 ||
    !HEX_SHA(p.manifestSha256) || !HEX_SHA(p.stateFingerprintSha256) ||
    !HEX_SHA(p.promptSha256) ||
    !stringIs(p.prompt) || p.prompt.length<50 || p.prompt.length>12000 ||
    createHash("sha256").update(p.prompt,"utf8").digest("hex")!==p.promptSha256 ||
    !p.prompt.includes("PUBLICATION_AUTHORIZED = NO") ||
    !p.prompt.includes("MANUAL_EXECUTION = YES") ||
    !p.prompt.includes("RUN_ID = "+p.runId) ||
    !UUID.test(p.runId) || !UUID.test(p.subject) ||
    !stringIs(p.clientId) || p.clientId.length<6 ||
    !stringIs(p.resource) ||
    !/^\d{4}-\d{2}-\d{2}$/.test(p.dataCutoff) ||
    !Number.isSafeInteger(p.runStateVersion) || p.runStateVersion<1 ||
    !Number.isSafeInteger(p.stageStateVersion) || p.stageStateVersion<1 ||
    !Number.isSafeInteger(p.verifiedArtifactsCount) ||
    p.verifiedArtifactsCount<1 || p.verifiedArtifactsCount>3000 ||
    !Number.isSafeInteger(p.issuedAt) || !Number.isSafeInteger(p.expiresAt) ||
    p.issuedAt>nowSeconds+5 || nowSeconds>=p.expiresAt ||
    nowSeconds-p.issuedAt>MAX_SIGNED_AGE_SECONDS ||
    p.expiresAt-p.issuedAt>MAX_SIGNED_AGE_SECONDS ||
    p.expiresAt<=p.issuedAt)
    return denied("D2_ATTESTATION_NOT_ADMITTED");

  if(!["RESEARCH","DEEP_DIVE","INTEGRATION"].includes(p.stage) ||
     !["RESEARCH","FUNDAMENTALS","VALUATION","CERTIFICATION_RECONCILIATION","INTEGRATION"].includes(p.phase) ||
     !["HANDOFF","CONTINUE"].includes(p.outcome) ||
     p.runStatus!=="ACTIVE")
    return denied("OUTCOME_NOT_ADMITTED");
  if(p.subject!==live.subject || p.clientId!==live.clientId ||
    p.resource!==live.resource || p.runId!==live.runId)
    return denied("OAUTH_OR_RUN_SCOPE_MISMATCH");
  if(live.runStatus!=="ACTIVE" || live.processVersion!=="3.1" ||
    live.pilotageVersion!=="3.1" ||
    live.stage!==p.stage || live.dataCutoff!==p.dataCutoff ||
    live.runStateVersion!==p.runStateVersion ||
    live.stageStateVersion!==p.stageStateVersion ||
    live.stageLifecycle!==p.stageLifecycle ||
    live.handoffGateState!==p.handoffGateState ||
    live.activeManifestKind!==p.activeManifestKind ||
    !Number.isSafeInteger(live.observedBlockerCount) ||
    live.observedBlockerCount>0 ||
    live.contractStatusCode?.startsWith("BLOCKED_")===true)
    return denied("REGISTRY_STATE_DRIFT_OR_BLOCKED");
  if(p.outcome==="HANDOFF" &&
    (p.stageLifecycle!=="COMPLETE" ||
     p.handoffGateState!=="YES" || p.activeManifestKind!=="FINAL"))
    return denied("HANDOFF_GATE_NOT_ADMITTED");
  if(p.outcome==="CONTINUE" &&
    !["IN_PROGRESS","PAUSED"].includes(p.stageLifecycle))
    return denied("CONTINUE_STATE_NOT_ADMITTED");

  return {
    status:"SIGNED_D2_HANDOFF_VERIFIED",
    reasonCode:"SIGNED_D2_ATTESTATION_AND_LIVE_CAS_MATCH",
    prompt:p.prompt,
    trust:"SIGNED_D2_ATTESTATION_WITH_LIVE_REGISTRY_CAS",
    nextStepAuthorized:true,
    publicationAuthorized:false,
    mutationPerformed:false,
  };
}
