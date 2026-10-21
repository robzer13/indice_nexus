/** D29b: stateless, RLS-backed receipt verification orchestration.
 * Never fetches artifacts or uses Supabase service_role; caller's JWT and
 * public verifier keys are obtained by a separately authenticated endpoint.
 * This module alone grants no prompt authority without a matching D2 receipt.
 */
import {
  verifyDelegatedD2Handoff,
  type CallerBoundState, type HandoffVerifyKey, type HandoffGateResult,
  type SignedHandoffEnvelope,
} from "./mcp-d29b-signed-handoff-gate";

export type ReceiptObservation={
  receiptId:string;
  keyId:string;
  payloadBase64Url:string;
  signatureBase64Url:string;
  expiresAt:string;
};

export type DelegatedReceiptPort={
  observeState:()=>Promise<CallerBoundState>;
  findReceipt:()=>Promise<ReceiptObservation|null>;
};

function reject(reasonCode:string):HandoffGateResult {
  return {
    status:"SIGNED_D2_HANDOFF_DENIED",reasonCode,prompt:null,
    trust:"REGISTRY_OBSERVATION_NOT_ARTIFACT_VERIFICATION",
    nextStepAuthorized:false,publicationAuthorized:false,mutationPerformed:false,
  };
}

function sameState(a:CallerBoundState,b:CallerBoundState):boolean {
  return a.subject===b.subject && a.clientId===b.clientId &&
    a.resource===b.resource && a.runId===b.runId &&
    a.runStatus===b.runStatus && a.stage===b.stage &&
    a.stageLifecycle===b.stageLifecycle &&
    a.handoffGateState===b.handoffGateState &&
    a.activeManifestKind===b.activeManifestKind &&
    a.contractStatusCode===b.contractStatusCode &&
    a.observedBlockerCount===b.observedBlockerCount &&
    a.processVersion===b.processVersion &&
    a.pilotageVersion===b.pilotageVersion &&
    a.dataCutoff===b.dataCutoff &&
    a.runStateVersion===b.runStateVersion &&
    a.stageStateVersion===b.stageStateVersion;
}
function sameReceipt(a:ReceiptObservation,b:ReceiptObservation):boolean {
  return a.receiptId===b.receiptId && a.keyId===b.keyId &&
    a.payloadBase64Url===b.payloadBase64Url &&
    a.signatureBase64Url===b.signatureBase64Url &&
    a.expiresAt===b.expiresAt;
}

/** Reads current Registry state twice and checks receipt presence twice.
 * Revocations between reads fail closed. This is optimistic CAS, not an
 * atomic cross-API lock: the caller still must revalidate before execution.
 */
export async function inspectD29bReceipt(
  port:DelegatedReceiptPort, keys:readonly HandoffVerifyKey[],
  nowSeconds:()=>number,
):Promise<HandoffGateResult> {
  try {
    const first=await port.observeState();
    const receipt=await port.findReceipt();
    if(!receipt) return reject("D2_RECEIPT_ABSENT_OR_REVOKED");
    if(!/^[0-9a-f-]{36}$/i.test(receipt.receiptId) ||
      typeof receipt.expiresAt!=="string" ||
      !Number.isFinite(Date.parse(receipt.expiresAt)) ||
      Date.parse(receipt.expiresAt)<=nowSeconds()*1000) {
      return reject("D2_RECEIPT_EXPIRED_OR_MALFORMED");
    }
    const last=await port.observeState();
    if(!sameState(first,last)) return reject("REGISTRY_CHANGED_DURING_READ");
    const envelope:SignedHandoffEnvelope={
      version:1,keyId:receipt.keyId,
      payloadBase64Url:receipt.payloadBase64Url,
      signatureBase64Url:receipt.signatureBase64Url,
    };
    const result=verifyDelegatedD2Handoff(envelope,last,keys,nowSeconds());
    if(result.status!=="SIGNED_D2_HANDOFF_VERIFIED") return result;
    const stillPresent=await port.findReceipt();
    if(!stillPresent || !sameReceipt(receipt,stillPresent)) {
      return reject("D2_RECEIPT_REVOKED_OR_ROTATED");
    }
    if(Date.parse(stillPresent.expiresAt)<=nowSeconds()*1000) {
      return reject("D2_RECEIPT_EXPIRED");
    }
    return result;
  } catch {
    // Database errors or unavailable issuer do not unlock a fallback prompt.
    return reject("D29B_VERIFICATION_UNAVAILABLE");
  }
}

/** Explicit public verifier keys, pinned server-side. No URL or keys from
 * the signed receipt, OAuth input, or user tool arguments are ever accepted.
 */
export function parseD29bVerifierKeys(raw:string|undefined):readonly HandoffVerifyKey[] {
  if(!raw || raw.length>9000) return [];
  try {
    const data:unknown=JSON.parse(raw);
    if(!Array.isArray(data) || data.length<1 || data.length>3 ||
      data.some((x:unknown)=>!x || typeof x!=="object" ||
        Array.isArray(x) || Object.keys(x).length!==2 ||
        !("keyId" in x) || !("publicKeySpkiPem" in x) ||
        typeof x.keyId!=="string" || typeof x.publicKeySpkiPem!=="string" ||
        !/^[A-Za-z0-9_-]{1,64}$/.test(x.keyId) ||
        x.publicKeySpkiPem.length>1800 ||
        !x.publicKeySpkiPem.startsWith("-----BEGIN PUBLIC KEY-----")))return [];
    const arr=data as HandoffVerifyKey[];
    if(new Set(arr.map(x=>x.keyId)).size!==arr.length)return [];
    return arr;
  }catch{return [];}
}
