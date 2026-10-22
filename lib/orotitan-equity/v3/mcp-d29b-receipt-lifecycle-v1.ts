/**
 * D29b trusted-receipt lifecycle — REVIEW ONLY.
 * No API route, no database writes by this module, no service_role fallback.
 * A separate privileged issuer MUST produce the receipt via immutable D2.
 * The injected writer MUST hold a transaction-scoped Registry CAS lock,
 * enforce explicit subject/client/resource/run allowlists and append an audit.
 */
import {
  verifyDelegatedD2Handoff,
  type CallerBoundState,
  type HandoffVerifyKey,
  type SignedHandoffEnvelope,
} from "./mcp-d29b-signed-handoff-gate";

export type D29bIssuedReceipt={
  receipt:SignedHandoffEnvelope;
  runId:string;subject:string;clientId:string;resource:string;
  issuedAt:number;expiresAt:number;
};
export type D29bReceiptForWrite={
  receipt:D29bIssuedReceipt;
  expectedRunStateVersion:number;
  expectedStageStateVersion:number;
  expectedStage:string;
  // Must be transactionally compared to current authenticated Registry state.
  expectedState:CallerBoundState;
};

export type D29bWriteResult=
  | {status:"STORED";receiptId:string}
  | {status:"STALE";receiptId:null}
  | {status:"DENIED";receiptId:null};

export type D29bAdminWritePort={
  /** Privileged, fresh Registry observation; NEVER derived from an OAuth tool argument. */
  observeCurrentState:()=>Promise<CallerBoundState>;
  /** REQUIRED: atomic row lock + CAS + replace + append-only audit transaction.
   * No best-effort, eventually consistent, in-memory or two-step implementation.
   */
  putReceiptAtomically:(input:D29bReceiptForWrite)=>Promise<D29bWriteResult>;
  /** Authenticated read-after-write. Must return the exact persisted envelope. */
  getStoredReceipt:(receiptId:string)=>Promise<{
    receiptId:string;keyId:string;payloadBase64Url:string;
    signatureBase64Url:string;expiresAt:string;revokedAt:string|null;
  }|null>;
  /** Transactionally revoke the current receipt and append immutable audit. */
  revokeReceiptAtomically:(runId:string,receiptId:string,reasonCode:string)=>
    Promise<"REVOKED"|"NOT_FOUND">;
};

export class D29bReceiptPersistenceError extends Error {
  constructor(readonly code:string){super(code);}
}
function deny(code:string):never {throw new D29bReceiptPersistenceError(code);}
function exactReceipt(a:SignedHandoffEnvelope,b:{
  keyId:string;payloadBase64Url:string;signatureBase64Url:string;
}):boolean {
  return a.version===1 && a.keyId===b.keyId &&
    a.payloadBase64Url===b.payloadBase64Url &&
    a.signatureBase64Url===b.signatureBase64Url;
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
    a.processVersion===b.processVersion && a.pilotageVersion===b.pilotageVersion &&
    a.dataCutoff===b.dataCutoff &&
    a.runStateVersion===b.runStateVersion &&
    a.stageStateVersion===b.stageStateVersion;
}

/**
 * The production caller must call the independently admin-guarded D2 issuer
 * BEFORE this function. This function rejects even valid-looking unsigned
 * and self-signed payloads unless their key is explicitly in the verifier ring.
 * Failures never return a prompt and do not transition any OroTitan stage.
 */
export async function persistVerifiedD29bReceipt(
  issued:D29bIssuedReceipt,
  keys:readonly HandoffVerifyKey[],
  port:D29bAdminWritePort,
  nowSeconds:()=>number,
):Promise<{status:"STORED";receiptId:string}> {
  const live=await port.observeCurrentState();
  if(live.subject!==issued.subject || live.clientId!==issued.clientId ||
    live.resource!==issued.resource || live.runId!==issued.runId ||
    !Number.isSafeInteger(live.runStateVersion) ||
    !Number.isSafeInteger(live.stageStateVersion))
    deny("D29B_PERSISTENCE_SCOPE_MISMATCH");
  const verdict=verifyDelegatedD2Handoff(issued.receipt,live,keys,nowSeconds());
  if(verdict.status!=="SIGNED_D2_HANDOFF_VERIFIED")
    deny("D29B_PERSISTENCE_PROOF_NOT_ADMITTED");
  // The outer metadata also goes to SQL. It must be identical to the already
  // signature-validated body; a forged metadata expiry cannot prolong a TTL.
  let signedTime:{issuedAt?:unknown;expiresAt?:unknown};
  try {
    signedTime=JSON.parse(Buffer.from(
      issued.receipt.payloadBase64Url,"base64url").toString("utf8"));
  } catch { deny("D29B_PERSISTENCE_SIGNED_METADATA_INVALID"); }
  if(!Number.isSafeInteger(issued.issuedAt) ||
    !Number.isSafeInteger(issued.expiresAt) ||
    signedTime.issuedAt!==issued.issuedAt ||
    signedTime.expiresAt!==issued.expiresAt ||
    issued.issuedAt>=issued.expiresAt ||
    issued.expiresAt<=nowSeconds())
    deny("D29B_PERSISTENCE_SIGNED_METADATA_MISMATCH");
  const second=await port.observeCurrentState();
  if(!sameState(live,second))deny("D29B_PERSISTENCE_STATE_CHANGED");
  if(second.stage===null || second.stageStateVersion===null)
    deny("D29B_PERSISTENCE_STAGE_MISSING");
  const result=await port.putReceiptAtomically({
    receipt:issued,expectedRunStateVersion:second.runStateVersion,
    expectedStageStateVersion:second.stageStateVersion,
    expectedStage:second.stage,expectedState:second,
  });
  if(result.status!=="STORED" || !result.receiptId)
    deny("D29B_PERSISTENCE_TRANSACTION_REFUSED");
  const stored=await port.getStoredReceipt(result.receiptId);
  if(!stored || stored.receiptId!==result.receiptId ||
    stored.revokedAt!==null ||
    !exactReceipt(issued.receipt,stored) ||
    !Number.isFinite(Date.parse(stored.expiresAt)) ||
    Date.parse(stored.expiresAt)!==issued.expiresAt*1000 ||
    Date.parse(stored.expiresAt)<=nowSeconds()*1000) {
    // Fail closed; best effort revoke, but never treat a failed revoke as success.
    await port.revokeReceiptAtomically(issued.runId,result.receiptId,
      "POST_STORE_MISMATCH");
    deny("D29B_PERSISTENCE_READBACK_MISMATCH");
  }
  return {status:"STORED",receiptId:result.receiptId};
}

/** Revocation is not an OAuth MCP operation. Caller must separately enforce
 * authenticated administrative authority and exact run-ID allowlist.
 */
export async function revokeD29bReceipt(
  runId:string,receiptId:string,port:D29bAdminWritePort,
):Promise<"REVOKED"|"NOT_FOUND"> {
  if(!/^[0-9a-f]{8}-[0-9a-f]{4}-[1-8][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i.test(runId) ||
    !/^[0-9a-f]{8}-[0-9a-f]{4}-[1-8][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i.test(receiptId))
    deny("D29B_REVOCATION_IDENTIFIERS_INVALID");
  return port.revokeReceiptAtomically(runId,receiptId,"ADMIN_REVOKED");
}
