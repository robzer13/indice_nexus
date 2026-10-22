/** D29b REVIEW ONLY: private, audit-gated receipt persistence boundary.
 * No Supabase client, HTTP route, secret key or write privilege is provided.
 * The adapter MUST be backed by a separately approved private transaction that
 * writes the live receipt and audit event atomically with its CAS guard.
 */
import { createHash } from "node:crypto";
import {
  verifyDelegatedD2Handoff,
  type SignedHandoffEnvelope, type CallerBoundState, type HandoffVerifyKey,
} from "./mcp-d29b-signed-handoff-gate";

export type AuthorizedIssuedReceipt={
  receipt: SignedHandoffEnvelope;
  runId:string;
  subject:string;
  clientId:string;
  resource:string;
  issuedAt:number;
  expiresAt:number;
};
export type ReceiptWriteContext={
  operatorId:string; // authenticated independent admin, not ChatGPT OAuth user
  approvalId:string; // separate explicit approval reference, audit record
  expectedRunStateVersion:number;
  expectedStageStateVersion:number;
};
export type ScopedReceiptWriter={
  /** Atomic store plus audit, or fail. Must verify current CAS inside DB txn.
   * Never use oauth/authenticated/anon table write roles.
   */
  storeWithAudit:(input:{
    issued:AuthorizedIssuedReceipt;
    operatorId:string;approvalId:string;
    expectedRunStateVersion:number;
    expectedStageStateVersion:number;
    contentSha256:string;
  })=>Promise<"STORED"|"CAS_FAILED"|"AUDIT_FAILED"|"NOT_AUTHORIZED">;
};
export class ReceiptWriteDenied extends Error{
  constructor(readonly code:string){super(code);}
}
const deny=(code:string):never=>{throw new ReceiptWriteDenied(code);};
const UUID=/^[0-9a-f]{8}-[0-9a-f]{4}-[1-8][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i;
const ID=/^[A-Za-z0-9_:-]{6,160}$/;

/** Caller MUST obtain live state independently and prove current D2 attestation.
 * This function cannot create authority. It receives a signing receipt from
 * the private issuer, authenticates it against pinned verifier key and the
 * caller's exact context, and delegates storage only to an injected atomic port.
 */
export async function persistScopedD29bReceipt(
  issued:AuthorizedIssuedReceipt,
  live:CallerBoundState,
  keyRing:readonly HandoffVerifyKey[],
  ctx:ReceiptWriteContext,
  writer:ScopedReceiptWriter,
  nowSeconds:number,
):Promise<{status:"STORED";contentSha256:string}>{
  if(!ctx || !UUID.test(ctx.operatorId) || !ID.test(ctx.approvalId) ||
    !Number.isSafeInteger(ctx.expectedRunStateVersion) ||
    !Number.isSafeInteger(ctx.expectedStageStateVersion) ||
    ctx.expectedRunStateVersion<1 || ctx.expectedStageStateVersion<1)
    deny("D29B_OPERATOR_APPROVAL_REQUIRED");
  if(!issued || issued.runId!==live.runId ||
    issued.subject!==live.subject || issued.clientId!==live.clientId ||
    issued.resource!==live.resource ||
    issued.issuedAt>nowSeconds+5 || issued.expiresAt<=nowSeconds ||
    issued.expiresAt-issued.issuedAt>300 ||
    !Number.isSafeInteger(issued.issuedAt) ||
    !Number.isSafeInteger(issued.expiresAt) ||
    live.runStateVersion!==ctx.expectedRunStateVersion ||
    live.stageStateVersion!==ctx.expectedStageStateVersion)
    deny("D29B_RECEIPT_BINDING_OR_CAS_FAILED");
  const proof=verifyDelegatedD2Handoff(issued.receipt,live,keyRing,nowSeconds);
  if(proof.status!=="SIGNED_D2_HANDOFF_VERIFIED")
    deny("D29B_SIGNATURE_OR_D2_ATTESTATION_DENIED");
  const digest=createHash("sha256").update(JSON.stringify(issued.receipt)).digest("hex");
  const result=await writer.storeWithAudit({
    issued,operatorId:ctx.operatorId,approvalId:ctx.approvalId,
    expectedRunStateVersion:ctx.expectedRunStateVersion,
    expectedStageStateVersion:ctx.expectedStageStateVersion,
    contentSha256:digest,
  });
  if(result!=="STORED")deny("D29B_ATOMIC_STORE_NOT_COMMITTED");
  return {status:"STORED",contentSha256:digest};
}
