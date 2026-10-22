/**
 * D29b admin-side PostgreSQL adapter — REVIEW ONLY.
 * Production connection / driver and IAM credentials are NOT provisioned.
 * This module relies on a PRIVATE trusted PostgreSQL pool wrapper, never on
 * Supabase service_role or the OAuth MCP JWT. It is not imported by an API.
 * SQL routines come from the unexecuted D29b private-writer proposal.
 */
import type {
  D29bAdminWritePort,D29bReceiptForWrite,D29bWriteResult,
} from "./mcp-d29b-receipt-lifecycle-v1";
import type { CallerBoundState } from "./mcp-d29b-signed-handoff-gate";
import { asOperatorBlockers } from "./pilotage-operator-view-v3-1";

export type D29bPgQueryResult<T>={rows:T[]};
export type D29bPgClient={
  query:<T>(sql:string,params?:unknown[])=>Promise<D29bPgQueryResult<T>>;
  release:()=>void;
};
export type D29bPgPool={
  connect:()=>Promise<D29bPgClient>;
};
export type D29bPgScope={
  runId:string;subject:string;clientId:string;resource:string;
};
const UUID=/^[0-9a-f]{8}-[0-9a-f]{4}-[1-8][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i;
const RESOURCE="https://orotitan-chatgpt-mcp-pilot.vercel.app/api/orotitan/mcp";
export class D29bPgWriterError extends Error {
  constructor(readonly code:string){super(code);}
}
function deny(code:string):never {throw new D29bPgWriterError(code);}
function safeVersion(value:unknown):number {
  const n=Number(value);
  if(!Number.isSafeInteger(n) || n<0)deny("D29B_PG_BAD_REGISTRY_VERSION");
  return n;
}
async function transaction<T>(
  pool:D29bPgPool,mode:"READ"|"WRITE",
  fn:(client:D29bPgClient)=>Promise<T>,
):Promise<T> {
  const client=await pool.connect();
  let started=false;
  try {
    await client.query(mode==="WRITE" ?
      "BEGIN ISOLATION LEVEL SERIALIZABLE" :
      "BEGIN ISOLATION LEVEL REPEATABLE READ READ ONLY");
    started=true;
    // The session principal must be a separately authorized DB member of
    // this NOLOGIN role. Neither anonymous nor authenticated OAuth can assume it.
    await client.query("SET LOCAL ROLE orotitan_d29b_writer");
    const result=await fn(client);
    await client.query("COMMIT");
    started=false;
    return result;
  } catch(error) {
    if(started)try{await client.query("ROLLBACK");}catch{/* fail closed */}
    throw error;
  } finally {client.release();}
}
type StateRow={
  run_id:string;run_status:string;current_stage:string|null;
  run_state_version:string|number;stage_state_version:string|number;
  data_cutoff:string;process_version:string;pilotage_contract_version:string;
  stage_lifecycle:string;handoff_gate_state:string;
  active_manifest_kind:string|null;contract_status_code:string|null;
  blocker_summary:unknown;
};
type StoredRow={
  receipt_id:string;key_id:string;payload_base64url:string;
  signature_base64url:string;expires_at:string;revoked_at:string|null;
};
function assertScope(scope:D29bPgScope) {
  if(!UUID.test(scope.runId) || !UUID.test(scope.subject) ||
    !/^[A-Za-z0-9_-]{6,160}$/.test(scope.clientId) ||
    scope.resource!==RESOURCE)
    deny("D29B_PG_PRIVATE_SCOPE_INVALID");
}
export function createD29bPgWriterPort(
  pool:D29bPgPool,scope:D29bPgScope,
):D29bAdminWritePort {
  assertScope(scope);
  async function observeCurrentState():Promise<CallerBoundState> {
    return transaction(pool,"READ",async(client)=>{
      const x=await client.query<StateRow>(`
        SELECT r.run_id,r.run_status,r.current_stage,
          r.state_version AS run_state_version,
          s.state_version AS stage_state_version,
          to_char(r.data_cutoff,'YYYY-MM-DD') AS data_cutoff,
          r.process_version,r.pilotage_contract_version,
          s.lifecycle_status AS stage_lifecycle,s.handoff_gate_state,
          s.active_manifest_kind,s.contract_status_code,s.blocker_summary
        FROM public.orotitan_runs r
        JOIN public.orotitan_run_stages s
          ON s.run_id=r.run_id AND s.stage_code=r.current_stage
        WHERE r.run_id=$1
      `,[scope.runId]);
      if(x.rows.length!==1)deny("D29B_PG_RUN_OR_STAGE_UNAVAILABLE");
      const r=x.rows[0];
      if(r.run_id!==scope.runId)deny("D29B_PG_CROSS_RUN_ROW");
      return {
        subject:scope.subject,clientId:scope.clientId,resource:scope.resource,
        runId:r.run_id,runStatus:r.run_status,stage:r.current_stage,
        stageLifecycle:r.stage_lifecycle,
        handoffGateState:r.handoff_gate_state,
        activeManifestKind:r.active_manifest_kind,
        contractStatusCode:r.contract_status_code,
        observedBlockerCount:asOperatorBlockers(r.blocker_summary).length,
        processVersion:r.process_version,
        pilotageVersion:r.pilotage_contract_version,
        dataCutoff:r.data_cutoff,
        runStateVersion:safeVersion(r.run_state_version),
        stageStateVersion:safeVersion(r.stage_state_version),
      };
    });
  }
  async function putReceiptAtomically(input:D29bReceiptForWrite):Promise<D29bWriteResult> {
    const s=input.expectedState;
    if(input.receipt.runId!==scope.runId ||
      input.receipt.subject!==scope.subject ||
      input.receipt.clientId!==scope.clientId ||
      input.receipt.resource!==scope.resource ||
      s.runId!==scope.runId || s.subject!==scope.subject ||
      s.clientId!==scope.clientId || s.resource!==scope.resource ||
      input.expectedRunStateVersion!==s.runStateVersion ||
      input.expectedStageStateVersion!==s.stageStateVersion ||
      input.expectedStage!==s.stage)
      return {status:"DENIED",receiptId:null};
    const x=input.receipt;
    const args=[
      scope.runId,scope.subject,scope.clientId,scope.resource,
      x.receipt.keyId,x.receipt.payloadBase64Url,x.receipt.signatureBase64Url,
      new Date(x.issuedAt*1000).toISOString(),
      new Date(x.expiresAt*1000).toISOString(),
      input.expectedStage,input.expectedRunStateVersion,
      input.expectedStageStateVersion,s.dataCutoff,
    ];
    try {
      return await transaction(pool,"WRITE",async(client)=>{
        const r=await client.query<{receipt_id:string}>(`
          SELECT orotitan_d29b_internal.put_receipt_v1(
            $1::uuid,$2::uuid,$3::text,$4::text,$5::text,$6::text,$7::text,
            $8::timestamptz,$9::timestamptz,$10::text,$11::bigint,
            $12::bigint,$13::date
          ) AS receipt_id
        `,args);
        const receiptId=r.rows[0]?.receipt_id;
        if(r.rows.length!==1 || !receiptId || !UUID.test(receiptId))
          deny("D29B_PG_WRITE_RESULT_INVALID");
        return {status:"STORED" as const,receiptId};
      });
    } catch(error) {
      // A CAS violation or serialization conflict is a denial, not an
      // instruction to retry a short-lived signed receipt automatically.
      const code=typeof error==="object" && error!==null &&
        "code" in error ? String(error.code) : "";
      const message=error instanceof Error?error.message:"";
      if(code==="40001" || (code==="P0001" && message.includes("D29b Registry CAS changed")))
        return {status:"STALE",receiptId:null};
      throw error;
    }
  }
  async function getStoredReceipt(receiptId:string) {
    if(!UUID.test(receiptId))deny("D29B_PG_RECEIPT_ID_INVALID");
    return transaction(pool,"READ",async(client)=>{
      const r=await client.query<StoredRow>(`
        SELECT receipt_id,key_id,payload_base64url,
          signature_base64url,expires_at::text,revoked_at::text
        FROM public.orotitan_mcp_verified_handoffs
        WHERE run_id=$1 AND receipt_id=$2
      `,[scope.runId,receiptId]);
      if(r.rows.length>1)deny("D29B_PG_DUPLICATE_RECEIPT");
      const x=r.rows[0];
      return x?{
        receiptId:x.receipt_id,keyId:x.key_id,
        payloadBase64Url:x.payload_base64url,
        signatureBase64Url:x.signature_base64url,
        expiresAt:x.expires_at,revokedAt:x.revoked_at,
      }:null;
    });
  }
  async function revokeReceiptAtomically(
    runId:string,receiptId:string,reasonCode:string,
  ):"REVOKED"|"NOT_FOUND"|Promise<"REVOKED"|"NOT_FOUND"> {
    if(runId!==scope.runId || !UUID.test(receiptId) ||
      !["ADMIN_REVOKED","POST_STORE_MISMATCH"].includes(reasonCode))
      deny("D29B_PG_REVOCATION_SCOPE_INVALID");
    return transaction(pool,"WRITE",async(client)=>{
      const r=await client.query<{did_revoke:boolean}>(`
        SELECT orotitan_d29b_internal.revoke_receipt_v1(
          $1::uuid,$2::uuid
        ) AS did_revoke
      `,[scope.runId,receiptId]);
      if(r.rows.length!==1 || typeof r.rows[0].did_revoke!=="boolean")
        deny("D29B_PG_REVOCATION_RESULT_INVALID");
      return r.rows[0].did_revoke?"REVOKED":"NOT_FOUND";
    });
  }
  return {observeCurrentState,putReceiptAtomically,getStoredReceipt,revokeReceiptAtomically};
}
