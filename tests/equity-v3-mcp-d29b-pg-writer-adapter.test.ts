import assert from "node:assert/strict";
import test from "node:test";
import {
  createD29bPgWriterPort, D29bPgWriterError,
  type D29bPgPool,
} from "../lib/orotitan-equity/v3/mcp-d29b-pg-writer-adapter";
import type { D29bReceiptForWrite } from
  "../lib/orotitan-equity/v3/mcp-d29b-receipt-lifecycle-v1";
import type { CallerBoundState } from
  "../lib/orotitan-equity/v3/mcp-d29b-signed-handoff-gate";

const runId="11111111-1111-4111-8111-111111111111";
const subject="33333333-3333-4333-8333-333333333333";
const resource="https://orotitan-chatgpt-mcp-pilot.vercel.app/api/orotitan/mcp";
const scope={runId,subject,clientId:"pilot-client-fixture",resource};
const receiptId="aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa";
function state():CallerBoundState {
  return {runId,subject,clientId:scope.clientId,resource,
    runStatus:"ACTIVE",stage:"RESEARCH",stageLifecycle:"IN_PROGRESS",
    handoffGateState:"NOT_EVALUATED",activeManifestKind:"CHECKPOINT",
    contractStatusCode:null,observedBlockerCount:0,
    processVersion:"3.1",pilotageVersion:"3.1",dataCutoff:"2026-10-12",
    runStateVersion:6,stageStateVersion:4};
}
function writeInput():D29bReceiptForWrite {
  return {receipt:{
    receipt:{version:1,keyId:"test-key-v1",
      payloadBase64Url:"a".repeat(200),signatureBase64Url:"b".repeat(86)},
    runId,subject,clientId:scope.clientId,resource,
    issuedAt:1800000000,expiresAt:1800000120,
  },expectedRunStateVersion:6,expectedStageStateVersion:4,
    expectedStage:"RESEARCH",expectedState:state()};
}
function fakePool(overrides?:{
  stale?:boolean;sqlError?:boolean;stateRows?:number;onConnect?:()=>void;
}) {
  const entries:{sql:string;params?:unknown[]}[]=[];
  let connectionCount=0,releases=0;
  const pool:D29bPgPool={
    connect:async()=>{
      connectionCount++;overrides?.onConnect?.();
      return {
        query:async<T>(sql:string,params?:unknown[])=>{
          entries.push({sql,params});
          if(overrides?.sqlError && sql.includes("FROM public.orotitan_runs"))
            throw new Error("Connection refused");
          if(sql.includes("FROM public.orotitan_runs"))return {rows:(
            overrides?.stateRows===0?[]:[{
              run_id:runId,run_status:"ACTIVE",current_stage:"RESEARCH",
              run_state_version:"6",stage_state_version:"4",
              data_cutoff:"2026-10-12",process_version:"3.1",
              pilotage_contract_version:"3.1",
              stage_lifecycle:"IN_PROGRESS",handoff_gate_state:"NOT_EVALUATED",
              active_manifest_kind:"CHECKPOINT",contract_status_code:null,
              blocker_summary:[],
            }]) as T[]};
          if(sql.includes("orotitan_d29b_internal.put_receipt_v1")) {
            if(overrides?.stale)throw Object.assign(new Error("serialization"),{code:"40001"});
            return {rows:[{receipt_id:receiptId}] as T[]};
          }
          if(sql.includes("FROM public.orotitan_mcp_verified_handoffs"))
            return {rows:[{receipt_id:receiptId,key_id:"test-key-v1",
              payload_base64url:"a".repeat(200),
              signature_base64url:"b".repeat(86),
              expires_at:"2027-01-15T12:00:00Z",revoked_at:null}] as T[]};
          if(sql.includes("orotitan_d29b_internal.revoke_receipt_v1"))
            return {rows:[{did_revoke:true}] as T[]};
          return {rows:[] as T[]};
        },
        release:()=>{releases++;},
      };
    },
  };
  return {pool,logs:()=>entries,counts:()=>({connectionCount,releases})};
}
test("D29b PG adapter: one read-only transaction and exact caller-free scoped state",async()=>{
  const x=fakePool(),port=createD29bPgWriterPort(x.pool,scope);
  assert.deepEqual(await port.observeCurrentState(),state());
  const sql=x.logs().map(z=>z.sql);
  assert.match(sql[0],/BEGIN ISOLATION LEVEL REPEATABLE READ READ ONLY/);
  assert.equal(sql[1],"SET LOCAL ROLE orotitan_d29b_writer");
  assert.match(sql[2],/WHERE r.run_id=\$1/);
  assert.deepEqual(x.logs()[2].params,[runId]);
  assert.equal(sql[3],"COMMIT");
  assert.deepEqual(x.counts(),{connectionCount:1,releases:1});
});
test("D29b PG adapter: serializable transaction, parameterized stored write and revocation",async()=>{
  const x=fakePool(),port=createD29bPgWriterPort(x.pool,scope);
  assert.deepEqual(await port.putReceiptAtomically(writeInput()),
    {status:"STORED",receiptId});
  assert.match(x.logs()[0].sql,/BEGIN ISOLATION LEVEL SERIALIZABLE/);
  assert.equal(x.logs()[1].sql,"SET LOCAL ROLE orotitan_d29b_writer");
  assert.match(x.logs()[2].sql,/put_receipt_v1/);
  assert.equal(x.logs()[2].params?.length,13);
  assert.equal(x.logs()[3].sql,"COMMIT");
  const stored=await port.getStoredReceipt(receiptId);
  assert.equal(stored?.receiptId,receiptId);
  assert.equal(stored?.keyId,"test-key-v1");
  assert.equal(await port.revokeReceiptAtomically(runId,receiptId,"ADMIN_REVOKED"),
    "REVOKED");
  assert.equal(x.logs().filter(z=>z.sql==="COMMIT").length,3);
  assert.equal(x.counts().connectionCount,x.counts().releases);
});
test("D29b PG adapter: scope mismatch fails before opening a connection",async()=>{
  const x=fakePool(),port=createD29bPgWriterPort(x.pool,scope);
  assert.deepEqual(await port.putReceiptAtomically({
    ...writeInput(),expectedRunStateVersion:7,
  }),{status:"DENIED",receiptId:null});
  await assert.rejects(()=>port.revokeReceiptAtomically(runId,
    receiptId,"arbitrary"),(e:unknown)=>e instanceof D29bPgWriterError);
  assert.deepEqual(x.counts(),{connectionCount:0,releases:0});
  assert.throws(()=>createD29bPgWriterPort(x.pool,{...scope,runId:
    "22222222-2222-4222-8222-222222222222",resource:"https://evil.example/mcp"}),
    D29bPgWriterError);
});
test("D29b PG adapter: stale serializable CAS rolls back and fails closed",async()=>{
  const x=fakePool({stale:true}),port=createD29bPgWriterPort(x.pool,scope);
  assert.deepEqual(await port.putReceiptAtomically(writeInput()),
    {status:"STALE",receiptId:null});
  assert.equal(x.logs().at(-1)?.sql,"ROLLBACK");
  assert.equal(x.logs().some(z=>z.sql==="COMMIT"),false);
  assert.deepEqual(x.counts(),{connectionCount:1,releases:1});
});
test("D29b PG adapter: missing Registry stage or unavailable DB returns no optimistic result",async()=>{
  const x=fakePool({stateRows:0}),port=createD29bPgWriterPort(x.pool,scope);
  await assert.rejects(()=>port.observeCurrentState(),(e:unknown)=>
    e instanceof D29bPgWriterError && e.code==="D29B_PG_RUN_OR_STAGE_UNAVAILABLE");
  assert.equal(x.logs().at(-1)?.sql,"ROLLBACK");
  const err=fakePool({sqlError:true});
  await assert.rejects(()=>createD29bPgWriterPort(err.pool,scope).observeCurrentState());
  assert.equal(err.logs().at(-1)?.sql,"ROLLBACK");
});
