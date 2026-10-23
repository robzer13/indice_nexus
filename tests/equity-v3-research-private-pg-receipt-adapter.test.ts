import assert from "node:assert/strict";
import test from "node:test";
import {
  ResearchIntakeError, type PreparedResearchDraft, type TrustedSubmissionGrant,
  type TrustedVerifierGrant,
} from "../lib/orotitan-equity/v3/research-evidence-intake-candidate-v1";
import {
  createPrivateResearchPgTransactionPort, type PrivatePgPool,
} from "../lib/orotitan-equity/v3/research-private-pg-receipt-adapter-v1";

const RUN="48b9426b-4d0d-4858-9d87-efcd13ac415d";
const HASH="c".repeat(64);
const PATH="research-intake/v1/"+RUN+"/RESEARCH/"+HASH+
  "/b78d8826-9f6b-4d9c-a1b8-ec5d13faac01.json";
const draft={
  runId:RUN,issuerId:"besi-issuer",stage:"RESEARCH",
  runStateVersion:6,stageStateVersion:4,dataCutoff:"2026-10-12",
  contractPinsSha256:"a".repeat(64),submittedBySubject:"oauth-sub",
  submittedByClientId:"chatgpt-client",sha256:HASH,sizeBytes:15,
  fileName:"evidence.json",mediaType:"application/json",
  sourceDate:"2026-07-23",bytes:new Uint8Array(15),
} as PreparedResearchDraft;
const grant:TrustedSubmissionGrant={
  grantId:"server-issued-grant",runId:RUN,issuerId:"besi-issuer",
  stage:"RESEARCH",subject:"oauth-sub",clientId:"chatgpt-client",
  resource:"https://orotitan-chatgpt-mcp-pilot.vercel.app/api/orotitan/mcp",
  runStateVersion:6,stageStateVersion:4,dataCutoff:"2026-10-12",
  contractPinsSha256:"a".repeat(64),issuedAtMs:100,
  expiresAtMs:200,maySubmitResearchDraft:true,
};
const row={
  receipt_id:"received-not-verified-id",run_id:RUN,issuer_id:"besi-issuer",
  stage:"RESEARCH",run_state_version:6,stage_state_version:4,
  sha256:HASH,file_name:"evidence.json",size_bytes:15,
  submitted_by_subject:"oauth-sub",duplicate:false,object_path:PATH,
};
function fakePg(options:{deny?:boolean;failCommit?:boolean;badRow?:boolean}={}){
  const calls:{sql:string;params?:unknown[]}[]=[];let released=0;
  const pool:PrivatePgPool={
    connect:async()=>({
      query:async<T>(sql:string,params?:unknown[])=>{
        calls.push({sql,params});
        if(options.failCommit && sql.includes("commit_receipt_v1")) {
          throw Object.assign(new Error("serialization"),{code:"40001"});
        }
        if(sql.includes("preflight_v1"))return {rows:[{allowed:!options.deny}] as T[]};
        if(sql.includes("commit_receipt_v1"))return {
          rows:[{...row,run_id:options.badRow?"forged-run":RUN}] as T[],
        };
        if(sql.includes("verifier_metadata_v1"))return {rows:[row] as T[]};
        return {rows:[] as T[]};
      },release:()=>{released++;},
    }),
  };
  return {pool,calls,get released(){return released;}};
}
async function rejects(code:string,fn:()=>Promise<unknown>){
  await assert.rejects(fn,e=>e instanceof ResearchIntakeError && e.code===code);
}

test("private PostgreSQL adapter uses scoped preflight, serializable commit and parameterized SQL",async()=>{
  const x=fakePg(),port=createPrivateResearchPgTransactionPort(x.pool);
  await port.checkBeforeUpload(draft,grant);
  const committed=await port.commitReceiptAndAudit({
    draft,grant,stored:{objectPath:PATH,sha256:HASH,sizeBytes:15},
  });
  assert.equal(committed.receipt.status,"RECEIVED_NOT_VERIFIED");
  assert.equal(committed.receipt.admitted,false);
  assert.equal(committed.receipt.mayAdvance,false);
  assert.equal(committed.authoritativeObjectPath,PATH);
  assert.equal(x.calls.filter(v=>v.sql.includes("BEGIN ISOLATION LEVEL SERIALIZABLE")).length,1);
  assert.equal(x.calls.filter(v=>v.sql==="SET LOCAL ROLE orotitan_research_intake_writer").length,2);
  assert.equal(x.calls.filter(v=>v.sql==="COMMIT").length,2);
  assert.equal(x.calls.find(v=>v.sql.includes("commit_receipt_v1"))?.params?.length,16);
  assert.equal(x.released,2);
});

test("preflight DENIED never issues a write transaction",async()=>{
  const x=fakePg({deny:true}),port=createPrivateResearchPgTransactionPort(x.pool);
  await rejects("INTAKE_PRIVATE_PG_PREFLIGHT_DENIED",()=>port.checkBeforeUpload(draft,grant));
  assert.equal(x.calls.some(v=>v.sql.includes("commit_receipt_v1")),false);
  assert.equal(x.released,1);
});

test("bad issuer/actor/resource/pins or blob scope rejected before private connection",async()=>{
  const x=fakePg(),port=createPrivateResearchPgTransactionPort(x.pool);
  await rejects("INTAKE_PRIVATE_PG_SCOPE_DENIED",()=>port.checkBeforeUpload(
    {...draft,submittedByClientId:"rogue-client"},grant));
  await rejects("INTAKE_PRIVATE_PG_SCOPE_DENIED",()=>port.checkBeforeUpload(
    draft,{...grant,resource:"https://untrusted.invalid/mcp"}));
  await rejects("INTAKE_PRIVATE_PG_SCOPE_DENIED",()=>port.checkBeforeUpload(
    draft,{...grant,maySubmitResearchDraft:false}));
  await rejects("INTAKE_PRIVATE_PG_BLOB_SCOPE_DENIED",()=>port.commitReceiptAndAudit({
    draft,grant,stored:{objectPath:"different-run/file",sha256:HASH,sizeBytes:15},
  }));
  assert.equal(x.calls.length,0);
});

test("commit serialization failure rolls back and never returns an optimistic receipt",async()=>{
  const x=fakePg({failCommit:true}),port=createPrivateResearchPgTransactionPort(x.pool);
  await assert.rejects(()=>port.commitReceiptAndAudit({
    draft,grant,stored:{objectPath:PATH,sha256:HASH,sizeBytes:15},
  }),/serialization/);
  assert.equal(x.calls.at(-1)?.sql,"ROLLBACK");
  assert.equal(x.calls.some(v=>v.sql==="COMMIT"),false);
  assert.equal(x.released,1);
});

test("forged return row rejected with rollback",async()=>{
  const x=fakePg({badRow:true}),port=createPrivateResearchPgTransactionPort(x.pool);
  await rejects("INTAKE_PRIVATE_PG_COMMIT_RESULT_INVALID",
    ()=>port.commitReceiptAndAudit({
      draft,grant,stored:{objectPath:PATH,sha256:HASH,sizeBytes:15},
    }));
  assert.equal(x.calls.at(-1)?.sql,"ROLLBACK");
});

test("independent metadata reads scoped and not source bytes",async()=>{
  const x=fakePg(),port=createPrivateResearchPgTransactionPort(x.pool);
  const g:TrustedVerifierGrant={
    reviewerId:"independent-reviewer",runId:RUN,issuerId:"besi-issuer",
    stage:"RESEARCH",runStateVersion:6,stageStateVersion:4,
    issuedAtMs:100,expiresAtMs:200,mayReadQuarantineMetadata:true,
  };
  const q={runId:RUN,expectedStage:"RESEARCH" as const,
    expectedRunStateVersion:6,expectedStageStateVersion:4,sha256:HASH};
  const receipt=await port.readVerifierMetadata(q,g);
  assert.equal(receipt?.status,"RECEIVED_NOT_VERIFIED");
  assert.equal(receipt?.sha256,HASH);
  assert.equal("bytes" in (receipt ?? {}),false);
  assert.equal(x.calls.some(v=>v.sql.includes("BEGIN ISOLATION LEVEL REPEATABLE READ READ ONLY")),true);
  assert.equal(x.calls.find(v=>v.sql.includes("verifier_metadata_v1"))?.params?.length,6);
  await rejects("INTAKE_PRIVATE_PG_VERIFIER_DENIED",()=>port.readVerifierMetadata(
    {...q,runId:"d56a0bc9-4e80-48b3-b326-74d1fea15e63"},g));
});
