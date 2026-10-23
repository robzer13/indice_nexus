import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import test from "node:test";
import type { SupabaseClient } from "@supabase/supabase-js";
import {
  ResearchIntakeError, submitResearchDraftCandidate,
  lookupResearchDraftForVerifierCandidate,
  type IntakeActor, type ResearchDraftInput, type PreparedResearchDraft,
  type TrustedSubmissionGrant, type TrustedVerifierGrant, type VerifierActor,
  type UnverifiedReceipt, type VerifierLookup,
} from "../lib/orotitan-equity/v3/research-evidence-intake-candidate-v1";
import {
  RESEARCH_PRIVATE_BUCKET, createPrivateResearchIntakePort,
  createSupabasePrivateResearchBlobPort, privateAttemptObjectPath,
  type PrivateResearchBlobPort, type ResearchReceiptTransactionPort,
  type ResearchScopedGrantReader,
} from "../lib/orotitan-equity/v3/research-private-quarantine-adapter-v1";

const RUN = "48b9426b-4d0d-4858-9d87-efcd13ac415d";
const OTHER = "d56a0bc9-4e80-48b3-b326-74d1fea15e63";
const ATTEMPT1 = "b78d8826-9f6b-4d9c-a1b8-ec5d13faac01";
const ATTEMPT2 = "b78d8826-9f6b-4d9c-a1b8-ec5d13faac02";
const NOW = Date.parse("2026-10-23T09:00:00.000Z");
const actor: IntakeActor = {
  subject: "approved-oauth-sub", clientId: "registered-mcp-client",
  resource: "https://orotitan-chatgpt-mcp-pilot.vercel.app/api/orotitan/mcp",
};
const reviewer: VerifierActor = { subject: "d2-independent-operator", reviewerId: "reviewer-d2" };
const bytes = new TextEncoder().encode('{"EVIDENCE_ID":"E001","status":"UNVERIFIED"}');
const hash = createHash("sha256").update(bytes).digest("hex");
const grant = (): TrustedSubmissionGrant => ({
  grantId: "approved-opaque-grant", runId: RUN, issuerId: "besi-issuer",
  stage: "RESEARCH", subject: actor.subject, clientId: actor.clientId,
  resource: actor.resource, runStateVersion: 6, stageStateVersion: 4,
  contractPinsSha256: "a".repeat(64), dataCutoff: "2026-10-12",
  issuedAtMs: NOW - 10_000, expiresAtMs: NOW + 60_000,
  maySubmitResearchDraft: true,
});
const draft = (patch: Partial<ResearchDraftInput> = {}): ResearchDraftInput => ({
  runId: RUN, expectedStage: "RESEARCH", expectedRunStateVersion: 6,
  expectedStageStateVersion: 4, fileName: "02_EVIDENCE_LEDGER.json",
  mediaType: "application/json", sourceDate: "2026-07-23",
  claimedSha256: hash, bytes, ...patch,
});
const lookup = (patch: Partial<VerifierLookup> = {}): VerifierLookup => ({
  runId: RUN, expectedStage: "RESEARCH", expectedRunStateVersion: 6,
  expectedStageStateVersion: 4, sha256: hash, ...patch,
});
const receipt = (d: PreparedResearchDraft, duplicate = false): UnverifiedReceipt => ({
  receiptId: "immutable-unverified-receipt", runId: d.runId, issuerId: d.issuerId,
  stage: "RESEARCH", runStateVersion: d.runStateVersion,
  stageStateVersion: d.stageStateVersion, sha256: d.sha256,
  fileName: d.fileName, sizeBytes: d.sizeBytes,
  submittedBySubject: d.submittedBySubject,
  status: "RECEIVED_NOT_VERIFIED", authority: "QUARANTINE_ONLY",
  duplicate, mayAdvance: false, mayPublish: false, admitted: false,
});

class FakeBlob implements PrivateResearchBlobPort {
  objects = new Map<string, Uint8Array>();
  tamper = false;
  async putImmutable(path: string, payload: Uint8Array) {
    if (this.objects.has(path)) throw new Error("upsert forbidden");
    this.objects.set(path, Uint8Array.from(payload));
  }
  async readExact(path: string) {
    const data = this.objects.get(path);
    if (!data) throw new Error("not found");
    return this.tamper ? new Uint8Array([0xff]) : Uint8Array.from(data);
  }
}

class FakeTransactions implements ResearchReceiptTransactionPort {
  liveVersion = 6;
  liveStageVersion = 4;
  revoked = false;
  failCommit = false;
  counterfeitPath = false;
  records = new Map<string, { receipt: UnverifiedReceipt; objectPath: string }>();
  audit: string[] = [];
  assertLive(d: PreparedResearchDraft, g: TrustedSubmissionGrant) {
    if (this.revoked || g.grantId !== "approved-opaque-grant" ||
      d.runId !== RUN || g.runId !== RUN || g.issuerId !== d.issuerId ||
      d.runStateVersion !== this.liveVersion ||
      d.stageStateVersion !== this.liveStageVersion ||
      d.submittedBySubject !== g.subject ||
      d.submittedByClientId !== g.clientId ||
      d.contractPinsSha256 !== g.contractPinsSha256 ||
      g.expiresAtMs <= NOW) {
      throw new ResearchIntakeError("PRIVATE_TXN_SCOPE_OR_CAS_DENIED");
    }
  }
  async checkBeforeUpload(d: PreparedResearchDraft, g: TrustedSubmissionGrant) {
    this.assertLive(d, g);
  }
  async commitReceiptAndAudit({draft:d,grant:g,stored}:{
    draft:PreparedResearchDraft;grant:TrustedSubmissionGrant;
    stored:{objectPath:string;sha256:string;sizeBytes:number};
  }) {
    if (this.failCommit) throw new Error("ambiguous COMMIT response");
    this.assertLive(d, g);
    if (stored.sha256 !== d.sha256 || stored.sizeBytes !== d.sizeBytes) {
      throw new ResearchIntakeError("PRIVATE_TXN_OBJECT_MISMATCH");
    }
    const old = this.records.get(d.sha256);
    if (old) {
      if (old.receipt.fileName !== d.fileName) throw new Error("filename collision");
      return {receipt:{...old.receipt,duplicate:true}, authoritativeObjectPath:old.objectPath};
    }
    const record = receipt(d);
    this.records.set(d.sha256,{receipt:record,objectPath:stored.objectPath});
    this.audit.push("STORED:"+d.sha256);
    return {receipt:record,authoritativeObjectPath:this.counterfeitPath?"unrelated/private/blob":stored.objectPath};
  }
  async readVerifierMetadata(q:VerifierLookup,g:TrustedVerifierGrant) {
    if (this.revoked || g.runId !== RUN || q.runId !== RUN ||
      this.liveVersion !== g.runStateVersion ||
      this.liveStageVersion !== g.stageStateVersion ||
      g.reviewerId !== reviewer.reviewerId) throw new Error("private verifier scope denied");
    return this.records.get(q.sha256)?.receipt ?? null;
  }
}

function make() {
  const blobs=new FakeBlob(), receipts=new FakeTransactions();
  const granted:ResearchScopedGrantReader={
    resolveSubmissionGrant:async(a,runId)=>runId===RUN &&
      a.subject===actor.subject && a.clientId===actor.clientId &&
      a.resource===actor.resource?grant():null,
    resolveVerifierGrant:async(a,runId):Promise<TrustedVerifierGrant|null>=>
      a.subject===reviewer.subject && a.reviewerId===reviewer.reviewerId &&
      runId===RUN?{
        reviewerId:reviewer.reviewerId,runId:RUN,issuerId:"besi-issuer",
        stage:"RESEARCH",runStateVersion:receipts.liveVersion,
        stageStateVersion:receipts.liveStageVersion,
        issuedAtMs:NOW-10_000,expiresAtMs:NOW+60_000,
        mayReadQuarantineMetadata:true,
      }:null,
  };
  let attempts=0;
  const port=createPrivateResearchIntakePort({
    grants:granted,blobs,receipts,newAttemptId:()=>++attempts===1?ATTEMPT1:ATTEMPT2,
  });
  return {port,blobs,receipts};
}
async function denied(fn:()=>Promise<unknown>,code:string) {
  await assert.rejects(fn,(e:unknown)=>e instanceof ResearchIntakeError && e.code===code);
}

test("Besi private adapter: upload -> reread sha -> atomic receipt/audit -> verifier metadata",async()=>{
  const x=make();
  const result=await submitResearchDraftCandidate(draft(),actor,x.port,NOW);
  assert.equal(result.status,"RECEIVED_NOT_VERIFIED");
  assert.equal(result.admitted,false);
  assert.equal(result.mayAdvance,false);
  assert.equal(result.mayPublish,false);
  assert.equal(x.receipts.audit.length,1);
  assert.equal(x.blobs.objects.size,1);
  assert.equal((await lookupResearchDraftForVerifierCandidate(lookup(),reviewer,x.port,NOW))?.sha256,hash);
  const repeated=await submitResearchDraftCandidate(draft(),actor,x.port,NOW);
  assert.equal(repeated.duplicate,true);
  assert.equal(x.receipts.audit.length,1);
  assert.equal(x.blobs.objects.size,2); // second unique blob: no request-path cleanup
});

test("wrong OAuth identity or cross-run scope never writes any blob",async()=>{
  const x=make();
  await denied(()=>submitResearchDraftCandidate(draft({runId:OTHER}),actor,x.port,NOW),
    "INTAKE_GRANT_NOT_FOUND");
  await denied(()=>submitResearchDraftCandidate(draft(),{...actor,clientId:"forged"},x.port,NOW),
    "INTAKE_GRANT_NOT_FOUND");
  assert.equal(x.blobs.objects.size,0);
});

test("stale or revoked before-upload preflight prevents all storage",async()=>{
  const x=make();x.receipts.liveStageVersion=5;
  await denied(()=>submitResearchDraftCandidate(draft(),actor,x.port,NOW),
    "PRIVATE_TXN_SCOPE_OR_CAS_DENIED");
  assert.equal(x.blobs.objects.size,0);
  x.receipts.liveStageVersion=4;x.receipts.revoked=true;
  await denied(()=>submitResearchDraftCandidate(draft(),actor,x.port,NOW),
    "PRIVATE_TXN_SCOPE_OR_CAS_DENIED");
  assert.equal(x.blobs.objects.size,0);
});

test("revocation AFTER preflight must be caught in transactional commit",async()=>{
  const x=make();
  const original=x.blobs.putImmutable.bind(x.blobs);
  x.blobs.putImmutable=async(path,bytes,type)=>{
    await original(path,bytes,type);
    x.receipts.revoked=true;
  };
  await denied(()=>submitResearchDraftCandidate(draft(),actor,x.port,NOW),
    "PRIVATE_TXN_SCOPE_OR_CAS_DENIED");
  assert.equal(x.receipts.records.size,0);
  assert.equal(x.receipts.audit.length,0);
  assert.equal(x.blobs.objects.size,1); // orphan retained for separate safe reconciler
});

test("tampered stored bytes block transaction, never create receipt",async()=>{
  const x=make();x.blobs.tamper=true;
  await denied(()=>submitResearchDraftCandidate(draft(),actor,x.port,NOW),
    "INTAKE_PRIVATE_STORED_HASH_MISMATCH");
  assert.equal(x.receipts.records.size,0);
  assert.equal(x.receipts.audit.length,0);
});

test("ambiguous DB commit preserves uploaded bytes, no dangerous deletion",async()=>{
  const x=make();x.receipts.failCommit=true;
  await assert.rejects(()=>submitResearchDraftCandidate(draft(),actor,x.port,NOW),
    /ambiguous COMMIT/);
  assert.equal(x.blobs.objects.size,1);
});

test("forged receipt object path rejected even if other metadata appears valid",async()=>{
  const x=make();x.receipts.counterfeitPath=true;
  await denied(()=>submitResearchDraftCandidate(draft(),actor,x.port,NOW),
    "INTAKE_PRIVATE_RECEIPT_INVALID");
});

test("unique private attempt paths cannot be changed by input filenames",async()=>{
  const x=make();
  await submitResearchDraftCandidate(draft(),actor,x.port,NOW);
  const key=[...x.blobs.objects.keys()][0];
  assert.match(key,/^research-intake\/v1\/48b9426b-4d0d-4858-9d87-efcd13ac415d\/RESEARCH\/[0-9a-f]{64}\//);
  assert.ok(key.endsWith(ATTEMPT1+".json"));
  const prepared={runId:RUN,stage:"RESEARCH",sha256:hash,fileName:"../evil.json"} as PreparedResearchDraft;
  assert.throws(()=>privateAttemptObjectPath(prepared,ATTEMPT2),ResearchIntakeError);
});

test("private Supabase Storage adapter sends upsert=false, reads exact bytes, no list",async()=>{
  let bucketName="",opts:Record<string,unknown>|undefined;
  const client={
    storage:{from:(name:string)=>{
      bucketName=name;
      return {
        upload:async(path:string,data:unknown,options:Record<string,unknown>)=>{
          assert.ok(path.includes("/RESEARCH/"));
          assert.ok(Buffer.isBuffer(data));
          opts=options;
          return {data:{path},error:null};
        },
        download:async(_path:string)=>({data:new Blob([bytes]),error:null}),
      };
    }},
  } as unknown as Pick<SupabaseClient,"storage">;
  const blob=createSupabasePrivateResearchBlobPort(client);
  const path="research-intake/v1/"+RUN+"/RESEARCH/"+hash+"/"+ATTEMPT1+".json";
  await blob.putImmutable(path,bytes,"application/json");
  assert.deepEqual(await blob.readExact(path),bytes);
  assert.equal(bucketName,RESEARCH_PRIVATE_BUCKET);
  assert.equal(opts?.upsert,false);
  assert.equal(opts?.contentType,"application/json");
  assert.throws(()=>createSupabasePrivateResearchBlobPort(client,"public"),ResearchIntakeError);
});
