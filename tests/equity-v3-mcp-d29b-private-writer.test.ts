import assert from "node:assert/strict";
import test from "node:test";
import { createHash, generateKeyPairSync, sign } from "node:crypto";
import pins from "../contracts/orotitan-equity/v3/contract-pin-pack-v3/OROTITAN_CONTRACT_PIN_PACK_V3_0_2.json";
import {
  HANDOFF_PURPOSE, HANDOFF_SIGNATURE_DOMAIN,
  type HandoffAttestationV1, type CallerBoundState,
} from "../lib/orotitan-equity/v3/mcp-d29b-signed-handoff-gate";
import {
  persistScopedD29bReceipt, ReceiptWriteDenied,
  type ReceiptWriteContext, type ScopedReceiptWriter,
} from "../lib/orotitan-equity/v3/mcp-d29b-private-writer-boundary-v1";

const now=1800000000;
const runId="11111111-1111-4111-8111-111111111111";
const subject="33333333-3333-4333-8333-333333333333";
const resource="https://orotitan-chatgpt-mcp-pilot.vercel.app/api/orotitan/mcp";
const {privateKey,publicKey}=generateKeyPairSync("ed25519");
const keyRing=[{keyId:"trusted-kms-test",publicKeySpkiPem:publicKey.export({type:"spki",format:"pem"}).toString()}];
const hash=(s:string)=>createHash("sha256").update(s).digest("hex");
const prompt=[
"OROTITAN V3.1 — PILOTAGE HANDOFF","RUN_ID = "+runId,
"DATA_CUTOFF = 2026-10-12","PUBLICATION_AUTHORIZED = NO",
"MANUAL_EXECUTION = YES; NO AUTOMATIC_MODEL_CALL_OR_REGISTRY_MUTATION",
"LOAD AND VERIFY AUTHORITATIVE REGISTRY STATE, PINNED CONTRACTS AND EXACT ARTIFACT BYTES BEFORE WORK.",
].join("\n");
function observation():CallerBoundState{return {
  subject,clientId:"pilot-client-fixture",resource,runId,runStatus:"ACTIVE",
  stage:"RESEARCH",stageLifecycle:"IN_PROGRESS",handoffGateState:"NOT_EVALUATED",
  activeManifestKind:"CHECKPOINT",contractStatusCode:null,
  observedBlockerCount:0,processVersion:"3.1",pilotageVersion:"3.1",
  dataCutoff:"2026-10-12",runStateVersion:6,stageStateVersion:4,
};}
function issued(){
  const p:HandoffAttestationV1={
    purpose:HANDOFF_PURPOSE,verifiedBy:"D2_IMMUTABLE_BYTES_AND_CONTRACTS_V3_1",
    runId,subject,clientId:"pilot-client-fixture",resource,runStatus:"ACTIVE",
    stage:"RESEARCH",stageLifecycle:"IN_PROGRESS",handoffGateState:"NOT_EVALUATED",
    activeManifestKind:"CHECKPOINT",processVersion:"3.1",pilotageVersion:"3.1",
    dataCutoff:"2026-10-12",runStateVersion:6,stageStateVersion:4,
    outcome:"CONTINUE",phase:"RESEARCH",contractSetSha256:pins.contract_set_sha256,
    manifestSha256:hash("manifest-fixture"),stateFingerprintSha256:hash("state-fixture"),
    verifiedArtifactsCount:3,prompt,promptSha256:hash(prompt),
    issuedAt:now-15,expiresAt:now+90,
  };
  const payloadBase64Url=Buffer.from(JSON.stringify(p)).toString("base64url");
  return {runId,subject,clientId:"pilot-client-fixture",resource,
    issuedAt:p.issuedAt,expiresAt:p.expiresAt,receipt:{
      version:1 as const,keyId:"trusted-kms-test",payloadBase64Url,
      signatureBase64Url:sign(null,Buffer.from(HANDOFF_SIGNATURE_DOMAIN+payloadBase64Url),
        privateKey).toString("base64url"),
  }};
}
function context():ReceiptWriteContext{return{
 operatorId:"55555555-5555-4555-8555-555555555555",approvalId:"D29B-PROPOSAL-0001",
 expectedRunStateVersion:6,expectedStageStateVersion:4,
};}
test("D29b private writer submits only independently signed and approved receipt",async()=>{
  let written=0;
  const port:ScopedReceiptWriter={storeWithAudit:async x=>{
    written++;
    assert.equal(x.expectedRunStateVersion,6);
    assert.equal(x.approvalId,"D29B-PROPOSAL-0001");
    assert.match(x.contentSha256,/^[0-9a-f]{64}$/);
    return "STORED";
  }};
  const result=await persistScopedD29bReceipt(issued(),observation(),keyRing,context(),port,now);
  assert.equal(result.status,"STORED");
  assert.equal(written,1);
});
test("D29b rejects forged, stale and unapproved receipts without calling writer",async()=>{
  let called=0;
  const port:ScopedReceiptWriter={storeWithAudit:async()=>{called++;return "STORED";}};
  const cases=[
   [issued(),observation(),{...context(),approvalId:""}],
   [issued(),{...observation(),runStateVersion:7},context()],
   [issued(),{...observation(),observedBlockerCount:1},context()],
   [issued(),{...observation(),subject:"44444444-4444-4444-8444-444444444444"},context()],
   [{...issued(),expiresAt:now-1},observation(),context()],
   [{...issued(),receipt:{...issued().receipt,signatureBase64Url:"forged"}},observation(),context()],
  ] as const;
  for(const [r,s,c] of cases){
    await assert.rejects(()=>persistScopedD29bReceipt(r,s,keyRing,c,port,now),
      (e:unknown)=>e instanceof ReceiptWriteDenied);
  }
  assert.equal(called,0);
});
test("D29b denies audit failure or CAS failure in atomic private adapter",async()=>{
  for(const failure of ["CAS_FAILED","AUDIT_FAILED","NOT_AUTHORIZED"] as const){
    const port:ScopedReceiptWriter={storeWithAudit:async()=>failure};
    await assert.rejects(()=>persistScopedD29bReceipt(issued(),observation(),keyRing,context(),port,now),
      (e:unknown)=>e instanceof ReceiptWriteDenied && e.code==="D29B_ATOMIC_STORE_NOT_COMMITTED");
  }
});
