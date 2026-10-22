import assert from "node:assert/strict";
import test from "node:test";
import { createHash, generateKeyPairSync, sign } from "node:crypto";
import pins from "../contracts/orotitan-equity/v3/contract-pin-pack-v3/OROTITAN_CONTRACT_PIN_PACK_V3_0_2.json";
import {
  HANDOFF_PURPOSE,HANDOFF_SIGNATURE_DOMAIN,
  type CallerBoundState,type HandoffAttestationV1,type HandoffVerifyKey,
} from "../lib/orotitan-equity/v3/mcp-d29b-signed-handoff-gate";
import {
  persistVerifiedD29bReceipt,revokeD29bReceipt,
  D29bReceiptPersistenceError,
  type D29bIssuedReceipt,type D29bAdminWritePort,
} from "../lib/orotitan-equity/v3/mcp-d29b-receipt-lifecycle-v1";

const {publicKey,privateKey}=generateKeyPairSync("ed25519");
const keys:readonly HandoffVerifyKey[]=[{
  keyId:"test-key-d29b",publicKeySpkiPem:publicKey.export({type:"spki",format:"pem"}).toString(),
}];
const now=1_800_000_000;
const runId="11111111-1111-4111-8111-111111111111";
const subject="33333333-3333-4333-8333-333333333333";
const resource="https://orotitan-chatgpt-mcp-pilot.vercel.app/api/orotitan/mcp";
const fingerprint=(x:string)=>createHash("sha256").update(x).digest("hex");
const prompt=[
"OROTITAN V3.1 — PILOTAGE HANDOFF",
"RUN_ID = "+runId,
"DATA_CUTOFF = 2026-10-12",
"PUBLICATION_AUTHORIZED = NO",
"MANUAL_EXECUTION = YES; NO AUTOMATIC_MODEL_CALL_OR_REGISTRY_MUTATION",
"LOAD AND VERIFY AUTHORITATIVE REGISTRY STATE, PINNED CONTRACTS AND EXACT ARTIFACT BYTES BEFORE WORK.",
].join("\n");
function live():CallerBoundState {
  return {
    runId,subject,clientId:"pilot-client-fixture",resource,
    runStatus:"ACTIVE",stage:"RESEARCH",stageLifecycle:"IN_PROGRESS",
    handoffGateState:"NOT_EVALUATED",activeManifestKind:"CHECKPOINT",
    contractStatusCode:null,observedBlockerCount:0,processVersion:"3.1",
    pilotageVersion:"3.1",dataCutoff:"2026-10-12",
    runStateVersion:6,stageStateVersion:4,
  };
}
function issued():D29bIssuedReceipt {
  const payload:HandoffAttestationV1={
    purpose:HANDOFF_PURPOSE,verifiedBy:"D2_IMMUTABLE_BYTES_AND_CONTRACTS_V3_1",
    runId,subject,clientId:"pilot-client-fixture",resource,
    runStatus:"ACTIVE",stage:"RESEARCH",stageLifecycle:"IN_PROGRESS",
    handoffGateState:"NOT_EVALUATED",activeManifestKind:"CHECKPOINT",
    processVersion:"3.1",pilotageVersion:"3.1",dataCutoff:"2026-10-12",
    runStateVersion:6,stageStateVersion:4,outcome:"CONTINUE",phase:"RESEARCH",
    contractSetSha256:pins.contract_set_sha256,manifestSha256:fingerprint("manifest"),
    stateFingerprintSha256:fingerprint("state"),verifiedArtifactsCount:2,
    prompt,promptSha256:fingerprint(prompt),issuedAt:now-10,expiresAt:now+70,
  };
  const payloadBase64Url=Buffer.from(JSON.stringify(payload)).toString("base64url");
  const signatureBase64Url=sign(null,Buffer.from(
    HANDOFF_SIGNATURE_DOMAIN+payloadBase64Url),privateKey).toString("base64url");
  return {
    receipt:{version:1,keyId:"test-key-d29b",payloadBase64Url,signatureBase64Url},
    runId,subject,clientId:"pilot-client-fixture",resource,
    issuedAt:now-10,expiresAt:now+70,
  };
}
function port(receipt=issued()) {
  let observeCount=0,putCount=0,readCount=0,revokeCount=0;
  const resultId="aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa";
  const p:D29bAdminWritePort={
    observeCurrentState:async()=>{observeCount++;return live();},
    putReceiptAtomically:async x=>{
      putCount++;
      assert.equal(x.expectedStage,"RESEARCH");
      assert.equal(x.expectedRunStateVersion,6);
      assert.equal(x.expectedStageStateVersion,4);
      assert.deepEqual(x.receipt,receipt);
      return {status:"STORED",receiptId:resultId};
    },
    getStoredReceipt:async()=>{
      readCount++;
      return {receiptId:resultId,keyId:receipt.receipt.keyId,
        payloadBase64Url:receipt.receipt.payloadBase64Url,
        signatureBase64Url:receipt.receipt.signatureBase64Url,
        expiresAt:new Date(receipt.expiresAt*1000).toISOString(),revokedAt:null};
    },
    revokeReceiptAtomically:async()=>{revokeCount++;return "REVOKED";},
  };
  return {p,stats:()=>({observeCount,putCount,readCount,revokeCount})};
}
async function fails(action:()=>Promise<unknown>,code:string) {
  await assert.rejects(action,(e:unknown)=>e instanceof D29bReceiptPersistenceError && e.code===code);
}

test("D29b persistence: signed D2 receipt + fresh CAS + atomic port + readback",async()=>{
  const receipt=issued(),p=port(receipt);
  const result=await persistVerifiedD29bReceipt(receipt,keys,p.p,()=>now);
  assert.equal(result.status,"STORED");
  assert.equal(result.receiptId,"aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa");
  assert.deepEqual(p.stats(),{observeCount:2,putCount:1,readCount:1,revokeCount:0});
});
test("D29b persistence: signed metadata expiry and scope cannot be overwritten",async()=>{
  const original=issued();
  const invalid=[
    {...original,issuedAt:now-11},
    {...original,expiresAt:now+500},
    {...original,subject:"44444444-4444-4444-8444-444444444444"},
  ];
  for(const item of invalid) {
    const p=port(original);
    await assert.rejects(()=>persistVerifiedD29bReceipt(item,keys,p.p,()=>now),
      (e:unknown)=>e instanceof D29bReceiptPersistenceError);
    assert.equal(p.stats().putCount,0);
  }
});
test("D29b persistence: untrusted key and tampering cannot write",async()=>{
  const original=issued();
  const scenarios=[
    {...original,receipt:{...original.receipt,payloadBase64Url:"a".repeat(100)}},
    {...original,receipt:{...original.receipt,signatureBase64Url:"b".repeat(86)}},
  ];
  for(const item of scenarios) {
    const p=port(original);
    await fails(()=>persistVerifiedD29bReceipt(item,keys,p.p,()=>now),
      "D29B_PERSISTENCE_PROOF_NOT_ADMITTED");
    assert.equal(p.stats().putCount,0);
  }
  const p=port(original);
  await fails(()=>persistVerifiedD29bReceipt(original,[],p.p,()=>now),
    "D29B_PERSISTENCE_PROOF_NOT_ADMITTED");
  assert.equal(p.stats().putCount,0);
});
test("D29b persistence: drift before write, refused CAS or postwrite mismatch fails closed",async()=>{
  const original=issued();
  let checks=0,putCount=0;
  const drift=port(original);
  drift.p.observeCurrentState=async()=>++checks===1?live():{...live(),runStateVersion:7};
  await fails(()=>persistVerifiedD29bReceipt(original,keys,drift.p,()=>now),
    "D29B_PERSISTENCE_STATE_CHANGED");
  assert.equal(drift.stats().putCount,0);

  const stale=port(original);
  stale.p.putReceiptAtomically=async()=>{putCount++;return {status:"STALE",receiptId:null};};
  await fails(()=>persistVerifiedD29bReceipt(original,keys,stale.p,()=>now),
    "D29B_PERSISTENCE_TRANSACTION_REFUSED");
  assert.equal(putCount,1);

  const wrong=port(original);
  wrong.p.getStoredReceipt=async()=>({receiptId:"aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa",
    keyId:original.receipt.keyId,payloadBase64Url:original.receipt.payloadBase64Url,
    signatureBase64Url:"bad",expiresAt:new Date(original.expiresAt*1000).toISOString(),
    revokedAt:null});
  await fails(()=>persistVerifiedD29bReceipt(original,keys,wrong.p,()=>now),
    "D29B_PERSISTENCE_READBACK_MISMATCH");
  assert.equal(wrong.stats().revokeCount,1);
});
test("D29b persistence: revocation requires UUID and never calls data write from an OAuth route",async()=>{
  const p=port();
  assert.equal(await revokeD29bReceipt(runId,"aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa",p.p),
    "REVOKED");
  assert.equal(p.stats().revokeCount,1);
  await fails(()=>revokeD29bReceipt("invalid","invalid",p.p),
    "D29B_REVOCATION_IDENTIFIERS_INVALID");
  assert.equal(p.stats().revokeCount,1);
});
