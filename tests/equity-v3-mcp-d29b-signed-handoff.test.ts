import assert from "node:assert/strict";
import test from "node:test";
import { createHash, generateKeyPairSync, sign } from "node:crypto";
import pins from "../contracts/orotitan-equity/v3/contract-pin-pack-v3/OROTITAN_CONTRACT_PIN_PACK_V3_0_2.json";
import {
  HANDOFF_PURPOSE, HANDOFF_SIGNATURE_DOMAIN, verifyDelegatedD2Handoff,
  type CallerBoundState, type HandoffAttestationV1,
  type SignedHandoffEnvelope, type HandoffVerifyKey,
} from "../lib/orotitan-equity/v3/mcp-d29b-signed-handoff-gate";

const {publicKey, privateKey}=generateKeyPairSync("ed25519");
const {publicKey: wrongPublicKey}=generateKeyPairSync("ed25519");
const ring:readonly HandoffVerifyKey[]=[{
  keyId:"test-d2-key-v1",
  publicKeySpkiPem:publicKey.export({type:"spki",format:"pem"}).toString(),
}];
const now=1_800_000_000;
const runId="11111111-1111-4111-8111-111111111111";
const subject="33333333-3333-4333-8333-333333333333";
const resource="https://orotitan-chatgpt-mcp-pilot.vercel.app/api/orotitan/mcp";
const sha=(v:string)=>createHash("sha256").update(v).digest("hex");
const prompt=[
  "OROTITAN V3.1 — PILOTAGE HANDOFF",
  "RUN_ID = "+runId,
  "DATA_CUTOFF = 2026-10-12",
  "EXECUTION_PHASE = RESEARCH",
  "PUBLICATION_AUTHORIZED = NO",
  "MANUAL_EXECUTION = YES; NO AUTOMATIC_MODEL_CALL_OR_REGISTRY_MUTATION",
  "LOAD AND VERIFY AUTHORITATIVE REGISTRY STATE, PINNED CONTRACTS AND EXACT ARTIFACT BYTES BEFORE WORK.",
].join("\n");

function fixture():HandoffAttestationV1 {
  return {
    purpose:HANDOFF_PURPOSE,
    verifiedBy:"D2_IMMUTABLE_BYTES_AND_CONTRACTS_V3_1",
    runId,subject,clientId:"test-oauth-client",
    resource,runStatus:"ACTIVE",stage:"RESEARCH",
    stageLifecycle:"IN_PROGRESS",handoffGateState:"NOT_EVALUATED",
    activeManifestKind:"CHECKPOINT",processVersion:"3.1",
    pilotageVersion:"3.1",dataCutoff:"2026-10-12",
    runStateVersion:6,stageStateVersion:4,
    outcome:"CONTINUE",phase:"RESEARCH",
    contractSetSha256:pins.contract_set_sha256,
    manifestSha256:sha("verified fixture manifest bytes"),
    stateFingerprintSha256:sha("verified fixture state"),
    verifiedArtifactsCount:3,prompt,promptSha256:sha(prompt),
    issuedAt:now-25,expiresAt:now+60,
  };
}
function live():CallerBoundState {
  return {
    subject,clientId:"test-oauth-client",resource,runId,
    runStatus:"ACTIVE",stage:"RESEARCH",stageLifecycle:"IN_PROGRESS",
    handoffGateState:"NOT_EVALUATED",activeManifestKind:"CHECKPOINT",
    contractStatusCode:null,observedBlockerCount:0,
    processVersion:"3.1",pilotageVersion:"3.1",dataCutoff:"2026-10-12",
    runStateVersion:6,stageStateVersion:4,
  };
}
function signed(payload:HandoffAttestationV1):SignedHandoffEnvelope {
  const payloadBase64Url=Buffer.from(JSON.stringify(payload)).toString("base64url");
  const signatureBase64Url=sign(null,
    Buffer.from(HANDOFF_SIGNATURE_DOMAIN+payloadBase64Url),privateKey).toString("base64url");
  return {version:1,keyId:"test-d2-key-v1",payloadBase64Url,signatureBase64Url};
}
function deny(receipt:unknown,context=live(),keys=ring,time=now){
  const r=verifyDelegatedD2Handoff(receipt,context,keys,time);
  assert.equal(r.status,"SIGNED_D2_HANDOFF_DENIED",r.reasonCode);
  assert.equal(r.prompt,null);
  assert.equal(r.nextStepAuthorized,false);
  assert.equal(r.publicationAuthorized,false);
  assert.equal(r.mutationPerformed,false);
  return r.reasonCode;
}

test("D29b: signed D2 evidence, caller binding, pinned contract and CAS release only manual prompt",()=>{
  const r=verifyDelegatedD2Handoff(signed(fixture()),live(),ring,now);
  assert.equal(r.status,"SIGNED_D2_HANDOFF_VERIFIED");
  assert.equal(r.prompt,prompt);
  assert.equal(r.nextStepAuthorized,true);
  assert.equal(r.trust,"SIGNED_D2_ATTESTATION_WITH_LIVE_REGISTRY_CAS");
  assert.equal(r.publicationAuthorized,false);
  assert.equal(r.mutationPerformed,false);
});
test("D29b: no receipt, arbitrary unsigned Registry row, unknown signer are denied",()=>{
  deny(null);
  deny({runId,prompt,runStatus:"ACTIVE",handoffGateState:"YES"});
  const r=signed(fixture());
  assert.equal(deny({...r,keyId:"untrusted"}),"SIGNER_NOT_TRUSTED");
  assert.equal(deny(r,live(),[]),"VERIFIER_CONFIGURATION_INVALID");
  const wrongRing=[{keyId:ring[0].keyId,
    publicKeySpkiPem:wrongPublicKey.export({type:"spki",format:"pem"}).toString()}];
  assert.equal(deny(r,live(),wrongRing),"SIGNATURE_INVALID");
});
test("D29b: signed payload tampering, wrong signature and unknown fields deny",()=>{
  const r=signed(fixture());
  const modified={...r,payloadBase64Url:Buffer.from(JSON.stringify({
    ...fixture(),prompt:"unauthorized replacement",
  })).toString("base64url")};
  assert.equal(deny(modified),"SIGNATURE_INVALID");
  assert.equal(deny({...r,signatureBase64Url:"not_base64!"}),"RECEIPT_BYTES_INVALID");
  assert.equal(deny(signed({...fixture(),unapproved:true} as HandoffAttestationV1)),
    "PAYLOAD_SHAPE_INVALID");
  assert.equal(deny({...r,extra:"x"}),"RECEIPT_SHAPE_INVALID");
});
test("D29b: OAuth subject/client/resource/run binding is exact",()=>{
  const receipt=signed(fixture());
  const cases=[
    {...live(),subject:"44444444-4444-4444-8444-444444444444"},
    {...live(),clientId:"attacker-client"},
    {...live(),resource:"https://other.example/mcp"},
    {...live(),runId:"22222222-2222-4222-8222-222222222222"},
  ];
  for (const x of cases)assert.equal(deny(receipt,x),"OAUTH_OR_RUN_SCOPE_MISMATCH");
});
test("D29b: stale CAS, changed cutoff, contract version and observed blockers fail closed",()=>{
  const receipt=signed(fixture());
  const cases=[
    {...live(),runStateVersion:7},
    {...live(),stageStateVersion:5},
    {...live(),stage:"DEEP_DIVE"},
    {...live(),dataCutoff:"2026-10-13"},
    {...live(),runStatus:"PUBLISHED"},
    {...live(),processVersion:"4.0"},
    {...live(),contractStatusCode:"BLOCKED_INSUFFICIENT_INPUT"},
    {...live(),observedBlockerCount:1},
  ];
  for (const x of cases)
    assert.equal(deny(receipt,x),"REGISTRY_STATE_DRIFT_OR_BLOCKED");
});
test("D29b: expiry/future issue and replay across time are denied",()=>{
  const receipt=signed(fixture());
  assert.equal(deny(receipt,live(),ring,now+61),"D2_ATTESTATION_NOT_ADMITTED");
  assert.equal(deny(signed({...fixture(),issuedAt:now+20})),"D2_ATTESTATION_NOT_ADMITTED");
  assert.equal(deny(signed({...fixture(),expiresAt:now+400})),"D2_ATTESTATION_NOT_ADMITTED");
  assert.equal(deny(signed({...fixture(),issuedAt:now-301,expiresAt:now+1})),
    "D2_ATTESTATION_NOT_ADMITTED");
});
test("D29b: signer cannot bypass pinned contracts, prompt hash or publication firewall",()=>{
  const cases=[
    {...fixture(),contractSetSha256:sha("wrong")},
    {...fixture(),promptSha256:sha("tampered")},
    {...fixture(),prompt:"dangerous prompt not bound to run"},
    {...fixture(),verifiedArtifactsCount:0},
    {...fixture(),dataCutoff:"bad-date"},
  ];
  for(const x of cases)
    assert.equal(deny(signed(x as HandoffAttestationV1)),"D2_ATTESTATION_NOT_ADMITTED");
});
test("D29b: unsigned final gate or signed invalid handoff cannot advance",()=>{
  const yes={...live(),stageLifecycle:"COMPLETE",
    handoffGateState:"YES",activeManifestKind:"FINAL"};
  const signedFinal=signed({...fixture(),outcome:"HANDOFF",
    stageLifecycle:"COMPLETE",handoffGateState:"YES",activeManifestKind:"FINAL"});
  const valid=verifyDelegatedD2Handoff(signedFinal,yes,ring,now);
  assert.equal(valid.status,"SIGNED_D2_HANDOFF_VERIFIED");
  assert.equal(valid.publicationAuthorized,false);
  assert.equal(deny(signed({...fixture(),outcome:"HANDOFF"})),"HANDOFF_GATE_NOT_ADMITTED");
  assert.equal(deny(signedFinal),"REGISTRY_STATE_DRIFT_OR_BLOCKED");
  assert.equal(deny(signed({...fixture(),runStatus:"PUBLISHED" as "ACTIVE"})),
    "OUTCOME_NOT_ADMITTED");
});
