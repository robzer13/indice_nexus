import assert from "node:assert/strict";
import test from "node:test";
import { createHash,generateKeyPairSync,sign,verify } from "node:crypto";
import { DOMAIN, KEY_VERSION, loadSignerPolicy, decodeSignablePayload, SignerPolicyError } from "./policy.mjs";
import { crc32c,signWithCloudKms,KmsTransportError } from "./kms.mjs";
import { createSignerServer } from "./server.mjs";

const {privateKey,publicKey}=generateKeyPairSync("ed25519");
const publicKeyPem=publicKey.export({type:"spki",format:"pem"}).toString();
const runId="11111111-1111-4111-8111-111111111111";
const subject="33333333-3333-4333-8333-333333333333";
const resource="https://orotitan-chatgpt-mcp-pilot.vercel.app/api/orotitan/mcp";
const sha=s=>createHash("sha256").update(s).digest("hex");
const prompt=[
  "OROTITAN V3.1 — PILOTAGE HANDOFF",
  "RUN_ID = "+runId,"DATA_CUTOFF = 2026-10-12",
  "PUBLICATION_AUTHORIZED = NO",
  "MANUAL_EXECUTION = YES; NO AUTOMATIC_MODEL_CALL_OR_REGISTRY_MUTATION",
  "LOAD AND VERIFY AUTHORITATIVE REGISTRY STATE, PINNED CONTRACTS AND EXACT ARTIFACT BYTES BEFORE WORK.",
].join("\n");
function policy(){
  return loadSignerPolicy({
    D29B_RUN_ID:runId,D29B_SUBJECT_ID:subject,
    D29B_OAUTH_CLIENT_ID:"pilot-client-fixture",
    D29B_RESOURCE:resource,
    D29B_CONTRACT_SET_SHA256:sha("pinned contracts"),
    D29B_KEY_ID:"gcp-d29b-v1",
    D29B_KMS_KEY_VERSION:KEY_VERSION,
    D29B_PUBLIC_KEY_SPKI_PEM:publicKeyPem,
  });
}
function attestation(now=Math.floor(Date.now()/1000)){
  return {
    purpose:"OROTITAN_D29B_D2_HANDOFF_V1",
    verifiedBy:"D2_IMMUTABLE_BYTES_AND_CONTRACTS_V3_1",
    runId,subject,clientId:"pilot-client-fixture",resource,
    runStatus:"ACTIVE",stage:"RESEARCH",stageLifecycle:"IN_PROGRESS",
    handoffGateState:"NOT_EVALUATED",activeManifestKind:"CHECKPOINT",
    processVersion:"3.1",pilotageVersion:"3.1",
    dataCutoff:"2026-10-12",
    runStateVersion:6,stageStateVersion:4,
    outcome:"CONTINUE",phase:"RESEARCH",
    contractSetSha256:sha("pinned contracts"),
    manifestSha256:sha("verified manifest"),
    stateFingerprintSha256:sha("current state"),
    verifiedArtifactsCount:3,prompt,promptSha256:sha(prompt),
    issuedAt:now-5,expiresAt:now+70,
  };
}
function envelope(p=attestation()){
  return {payloadBase64Url:Buffer.from(JSON.stringify(p)).toString("base64url")};
}
test("D29b Cloud Run signer: canonical, scoped D2 payload signs raw Ed25519 bytes",()=>{
  const request=envelope();
  const data=decodeSignablePayload(request,policy(),Math.floor(Date.now()/1000));
  assert.equal(data.toString(),DOMAIN+request.payloadBase64Url);
  const signature=sign(null,data,privateKey);
  assert.equal(verify(null,data,publicKey,signature),true);
  assert.equal(signature.length,64);
});
test("D29b Cloud Run signer: missing config or mismatched resource, trust pin and version deny",()=>{
  assert.throws(()=>loadSignerPolicy({D29B_RUN_ID:runId}),
    (e)=>e instanceof SignerPolicyError && e.code==="SIGNER_CONFIG_NOT_PINNED");
  const p=policy();
  const now=Math.floor(Date.now()/1000);
  for(const altered of [
    {...attestation(now),runId:"22222222-2222-4222-8222-222222222222"},
    {...attestation(now),subject:"44444444-4444-4444-8444-444444444444"},
    {...attestation(now),clientId:"attacker"},
    {...attestation(now),resource:"https://attacker.invalid/mcp"},
    {...attestation(now),contractSetSha256:sha("incorrect")},
    {...attestation(now),verifiedBy:"REGISTRY_ONLY"},
    {...attestation(now),manifestSha256:"00"},
    {...attestation(now),promptSha256:sha("wrong prompt")},
    {...attestation(now),issuedAt:now-1000,expiresAt:now+20},
    {...attestation(now),expiresAt:now+600},
    {...attestation(now),outcome:"HANDOFF"},
    {...attestation(now),phase:"UNTRUSTED_PHASE"},
    {...attestation(now),unexpected:"duplicate-shape"},
  ]){
    assert.throws(()=>decodeSignablePayload(envelope(altered),p,now),
      (e)=>e instanceof SignerPolicyError);
  }
  assert.throws(()=>decodeSignablePayload({
    payloadBase64Url:envelope().payloadBase64Url,signatureBase64Url:"forged",
  },p,now),(e)=>e instanceof SignerPolicyError);
});
test("D29b Cloud Run signer: CRC32C and KMS metadata identity are strictly checked",async()=>{
  assert.equal(crc32c(Buffer.from("123456789")),0xe3069283);
  const msg=decodeSignablePayload(envelope(),policy(),Math.floor(Date.now()/1000));
  const signature=sign(null,msg,privateKey);
  let seen=0;
  const f=async(url,options)=>{
    seen++;
    assert.equal(options.redirect,"error");
    assert.ok(options.signal instanceof AbortSignal);
    if(seen===1){
      assert.equal(url,"http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/token");
      assert.equal(options.headers["Metadata-Flavor"],"Google");
      return Response.json({access_token:"ya29."+"a".repeat(50)});
    }
    assert.equal(url,"https://cloudkms.googleapis.com/v1/"+KEY_VERSION+":asymmetricSign");
    assert.match(options.headers.Authorization,/^Bearer ya29\./);
    const body=JSON.parse(options.body);
    assert.equal(body.data,Buffer.from(msg).toString("base64"));
    assert.equal(body.dataCrc32c,String(crc32c(msg)));
    return Response.json({
      name:KEY_VERSION,verifiedDataCrc32c:true,protectionLevel:"SOFTWARE",
      signature:signature.toString("base64"),
      signatureCrc32c:String(crc32c(signature)),
    });
  };
  assert.deepEqual(await signWithCloudKms(msg,policy(),f),signature);
  assert.equal(seen,2);
});
test("D29b Cloud Run signer: KMS wrong signing key, checksum or signature denies",async()=>{
  const msg=decodeSignablePayload(envelope(),policy(),Math.floor(Date.now()/1000));
  const signature=sign(null,msg,privateKey);
  const base={name:KEY_VERSION,verifiedDataCrc32c:true,
    protectionLevel:"SOFTWARE",signature:signature.toString("base64"),
    signatureCrc32c:String(crc32c(signature))};
  for(const [patch,code] of [
    [{name:"projects/wrong-key"},"KMS_RESPONSE_NOT_ADMITTED"],
    [{verifiedDataCrc32c:false},"KMS_RESPONSE_NOT_ADMITTED"],
    [{protectionLevel:"HSM"},"KMS_RESPONSE_NOT_ADMITTED"],
    [{signatureCrc32c:"0"},"KMS_SIGNATURE_INTEGRITY_INVALID"],
    [{signature:Buffer.alloc(64).toString("base64"),signatureCrc32c:String(crc32c(Buffer.alloc(64)))},"KMS_SIGNATURE_INTEGRITY_INVALID"],
  ]){
    let n=0;
    const t=async()=>{
      n++;
      return n===1?Response.json({access_token:"ya29."+"a".repeat(50)}):
        Response.json({...base,...patch});
    };
    await assert.rejects(()=>signWithCloudKms(msg,policy(),t),
      (e)=>e instanceof KmsTransportError && e.code===code);
  }
  await assert.rejects(()=>signWithCloudKms(msg,policy(),async()=>{throw Error("offline")}),
    (e)=>e instanceof KmsTransportError && e.code==="METADATA_UNAVAILABLE");
});
test("D29b Cloud Run signer: HTTP routes reject malformed and untrusted envelopes",async()=>{
  const server=createSignerServer(policy(),async(msg)=>sign(null,msg,privateKey));
  await new Promise(resolve=>server.listen(0,"127.0.0.1",resolve));
  const addr=server.address();
  const url="http://127.0.0.1:"+addr.port;
  try{
    let res=await fetch(url+"/healthz");
    assert.equal(res.status,204);
    res=await fetch(url+"/unrelated",{method:"POST",body:"{}"});
    assert.equal(res.status,404);
    res=await fetch(url+"/v1/sign-d2-handoff",{method:"POST",
      headers:{"Content-Type":"application/json"},body:JSON.stringify({hello:"world"})});
    assert.equal(res.status,403);
    res=await fetch(url+"/v1/sign-d2-handoff",{method:"POST",
      headers:{"Content-Type":"application/json"},body:JSON.stringify(envelope())});
    assert.equal(res.status,200);
    const body=await res.json();
    assert.deepEqual(Object.keys(body),["version","keyId","payloadBase64Url","signatureBase64Url"]);
    assert.equal(body.keyId,"gcp-d29b-v1");
    assert.equal(verify(null,Buffer.from(DOMAIN+body.payloadBase64Url),
      publicKey,Buffer.from(body.signatureBase64Url,"base64url")),true);
  }finally{await new Promise(resolve=>server.close(resolve));}
});
