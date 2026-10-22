import assert from "node:assert/strict";
import test from "node:test";
import { generateKeyPairSync, sign } from "node:crypto";
import {
  crc32c,createGcpD29bEd25519Signer,D29bKmsError,
} from "../lib/orotitan-equity/v3/mcp-d29b-gcp-kms-signer";

const keyVersion="projects/example-demo-123/locations/europe-west1/keyRings/orotitan-test/cryptoKeys/d29b/cryptoKeyVersions/1";
const {publicKey,privateKey}=generateKeyPairSync("ed25519");
const {publicKey:otherKey}=generateKeyPairSync("ed25519");
const pinned=publicKey.export({type:"spki",format:"pem"}).toString();
const payload=Buffer.from("OROTITAN:D29B:D2_HANDOFF:V1\ntest-domain-separated-receipt");
const token="mockshortlivedgoogletoken";

type SimulatedResponse={
 signature:string;signatureCrc32c:string;verifiedDataCrc32c:boolean;
 name:string;protectionLevel:string;
};
const goodResponse=():SimulatedResponse=>{
  const signature=sign(null,payload,privateKey);
  return {signature:signature.toString("base64"),
    signatureCrc32c:String(crc32c(signature)),
    verifiedDataCrc32c:true,name:keyVersion,protectionLevel:"SOFTWARE"};
};
function signer(overrides?:Partial<SimulatedResponse>,opts?:{
  rejected?:boolean;key?:string;token?:()=>Promise<string>;
}) {
  let calls=0;
  let observedUrl="",observedMethod="",observedPayload:unknown=null;
  const adapter=createGcpD29bEd25519Signer({
    keyId:"kms-d29b-test-v1",versionName:keyVersion,
    pinnedPublicKeySpkiPem:opts?.key??pinned,
    accessToken:opts?.token??(async()=>token),
    fetcher:async(url,init)=>{
      calls++;
      observedUrl=url;observedMethod=init.method;
      assert.equal(init.headers.Authorization,"Bearer "+token);
      observedPayload=JSON.parse(init.body);
      return {ok:!opts?.rejected,status:opts?.rejected?403:200,
        json:async()=>({...goodResponse(),...overrides})};
    },
  });
  return {adapter,stats:()=>({calls,observedUrl,observedMethod,observedPayload})};
}
async function rejects(run:()=>Promise<unknown>,code:string) {
  await assert.rejects(run,(e:unknown)=>e instanceof D29bKmsError && e.code===code);
}
test("D29b Cloud KMS Ed25519 signs exact raw bytes with request CRC and verifies signature CRC",async()=>{
  assert.equal(crc32c(Buffer.from("123456789")),0xe3069283);
  const fixture=signer();
  const sig=await fixture.adapter.sign(payload);
  assert.equal(Buffer.from(sig).toString("base64"),goodResponse().signature);
  const state=fixture.stats();
  assert.equal(state.calls,1);
  assert.equal(state.observedMethod,"POST");
  assert.equal(state.observedUrl,
    "https://cloudkms.googleapis.com/v1/"+keyVersion+":asymmetricSign");
  assert.deepEqual(state.observedPayload,{
    data:payload.toString("base64"),dataCrc32c:String(crc32c(payload)),
  });
});
test("D29b Cloud KMS refuses unpinned key, invalid resource, bad IAM, malformed bytes",async()=>{
  assert.throws(()=>createGcpD29bEd25519Signer({
    keyId:"test",versionName:"https://attacker.example/sign",
    pinnedPublicKeySpkiPem:pinned,accessToken:async()=>token,
    fetcher:async()=>({ok:true,status:200,json:async()=>goodResponse()}),
  }),D29bKmsError);
  const fixture=signer();
  await rejects(()=>fixture.adapter.sign(new Uint8Array(0)),"D29B_GCP_KMS_DATA_INVALID");
  assert.equal(fixture.stats().calls,0);
  const failedAuth=signer(undefined,{token:async()=>"bad"});
  await rejects(()=>failedAuth.adapter.sign(payload),"D29B_GCP_KMS_IDENTITY_UNAVAILABLE");
  assert.equal(failedAuth.stats().calls,0);
  await rejects(()=>signer(undefined,{rejected:true}).adapter.sign(payload),
    "D29B_GCP_KMS_SIGN_REFUSED");
});
test("D29b Cloud KMS rejects changed key name, corruption and forged public key",async()=>{
  for(const override of [
    {name:"projects/attacker/locations/global/keyRings/x/cryptoKeys/y/cryptoKeyVersions/1"},
    {verifiedDataCrc32c:false},
    {signatureCrc32c:"1"},
    {protectionLevel:"HSM"},
    {signature:Buffer.alloc(64,6).toString("base64")},
  ]) {
    const f=signer(override);
    await assert.rejects(()=>f.adapter.sign(payload),
      (e:unknown)=>e instanceof D29bKmsError);
  }
  await rejects(()=>signer(undefined,{key:otherKey.export({type:"spki",format:"pem"}).toString()})
    .adapter.sign(payload),"D29B_GCP_KMS_PINNED_PUBLIC_KEY_MISMATCH");
});
test("D29b Cloud KMS denies network outages, never silently supplies a local private key",async()=>{
  const f=createGcpD29bEd25519Signer({
    keyId:"test",versionName:keyVersion,pinnedPublicKeySpkiPem:pinned,
    accessToken:async()=>token,fetcher:async()=>{throw new Error("unavailable");},
  });
  await rejects(()=>f.sign(payload),"D29B_GCP_KMS_NETWORK_UNAVAILABLE");
});
