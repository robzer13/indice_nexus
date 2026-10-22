import assert from "node:assert/strict";
import test from "node:test";
import {generateKeyPairSync,sign,verify} from "node:crypto";
import {
 createD29bGoogleKmsEd25519Adapter,D29bKmsError,
 type D29bKmsTransport,
} from "../lib/orotitan-equity/v3/mcp-d29b-google-kms-ed25519-v1";

const path="projects/orotitan-demo/locations/europe-west1/keyRings/d29b/cryptoKeys/receipt-sign/cryptoKeyVersions/1";
const {privateKey,publicKey}=generateKeyPairSync("ed25519");
const signedInput=Buffer.from("OROTITAN:D29B:D2_HANDOFF:V1\n"+ "x".repeat(85));
const token="ya29."+"a".repeat(50);

test("D29b KMS: sign exact raw Ed25519 message with pinned key-version URL",async()=>{
  let requests=0;
  const adapter=createD29bGoogleKmsEd25519Adapter({keyId:"d2-2026-10",cryptoKeyVersion:path},{
    accessToken:async()=>token,
    request:async(url,arg)=>{
      requests++;
      assert.equal(url,"https://cloudkms.googleapis.com/v1/"+path+":asymmetricSign");
      assert.equal(arg.headers.Authorization,"Bearer "+token);
      assert.equal(arg.headers["Content-Type"],"application/json");
      const body=JSON.parse(arg.body);
      assert.deepEqual(Object.keys(body),["data"]);
      assert.equal(Buffer.from(body.data,"base64").toString("utf8"),signedInput.toString("utf8"));
      return {ok:true,status:200,json:async()=>({
        signature:sign(null,signedInput,privateKey).toString("base64"),
      })};
    },
  });
  const signature=await adapter.sign(signedInput);
  assert.equal(adapter.keyId,"d2-2026-10");
  assert.equal(verify(null,signedInput,publicKey,signature),true);
  assert.equal(requests,1);
});

test("D29b KMS: reject redirectable or attacker-controlled endpoints",()=>{
  for(const bad of [
    "https://evil.example/v1/projects/abc",
    "projects/orotitan-demo/locations/europe-west1/keyRings/k/cryptoKeys/key/cryptoKeyVersions/latest",
    "projects/orotitan-demo/locations/europe-west1/keyRings/k/cryptoKeys/key/cryptoKeyVersions/1?x=1",
    "../projects/orotitan-demo/locations/europe-west1/keyRings/k/cryptoKeys/key/cryptoKeyVersions/1",
  ])assert.throws(()=>createD29bGoogleKmsEd25519Adapter({
    keyId:"test",cryptoKeyVersion:bad,
  },{accessToken:async()=>token,request:async()=>({ok:true,status:200,json:async()=>({})})}),
   (e:unknown)=>e instanceof D29bKmsError && e.code==="D29B_KMS_CONFIGURATION_INVALID");
});

test("D29b KMS: malformed bearer, signatures, HTTP errors and outages fail closed",async()=>{
  async function fails(arg:{
    response?:unknown;status?:number;identity?:string;throws?:boolean;
  },expected:string) {
    const transport:D29bKmsTransport={
      accessToken:async()=>arg.identity??token,
      request:async()=>{
        if(arg.throws)throw Error("network error");
        return {ok:arg.status===undefined||arg.status===200,
          status:arg.status??200,json:async()=>arg.response};
      },
    };
    const adapter=createD29bGoogleKmsEd25519Adapter({
      keyId:"test",cryptoKeyVersion:path,
    },transport);
    await assert.rejects(()=>adapter.sign(signedInput),
     (e:unknown)=>e instanceof D29bKmsError && e.code===expected);
  }
  await fails({identity:"short"},"D29B_KMS_IDENTITY_UNAVAILABLE");
  await fails({status:403},"D29B_KMS_SIGNATURE_UNAVAILABLE");
  await fails({throws:true},"D29B_KMS_REQUEST_FAILED");
  await fails({response:{signature:"notvalid"}},"D29B_KMS_SIGNATURE_INVALID");
  await fails({response:{signature:Buffer.alloc(64).toString("base64")+"?"}},
    "D29B_KMS_SIGNATURE_INVALID");
  await fails({response:{signature:Buffer.alloc(48).toString("base64")}},
    "D29B_KMS_SIGNATURE_INVALID");
});
