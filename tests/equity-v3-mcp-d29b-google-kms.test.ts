import assert from "node:assert/strict";
import test from "node:test";
import {generateKeyPairSync,sign,verify} from "node:crypto";
import {crc32c} from "../lib/orotitan-equity/v3/mcp-d29b-crc32c-v1";
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
      assert.deepEqual(Object.keys(body),["data","dataCrc32c"]);
      assert.equal(Buffer.from(body.data,"base64").toString("utf8"),signedInput.toString("utf8"));
      assert.equal(body.dataCrc32c,String(crc32c(signedInput)));
      assert.ok(arg.signal instanceof AbortSignal);
      const signature=sign(null,signedInput,privateKey);
      return {ok:true,status:200,json:async()=>({
        name:path,signature:signature.toString("base64"),
        signatureCrc32c:String(crc32c(signature)),
        verifiedDataCrc32c:true,protectionLevel:"SOFTWARE",
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

test("D29b CRC32C: standard Castagnoli test vector",()=>{
  assert.equal(crc32c(Buffer.from("123456789")),0xe3069283);
  assert.equal(crc32c(Buffer.alloc(0)),0);
  assert.equal(crc32c(Buffer.from("a")),0xc1d04330);
});

test("D29b KMS: integrity, protection-level and key-version failures reject signatures",async()=>{
  const signed=sign(null,signedInput,privateKey);
  const valid={
    name:path,signature:signed.toString("base64"),
    signatureCrc32c:String(crc32c(signed)),
    verifiedDataCrc32c:true,protectionLevel:"SOFTWARE",
  };
  for(const [patch,code] of [
    [{name:path+"/other"},"D29B_KMS_KEY_OR_REQUEST_INTEGRITY_INVALID"],
    [{name:"projects/other/locations/europe-west1/keyRings/k/cryptoKeys/key/cryptoKeyVersions/1"},"D29B_KMS_KEY_OR_REQUEST_INTEGRITY_INVALID"],
    [{verifiedDataCrc32c:false},"D29B_KMS_KEY_OR_REQUEST_INTEGRITY_INVALID"],
    [{protectionLevel:"HSM"},"D29B_KMS_KEY_OR_REQUEST_INTEGRITY_INVALID"],
    [{signatureCrc32c:"0"},"D29B_KMS_SIGNATURE_INTEGRITY_INVALID"],
    [{signatureCrc32c:"4294967296"},"D29B_KMS_SIGNATURE_INTEGRITY_INVALID"],
    [{signatureCrc32c:"-1"},"D29B_KMS_SIGNATURE_INTEGRITY_INVALID"],
    [{signatureCrc32c:null},"D29B_KMS_SIGNATURE_INTEGRITY_INVALID"],
  ] as [Record<string,unknown>,string][]) {
    const adapter=createD29bGoogleKmsEd25519Adapter({
      keyId:"test",cryptoKeyVersion:path,
    },{accessToken:async()=>token,request:async()=>({
      status:200,ok:true,json:async()=>({...valid,...patch}),
    })});
    await assert.rejects(()=>adapter.sign(signedInput),
      (err:unknown)=>err instanceof D29bKmsError && err.code===code,
      "expected "+code);
  }
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
