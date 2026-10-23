import assert from "node:assert/strict";
import {randomBytes,randomUUID} from "node:crypto";
import test from "node:test";
import {
 sealD34ServerCapability,openD34ServerCapability,D34VaultError,
 type D34VaultContext,
} from "../lib/orotitan-equity/v3/mobile-command-private-encrypted-vault-v1";
const ctx:D34VaultContext={
  previewId:randomUUID(),kind:"add_company",
  actor:{subject:"d34-subject",clientId:"d34-client",resource:
    "https://orotitan-chatgpt-mcp-pilot.vercel.app/api/orotitan/mcp"},
  fingerprintSha256:"a".repeat(64),
};
const TOKEN={sealed:randomBytes(200).toString("base64url")+"."+randomBytes(32).toString("base64url")};
function expectDenied(fn:()=>unknown,code:string){
 assert.throws(fn,(e:unknown)=>e instanceof D34VaultError&&e.code===code);
}
test("D34 AES-256-GCM: exact sealed capability roundtrip with actor-specific AAD",()=>{
 const key=randomBytes(32);
 const stored=sealD34ServerCapability({token:TOKEN,context:ctx,key,keyVersion:"test_v1"});
 assert.equal(stored.envelope.keyVersion,"test_v1");
 assert.notEqual(stored.envelope.ciphertext,TOKEN.sealed);
 assert.equal(JSON.stringify(stored).includes(TOKEN.sealed),false);
 const recovered=openD34ServerCapability({envelope:stored.envelope,context:ctx,key,
  expectedSha256:stored.capabilitySha256,expectedKeyVersion:"test_v1"});
 assert.equal(recovered.sealed,TOKEN.sealed);
});
test("D34 AES-GCM: wrong key, actor, client, kind, preview, fingerprint fail",()=>{
 const key=randomBytes(32);
 const stored=sealD34ServerCapability({token:TOKEN,context:ctx,key,keyVersion:"test_v1"});
 const open=(context=ctx,k=key)=>openD34ServerCapability({
  envelope:stored.envelope,context,key:k,
  expectedSha256:stored.capabilitySha256,expectedKeyVersion:"test_v1"});
 expectDenied(()=>open(ctx,randomBytes(32)),"D34_DECRYPTION_DENIED");
 for(const change of [
  {...ctx,previewId:randomUUID()},
  {...ctx,kind:"request_research" as const},
  {...ctx,actor:{...ctx.actor,subject:"changed"}},
  {...ctx,actor:{...ctx.actor,clientId:"changed"}},
  {...ctx,fingerprintSha256:"b".repeat(64)},
 ]) expectDenied(()=>open(change),"D34_DECRYPTION_DENIED");
});
test("D34 AES-GCM: corrupted ciphertext/tag and replay to wrong version fail",()=>{
 const key=randomBytes(32);
 const stored=sealD34ServerCapability({token:TOKEN,context:ctx,key,keyVersion:"test_v1"});
 const read=(envelope=stored.envelope,version="test_v1",digest=stored.capabilitySha256)=>
   openD34ServerCapability({envelope,context:ctx,key,
    expectedSha256:digest,expectedKeyVersion:version});
 expectDenied(()=>read({...stored.envelope,authTag:randomBytes(16).toString("base64")}),
  "D34_DECRYPTION_DENIED");
 expectDenied(()=>read({...stored.envelope,ciphertext:randomBytes(250).toString("base64")}),
  "D34_DECRYPTION_DENIED");
 expectDenied(()=>read(stored.envelope,"test_v2"),"D34_KEY_VERSION_OR_DIGEST_DENIED");
 expectDenied(()=>read(stored.envelope,"test_v1","b".repeat(64)),
  "D34_CAPABILITY_DIGEST_DENIED");
});
test("D34 AES-GCM: invalid keys, noncanonical base64, and untrusted contexts rejected",()=>{
 const key=randomBytes(32);
 expectDenied(()=>sealD34ServerCapability({token:TOKEN,context:ctx,
   key:randomBytes(16),keyVersion:"test_v1"}),"D34_VAULT_AES_KEY_REQUIRED");
 const stored=sealD34ServerCapability({token:TOKEN,context:ctx,key,keyVersion:"test_v1"});
 expectDenied(()=>openD34ServerCapability({envelope:{
  ...stored.envelope,nonce:"!!!"},context:ctx,key,
  expectedSha256:stored.capabilitySha256,expectedKeyVersion:"test_v1"}),
  "D34_ENVELOPE_FORMAT_DENIED");
 expectDenied(()=>sealD34ServerCapability({token:TOKEN,
  context:{...ctx,actor:{...ctx.actor,resource:"https://attacker.invalid"}},
  key,keyVersion:"test_v1"}),"D34_CONTEXT_INVALID");
});
