/**
 * D34 disposable PG17 integration, NOT imported by MCP routes.
 * Signed ES256 OAuth -> D30 preview using restricted D34 preview issuer ->
 * D34 preview reader -> D31 HMAC capability -> AES-256-GCM vault issuer ->
 * D34 vault reader/decrypt -> D32 confirmed request (no Research execution).
 */
import assert from "node:assert/strict";
import {execFileSync} from "node:child_process";
import {createHash,randomBytes,randomUUID} from "node:crypto";
import {createLocalJWKSet,exportJWK,generateKeyPair,SignJWT} from "jose";
import {
  previewAddCompany, type DraftCommand, type MobileActor, type MobileCommandPrivatePort,
} from "../../lib/orotitan-equity/v3/mobile-command-intents-candidate-v1";
import {
  issueServerPrivateCommandCapability,confirmWithServerPrivateCapability,
  type IndependentCommandPolicy,
} from "../../lib/orotitan-equity/v3/mobile-command-oauth-capabilities-v1";
import {
  sealD34ServerCapability,openD34ServerCapability,
} from "../../lib/orotitan-equity/v3/mobile-command-private-encrypted-vault-v1";
import type {McpOAuthConfig} from "../../lib/orotitan-equity/v3/mcp-oauth-v1";

const db=process.argv[2];
if(!/^orotitan_d32_mobile_[0-9]+$/.test(db??"")||
  !["localhost","127.0.0.1","::1"].includes(process.env.PGHOST??"127.0.0.1")||
  process.env.PGPORT!=="5432"||process.env.PGUSER!=="postgres")
  throw new Error("D34 requires disposable loopback PG17 fixture");
const resource="https://orotitan-chatgpt-mcp-pilot.vercel.app/api/orotitan/mcp";
const actor:MobileActor={
  subject:"d34-reviewed-operator",clientId:"d34-approved-client",resource,
};
const company={name:"D34 Synthetic Preview",ticker:"D34X",
  exchange:"XAMS",isin:"NL5555555555"};
const oauth:McpOAuthConfig={
  issuer:"https://oauth-d34.example.org",audience:resource,
  tokenAudience:"authenticated",jwksUri:"https://oauth-d34.example.org/jwks",
  resourceMetadataUrl:"https://orotitan-chatgpt-mcp-pilot.vercel.app/.well-known/oauth-protected-resource",
  allowedSubjects:new Set([actor.subject]),
  allowedClients:new Set([actor.clientId,"another-approved-client"]),
};
const signingKey=randomBytes(32),aesKey=randomBytes(32);
const t=(value:string)=>"convert_from(decode('"+
  Buffer.from(value,"utf8").toString("base64")+"','base64'),'UTF8')";
const u=(value:string)=>t(value)+"::uuid";
const b=(value:string)=>"decode("+t(value)+",'base64')";
function query(sql:string):string{
 return execFileSync("psql",["-X","-q","-A","-t","-v","ON_ERROR_STOP=1",
  "-d",db,"-c",sql],{encoding:"utf8",timeout:10000,
  stdio:["ignore","pipe","pipe"]}).trim();
}
function privateCall(role:string,sql:string):string {
 if(!/^orotitan_d(32|34)_[a-z0-9_]+$/.test(role))throw new Error("role denied");
 return query("SET ROLE "+role+"; "+sql+"; RESET ROLE;");
}
const backend:Omit<MobileCommandPrivatePort,"atomicConfirm">={
 async authorize(a){return a.subject===actor.subject &&
   a.clientId===actor.clientId&&a.resource===actor.resource;},
 async lookupCompany(c){assert.equal(c.isin,company.isin);return "NEW_UNAMBIGUOUS";},
 async lookupResearchTarget(){return null;},
 async storePreview(p:DraftCommand){
  const x=privateCall("orotitan_d34_preview_issuer",
    "SELECT orotitan_mobile_private.create_preview_v3("+
    [u(p.previewId),t(p.kind),t(p.actor.subject),t(p.actor.clientId),
      t(p.actor.resource),t(JSON.stringify(p.payload))+"::jsonb",
      t(p.fingerprintSha256),"to_timestamp("+p.expiresAtMs+"/1000.0)"].join(",")+")");
  assert.equal(x,"t");
 },
};
function privatePreview(id:string,subject=actor.subject,client=actor.clientId){
 const result=privateCall("orotitan_d34_preview_reader",
  "SELECT orotitan_mobile_private.read_preview_v3("+
  [u(id),t(subject),t(client),t(resource)].join(",")+")::text");
 return result ? JSON.parse(result) as DraftCommand:null;
}
const policy:IndependentCommandPolicy={
 async readPreview(id){return privatePreview(id);},
 async allows(c){
  const p=privatePreview(c.previewId);
  return !!p&&p.kind===c.kind&&p.fingerprintSha256===c.fingerprintSha256&&
  p.actor.subject===c.actor.subject&&p.actor.clientId===c.actor.clientId&&
  p.actor.resource===c.actor.resource;
 },
};
async function run(){
 assert.match(query("SHOW server_version_num"),/^17[0-9]{4}$/);
 const before=Number(query("SELECT count(*) FROM orotitan_mobile_private.command_requests"));
 const preview=await previewAddCompany(company,actor,backend);
 assert.equal(preview.kind,"add_company");
 assert.equal(privatePreview(preview.previewId,"forged-subject"),null);
 assert.equal(privatePreview(preview.previewId,actor.subject,"different-client"),null);
 // The limited preview reader receives a precise preview, not all rows.
 const loaded=privatePreview(preview.previewId);
 assert.equal(loaded?.fingerprintSha256,preview.fingerprintSha256);

 // D35: even the restricted SQL issuer must fail closed on extra command
 // fields or an unverified issuer/security crosswalk, despite a correct hash.
 const fingerprint=(kind:string,payload:Record<string,string>)=>
  createHash("sha256").update(JSON.stringify([
   kind,Object.keys(payload).sort().map(k=>[k,payload[k]])
  ])).digest("hex");
 const previewSql=(kind:"add_company"|"request_research",
   payload:Record<string,string>)=>
  "SELECT orotitan_mobile_private.create_preview_v3("+
   [u(randomUUID()),t(kind),t(actor.subject),t(actor.clientId),
    t(resource),t(JSON.stringify(payload))+"::jsonb",
    t(fingerprint(kind,payload)),"clock_timestamp()+interval '60 seconds'"].join(",")+")";
 await assert.rejects(async()=>privateCall("orotitan_d34_preview_issuer",
  previewSql("add_company",{...company,permission:"administrator"})));
 const fakeResearch={
  issuerId:"00000000-0000-4000-8000-000000000002",
  name:"Unverified Synthetic Issuer",ticker:"D35BAD",
  exchange:"XAMS",isin:"FR1234567890",
 };
 await assert.rejects(async()=>privateCall("orotitan_d34_preview_issuer",
  previewSql("request_research",fakeResearch)));
 assert.equal(privatePreview(preview.previewId)?.kind,"add_company");

 // A forged SHA cannot create another preview even through issuer role.
 const bad=preview.previewId.replace(/.$/,"0");
 await assert.rejects(async()=>privateCall("orotitan_d34_preview_issuer",
   "SELECT orotitan_mobile_private.create_preview_v3("+
   [u(bad),t(preview.kind),t(actor.subject),t(actor.clientId),t(resource),
    t(JSON.stringify(preview.payload))+"::jsonb",t("f".repeat(64)),
    "clock_timestamp()+interval '30 seconds'"].join(",")+")"));
 const {privateKey,publicKey}=await generateKeyPair("ES256",{extractable:true});
 const jwk=await exportJWK(publicKey);
 const keys=createLocalJWKSet({keys:[{...jwk,alg:"ES256",kid:"d34-key",use:"sig"}]});
 const bearer="Bearer "+await new SignJWT({
   role:"authenticated",client_id:actor.clientId,resource,
 }).setProtectedHeader({alg:"ES256",kid:"d34-key"})
  .setIssuer(oauth.issuer).setAudience(oauth.tokenAudience)
  .setSubject(actor.subject).setIssuedAt().setExpirationTime("10m").sign(privateKey);
 const token=await issueServerPrivateCommandCapability({
  bearer,oauth,keys,previewId:preview.previewId,policy,signingKey,
 });
 const context={previewId:preview.previewId,kind:preview.kind,
   actor,fingerprintSha256:preview.fingerprintSha256};
 const sealed=sealD34ServerCapability({
  token,context,key:aesKey,keyVersion:"local_test_v1",
 });
 assert.equal(Number(query("SELECT count(*) FROM orotitan_mobile_private.command_requests")),before);
 const x=privateCall("orotitan_d34_vault_issuer",
   "SELECT orotitan_mobile_private.approve_and_store_capability_v3("+
   [u(preview.previewId),t(preview.kind),t(actor.subject),t(actor.clientId),
     t(resource),t(preview.fingerprintSha256),t(sealed.capabilitySha256),
     t(sealed.envelope.keyVersion),b(sealed.envelope.nonce),
     b(sealed.envelope.ciphertext),b(sealed.envelope.authTag)].join(",")+")");
 assert.equal(x,"t");
 // The vault reader has no rights to the plaintext or broad underlying table.
 assert.equal(query("SELECT has_table_privilege('orotitan_d34_vault_reader',"+
  "'orotitan_mobile_private.encrypted_capability_vault','SELECT')"),"f");
 const getEncrypted=(sub=actor.subject)=>{
  const raw=privateCall("orotitan_d34_vault_reader",
   "SELECT orotitan_mobile_private.read_encrypted_capability_v3("+
    [u(preview.previewId),t(sub),t(actor.clientId),t(resource)].join(",")+")::text");
  return raw?JSON.parse(raw) as Record<string,string>:null;
 };
 assert.equal(getEncrypted("unauthorized-actor"),null);
 const envelope=getEncrypted();
 assert.ok(envelope);
 assert.equal(JSON.stringify(envelope).includes(token.sealed),false);
 const restored=openD34ServerCapability({
  envelope:{keyVersion:envelope!.keyVersion,nonce:envelope!.nonce,
    ciphertext:envelope!.ciphertext,authTag:envelope!.authTag},
  context,key:aesKey,expectedSha256:sealed.capabilitySha256,
  expectedKeyVersion:"local_test_v1",
 });
 assert.equal(restored.sealed,token.sealed);
 const queue:MobileCommandPrivatePort={
   ...backend,
   async atomicConfirm(q){
     const raw=privateCall("orotitan_d32_command_submitter",
      "SELECT json_build_object('requestId',r.request_id::text,'kind',r.kind,"+
      "'state',r.state,'duplicate',r.duplicate)::text "+
      "FROM orotitan_mobile_private.confirm_approved_request_v2("+
      [u(q.previewId),t(restored.sealed)].join(",")+") r");
     const receipt=JSON.parse(raw) as {requestId:string;kind:"add_company"|"request_research";
       state:"REQUESTED_NOT_EXECUTED";duplicate:boolean};
     return {requestId:receipt.requestId,kind:receipt.kind,
      fingerprintSha256:q.fingerprintSha256,state:receipt.state,
      duplicate:receipt.duplicate,confirmed:true,
      registryRunId:null,registryMutationPerformed:false,
      companyInserted:false,researchStarted:false};
   },
 };
 const res=await confirmWithServerPrivateCapability({
  bearer,oauth,keys,token:restored,signingKey,
  previewId:preview.previewId,kind:preview.kind,
  fingerprintSha256:preview.fingerprintSha256,confirmation:"CONFIRM",
  policy,queue,
 });
 assert.equal(res.state,"REQUESTED_NOT_EXECUTED");
 assert.equal(res.researchStarted,false);
 assert.equal(Number(query("SELECT count(*) FROM orotitan_mobile_private.command_requests")),before+1);
 assert.equal(getEncrypted(),null); // vault read disabled after confirmation
 assert.equal(privatePreview(preview.previewId),null); // post-confirm preview is closed
 assert.equal(query("SELECT to_regclass('public.orotitan_runs') IS NULL"),"t");
 console.log("D34 limited preview + encrypted AES-GCM vault -> D32 confirmation PASSED (local PG17 only).");
}
run().catch(e=>{console.error("D34 offline integration failure:",e instanceof Error?e.message:"blocked");process.exitCode=1;});
