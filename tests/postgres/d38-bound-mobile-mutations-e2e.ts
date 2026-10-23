import assert from "node:assert/strict";
import {execFileSync} from "node:child_process";
import {createHash,randomBytes,randomUUID} from "node:crypto";
import {createLocalJWKSet,exportJWK,generateKeyPair,SignJWT} from "jose";
import {
 issueD38BoundProof,d38PreviewBinding,d38VaultBinding,d38CapabilityBinding,
 type D38Operation,type D38Proof,type D38SessionPolicy,
} from "../../lib/orotitan-equity/v3/mobile-command-bound-session-operations-v1";
import {issueServerPrivateCommandCapability,
 type IndependentCommandPolicy} from "../../lib/orotitan-equity/v3/mobile-command-oauth-capabilities-v1";
import {
 sealD34ServerCapability,openD34ServerCapability,
} from "../../lib/orotitan-equity/v3/mobile-command-private-encrypted-vault-v1";
import type {MobileActor,CommandKind,DraftCommand} from "../../lib/orotitan-equity/v3/mobile-command-intents-candidate-v1";
import type {McpOAuthConfig} from "../../lib/orotitan-equity/v3/mcp-oauth-v1";

const db=process.argv[2];
if(!/^orotitan_d32_mobile_[0-9]+$/.test(db??"") ||
 !["localhost","127.0.0.1","::1"].includes(process.env.PGHOST??"127.0.0.1")||
 process.env.PGPORT!=="5432"||process.env.PGUSER!=="postgres")
 throw new Error("D38_DISPOSABLE_PG17_ONLY");
const resource="https://orotitan-chatgpt-mcp-pilot.vercel.app/api/orotitan/mcp";
const actor:MobileActor={
 subject:"d38-reviewed-operator",clientId:"d38-approved-client",resource,
};
const sessionId=randomUUID(),macKey=Buffer.alloc(32,0x37);
const signKey=randomBytes(32),aesKey=randomBytes(32),ZERO="0".repeat(64);
const company={name:"D38 Synthetic Company",ticker:"D38X",exchange:"XAMS",isin:"NL6666666666"};
const research={issuerId:"00000000-0000-4000-8000-000000000002",
 name:"Beta Display",ticker:"BET",exchange:"XPAR",isin:"FR1234567890"};
const oauth:McpOAuthConfig={
 issuer:"https://oauth-d38.example.org",audience:resource,
 tokenAudience:"authenticated",jwksUri:"https://oauth-d38.example.org/jwks",
 resourceMetadataUrl:"https://orotitan-chatgpt-mcp-pilot.vercel.app/.well-known/oauth-protected-resource",
 allowedSubjects:new Set([actor.subject]),
 allowedClients:new Set([actor.clientId]),
};
const str=(x:string)=>"convert_from(decode('"+Buffer.from(x).toString("base64")+"','base64'),'UTF8')";
const uid=(x:string)=>str(x)+"::uuid";
const bytes=(x:string)=>"decode("+str(x)+",'base64')";
function sql(s:string):string{
 return execFileSync("psql",["-X","-q","-A","-t","-v","ON_ERROR_STOP=1","-d",db,"-c",s],
  {encoding:"utf8",timeout:10000,stdio:["ignore","pipe","pipe"]}).trim();
}
function limited(s:string){return sql("SET ROLE orotitan_d38_mutation_gateway; "+s+"; RESET ROLE;");}
function call(p:D38Proof,fn:string,args:string[]=[]){
 return limited("SELECT orotitan_mobile_private."+fn+"("+
 [str(JSON.stringify(p.claims))+"::jsonb",str(p.macSha256),...args].join(",")+")::text");
}
const fingerprint=(kind:CommandKind,payload:Record<string,string>)=>
 createHash("sha256").update(JSON.stringify([kind,Object.keys(payload).sort().map(k=>[k,payload[k]])])).digest("hex");
const previews=new Map<string,DraftCommand>(),approved=new Set<string>();
let active=true;
const keyFor=(op:D38Operation,id:string,digest:string)=>[op,id,digest].join(":");
const policy:D38SessionPolicy={
 async isSessionActive({subject,sessionId:id}){return active&&subject===actor.subject&&id===sessionId;},
 async allows({actor:a,sessionId:id,operation,previewId,bindingSha256}){
  return a.subject===actor.subject&&a.clientId===actor.clientId&&a.resource===resource&&
   id===sessionId&&previews.has(previewId)&&approved.has(keyFor(operation,previewId,bindingSha256));
 },
};
const d31Policy:IndependentCommandPolicy={
 async readPreview(id){return previews.get(id)??null;},
 async allows({actor:a,previewId:id,kind,fingerprintSha256:sha}){
  const p=previews.get(id);
  return !!p&&a.subject===actor.subject&&a.clientId===actor.clientId&&
   a.resource===resource&&p.kind===kind&&p.fingerprintSha256===sha;
 },
};
async function reject(fn:()=>Promise<unknown>|unknown){await assert.rejects(async()=>fn());}
async function main(){
 assert.match(sql("SHOW server_version_num"),/^17[0-9]{4}$/);
 assert.equal(sql("SELECT encode(secret,'hex') FROM orotitan_mobile_private.d37_attestation_keys"),
  macKey.toString("hex"));
 const initialCompany=Number(sql("SELECT count(*) FROM orotitan_mobile_private.command_requests"));
 const initialResearch=Number(sql("SELECT count(*) FROM orotitan_mobile_private.research_requests_v4"));
 sql("INSERT INTO orotitan_mobile_private.d38_test_active_sessions(subject,session_id) VALUES("+
  str(actor.subject)+","+uid(sessionId)+")");
 const {publicKey,privateKey}=await generateKeyPair("ES256",{extractable:true});
 const jwk=await exportJWK(publicKey);
 const keys=createLocalJWKSet({keys:[{...jwk,kid:"d38-test",alg:"ES256",use:"sig"}]});
 const bearer="Bearer "+await new SignJWT({
  role:"authenticated",client_id:actor.clientId,resource,session_id:sessionId,
 }).setProtectedHeader({alg:"ES256",kid:"d38-test"})
  .setIssuer(oauth.issuer).setAudience("authenticated")
  .setSubject(actor.subject).setIssuedAt().setExpirationTime("10m").sign(privateKey);
 async function proof(op:D38Operation,id:string,binding:string,nowMs?:number){
  approved.add(keyFor(op,id,binding));
  return issueD38BoundProof({bearer,oauth,keys,operation:op,previewId:id,
   bindingSha256:binding,privateMacKey:macKey,policy,nowMs});
 }
 async function create(kind:CommandKind,payload:Record<string,string>){
  const previewId=randomUUID(),fp=fingerprint(kind,payload),expiresAtMs=Date.now()+75_000;
  const p:DraftCommand={previewId,kind,actor,payload,fingerprintSha256:fp,expiresAtMs};
  previews.set(previewId,p);
  const signed=await proof("create_preview",previewId,d38PreviewBinding(kind,fp,expiresAtMs));
  const args=[str(kind),str(JSON.stringify(payload))+"::jsonb",str(fp),expiresAtMs+"::bigint"];
  assert.equal(call(signed,"create_oauth_bound_preview_v6",args),"true");
  await reject(()=>call(signed,"create_oauth_bound_preview_v6",args));
  return p;
 }
 async function commit(p:DraftCommand){
  const token=await issueServerPrivateCommandCapability({
   bearer,oauth,keys,previewId:p.previewId,policy:d31Policy,signingKey:signKey,
  });
  const context={previewId:p.previewId,kind:p.kind,actor,fingerprintSha256:p.fingerprintSha256};
  const sealed=sealD34ServerCapability({token,context,key:aesKey,keyVersion:"d38_test_v1"});
  const e=sealed.envelope;
  const digest=d38VaultBinding({kind:p.kind,fingerprintSha256:p.fingerprintSha256,
   capabilitySha256:sealed.capabilitySha256,keyVersion:e.keyVersion,
   nonce:Buffer.from(e.nonce,"base64"),ciphertext:Buffer.from(e.ciphertext,"base64"),
   authTag:Buffer.from(e.authTag,"base64")});
  const signed=await proof("approve_vault",p.previewId,digest);
  const args=[str(p.kind),str(p.fingerprintSha256),str(sealed.capabilitySha256),
   str(e.keyVersion),bytes(e.nonce),bytes(e.ciphertext),bytes(e.authTag)];
  await reject(()=>call(signed,"approve_oauth_bound_vault_v6",[
   ...args.slice(0,5),bytes(randomBytes(200).toString("base64")),args[6],
  ]));
  assert.equal(call(signed,"approve_oauth_bound_vault_v6",args),"true");
  const readProof=await proof("read_vault",p.previewId,ZERO);
  const env=JSON.parse(call(readProof,"read_oauth_bound_vault_v6")) as {
   keyVersion:string;nonce:string;ciphertext:string;authTag:string;
  };
  assert.equal(JSON.stringify(env).includes(token.sealed),false);
  const restored=openD34ServerCapability({envelope:env,context,key:aesKey,
   expectedSha256:sealed.capabilitySha256,expectedKeyVersion:"d38_test_v1"});
  assert.equal(restored.sealed,token.sealed);
  const operation=p.kind==="add_company"?"confirm_company":"confirm_research";
  const confirmed=await proof(operation,p.previewId,d38CapabilityBinding(restored.sealed));
  await reject(()=>call(confirmed,"confirm_oauth_bound_request_v6",[
   str("invalid."+randomBytes(32).toString("base64url")),
  ]));
  const receipt=JSON.parse(call(confirmed,"confirm_oauth_bound_request_v6",[
   str(restored.sealed),
  ])) as {requestId:string;kind:string;state:string;duplicate:boolean};
  assert.equal(receipt.kind,p.kind);
  assert.equal(receipt.state,"REQUESTED_NOT_EXECUTED");
  assert.equal(receipt.duplicate,false);
  await reject(()=>call(confirmed,"confirm_oauth_bound_request_v6",[str(restored.sealed)]));
 }
 const companyPreview=await create("add_company",company);
 // Pre-issued valid signed proofs fail after revocation at the DB boundary.
 const pending=await proof("read_vault",companyPreview.previewId,ZERO);
 sql("UPDATE orotitan_mobile_private.d38_test_active_sessions "+
 "SET active=false,revoked_at=clock_timestamp() WHERE subject="+
 str(actor.subject)+" AND session_id="+uid(sessionId));
 active=false;
 await reject(()=>proof("read_vault",companyPreview.previewId,ZERO));
 await reject(()=>call(pending,"read_oauth_bound_vault_v6"));
 assert.equal(sql("SELECT count(*) FROM orotitan_mobile_private.d37_consumed_nonces WHERE nonce="+
 str(pending.claims.nonce)),"0");
 // Reenable synthetic session for remaining local-only tests.
 sql("UPDATE orotitan_mobile_private.d38_test_active_sessions "+
 "SET active=true,revoked_at=NULL WHERE subject="+str(actor.subject)+
 " AND session_id="+uid(sessionId));
 active=true;
 await commit(companyPreview);
 assert.equal(Number(sql("SELECT count(*) FROM orotitan_mobile_private.command_requests")),initialCompany+1);
 const researchPreview=await create("request_research",research);
 // Scope is seeded by the fixture owner; its D37-bound issuance was already
 // verified separately. This test covers the final D38 Research confirmation.
 sql("INSERT INTO orotitan_mobile_private.research_preview_scope_v4("+
 "preview_id,issuer_id,contract_pack_sha256,cutoff_date) VALUES("+
 [uid(researchPreview.previewId),uid(research.issuerId),str("e".repeat(64)),"current_date"].join(",")+")");
 await commit(researchPreview);
 assert.equal(Number(sql("SELECT count(*) FROM orotitan_mobile_private.research_requests_v4")),initialResearch+1);
 for(const [role,fn] of [
  ["orotitan_d34_preview_issuer","create_preview_v3(uuid,text,text,text,text,jsonb,text,timestamptz)"],
  ["orotitan_d34_vault_issuer","approve_and_store_capability_v3(uuid,text,text,text,text,text,text,text,bytea,bytea,bytea)"],
  ["orotitan_d34_vault_reader","read_encrypted_capability_v3(uuid,text,text,text)"],
  ["orotitan_d32_command_submitter","confirm_approved_request_v2(uuid,text)"],
  ["orotitan_d36_research_submitter","confirm_scoped_research_v4(uuid,text)"],
  ["authenticated","confirm_oauth_bound_request_v6(jsonb,text,text)"],
 ]){
  assert.equal(sql("SELECT has_function_privilege("+str(role)+","+
   str("orotitan_mobile_private."+fn)+",'EXECUTE')"),"f");
 }
 assert.equal(sql("SELECT has_table_privilege('orotitan_d38_mutation_gateway',"+
 "'orotitan_mobile_private.d38_test_active_sessions','SELECT')"),"f");
 assert.equal(sql("SELECT to_regclass('public.orotitan_runs') IS NULL"),"t");
 console.log("D38 signed OAuth -> revoked-session SQL check -> preview/vault/confirmation PASS (local PG17).");
}
main().catch(e=>{console.error("D38 test failed:",e instanceof Error?e.message:String(e));process.exitCode=1;});
