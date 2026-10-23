/**
 * D37 / disposable PG17 only:
 * signed ES256 OAuth JWT + real signed session_id -> independent live-session
 * policy -> short HMAC attestation -> DB one-use verification -> private SQL.
 * NO network/hosted Supabase, NO real companies, NO Research execution.
 */
import assert from "node:assert/strict";
import {execFileSync} from "node:child_process";
import {createHash,randomUUID} from "node:crypto";
import {createLocalJWKSet,exportJWK,generateKeyPair,SignJWT} from "jose";
import {
  issueD37OAuthSessionProof,D37AuthorizationError,
  type D37ServerProof,type D37IndependentSessionPolicy,
} from "../../lib/orotitan-equity/v3/mobile-command-verified-db-session-proof-v1";
import type {McpOAuthConfig} from "../../lib/orotitan-equity/v3/mcp-oauth-v1";

const db=process.argv[2];
if(!/^orotitan_d32_mobile_[0-9]+$/.test(db??"") ||
 !["localhost","127.0.0.1","::1"].includes(process.env.PGHOST??"127.0.0.1") ||
 process.env.PGPORT!=="5432"||process.env.PGUSER!=="postgres")
 throw new Error("D37_DISPOSABLE_LOOPBACK_PG17_ONLY");

const resource="https://orotitan-chatgpt-mcp-pilot.vercel.app/api/orotitan/mcp";
const SUBJECT="person-d-operator",OTHER_SUBJECT="person-e-operator";
const CLIENT="approved-mobile-client",OTHER_CLIENT="approved-desktop-client";
const sessionId=randomUUID(),secondSessionId=randomUUID();
const previewId=randomUUID();
const issuerId="00000000-0000-4000-8000-000000000002";
const contractSha256="d".repeat(64);
const cutoff=new Date().toISOString().slice(0,10);
const key=Buffer.alloc(32,0x37); // SYNTHETIC LOCAL TEST KEY only.
const payload={
 issuerId,name:"Beta Display",ticker:"BET",exchange:"XPAR",isin:"FR1234567890",
};
const fingerprint=createHash("sha256").update(JSON.stringify([
 "request_research",Object.keys(payload).sort().map(k=>[k,payload[k as keyof typeof payload]])
])).digest("hex");
const oauth:McpOAuthConfig={
 issuer:"https://oauth-d37.example.org",audience:resource,
 tokenAudience:"authenticated",jwksUri:"https://oauth-d37.example.org/jwks",
 resourceMetadataUrl:"https://orotitan-chatgpt-mcp-pilot.vercel.app/.well-known/oauth-protected-resource",
 allowedSubjects:new Set([SUBJECT,OTHER_SUBJECT]),
 allowedClients:new Set([CLIENT,OTHER_CLIENT]),
};
const str=(s:string)=>"convert_from(decode('"+
 Buffer.from(s,"utf8").toString("base64")+"','base64'),'UTF8')";
const uuid=(s:string)=>str(s)+"::uuid";
function sql(body:string):string{
 return execFileSync("psql",["-X","-q","-A","-t","-v","ON_ERROR_STOP=1","-d",db,"-c",body],
 {encoding:"utf8",timeout:10000,stdio:["ignore","pipe","pipe"]}).trim();
}
function limited(role:string,statement:string):string{
 if(!/^orotitan_d(34|36|37)_[a-z0-9_]+$/.test(role))throw Error("D37_ROLE_INVALID");
 return sql("SET ROLE "+role+"; "+statement+"; RESET ROLE;");
}
function call(proof:D37ServerProof,action:"read"|"scope"){
 const routine=action==="read"?"read_oauth_bound_preview_v5":"issue_oauth_bound_scope_v5";
 return limited("orotitan_d37_session_gateway",
  "SELECT orotitan_mobile_private."+routine+"("+
  str(JSON.stringify(proof.claims))+"::jsonb,"+str(proof.macSha256)+")::text");
}
const policy:D37IndependentSessionPolicy={
 async isSessionActive({subject,sessionId:id}){
  return subject===SUBJECT&&id===sessionId;
 },
 async allows({actor,sessionId:id,operation,previewId:idPreview,issuerId:idIssuer,
  contractSha256:pack,cutoff:day}){
  if(actor.subject!==SUBJECT || actor.clientId!==CLIENT ||
    actor.resource!==resource || id!==sessionId || idPreview!==previewId)
   return false;
  if(operation==="read_preview")return idIssuer==="00000000-0000-0000-0000-000000000000";
  return idIssuer===issuerId&&pack===contractSha256&&day===cutoff;
 },
};
async function rejects(run:()=>Promise<unknown>|unknown){
 await assert.rejects(async()=>run());
}
async function main(){
 assert.match(sql("SHOW server_version_num"),/^17[0-9]{4}$/);
 // Local superuser only seeds synthetic preview and key; D37 runtime roles
 // NEVER receive key/table SELECT and must use proof-guarded functions.
 sql("INSERT INTO orotitan_mobile_private.d37_attestation_keys(key_id,secret)"+
 " VALUES('local_v1',decode("+str(key.toString("hex"))+",'hex'))");
 sql("INSERT INTO orotitan_mobile_private.previews("+
 "preview_id,kind,actor_subject,actor_client,actor_resource,payload,"+
 "fingerprint_sha256,expires_at) VALUES("+
 [uuid(previewId),str("request_research"),str(SUBJECT),str(CLIENT),str(resource),
 str(JSON.stringify(payload))+"::jsonb",str(fingerprint),
 "clock_timestamp()+interval '100 seconds'"].join(",")+")");
 const {publicKey,privateKey}=await generateKeyPair("ES256",{extractable:true});
 const jwk=await exportJWK(publicKey);
 const keys=createLocalJWKSet({keys:[{...jwk,kid:"d37-fixture",alg:"ES256",use:"sig"}]});
 async function bearer({
  subject=SUBJECT,client=CLIENT,session=sessionId,withSession=true,
  resourceClaim=resource,
 }:{subject?:string;client?:string;session?:string;withSession?:boolean;
  resourceClaim?:string}={}){
   return "Bearer "+await new SignJWT({
    role:"authenticated",client_id:client,resource:resourceClaim,
    ...(withSession?{session_id:session}:{}),
   }).setProtectedHeader({alg:"ES256",kid:"d37-fixture"})
    .setIssuer(oauth.issuer).setAudience("authenticated")
    .setSubject(subject).setIssuedAt().setExpirationTime("10m")
    .sign(privateKey);
 }
 const validBearer=await bearer();
 const issue=(operation:"read_preview"|"issue_research_scope",
  opts:Partial<{
    bearer:string;issuerId:string;contractSha256:string;cutoff:string;
    nowMs:number;privateMacKey:Uint8Array;
  }>={})=>issueD37OAuthSessionProof({
   bearer:opts.bearer??validBearer,oauth,keys,operation,
   previewId,issuerId:opts.issuerId??issuerId,
   contractSha256:opts.contractSha256??contractSha256,
   cutoff:opts.cutoff??cutoff,sessionPolicy:policy,
   privateMacKey:opts.privateMacKey??key,nowMs:opts.nowMs,
  });
 const read=await issue("read_preview");
 assert.equal(read.claims.subject,SUBJECT);
 assert.equal(read.claims.clientId,CLIENT);
 assert.equal(read.claims.sessionId,sessionId);
 const preview=JSON.parse(call(read,"read")) as {previewId:string;actor:{subject:string}};
 assert.equal(preview.previewId,previewId);
 assert.equal(preview.actor.subject,SUBJECT);
 await rejects(()=>call(read,"read")); // replay, consumed nonce
 await rejects(()=>call({...read,macSha256:"f".repeat(64)},"read"));
 const forged={...read,claims:{...read.claims,subject:OTHER_SUBJECT}};
 await rejects(()=>call(forged,"read"));
 const wrongClient={...read,claims:{...read.claims,clientId:OTHER_CLIENT}};
 await rejects(()=>call(wrongClient,"read"));
 const wrongSession={...read,claims:{...read.claims,sessionId:secondSessionId}};
 await rejects(()=>call(wrongSession,"read"));
 const changedScope={...read,claims:{...read.claims,contractSha256:"e".repeat(64)}};
 await rejects(()=>call(changedScope,"read"));
 const expiry=await issue("read_preview",{nowMs:Date.now()-70_000});
 await rejects(()=>call(expiry,"read"));
 await rejects(()=>issue("read_preview",{bearer:await bearer({subject:OTHER_SUBJECT})}));
 await rejects(()=>issue("read_preview",{bearer:await bearer({client:OTHER_CLIENT})}));
 await rejects(()=>issue("read_preview",{bearer:await bearer({session:secondSessionId})}));
 await rejects(()=>issue("read_preview",{bearer:await bearer({withSession:false})}));
 await rejects(()=>issue("issue_research_scope",{contractSha256:"f".repeat(64)}));
 await rejects(()=>issue("read_preview",{privateMacKey:Buffer.alloc(12)}));
 assert.equal(sql("SELECT count(*) FROM orotitan_mobile_private.d37_consumed_nonces"),"1");

 const scoped=await issue("issue_research_scope");
 await rejects(()=>call(scoped,"read")); // wrong operation, no nonce consumed
 assert.equal(call(scoped,"scope"),"t");
 const scope=JSON.parse(sql("SELECT json_build_object("+
 "'issuer',issuer_id::text,'contract',contract_pack_sha256,"+
 "'cutoff',cutoff_date::text)::text FROM "+
 "orotitan_mobile_private.research_preview_scope_v4 WHERE preview_id="+uuid(previewId))) as {
 issuer:string;contract:string;cutoff:string};
 assert.equal(scope.issuer,issuerId);
 assert.equal(scope.contract,contractSha256);
 assert.equal(scope.cutoff,cutoff);
 await rejects(()=>call(scoped,"scope")); // replay rejected
 assert.equal(sql("SELECT count(*) FROM orotitan_mobile_private.d37_consumed_nonces"),"2");
 // Old direct endpoints must be inaccessible to previous runtime roles.
 await rejects(()=>limited("orotitan_d34_preview_reader",
  "SELECT orotitan_mobile_private.read_preview_v3("+
  [uuid(previewId),str(SUBJECT),str(CLIENT),str(resource)].join(",")+")"));
 await rejects(()=>limited("orotitan_d36_scope_issuer",
  "SELECT orotitan_mobile_private.issue_research_scope_v4("+
  [uuid(previewId),str(SUBJECT),str(CLIENT),str(resource),
   uuid(issuerId),str(contractSha256),str(cutoff)+"::date"].join(",")+")"));
 assert.equal(sql("SELECT has_table_privilege('orotitan_d37_session_gateway',"+
 "'orotitan_mobile_private.d37_attestation_keys','SELECT')"),"f");
 assert.equal(sql("SELECT has_table_privilege('orotitan_d37_session_gateway',"+
 "'orotitan_mobile_private.d37_consumed_nonces','INSERT')"),"f");
 assert.equal(sql("SELECT has_function_privilege('authenticated',"+
 "'orotitan_mobile_private.issue_oauth_bound_scope_v5(jsonb,text)','EXECUTE')"),"f");
 assert.equal(sql("SELECT count(*) FROM orotitan_mobile_private.research_requests_v4 WHERE "+
 "actor_subject="+str(SUBJECT)),"0");
 assert.equal(sql("SELECT to_regclass('public.orotitan_runs') IS NULL"),"t");
 console.log("D37 verified ES256 session -> signed SQL proof -> scoped PG17 PASS (test only).");
}
main().catch(e=>{console.error("D37 test failed:",e instanceof D37AuthorizationError?e.code:
 e instanceof Error?e.message:"unknown");process.exitCode=1;});
