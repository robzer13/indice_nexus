/**
 * D33 local PG17 end-to-end exercise, never included in production bundles.
 * Real ES256 OAuth JWT -> D31 HMAC authorization -> D32 independent approval
 * -> D32 EXECUTE-only command confirmation -> actual durable D30 PG receipt.
 *
 * Superuser psql is used solely to seed/read SYNTHETIC local fixtures.
 * All approval/confirmation calls use SQL SET ROLE restricted D32 principals.
 */
import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import { randomBytes, randomUUID } from "node:crypto";
import {
  createLocalJWKSet, exportJWK, generateKeyPair, SignJWT,
  type JWTVerifyGetKey,
} from "jose";
import {
  previewAddCompany, previewRequestResearch,
  type DraftCommand, type MobileActor, type CompanyCandidate,
  type ResearchTarget, type MobileCommandPrivatePort,
} from "../../lib/orotitan-equity/v3/mobile-command-intents-candidate-v1";
import {
  issueAndRegisterD33PrivateApproval,confirmD33AuthorizedRequest,
  type D33PrivateDatabaseProcedures,
} from "../../lib/orotitan-equity/v3/mobile-command-private-pg-bridge-candidate-v1";
import type { IndependentCommandPolicy } from "../../lib/orotitan-equity/v3/mobile-command-oauth-capabilities-v1";
import type { McpOAuthConfig } from "../../lib/orotitan-equity/v3/mcp-oauth-v1";

const db = process.argv[2];
if (!/^orotitan_d32_mobile_[0-9]+$/.test(db??"") ||
  !["127.0.0.1","localhost","::1"].includes(process.env.PGHOST??"127.0.0.1") ||
  process.env.PGPORT !== "5432" || process.env.PGUSER !== "postgres") {
  throw new Error("D33 refuses non-disposable PostgreSQL 17 or remote fixture");
}
const RESOURCE="https://orotitan-chatgpt-mcp-pilot.vercel.app/api/orotitan/mcp";
const ISSUER="00000000-0000-4000-8000-000000000001";
const SUBJECT="d33-approved-operator";
const CLIENT="d33-approved-client";
const SECOND_CLIENT="d33-second-client";
const actor:MobileActor={subject:SUBJECT,clientId:CLIENT,resource:RESOURCE};
const company:CompanyCandidate={
  name:"D33 Synthetic Co",ticker:"D33X",exchange:"XAMS",isin:"NL4444444444",
};
const oauth:McpOAuthConfig={
  issuer:"https://oauth-d33.example.org",audience:RESOURCE,
  tokenAudience:"authenticated",
  jwksUri:"https://oauth-d33.example.org/jwks",
  resourceMetadataUrl:"https://orotitan-chatgpt-mcp-pilot.vercel.app/.well-known/oauth-protected-resource",
  allowedSubjects:new Set([SUBJECT]),
  allowedClients:new Set([CLIENT,SECOND_CLIENT]),
};
const secret=randomBytes(32); // ephemeral CI-only; never printed or persisted.
// Every variable is passed as strict base64-encoded text inside a SQL string
// using only [A-Za-z0-9+/=]; no caller data is interpolated as SQL syntax.
const str=(s:string)=>"convert_from(decode('"+Buffer.from(s,"utf8").toString("base64")+"','base64'),'UTF8')";
const uuid=(s:string)=>str(s)+"::uuid";
function query(sql:string):string {
  return execFileSync("psql",["-X","-q","-A","-t","-v","ON_ERROR_STOP=1",
    "-d",db,"-c",sql],{encoding:"utf8",timeout:10000,
    stdio:["ignore","pipe","pipe"]}).trim();
}
function readRow(sql:string) {
  const result=query(sql);
  if (!result) throw new Error("D33_SQL_EXPECTED_ROW");
  return JSON.parse(result) as Record<string,unknown>;
}
function countRequests() {
  return Number(query("SELECT count(*) FROM orotitan_mobile_private.command_requests"));
}
function updateResearchFixture() {
  query("UPDATE public.securities SET isin='GB1234567890' WHERE issuer_id="+uuid(ISSUER));
}
class PostgresPreview implements Omit<MobileCommandPrivatePort,"atomicConfirm"> {
  async authorize(value:MobileActor):Promise<boolean>{
    return value.subject===SUBJECT && oauth.allowedClients.has(value.clientId) &&
      value.resource===RESOURCE;
  }
  async lookupCompany(input:CompanyCandidate) {
    // This is a SYNTHETIC authoritative resolver fixture, NOT a market feed.
    assert.equal(input.isin,company.isin);
    const n=Number(query("SELECT count(*) FROM public.companies WHERE "+
      "upper(ticker)="+str(input.ticker)+" AND upper(exchange)="+str(input.exchange)));
    return n ? "ALREADY_TRACKED" as const : "NEW_UNAMBIGUOUS" as const;
  }
  async lookupResearchTarget(issuerId:string):Promise<ResearchTarget|null>{
    const s=query("SELECT json_build_object("+
      "'issuerId',i.issuer_id::text,'displayName',i.display_name,"+
      "'ticker',s.ticker,'exchange',s.exchange,'isin',s.isin)::text "+
      "FROM public.issuers i JOIN public.securities s USING (issuer_id) "+
      "WHERE i.issuer_id="+uuid(issuerId)+" LIMIT 1");
    if(!s)return null;
    const row=JSON.parse(s) as ResearchTarget;
    return {...row,eligibleForRequest:row.isin!=="UNKNOWN"};
  }
  async storePreview(p:DraftCommand) {
    const sql="INSERT INTO orotitan_mobile_private.previews "+
      "(preview_id,kind,actor_subject,actor_client,actor_resource,"+
      "payload,fingerprint_sha256,expires_at) VALUES ("+
      [uuid(p.previewId),str(p.kind),str(p.actor.subject),
        str(p.actor.clientId),str(p.actor.resource),
        str(JSON.stringify(p.payload))+"::jsonb",
        str(p.fingerprintSha256),
        "to_timestamp("+Number(p.expiresAtMs)+"/1000.0)"].join(",")+")";
    query(sql);
  }
}
class PrivatePolicy implements IndependentCommandPolicy {
  allowed=true;
  async readPreview(previewId:string):Promise<DraftCommand|null> {
    const sql="SELECT json_build_object("+
      "'previewId',preview_id::text,'kind',kind,"+
      "'actor',json_build_object('subject',actor_subject,"+
      "'clientId',actor_client,'resource',actor_resource),"+
      "'payload',payload,'fingerprintSha256',fingerprint_sha256,"+
      "'expiresAtMs',(extract(epoch from expires_at)*1000)::bigint)::text "+
      "FROM orotitan_mobile_private.previews WHERE preview_id="+uuid(previewId);
    const result=query(sql);
    return result ? JSON.parse(result) as DraftCommand : null;
  }
  async allows(value:{
    actor:MobileActor;kind:string;previewId:string;fingerprintSha256:string;
  }):Promise<boolean> {
    const p=await this.readPreview(value.previewId);
    return this.allowed && !!p && p.kind===value.kind &&
      p.actor.subject===value.actor.subject &&
      p.actor.clientId===value.actor.clientId &&
      p.actor.resource===value.actor.resource &&
      p.fingerprintSha256===value.fingerprintSha256;
  }
}
const procedures:D33PrivateDatabaseProcedures={
  async approve(input) {
    const fields=[
      uuid(input.previewId),str(input.kind),str(input.actor.subject),
      str(input.actor.clientId),str(input.actor.resource),
      str(input.fingerprintSha256),str(input.capabilitySha256),
    ];
    const out=query("SET ROLE orotitan_d32_approval_issuer; "+
      "SELECT orotitan_mobile_private.approve_preview_v2("+
      fields.join(",")+"); RESET ROLE;");
    if(out!=="t"&&out!=="f")throw new Error("D33_DB_APPROVAL_RESULT_INVALID");
    return out==="t";
  },
  async confirm(input) {
    const out=readRow("SET ROLE orotitan_d32_command_submitter; "+
      "SELECT json_build_object('requestId',r.request_id::text,"+
      "'kind',r.kind,'state',r.state,'duplicate',r.duplicate)::text "+
      "FROM orotitan_mobile_private.confirm_approved_request_v2("+
      uuid(input.previewId)+","+str(input.sealedCapability)+") r; RESET ROLE;");
    return out as unknown as Awaited<ReturnType<D33PrivateDatabaseProcedures["confirm"]>>;
  },
};
async function run() {
  const ver=query("SHOW server_version_num");
  assert.match(ver,/^17[0-9]{4}$/);
  updateResearchFixture();
  const {publicKey,privateKey}=await generateKeyPair("ES256",{extractable:true});
  const jwk=await exportJWK(publicKey);
  const keys:JWTVerifyGetKey=createLocalJWKSet({
    keys:[{...jwk,kid:"d33-disposable",alg:"ES256",use:"sig"}],
  });
  async function bearer(id:Partial<{
    subject:string;client:string;resource:string;
  }>={}) {
    return "Bearer "+await new SignJWT({
      role:"authenticated",client_id:id.client??CLIENT,
      resource:id.resource??RESOURCE,
    }).setProtectedHeader({alg:"ES256",kid:"d33-disposable"})
      .setIssuer(oauth.issuer).setAudience(oauth.tokenAudience)
      .setSubject(id.subject??SUBJECT).setIssuedAt()
      .setExpirationTime("10m").sign(privateKey);
  }
  const valid=await bearer();
  const backend=new PostgresPreview();
  const policy=new PrivatePolicy();
  async function makePreview(kind:"add_company"|"request_research") {
    return kind==="add_company"
      ?previewAddCompany(company,actor,backend)
      :previewRequestResearch(ISSUER,actor,backend);
  }
  async function approve(p:DraftCommand,auth=valid) {
    return issueAndRegisterD33PrivateApproval({
      bearer:auth,oauth,keys,previewId:p.previewId,
      policy,signingKey:secret,db:procedures,
    });
  }
  async function confirm(p:DraftCommand,token:Awaited<ReturnType<typeof approve>>,
    auth=valid) {
    return confirmD33AuthorizedRequest({
      bearer:auth,oauth,keys,previewId:p.previewId,
      policy,signingKey:secret,db:procedures,token,
      kind:p.kind,fingerprintSha256:p.fingerprintSha256,
      confirmation:"CONFIRM",baseQueue:backend,
    });
  }
  // D33 confirmation is blocked even when D30 preview exists but approval is
  // absent; only D32's limited submitter gets a private DB procedure.
  const pending=await makePreview("add_company");
  const dummy="a".repeat(100)+"."+"b".repeat(43);
  assert.throws(()=>procedures.confirm({previewId:pending.previewId,
    sealedCapability:dummy}),/D33_UNUSED_SYNC_ASSERT/);
  // Above is deliberately a promise, not synchronous: test rejection below.
  await assert.rejects(procedures.confirm({
    previewId:pending.previewId,sealedCapability:dummy,
  }));
  assert.equal(countRequests(),0);
  policy.allowed=false;
  await assert.rejects(approve(pending),e=>(e as Error).message.includes("D31_INDEPENDENT_AUTHORIZATION_DENIED"));
  policy.allowed=true;
  await assert.rejects(approve(pending,await bearer({client:SECOND_CLIENT})));
  assert.equal(countRequests(),0);

  const capability=await approve(pending);
  assert.equal(countRequests(),0); // Approval is not a queue insert.
  await assert.rejects(confirm(pending,capability,
    await bearer({client:SECOND_CLIENT})));
  assert.equal(countRequests(),0);
  const [first,replayed]=await Promise.all([
    confirm(pending,capability),confirm(pending,capability),
  ]);
  assert.equal(first.requestId,replayed.requestId);
  assert.equal([first.duplicate,replayed.duplicate].filter(Boolean).length,1);
  assert.equal(first.state,"REQUESTED_NOT_EXECUTED");
  assert.equal(first.registryMutationPerformed,false);
  assert.equal(first.companyInserted,false);
  assert.equal(countRequests(),1);

  const research=await makePreview("request_research");
  const researchCap=await approve(research);
  const result=await confirm(research,researchCap);
  assert.equal(result.kind,"request_research");
  assert.equal(result.researchStarted,false);
  assert.equal(result.registryRunId,null);
  assert.equal(countRequests(),2);
  const revoked=await makePreview("add_company");
  const revokedToken=await approve(revoked);
  query("SET ROLE orotitan_d32_approval_issuer; "+
    "SELECT orotitan_mobile_private.revoke_preview_v2("+
    uuid(revoked.previewId)+"); RESET ROLE;");
  await assert.rejects(confirm(revoked,revokedToken));
  assert.equal(countRequests(),2);
  const last=query("SELECT count(*) FROM orotitan_mobile_private.command_requests "+
    "WHERE state<>'REQUESTED_NOT_EXECUTED'");
  assert.equal(last,"0");
  // D32 local data never mutates canonical identities or Registry tables.
  assert.equal(query("SELECT count(*) FROM public.issuers"),"2");
  assert.equal(query("SELECT count(*) FROM public.research_dossiers"),"2");
  assert.equal(query("SELECT to_regclass('public.orotitan_runs') IS NULL"),"t");
  console.log("D33 ES256 OAuth -> D31 approval -> D32 limited PG17 submitter PASSED (synthetic only).");
}
run().catch(e=>{console.error("D33 isolated E2E failed:",e instanceof Error?e.message:String(e));process.exitCode=1;});
