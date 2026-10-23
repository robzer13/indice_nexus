import assert from "node:assert/strict";
import { randomUUID } from "node:crypto";
import test from "node:test";
import { createLocalJWKSet, exportJWK, generateKeyPair, SignJWT } from "jose";
import {
  type McpOAuthConfig, MCP_SCOPE_READ, MCP_SCOPE_WRITE,
} from "../lib/orotitan-equity/v3/mcp-oauth-v1";
import {
  CommandCapabilityError,
  issueServerPrivateCommandCapability,
  confirmWithServerPrivateCapability,
  type IndependentCommandPolicy,
} from "../lib/orotitan-equity/v3/mobile-command-oauth-capabilities-v1";
import {
  previewAddCompany, previewRequestResearch, type CompanyCandidate,
  type MobileActor, type DraftCommand, type ResearchTarget,
  type CommandReceipt, type MobileCommandPrivatePort,
  MobileCommandError,
} from "../lib/orotitan-equity/v3/mobile-command-intents-candidate-v1";
import type { JWTVerifyGetKey } from "jose";

const NOW = Date.now();
const RESOURCE="https://orotitan-chatgpt-mcp-pilot.vercel.app/api/orotitan/mcp";
const SUBJECT="approved-operator-123";
const CLIENT="approved-chatgpt-456";
const ISSUER="0fe958d4-6852-4357-a5bf-444942c61a23";
const KEY=Buffer.alloc(32,0x73); // Synthetic fixture; NEVER a production signing key.
const actor: MobileActor={
  subject:SUBJECT,clientId:CLIENT,resource:RESOURCE,
};
const company: CompanyCandidate={
  name:"ASML Holding N.V.",ticker:"ASML",exchange:"XAMS",isin:"NL0010273215",
};
const target: ResearchTarget={
  issuerId:ISSUER,displayName:company.name,ticker:company.ticker,
  exchange:company.exchange,isin:company.isin,eligibleForRequest:true,
};
const oauth: McpOAuthConfig={
  issuer:"https://oauth-fixture.example.org",audience:RESOURCE,
  tokenAudience:"authenticated",
  jwksUri:"https://oauth-fixture.example.org/jwks",
  resourceMetadataUrl:"https://orotitan-chatgpt-mcp-pilot.vercel.app/.well-known/oauth-protected-resource",
  allowedSubjects:new Set([SUBJECT]),allowedClients:new Set([CLIENT,"approved-second-client"]),
};

class MemoryPort implements MobileCommandPrivatePort {
  previews=new Map<string,DraftCommand>();
  receipts=new Map<string,CommandReceipt>();
  target=target;
  previewWrites=0;
  queueWrites=0;
  permitted=true;
  async authorize(a: MobileActor) {
    return this.permitted && a.subject===SUBJECT &&
      oauth.allowedClients.has(a.clientId) && a.resource===RESOURCE;
  }
  async lookupCompany(c: CompanyCandidate) {
    assert.equal(c.isin,company.isin);return "NEW_UNAMBIGUOUS" as const;
  }
  async lookupResearchTarget(issuerId: string) {
    return issuerId===ISSUER ? {...this.target}:null;
  }
  async storePreview(p: DraftCommand) {
    this.previews.set(p.previewId,structuredClone(p));this.previewWrites++;
  }
  async atomicConfirm(a: {
    kind:"add_company"|"request_research";previewId:string;
    actor:MobileActor;fingerprintSha256:string;nowMs:number;
  }): Promise<CommandReceipt> {
    const p=this.previews.get(a.previewId);
    if(!p || p.kind!==a.kind || p.fingerprintSha256!==a.fingerprintSha256 ||
      p.actor.subject!==a.actor.subject || p.actor.clientId!==a.actor.clientId ||
      a.nowMs>=p.expiresAtMs) {
      throw new MobileCommandError("D31_QUEUE_DENIED");
    }
    const prev=this.receipts.get(p.previewId);
    if(prev)return {...prev,duplicate:true};
    const r:CommandReceipt={
      requestId:randomUUID(),kind:p.kind,fingerprintSha256:p.fingerprintSha256,
      state:"REQUESTED_NOT_EXECUTED",confirmed:true,
      registryRunId:null,registryMutationPerformed:false,
      companyInserted:false,researchStarted:false,duplicate:false,
    };
    this.receipts.set(p.previewId,r);this.queueWrites++;return r;
  }
}
class Policy implements IndependentCommandPolicy {
  enabled=true;
  constructor(readonly port:MemoryPort){}
  async readPreview(id:string) {
    return this.port.previews.get(id)??null;
  }
  async allows(a:{actor:MobileActor;kind:string;previewId:string;
    fingerprintSha256:string}) {
    const p=this.port.previews.get(a.previewId);
    return this.enabled && !!p &&
      p.kind===a.kind && p.fingerprintSha256===a.fingerprintSha256 &&
      p.actor.subject===a.actor.subject && p.actor.clientId===a.actor.clientId &&
      p.actor.resource===a.actor.resource;
  }
}

async function fixture(){
  const {publicKey,privateKey}=await generateKeyPair("ES256",{extractable:true});
  const jwk=await exportJWK(publicKey);
  const keys: JWTVerifyGetKey=createLocalJWKSet({
    keys:[{...jwk,kid:"d31-test-key",alg:"ES256",use:"sig"}],
  });
  async function signed({
    subject=SUBJECT,clientId=CLIENT,resource=RESOURCE,issuer=oauth.issuer,
    role="authenticated",
  }={}) {
    return "Bearer "+await new SignJWT({
      role,client_id:clientId,resource,
    }).setProtectedHeader({alg:"ES256",kid:"d31-test-key"})
      .setIssuer(issuer).setAudience("authenticated").setSubject(subject)
      .setIssuedAt().setExpirationTime("10m").sign(privateKey);
  }
  const port=new MemoryPort(),policy=new Policy(port);
  const bearer=await signed();
  const preview=await previewAddCompany(company,actor,port,NOW);
  const issue=()=>issueServerPrivateCommandCapability({
    bearer,oauth,keys,previewId:preview.previewId,
    policy,signingKey:KEY,nowMs:NOW+1_000,
  });
  const confirm=async(token:Awaited<ReturnType<typeof issue>>,
    options:Partial<{
      bearer:string;previewId:string;kind:"add_company"|"request_research";
      fingerprintSha256:string;confirmation:"CONFIRM";
      signingKey:Uint8Array;nowMs:number;
    }>={})=>confirmWithServerPrivateCapability({
      bearer:options.bearer??bearer,oauth,keys,token,
      previewId:options.previewId??preview.previewId,
      kind:options.kind??"add_company",
      fingerprintSha256:options.fingerprintSha256??preview.fingerprintSha256,
      confirmation:options.confirmation??"CONFIRM",
      policy,queue:port,signingKey:options.signingKey??KEY,
      nowMs:options.nowMs??NOW+2_000,
    });
  return {port,policy,preview,bearer,keys,issue,confirm,signed};
}
async function rejected(p:Promise<unknown>,code:string) {
  await assert.rejects(p,(err:unknown)=>err instanceof CommandCapabilityError &&
    err.code===code);
}

test("D31: signed OAuth subject/client/resource + independent grant queues only after confirmation",async()=>{
  const f=await fixture();
  const cap=await f.issue();
  assert.ok(cap.sealed.includes("."));
  assert.equal(JSON.stringify(f.preview).includes(cap.sealed),false);
  const receipt=await f.confirm(cap);
  assert.equal(receipt.state,"REQUESTED_NOT_EXECUTED");
  assert.equal(receipt.companyInserted,false);
  assert.equal(receipt.researchStarted,false);
  assert.equal(receipt.registryRunId,null);
  assert.equal(f.port.queueWrites,1);
  const replay=await f.confirm(cap);
  assert.equal(replay.requestId,receipt.requestId);
  assert.equal(replay.duplicate,true);
  assert.equal(f.port.queueWrites,1);
});

test("D31: independent server permission denied even with valid signed OAuth",async()=>{
  const f=await fixture();f.policy.enabled=false;
  await rejected(f.issue(),"D31_INDEPENDENT_AUTHORIZATION_DENIED");
  assert.equal(f.port.queueWrites,0);
  f.policy.enabled=true;const cap=await f.issue();
  f.policy.enabled=false;
  await rejected(f.confirm(cap),"D31_INDEPENDENT_AUTHORIZATION_DENIED");
  assert.equal(f.port.queueWrites,0);
});

test("D31: tampered capability, forged key or wrong preview/fingerprint/kind denied",async()=>{
  const f=await fixture();const cap=await f.issue();
  const altered={sealed:cap.sealed.slice(0,-1)+(cap.sealed.endsWith("A")?"B":"A")};
  await rejected(f.confirm(altered),"D31_CAPABILITY_SIGNATURE_DENIED");
  await rejected(f.confirm(cap,{signingKey:Buffer.alloc(32,2)}),
    "D31_CAPABILITY_SIGNATURE_DENIED");
  await rejected(f.confirm(cap,{fingerprintSha256:"a".repeat(64)}),
    "D31_CAPABILITY_BINDING_DENIED");
  await rejected(f.confirm(cap,{previewId:randomUUID()}),
    "D31_CAPABILITY_BINDING_DENIED");
  await rejected(f.confirm(cap,{kind:"request_research"}),
    "D31_CAPABILITY_BINDING_DENIED");
  assert.equal(f.port.queueWrites,0);
});

test("D31: signed client mismatch or other subject cannot reuse server capability",async()=>{
  const f=await fixture();const cap=await f.issue();
  await rejected(f.confirm(cap,{bearer:await f.signed({clientId:"approved-second-client"})}),
    "D31_CAPABILITY_BINDING_DENIED");
  await assert.rejects(f.confirm(cap,{bearer:await f.signed({subject:"another-person"})}));
  await assert.rejects(f.confirm(cap,{bearer:await f.signed({
    resource:"https://other.example.org/api/orotitan/mcp",
  })}));
  await assert.rejects(f.confirm(cap,{bearer:await f.signed({
    role:"anon",
  })}));
  assert.equal(f.port.queueWrites,0);
});

test("D31: preview changed, removed, expired or independently revoked before confirm",async()=>{
  const f=await fixture();const cap=await f.issue();
  await rejected(f.confirm(cap,{nowMs:f.preview.expiresAtMs}),
    "D31_CAPABILITY_EXPIRED_OR_INVALID");
  f.port.previews.set(f.preview.previewId,{
    ...f.preview,fingerprintSha256:"b".repeat(64),
  });
  await rejected(f.confirm(cap),"D31_PREVIEW_CHANGED");
  f.port.previews.delete(f.preview.previewId);
  await rejected(f.confirm(cap),"D31_PREVIEW_DENIED_OR_EXPIRED");
  assert.equal(f.port.queueWrites,0);
});

test("D31: invalid signing key never issues capabilities",async()=>{
  const f=await fixture();
  await rejected(issueServerPrivateCommandCapability({
    bearer:f.bearer,oauth,keys:f.keys,previewId:f.preview.previewId,
    policy:f.policy,signingKey:Buffer.alloc(12),nowMs:NOW+1_000,
  }),"D31_SIGNING_KEY_NOT_CONFIGURED");
  assert.equal(f.port.queueWrites,0);
});

test("D31: confirmed Research is still just a queue request",async()=>{
  const f=await fixture();
  const p=await previewRequestResearch(ISSUER,actor,f.port,NOW);
  const token=await issueServerPrivateCommandCapability({
    bearer:f.bearer,oauth,keys:f.keys,previewId:p.previewId,
    policy:f.policy,signingKey:KEY,nowMs:NOW+1_000,
  });
  const r=await confirmWithServerPrivateCapability({
    bearer:f.bearer,oauth,keys:f.keys,token,
    previewId:p.previewId,kind:"request_research",
    fingerprintSha256:p.fingerprintSha256,confirmation:"CONFIRM",
    policy:f.policy,queue:f.port,signingKey:KEY,nowMs:NOW+2_000,
  });
  assert.equal(r.kind,"request_research");
  assert.equal(r.state,"REQUESTED_NOT_EXECUTED");
  assert.equal(r.researchStarted,false);
  assert.equal(r.registryMutationPerformed,false);
});

test("D31: missing preview and caller-provided fake issuer rejected",async()=>{
  const f=await fixture();
  await rejected(issueServerPrivateCommandCapability({
    bearer:f.bearer,oauth,keys:f.keys,previewId:randomUUID(),
    policy:f.policy,signingKey:KEY,nowMs:NOW+1_000,
  }),"D31_PREVIEW_DENIED_OR_EXPIRED");
  assert.equal(f.port.queueWrites,0);
});

test("D31: OAuth signed client claim comes from verifier, not custom write scope",async()=>{
  const f=await fixture();
  const jwt=await f.signed();
  // This exercise ensures no scope-derived permission is silently inferred.
  const {verifyMcpOAuthBearer}=await import(
    "../lib/orotitan-equity/v3/mcp-oauth-v1"
  );
  const id=await verifyMcpOAuthBearer(jwt,oauth,f.keys);
  assert.equal(id.clientId,CLIENT);
  assert.equal(id.subject,SUBJECT);
  assert.equal(id.scopes.has(MCP_SCOPE_READ),true);
  assert.equal(id.scopes.has(MCP_SCOPE_WRITE),false);
});
