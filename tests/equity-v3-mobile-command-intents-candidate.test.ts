import assert from "node:assert/strict";
import { randomUUID } from "node:crypto";
import test from "node:test";
import {
  MOBILE_PREVIEW_TTL_MS, MobileCommandError,
  previewAddCompany, previewRequestResearch, confirmMobileCommand,
  type MobileActor, type CompanyCandidate, type CommandKind,
  type DraftCommand, type CommandReceipt, type ResearchTarget,
  type MobileCommandPrivatePort,
} from "../lib/orotitan-equity/v3/mobile-command-intents-candidate-v1";

const NOW = Date.parse("2026-10-23T11:00:00.000Z");
const ISSUER = "0fe958d4-6852-4357-a5bf-444942c61a23";
const actor: MobileActor = {
  subject: "authorized-user-123",
  clientId: "chatgpt-mobile-123",
  resource: "https://orotitan-chatgpt-mcp-pilot.vercel.app/api/orotitan/mcp",
};
const otherActor: MobileActor = { ...actor, subject: "unauthorized-user" };
const company: CompanyCandidate = {
  name: "ASML Holding N.V.", ticker: "ASML",
  exchange: "XAMS", isin: "NL0010273215",
};
const originalTarget: ResearchTarget = {
  issuerId: ISSUER, displayName: company.name, ticker: company.ticker,
  exchange: company.exchange, isin: company.isin,
  eligibleForRequest: true,
};

function assertDenied(p: Promise<unknown>, code: string) {
  return assert.rejects(p, (e: unknown) =>
    e instanceof MobileCommandError && e.code === code);
}
class FakePrivateBackend implements MobileCommandPrivatePort {
  previews = new Map<string, DraftCommand>();
  receipts = new Map<string, CommandReceipt>();
  byTarget = new Map<string, CommandReceipt>();
  target = { ...originalTarget };
  ambiguity = false;
  tracked = false;
  declined = false;
  writesToRegistry = 0;
  writesToCompanies = 0;
  callsToStartResearch = 0;
  confirms = 0;
  async authorize(a: MobileActor) {
    return !this.declined && a.subject === actor.subject &&
      a.clientId === actor.clientId && a.resource === actor.resource;
  }
  async lookupCompany(c: CompanyCandidate) {
    if (this.ambiguity) return "AMBIGUOUS" as const;
    if (this.tracked) return "ALREADY_TRACKED" as const;
    assert.equal(c.isin, company.isin);
    return "NEW_UNAMBIGUOUS" as const;
  }
  async lookupResearchTarget(issuerId: string) {
    return issuerId === ISSUER ? {...this.target} : null;
  }
  async storePreview(p: DraftCommand) {
    if (this.previews.has(p.previewId)) throw new Error("nonce collision");
    this.previews.set(p.previewId, structuredClone(p));
  }
  async atomicConfirm({
    previewId, actor: current, fingerprintSha256, nowMs, kind,
  }: {
    previewId: string; actor: MobileActor; fingerprintSha256: string;
    nowMs: number; kind: CommandKind;
  }): Promise<CommandReceipt> {
    const p = this.previews.get(previewId);
    if (!p || p.kind !== kind ||
      p.actor.subject !== current.subject ||
      p.actor.clientId !== current.clientId ||
      p.actor.resource !== current.resource) {
      throw new MobileCommandError("MOBILE_PREVIEW_ACTOR_OR_KIND_DENIED");
    }
    if (nowMs >= p.expiresAtMs) {
      throw new MobileCommandError("MOBILE_PREVIEW_EXPIRED");
    }
    if (fingerprintSha256 !== p.fingerprintSha256) {
      throw new MobileCommandError("MOBILE_PREVIEW_CHANGED");
    }
    const cached = this.receipts.get(previewId);
    if (cached) return {...cached,duplicate:true};
    if (p.kind === "add_company") {
      if (this.tracked || this.ambiguity) {
        throw new MobileCommandError("MOBILE_COMPANY_CHANGED_REVIEW");
      }
    } else {
      if (this.target.issuerId !== p.payload.issuerId ||
        !this.target.eligibleForRequest ||
        this.target.ticker !== p.payload.ticker ||
        this.target.exchange !== p.payload.exchange ||
        this.target.isin !== p.payload.isin ||
        this.target.displayName !== p.payload.name) {
        throw new MobileCommandError("MOBILE_TARGET_CHANGED_REVIEW");
      }
    }
    const targetKey = p.kind === "add_company"
      ? p.payload.exchange + ":" + p.payload.ticker + ":" + p.payload.isin
      : p.payload.issuerId;
    const key = p.kind + ":" + targetKey;
    const previous = this.byTarget.get(key);
    if (previous) {
      this.receipts.set(previewId,previous);
      return {...previous,duplicate:true};
    }
    const result: CommandReceipt = {
      requestId: randomUUID(), kind, fingerprintSha256: p.fingerprintSha256,
      state: "REQUESTED_NOT_EXECUTED", confirmed:true,
      registryRunId:null, registryMutationPerformed:false,
      companyInserted:false, researchStarted:false, duplicate:false,
    };
    // Represents one serialized DB transaction in the fake; NOT durable.
    this.byTarget.set(key,result);
    this.receipts.set(previewId,result);
    this.confirms++;
    return result;
  }
}
const confirm = (p: DraftCommand, backend: FakePrivateBackend, a = actor,
  now = NOW + 1000) => confirmMobileCommand(p.kind,{
  previewId:p.previewId,fingerprintSha256:p.fingerprintSha256,
  confirmation:"CONFIRM",
},a,backend,now);

test("mobile: issuer onboarding preview + explicit confirmation queues only",async()=>{
  const x=new FakePrivateBackend();
  const p=await previewAddCompany(company,actor,x,NOW);
  assert.equal(p.kind,"add_company");
  assert.equal(p.payload.isin,company.isin);
  assert.equal(p.expiresAtMs,NOW+MOBILE_PREVIEW_TTL_MS);
  assert.match(p.fingerprintSha256,/^[0-9a-f]{64}$/);
  assert.equal(x.confirms,0);
  const receipt=await confirm(p,x);
  assert.equal(receipt.kind,"add_company");
  assert.equal(receipt.state,"REQUESTED_NOT_EXECUTED");
  assert.equal(receipt.companyInserted,false);
  assert.equal(receipt.researchStarted,false);
  assert.equal(receipt.registryRunId,null);
  assert.equal(x.writesToRegistry,0);
  assert.equal(x.writesToCompanies,0);
});

test("mobile: canonical issuer Research intent never starts Registry run",async()=>{
  const x=new FakePrivateBackend();
  const p=await previewRequestResearch(ISSUER,actor,x,NOW);
  assert.equal(p.kind,"request_research");
  assert.equal(p.payload.issuerId,ISSUER);
  const r=await confirm(p,x);
  assert.equal(r.state,"REQUESTED_NOT_EXECUTED");
  assert.equal(r.researchStarted,false);
  assert.equal(r.registryMutationPerformed,false);
  assert.equal(x.callsToStartResearch,0);
  assert.equal(x.writesToRegistry,0);
});

test("mobile: repeated confirmations/repeated tabs are idempotent",async()=>{
  const x=new FakePrivateBackend();
  const p=await previewRequestResearch(ISSUER,actor,x,NOW);
  const [r1,r2]=await Promise.all([confirm(p,x),confirm(p,x)]);
  assert.equal(r1.requestId,r2.requestId);
  assert.equal(r1.duplicate,false);
  assert.equal(r2.duplicate,true);
  const p2=await previewRequestResearch(ISSUER,actor,x,NOW);
  const r3=await confirm(p2,x);
  assert.equal(r3.requestId,r1.requestId);
  assert.equal(r3.duplicate,true);
  assert.equal(x.confirms,1);
});

test("mobile: only previously authenticated subject and client can confirm",async()=>{
  const x=new FakePrivateBackend();
  const p=await previewAddCompany(company,actor,x,NOW);
  await assertDenied(confirm(p,x,otherActor),"MOBILE_COMMAND_NOT_AUTHORIZED");
  await assertDenied(confirm(p,x,{...actor,clientId:"another-chatgpt-client-123"}),
    "MOBILE_COMMAND_NOT_AUTHORIZED");
  assert.equal(x.confirms,0);
});

test("mobile: arbitrary user-text cannot bypass explicit confirmation",async()=>{
  const x=new FakePrivateBackend();
  const p=await previewAddCompany(company,actor,x,NOW);
  await assertDenied(confirmMobileCommand("add_company",{
    previewId:p.previewId,fingerprintSha256:p.fingerprintSha256,
    confirmation:"YES" as "CONFIRM",
  },actor,x,NOW+1000),"MOBILE_EXPLICIT_CONFIRMATION_REQUIRED");
  await assertDenied(confirmMobileCommand("request_research",{
    previewId:p.previewId,fingerprintSha256:p.fingerprintSha256,
    confirmation:"CONFIRM",
  },actor,x,NOW+1000),"MOBILE_PREVIEW_ACTOR_OR_KIND_DENIED");
  assert.equal(x.confirms,0);
});

test("mobile: stale fingerprint and expired preview cannot enqueue",async()=>{
  const x=new FakePrivateBackend();
  const p=await previewAddCompany(company,actor,x,NOW);
  await assertDenied(confirmMobileCommand("add_company",{
    previewId:p.previewId,fingerprintSha256:"f".repeat(64),
    confirmation:"CONFIRM",
  },actor,x,NOW+1000),"MOBILE_PREVIEW_CHANGED");
  await assertDenied(confirm(p,x,actor,NOW+MOBILE_PREVIEW_TTL_MS),
    "MOBILE_PREVIEW_EXPIRED");
  assert.equal(x.confirms,0);
});

test("mobile: ambiguous or already tracked company is never onboarded",async()=>{
  const x=new FakePrivateBackend();x.ambiguity=true;
  await assertDenied(previewAddCompany(company,actor,x,NOW),
    "MOBILE_IDENTITY_NEEDS_REVIEW");
  x.ambiguity=false;x.tracked=true;
  await assertDenied(previewAddCompany(company,actor,x,NOW),
    "MOBILE_COMPANY_ALREADY_TRACKED");
  x.tracked=false;
  const p=await previewAddCompany(company,actor,x,NOW);
  x.tracked=true;
  await assertDenied(confirm(p,x),"MOBILE_COMPANY_CHANGED_REVIEW");
  assert.equal(x.confirms,0);
});

test("mobile: malformed ticker/ISIN/exchange, missing canonical issuer fail closed",async()=>{
  const x=new FakePrivateBackend();
  for(const broken of [
    {...company,ticker:"ASML/../"},
    {...company,isin:"UNKNOWN"},
    {...company,exchange:""},
    {...company,name:"X"},
  ]) await assertDenied(previewAddCompany(broken,actor,x,NOW),
    "MOBILE_COMPANY_IDENTITY_INVALID");
  await assertDenied(previewRequestResearch("arbitrary",actor,x,NOW),
    "MOBILE_ISSUER_ID_INVALID");
  await assertDenied(previewRequestResearch(
    "48b9426b-4d0d-4858-9d87-efcd13ac415d",actor,x,NOW),
    "MOBILE_RESEARCH_NOT_ELIGIBLE");
  assert.equal(x.previews.size,0);
});

test("mobile: changed issuer identity and revoked eligibility between steps fail closed",async()=>{
  const x=new FakePrivateBackend();
  const p=await previewRequestResearch(ISSUER,actor,x,NOW);
  x.target.ticker="WRONG";
  await assertDenied(confirm(p,x),"MOBILE_TARGET_CHANGED_REVIEW");
  x.target.ticker=company.ticker;
  x.target.eligibleForRequest=false;
  await assertDenied(confirm(p,x),"MOBILE_TARGET_CHANGED_REVIEW");
  assert.equal(x.confirms,0);
});

test("mobile: backend cannot fabricate completed analysis or company insertion",async()=>{
  const x=new FakePrivateBackend();
  const p=await previewAddCompany(company,actor,x,NOW);
  const orig=x.atomicConfirm.bind(x);
  x.atomicConfirm=async a=>({...await orig(a),companyInserted:true} as unknown as CommandReceipt);
  await assertDenied(confirm(p,x),"MOBILE_UNTRUSTED_CONFIRM_RESULT");
});
