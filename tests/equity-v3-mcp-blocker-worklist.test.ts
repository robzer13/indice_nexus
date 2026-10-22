import assert from "node:assert/strict";
import test from "node:test";
import { readFileSync } from "node:fs";
import { buildRecoveryWorklist } from "../lib/orotitan-equity/v3/pilotage-blocker-recovery-v3-1";
import type { OperatorRun } from "../lib/orotitan-equity/v3/pilotage-operator-view-v3-1";

const run:OperatorRun={
  runId:"48b9426b-4d0d-4858-9d87-efcd13ac415d",
  issuerName:"Besi",ticker:"BESI",isin:"NL0012866412",
  status:"ACTIVE",stage:"RESEARCH",mode:"ANALYZE",
  cutoff:"2026-10-12",stateVersion:6,createdAt:"",
  processVersion:"3.1",pilotageVersion:"3.1",
  stageLifecycle:"IN_PROGRESS",contractStatusCode:"BLOCKED_INSUFFICIENT_INPUT",
  handoffGateState:"NO",activeManifestKind:"CHECKPOINT",
  blockers:[
    {code:null,text:"C003 FY2025 18 vs 15 adopting clients",resolution:null},
    {code:null,text:"No independent hybrid bonding commercial qualification or fab yield evidence",resolution:null},
    {code:null,text:"No technology adoption curve separating HBM and logic",resolution:null},
    {code:null,text:"Economic through-cycle ROIC calculations incomplete",resolution:null},
    {code:null,text:"Owner cash needs, recurring development capital and tax benefits",resolution:null},
    {code:null,text:"Standalone FY2021-FY2022 audited ESEF full document cross check",resolution:null},
  ],supportsV31Verifier:true,
};

test("MCP worklist offers all six Besi tasks without closing gates or claiming work completed",()=>{
  const result=buildRecoveryWorklist(run);
  assert.equal(result.tasks.length,6);
  assert.deepEqual(result.tasks.map(x=>x.kind),[
    "EVIDENCE_CONTRADICTION","INDEPENDENT_CORROBORATION","TECHNOLOGY_ADOPTION",
    "FINANCIAL_RECONCILIATION","FINANCIAL_RECONCILIATION","PRIMARY_FILING",
  ]);
  assert.ok(result.tasks.every(x=>x.status==="OPEN_UNVERIFIED" && x.acceptanceEvidence.length>0));
  assert.equal(result.mayMutate,false);
  assert.equal(result.mayAdvance,false);
  assert.equal(result.mayPublish,false);
  assert.equal(result.verifiedResearchPerformed,false);
});

test("MCP recovery route is available only to verified OAuth delegation; not a service-role escape",()=>{
  const route=readFileSync(new URL("../app/api/orotitan/mcp/route.ts",import.meta.url),"utf8");
  const service=readFileSync(new URL("../lib/orotitan-equity/v3/mcp-bridge-service-v1.ts",import.meta.url),"utf8");
  assert.match(route,/registerTool\("get_orotitan_blocker_worklist"/);
  assert.match(route,/readOnlyHint:true,destructiveHint:false,openWorldHint:false/);
  assert.match(route,/if\(!delegatedJwt\) throw new McpBridgeError\("RECOVERY_OAUTH_REQUIRED"\)/);
  assert.match(route,/getMcpRecoveryWorklistDelegated\(run_id,delegatedJwt,process\.env\)/);
  const wrapper=service.split("export async function getMcpRecoveryWorklistDelegated(")[1]
    .split("Reuses D2 trusted immutable bytes verification")[0];
  assert.match(wrapper,/getMcpRunContextDelegated\(runIdRaw,jwt,env\)/);
  assert.match(wrapper,/buildRecoveryWorklist\(asRun\)/);
  assert.match(wrapper,/RECOVERY_UNSUPPORTED_CONTRACT_VERSION/);
  assert.match(wrapper,/RECOVERY_STAGE_STATE_UNAVAILABLE/);
  assert.match(wrapper,/runStateVersion:observed.runStateVersion/);
  assert.match(wrapper,/stageStateVersion:observed.stageStateVersion/);
  for(const blocked of ["createServerSupabaseClient\(","loadTrustedPilotageDecision\(","listMcpSubmissions\(",
    "\.insert\(","\.update\(","\.rpc\(","submitMcpWorkProduct\("]){
    assert.doesNotMatch(wrapper,new RegExp(blocked));
  }
  assert.match(wrapper,/mayAdvance:false/);
  assert.match(wrapper,/mayPublish:false/);
  assert.match(wrapper,/mayMutate:false/);
  assert.match(wrapper,/mutationPerformed:false/);
  assert.match(wrapper,/verifiedResearchPerformed:false/);
});

test("MCP read-only tool does not permit a quarantine or Registry write despite OAuth consent",()=>{
  const route=readFileSync(new URL("../app/api/orotitan/mcp/route.ts",import.meta.url),"utf8");
  const access=readFileSync(new URL("../lib/orotitan-equity/v3/mcp-bridge-access-v1.ts",import.meta.url),"utf8");
  assert.match(route,/DELEGATED_QUARANTINE_READ_DISABLED/);
  assert.match(route,/if\(mcpWriteEnabled\(process\.env\)\)/);
  assert.match(access,/export function mcpWriteEnabled/);
  assert.match(access,/return false/);
  // The worklist must not send a fake Research FINAL, GO or accepted manifest.
  const classify=readFileSync(new URL("../lib/orotitan-equity/v3/pilotage-blocker-recovery-v3-1.ts",import.meta.url),"utf8");
  assert.doesNotMatch(classify,/\.rpc\s*\(|\.insert\s*\(|\.update\s*\(/);
});

test("missing manifest is a remediation task, not an authorization",()=>{
  const result=buildRecoveryWorklist({...run,activeManifestKind:null,blockers:[]});
  assert.equal(result.tasks[0].kind,"MISSING_CHECKPOINT");
  assert.equal(result.mayAdvance,false);
});
