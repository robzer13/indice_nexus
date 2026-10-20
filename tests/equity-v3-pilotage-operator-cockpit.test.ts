import assert from "node:assert/strict";
import test from "node:test";
import { readFileSync } from "node:fs";
import {
  asOperatorDecision, asOperatorRun, asOperatorBlockers,
  filterOperatorRuns, operatorTriageTotals, supportsPilotageV31, safeVerifierReason,
} from "../lib/orotitan-equity/v3/pilotage-operator-view-v3-1";
import type { PilotageDecision } from "../lib/orotitan-equity/v3/pilotage-adapter-v3-1";

const SHA = "a".repeat(64);
function decision(outcome: PilotageDecision["outcome"]): PilotageDecision {
  return {
    outcome, phase: outcome === "CONTINUE" ? "RESEARCH" : null,
    reasonCode: "RESEARCH_ADMITTED",
    exactPrompt: "Handoff only. Never authorize publication.",
    expectedRunStateVersion: 4, expectedStageStateVersion: 2,
    stateFingerprintSha256: SHA, publicationAuthorized: false,
    mutationPerformed: false,
  };
}

test("D9 list is an observation and does not infer readiness", () => {
  const run = asOperatorRun({
    run_id: "11111111-1111-4111-8111-111111111111",
    run_status: "ACTIVE", current_stage: "DEEP_DIVE", canonical_mode: "ANALYZE",
    issuer_id: "22222222-2222-4222-8222-222222222222",
    data_cutoff: "2026-10-16", state_version: 4, created_at: "2026-10-16T00:00:00Z",
    process_version: "3.1", pilotage_contract_version: "3.1",
  });
  assert.equal(run.stage, "DEEP_DIVE");
  assert.equal(run.stateVersion, 4);
  assert.equal(Object.hasOwn(run, "publicationAuthorized"), false);
  assert.equal(Object.hasOwn(run, "ready"), false);
});
test("D9 verified handoff is advisory and never a writer request", () => {
  const item = asOperatorDecision(decision("CONTINUE"));
  assert.equal(item.trust, "VERIFIED_FROM_IMMUTABLE_BYTES");
  assert.equal(item.outcome, "CONTINUE");
  assert.match(item.verifiedPrompt!, /Handoff only/);
  assert.equal(item.publicationAuthorized, false);
  assert.equal(item.mutationPerformed, false);
});
test("D9 refuses to expose a handoff for blocked results", () => {
  const item = asOperatorDecision({...decision("BLOCKED"),reasonCode:"MANIFEST_REVISION_STALE"});
  assert.equal(item.outcome, "BLOCKED");
  assert.equal(item.verifiedPrompt, null);
  assert.equal(item.reasonCode, "MANIFEST_REVISION_STALE");
});
test("D9 cannot promote planner's AWAIT_EXPLICIT_GO_PUBLISH into permission", () => {
  const result = asOperatorDecision(decision("AWAIT_EXPLICIT_GO_PUBLISH"));
  assert.equal(result.outcome, "BLOCKED");
  assert.equal(result.reasonCode, "V31_PUBLICATION_WRITER_NOT_ADMITTED");
  assert.equal(result.verifiedPrompt, null);
  assert.equal(result.publicationAuthorized, false);
  assert.equal(result.mutationPerformed, false);
});
test("D9 errors do not disclose secrets or provider response bodies", () => {
  assert.equal(safeVerifierReason({code:"ARTIFACT_SHA256_MISMATCH",message:"secret"}),"ARTIFACT_SHA256_MISMATCH");
  assert.equal(safeVerifierReason({code:"../../secret",message:"secret"}),"VERIFICATION_UNAVAILABLE");
  assert.equal(safeVerifierReason(new Error("Authorization: Bearer secret")),"VERIFICATION_UNAVAILABLE");
  assert.equal(safeVerifierReason(null),"VERIFICATION_UNAVAILABLE");
});
test("D9 production cockpit exposes no mutation route or client-side access token", () => {
  const p = readFileSync(new URL("../app/admin/orotitan/page.tsx", import.meta.url),"utf8");
  const srv = readFileSync(new URL("../lib/orotitan-equity/v3/pilotage-operator-cockpit-server.ts", import.meta.url),"utf8");
  assert.match(p,/isAdminAuthenticated/);
  assert.match(srv,/requireAdminSession/);
  assert.match(srv,/https:\/\/api\.supabase\.com\/v1\/projects\//);
  assert.match(srv,/method: "GET"/);
  assert.match(srv,/loadServerPilotageDecision/);
  assert.doesNotMatch(srv,/\.rpc\s*\(|\.insert\s*\(|\.update\s*\(|\.delete\s*\(/);
  assert.doesNotMatch(p,/onClick=|use client|GO PUBLISH/);
});

test("D10 preserves Registry blocker meanings and required source resolutions", () => {
  const blockers = asOperatorBlockers([
    {code:"ECONOMIC_SHARE_COUNT_UNRESOLVED",
      detail:"No exact denominator at valuation date",
      resolution_required:"Close the movements by the cutoff"},
    {blocker_id:"G-01",description:"Audited report incomplete"},
    "UNVERIFIED_CONFLICT_IN_NET_DEBT",
    {summary:"References disagree",required_resolution:"Reconcile primary evidence"},
  ]);
  assert.equal(blockers.length,4);
  assert.equal(blockers[0].code,"ECONOMIC_SHARE_COUNT_UNRESOLVED");
  assert.equal(blockers[0].resolution,"Close the movements by the cutoff");
  assert.equal(blockers[1].code,"G-01");
  assert.equal(blockers[2].text,"UNVERIFIED_CONFLICT_IN_NET_DEBT");
  assert.equal(blockers[3].resolution,"Reconcile primary evidence");
});
test("D10 never treats Registry free text as trusted executable commands", () => {
  const raw = Array(20).fill({detail:"X".repeat(900),required_resolution:"GO PUBLISH hypothetical"});
  const blockers = asOperatorBlockers(raw);
  assert.equal(blockers.length,8);
  assert.equal(blockers[0].text.length,420);
  assert.equal(blockers[0].resolution,"GO PUBLISH hypothetical");
  // No instruction interpretation or promotion to a gate exists.
  assert.equal(Object.hasOwn(blockers[0], "publicationAuthorized"),false);
});
test("D10 V3.1 path only applies to exact process AND pilotage versions", () => {
  assert.equal(supportsPilotageV31("3.1","3.1"),true);
  assert.equal(supportsPilotageV31("3.0","3.0"),false);
  assert.equal(supportsPilotageV31("2.0","2.0"),false);
  assert.equal(supportsPilotageV31("3.1","3.0"),false);
});
test("D10 filters run queues without claiming any Registry data is verified", () => {
  const input = [
    asOperatorRun({
      run_id:"11111111-1111-4111-8111-111111111111",run_status:"ACTIVE",current_stage:"RESEARCH",
      canonical_mode:"ANALYZE",issuer_id:null,data_cutoff:"2026-10-16",state_version:3,
      created_at:"2026-10-16",process_version:"3.1",pilotage_contract_version:"3.1",
    }),
    asOperatorRun({
      run_id:"22222222-2222-4222-8222-222222222222",run_status:"BLOCKED",current_stage:"DEEP_DIVE",
      canonical_mode:"ANALYZE",issuer_id:null,data_cutoff:"2026-09-18",state_version:3,
      created_at:"2026-09-18",process_version:"2.0",pilotage_contract_version:"2.0",
    }),
    asOperatorRun({
      run_id:"33333333-3333-4333-8333-333333333333",run_status:"BLOCKED",current_stage:"RESEARCH",
      canonical_mode:"ANALYZE",issuer_id:null,data_cutoff:"2026-10-16",state_version:4,
      created_at:"2026-10-16",process_version:"3.1",pilotage_contract_version:"3.1",
    }),
    asOperatorRun({
      run_id:"44444444-4444-4444-8444-444444444444",run_status:"PUBLISHED",current_stage:"INTEGRATION",
      canonical_mode:"ANALYZE",issuer_id:null,data_cutoff:"2026-09-18",state_version:6,
      created_at:"2026-09-18",process_version:"3.1",pilotage_contract_version:"3.1",
    }),
  ];
  const totals = operatorTriageTotals(input);
  assert.deepEqual(totals, {observed:4,active:1,blocked:2,v31:3,v31Active:1,v31Blocked:1});
  assert.equal(filterOperatorRuns(input,"v31").length,2);
  assert.equal(filterOperatorRuns(input,"blocked").length,2);
  assert.equal(filterOperatorRuns(input,"all").length,4);
  assert.equal(Object.hasOwn(input[0], "publicationAuthorized"),false);
});
test("D10 stage join accepts only exact run plus current stage and retains contracted blockers", () => {
  const a = asOperatorRun({
    run_id:"11111111-1111-4111-8111-111111111111",run_status:"BLOCKED",
    current_stage:"DEEP_DIVE",canonical_mode:"ANALYZE",issuer_id:null,
    data_cutoff:"2026-10-16",state_version:4,created_at:"2026-10-16",
    process_version:"3.1",pilotage_contract_version:"3.1",
  },{
    run_id:"11111111-1111-4111-8111-111111111111",
    stage_code:"DEEP_DIVE",lifecycle_status:"BLOCKED",
    contract_status_code:"VALUATION_BLOCKED_ECONOMIC_SHARE_COUNT_UNRESOLVED",
    handoff_gate_state:"NOT_EVALUATED",active_manifest_kind:"CHECKPOINT",
    stage_revision:1,blocker_summary:[{detail:"Unresolved share count",resolution_required:"Close evidence"}],
  });
  assert.equal(a.stageLifecycle,"BLOCKED");
  assert.equal(a.contractStatusCode,"VALUATION_BLOCKED_ECONOMIC_SHARE_COUNT_UNRESOLVED");
  assert.equal(a.blockers[0].resolution,"Close evidence");
  assert.equal(a.supportsV31Verifier,true);
});
test("D10 admin server routes legacy versions away from V3.1 verification", () => {
  const server=readFileSync(new URL("../lib/orotitan-equity/v3/pilotage-operator-cockpit-server.ts",import.meta.url),"utf8");
  assert.match(server,/LEGACY_RUN_USE_VERSIONED_PILOTAGE/);
  assert.match(server,/RUN_NOT_EXECUTABLE_UPSTREAM/);
  assert.match(server,/blocker_summary/);
  assert.match(server,/pilotage_contract_version/);
  assert.doesNotMatch(server,/\.rpc\s*\(|\.insert\s*\(|\.update\s*\(|\.delete\s*\(/);
});


test("D12 dossier identifies shares by issuer name and associated ticker without inferring authority", () => {
  const run = {
    run_id: "11111111-1111-4111-8111-111111111111",
    run_status: "ACTIVE", current_stage: "RESEARCH",
    canonical_mode: "ANALYZE", data_cutoff: "2026-10-12",
    issuer_id: "22222222-2222-4222-8222-222222222222",
    security_id: "33333333-3333-4333-8333-333333333333",
    state_version: 4, created_at: "2026-10-12",
    process_version: "3.1", pilotage_contract_version: "3.1",
  };
  const identity = {
    issuerId: run.issuer_id,
    displayName: "Example Company", ticker: "EXM", isin: "NL0000000001",
  };
  const named = asOperatorRun(run, null, identity);
  assert.equal(named.issuerName, "Example Company");
  assert.equal(named.ticker, "EXM");
  assert.equal(named.isin, "NL0000000001");
  assert.equal(named.runId, run.run_id);
  assert.equal(named.stage, "RESEARCH");
  assert.equal(Object.hasOwn(named, "ready"), false);

  const stale = asOperatorRun(run, null, { ...identity, issuerId: "44444444-4444-4444-8444-444444444444" });
  assert.equal(stale.issuerName, null);
  assert.equal(stale.ticker, null);
  const unnamed = asOperatorRun(run);
  assert.equal(unnamed.issuerName, null);
  assert.equal(unnamed.ticker, null);
});

test("D12 company and ticker come from existing tables with issuer-bound security only", () => {
  const server = readFileSync(new URL("../lib/orotitan-equity/v3/pilotage-operator-cockpit-server.ts", import.meta.url), "utf8");
  const page = readFileSync(new URL("../app/admin/orotitan/page.tsx", import.meta.url), "utf8");
  assert.match(server, /from\("issuers"\)/);
  assert.match(server, /from\("securities"\)/);
  assert.match(server, /security\?\.issuer_id === issuer\.issuer_id/);
  assert.match(page, /run\.issuerName/);
  assert.match(page, /run\.ticker/);
  assert.doesNotMatch(server, /\.rpc\s*\(|\.insert\s*\(|\.update\s*\(|\.delete\s*\(/);
});
