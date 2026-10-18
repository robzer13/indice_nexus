import assert from "node:assert/strict";
import test from "node:test";
import { readFileSync } from "node:fs";
import {
  asOperatorDecision, asOperatorRun, safeVerifierReason,
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
