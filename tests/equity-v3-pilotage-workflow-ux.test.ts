import assert from "node:assert/strict";
import test from "node:test";
import { readFileSync } from "node:fs";
import {
  asOperatorDecision, asOperatorRun, filterOperatorRuns,
  hasObservedBlockers, operatorTriageTotals,
  type OperatorRun,
} from "../lib/orotitan-equity/v3/pilotage-operator-view-v3-1";
import {
  WORKFLOW_STEPS, workflowPosition,
} from "../lib/orotitan-equity/v3/pilotage-workflow-view-v3-1";
import type { PilotageDecision } from "../lib/orotitan-equity/v3/pilotage-adapter-v3-1";

const RUN = "11111111-1111-4111-8111-111111111111";
function run(stage: string = "RESEARCH"): OperatorRun {
  return asOperatorRun({
    run_id: RUN, run_status: "ACTIVE", current_stage: stage,
    canonical_mode: "ANALYZE", data_cutoff: "2026-10-12",
    issuer_id: "22222222-2222-4222-8222-222222222222",
    state_version: 5, created_at: "2026-10-12T00:00:00Z",
    process_version: "3.1", pilotage_contract_version: "3.1",
  });
}
function verified(phase: PilotageDecision["phase"], outcome: "HANDOFF" | "CONTINUE" | "BLOCKED" = "HANDOFF") {
  return asOperatorDecision({
    outcome, phase, reasonCode: "RESEARCH_FINAL_ADMITTED_START_REQUIRED",
    exactPrompt: "Manual analysis prompt only",
    expectedRunStateVersion: 5, expectedStageStateVersion: 3,
    stateFingerprintSha256: "a".repeat(64),
    publicationAuthorized: false, mutationPerformed: false,
  });
}

test("UX tracks exactly five manual analytical phases, not three Registry stages", () => {
  assert.deepEqual(WORKFLOW_STEPS.map(step => step.phase), [
    "RESEARCH", "FUNDAMENTALS", "VALUATION",
    "CERTIFICATION_RECONCILIATION", "INTEGRATION",
  ]);
});

test("UX Besi-like Research CHECKPOINT with material insufficiency is action-required, not complete", () => {
  const item = { ...run(), contractStatusCode: "BLOCKED_INSUFFICIENT_INPUT",
    activeManifestKind: "CHECKPOINT", blockers: [{ code: "C003", text: "Conflict 18 vs 15", resolution: null }] };
  const j = workflowPosition({ run: item, preview: null, previewState: "NOT_SELECTED" });
  assert.equal(j.source, "REGISTRY_OBSERVATION");
  assert.equal(j.index, 1);
  assert.equal(j.isBlocked, true);
  assert.match(j.actionLabel, /Examiner les 1 point/);
  assert.equal(j.steps[0].state, "current");
  assert.equal(j.steps[1].state, "upcoming");
  assert.equal(hasObservedBlockers(item), true);
  assert.equal(filterOperatorRuns([item], "blocked").length, 1);
  const totals = operatorTriageTotals([item]);
  assert.equal(totals.active, 1);
  assert.equal(totals.blocked, 1);
});

test("UX verified Research handoff locates Fundamentals but never calls it completed", () => {
  const j = workflowPosition({
    run: run(), preview: verified("FUNDAMENTALS"), previewState: "VERIFIED",
  });
  assert.equal(j.index, 2);
  assert.equal(j.source, "VERIFIED_HANDOFF");
  assert.equal(j.title, "Prochaine phase admissible");
  assert.equal(j.steps[0].state, "previous");
  assert.equal(j.steps[1].state, "current");
  assert.match(j.actionLabel, /analyse fondamentale/);
});

test("UX differentiates observed Deep Dive from verified Valuation or Certification", () => {
  const unverified = workflowPosition({
    run: run("DEEP_DIVE"), preview: null, previewState: "BLOCKED",
  });
  assert.equal(unverified.index, 2);
  assert.equal(unverified.source, "REGISTRY_OBSERVATION");
  assert.equal(unverified.isBlocked, true);
  const valuation = workflowPosition({
    run: run("DEEP_DIVE"), preview: verified("VALUATION"), previewState: "VERIFIED",
  });
  assert.equal(valuation.index, 3);
  assert.match(valuation.actionLabel, /valorisation/);
  const certification = workflowPosition({
    run: run("DEEP_DIVE"), preview: verified("CERTIFICATION_RECONCILIATION"), previewState: "VERIFIED",
  });
  assert.equal(certification.index, 4);
});

test("UX blocked verification must not elevate phase from an untrusted decision", () => {
  const d = verified("INTEGRATION", "BLOCKED");
  const j = workflowPosition({
    run: run("RESEARCH"), preview: d, previewState: "VERIFIED",
  });
  assert.equal(j.index, 1);
  assert.equal(j.source, "REGISTRY_OBSERVATION");
  assert.equal(j.isBlocked, true);
  assert.equal(d.verifiedPrompt, null);
  const unknown = workflowPosition({
    run: run("NOT_ASSIGNED"), preview: null, previewState: "NOT_SELECTED",
  });
  assert.equal(unknown.index, 0);
  assert.equal(unknown.source, "UNKNOWN");
});

test("UX manual prompt button stays separate from analytical progress and any writer", () => {
  const ui = readFileSync(new URL("../components/admin/orotitan-workflow-journey.tsx", import.meta.url), "utf8");
  const copy = readFileSync(new URL("../components/admin/orotitan-prompt-copy.tsx", import.meta.url), "utf8");
  const page = readFileSync(new URL("../app/admin/orotitan/page.tsx", import.meta.url), "utf8");
  assert.match(ui, /role="progressbar"/);
  assert.match(ui, /Position, non taux/);
  assert.match(ui, /OroTitanPromptCopy/);
  assert.match(copy, /navigator\.clipboard\.writeText/);
  assert.match(page, /OroTitanWorkflowJourney/);
  assert.doesNotMatch(ui, /\.rpc\s*\(|\.insert\s*\(|\.update\s*\(|\.delete\s*\(/);
  assert.doesNotMatch(copy, /\.rpc\s*\(|fetch\s*\(|start_orotitan_stage|GO PUBLISH/);
});
