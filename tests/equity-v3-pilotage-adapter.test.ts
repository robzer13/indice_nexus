import assert from "node:assert/strict";
import test from "node:test";
import {
  planPilotageV31,
  type ArtifactProof,
  type ManifestProof,
  type PilotageSnapshot,
  type StageCode,
  type StageState,
} from "../lib/orotitan-equity/v3/pilotage-adapter-v3-1";
import pins from "../contracts/orotitan-equity/v3/contract-pin-pack-v3/OROTITAN_CONTRACT_PIN_PACK_V3_0_2.json";

const IDS = {
  run: "11111111-1111-4111-8111-111111111111",
  issuer: "22222222-2222-4222-8222-222222222222",
  security: "33333333-3333-4333-8333-333333333333",
  dossier: "44444444-4444-4444-8444-444444444444",
};
const SHA = "a".repeat(64);
const stagePins = {
  RESEARCH: pins.contract_pins.research_stage,
  DEEP_DIVE: pins.contract_pins.deep_dive_stage,
  INTEGRATION: pins.contract_pins.integration_stage,
};

function artifact(stageCode: StageCode, artifactType: string, number = 0): ArtifactProof {
  return {
    runId: IDS.run, stageCode, artifactType,
    artifactId: "77777777-7777-4777-8777-" + (20 + number).toString(16).padStart(12, "0"),
    version: 1, sha256: SHA, available: true, bytesVerified: true,
  };
}
function manifest(stageCode: StageCode, kind: "FINAL" | "CHECKPOINT"): ManifestProof {
  return { ...artifact(stageCode, stageCode + "_STAGE_MANIFEST_" + kind, 85),
    kind, stageRevision: 1, gatesVerifiedAgainstBytes: true };
}
function stage(code: StageCode, lifecycle: StageState["lifecycle"], previous: number | null): StageState {
  const p = stagePins[code];
  return { code, lifecycle, revision: 1, stateVersion: 2,
    contractName: p.name, contractVersion: p.version, contractSha256: p.content_sha256,
    consumedPredecessorRevision: previous, manifest: null, artifacts: [], gates: {} };
}

const RESEARCH = [
  "RESEARCH_SOURCE_MANIFEST", "EVIDENCE_LEDGER", "CONFLICT_LEDGER",
  "MATERIAL_RESEARCH_HYPOTHESIS_REGISTER", "RESEARCH_GAP_REGISTER",
  "INPUT_SUFFICIENCY_ASSESSMENT", "CLASSIFICATION_INPUT_RECORD",
];
const DEEP_DIVE = [
  "DEEP_DIVE_REPORT", "EVIDENCE_LEDGER", "CONFLICT_LEDGER", "CALCULATION_LEDGER",
  "MATERIAL_ASSUMPTION_REGISTER", "ANALYTICAL_BLOCK_OUTPUTS", "CROSS_BLOCK_RECONCILIATION_RECORD",
  "RED_TEAM_PREMORTEM_RECORD", "VALUATION_ARTIFACT", "CERTIFICATION_ARTIFACT",
  "OROTITAN_TERMINAL_GATE_ARTIFACT", "READINESS_NEXT_ACTION_ARTIFACT",
];
const INTEGRATION = [
  "CANONICAL_SNAPSHOT_CANDIDATE", "INTEGRATION_MAPPING_RECORD", "SCHEMA_VALIDATION_REPORT",
  "I2_RECONCILIATION_REPORT", "HISTORY_TRANSITION_VALIDATION_REPORT",
  "I3B_ADMISSION_REPORT", "PRE_PUBLICATION_CONTROL_CARD",
];

function snapshot(code: StageCode = "RESEARCH"): PilotageSnapshot {
  const r = stage("RESEARCH", code === "RESEARCH" ? "IN_PROGRESS" : "COMPLETE", null);
  if (code !== "RESEARCH") {
    r.manifest = manifest("RESEARCH", "FINAL");
    r.artifacts = RESEARCH.map((type, i) => artifact("RESEARCH", type, i));
    r.gates = { READY_FOR_DEEP_DIVE: true, FINAL_OUTPUTS_VERIFIED: true };
  }
  const d = stage("DEEP_DIVE", code === "INTEGRATION" ? "COMPLETE" : "NOT_STARTED", 1);
  if (code === "INTEGRATION") {
    d.manifest = manifest("DEEP_DIVE", "FINAL");
    d.artifacts = DEEP_DIVE.map((type, i) => artifact("DEEP_DIVE", type, i));
    d.gates = { READY_FOR_INTEGRATION: true, FINAL_OUTPUTS_VERIFIED: true };
  }
  const x = stage("INTEGRATION", "NOT_STARTED", 1);
  return {
    environment: { projectRef: "cugpgtzygqqlxetyeven", projectName: "orotitan-screener", region: "eu-west-3", status: "ACTIVE_HEALTHY" },
    run: { runId: IDS.run, company: "Example SA", companyCommandName: "EXAMPLE SA",
      issuerId: IDS.issuer, securityId: IDS.security, dossierId: IDS.dossier,
      dataCutoff: "2026-10-16", processVersion: "3.1", pilotageVersion: "3.1",
      contractSetSha256: pins.contract_set_sha256, stateVersion: 4,
      status: "ACTIVE", currentStage: code },
    stages: { RESEARCH: r, DEEP_DIVE: d, INTEGRATION: x },
    expectedRunStateVersion: 4,
    expectedCurrentStageStateVersion: 2,
  };
}
function blocked(s: PilotageSnapshot, expected: string): void {
  const value = planPilotageV31(s);
  assert.equal(value.outcome, "BLOCKED");
  assert.equal(value.reasonCode, expected);
  assert.equal(value.exactPrompt, null);
  assert.equal(value.publicationAuthorized, false);
  assert.equal(value.mutationPerformed, false);
}

test("Research starts or continues with a guarded bootstrap and never authorizes publication", () => {
  const result = planPilotageV31(snapshot());
  assert.equal(result.outcome, "CONTINUE");
  assert.equal(result.phase, "RESEARCH");
  assert.match(result.exactPrompt ?? "", /DO NOT USE THIS BOOTSTRAP AS ANALYTICAL EVIDENCE/);
  assert.equal(result.publicationAuthorized, false);
  assert.equal(result.mutationPerformed, false);
});
test("Research FINAL admits Fundamentals but no Valuation shortcut", () => {
  const value = planPilotageV31(snapshot("DEEP_DIVE"));
  assert.equal(value.outcome, "HANDOFF");
  assert.equal(value.phase, "FUNDAMENTALS");
  assert.match(value.exactPrompt ?? "", /RESEARCH_SOURCE_MANIFEST/);
  assert.match(value.exactPrompt ?? "", /RESEARCH_STAGE_MANIFEST_FINAL/);
});
test("wrong environment is a hard stop", () => {
  const s = snapshot();
  s.environment.projectRef = "awgsurdyvsyolcgpnygh";
  blocked(s, "WRONG_ENVIRONMENT");
});
test("wrong contract set is a hard stop", () => {
  const s = snapshot();
  s.run.contractSetSha256 = "f".repeat(64);
  blocked(s, "RUN_CONTRACT_SET_MISMATCH");
});
test("wrong stage contract hash is a hard stop", () => {
  const s = snapshot();
  s.stages.RESEARCH!.contractSha256 = "f".repeat(64);
  blocked(s, "STAGE_CONTRACT_PIN_MISMATCH");
});
test("stale run optimistic concurrency lease is a hard stop", () => {
  const s = snapshot();
  s.expectedRunStateVersion--;
  blocked(s, "STALE_RUN_CAS");
});
test("stale stage optimistic concurrency lease is a hard stop", () => {
  const s = snapshot();
  s.expectedCurrentStageStateVersion--;
  blocked(s, "STALE_STAGE_CAS");
});
test("missing Research FINAL cannot admit Deep Dive", () => {
  const s = snapshot("DEEP_DIVE");
  s.stages.RESEARCH!.manifest = null;
  blocked(s, "MANIFEST_KIND_MISMATCH");
});
test("unverified Research evidence cannot admit Deep Dive", () => {
  const s = snapshot("DEEP_DIVE");
  s.stages.RESEARCH!.artifacts = s.stages.RESEARCH!.artifacts.map(a =>
    a.artifactType === "EVIDENCE_LEDGER" ? { ...a, bytesVerified: false } : a);
  blocked(s, "ARTIFACT_MISSING_UNVERIFIED_OR_AMBIGUOUS_EVIDENCE_LEDGER");
});
test("Research stage reopen invalidates dependent Deep Dive", () => {
  const s = snapshot("DEEP_DIVE");
  s.stages.RESEARCH!.revision++;
  blocked(s, "UPSTREAM_REOPENED_OR_STALE_LINEAGE");
});
test("run lifecycle BLOCKED cannot dispatch", () => {
  const s = snapshot();
  s.run.status = "BLOCKED";
  blocked(s, "RUN_NOT_ACTIVE");
});
function fundamentals(): PilotageSnapshot {
  const s = snapshot("DEEP_DIVE");
  const dd = s.stages.DEEP_DIVE!;
  dd.lifecycle = "IN_PROGRESS";
  dd.manifest = manifest("DEEP_DIVE", "CHECKPOINT");
  dd.artifacts = ["FUNDAMENTALS_LOCK", "FUNDAMENTAL_RED_TEAM_RECORD"].map((type, i) =>
    artifact("DEEP_DIVE", type, i));
  dd.gates = {
    READY_FOR_VALUATION: true, FUNDAMENTAL_RED_TEAM_COMPLETE: true,
    DENOMINATOR_ADMISSION_VALID: true, VALUATION_DATE_ALIGNMENT_VALID: true,
    DCF_BASIS_ADMITTED: true, EV_BRIDGE_GOVERNED: true,
  };
  return s;
}
test("Fundamentals admits Valuation before valuation-only economic gates are known", () => {
  const s = fundamentals();
  s.stages.DEEP_DIVE!.gates.DENOMINATOR_ADMISSION_VALID = false;
  s.stages.DEEP_DIVE!.gates.VALUATION_DATE_ALIGNMENT_VALID = false;
  s.stages.DEEP_DIVE!.gates.DCF_BASIS_ADMITTED = false;
  s.stages.DEEP_DIVE!.gates.EV_BRIDGE_GOVERNED = false;
  assert.equal(planPilotageV31(s).phase, "VALUATION");
  s.stages.DEEP_DIVE!.gates.FUNDAMENTAL_RED_TEAM_COMPLETE = false;
  blocked(s, "GATE_NOT_ADMITTED_FUNDAMENTAL_RED_TEAM_COMPLETE");
});
test("valuation date mismatch blocks Certification, not Valuation entry", () => {
  const s = fundamentals();
  const dd = s.stages.DEEP_DIVE!;
  dd.artifacts = [...dd.artifacts, artifact("DEEP_DIVE", "VALUATION_LOCK", 6)];
  dd.gates.READY_FOR_CERTIFICATION = true;
  dd.gates.VALUATION_DATE_ALIGNMENT_VALID = false;
  blocked(s, "GATE_NOT_ADMITTED_VALUATION_DATE_ALIGNMENT_VALID");
});
test("Valuation lock admits Certification only, never Integration", () => {
  const s = fundamentals();
  const dd = s.stages.DEEP_DIVE!;
  dd.artifacts = [...dd.artifacts, artifact("DEEP_DIVE", "VALUATION_LOCK", 6)];
  dd.gates.READY_FOR_CERTIFICATION = true;
  const value = planPilotageV31(s);
  assert.equal(value.outcome, "HANDOFF");
  assert.equal(value.phase, "CERTIFICATION_RECONCILIATION");
  assert.equal(value.publicationAuthorized, false);
});
test("Valuation CHECKPOINT cannot stand in for Deep Dive FINAL", () => {
  const s = snapshot("INTEGRATION");
  s.stages.DEEP_DIVE!.lifecycle = "IN_PROGRESS";
  s.stages.DEEP_DIVE!.manifest = manifest("DEEP_DIVE", "CHECKPOINT");
  blocked(s, "UPSTREAM_STAGE_NOT_COMPLETE");
});
test("Deep Dive FINAL admits Integration", () => {
  const d = planPilotageV31(snapshot("INTEGRATION"));
  assert.equal(d.outcome, "HANDOFF");
  assert.equal(d.phase, "INTEGRATION");
  assert.match(d.exactPrompt ?? "", /DEEP_DIVE_REPORT/);
  assert.match(d.exactPrompt ?? "", /DEEP_DIVE_STAGE_MANIFEST_FINAL/);
});
test("Deep Dive reopen invalidates dependent Integration", () => {
  const s = snapshot("INTEGRATION");
  s.stages.DEEP_DIVE!.revision++;
  blocked(s, "UPSTREAM_REOPENED_OR_STALE_LINEAGE");
});
function candidate(): PilotageSnapshot {
  const s = snapshot("INTEGRATION");
  const integration = s.stages.INTEGRATION!;
  integration.lifecycle = "COMPLETE";
  integration.manifest = manifest("INTEGRATION", "FINAL");
  integration.artifacts = INTEGRATION.map((type, i) => artifact("INTEGRATION", type, i));
  integration.gates = { READY_TO_PUBLISH: true, CANONICAL_CANDIDATE_ADMITTED: true,
    I2_PASS: true, I3B_PASS: true, FINAL_OUTPUTS_VERIFIED: true };
  s.run.status = "READY_TO_PUBLISH";
  return s;
}
test("Integration FINAL produces only a distinct GO PUBLISH user prompt", () => {
  const value = planPilotageV31(candidate());
  assert.equal(value.outcome, "AWAIT_EXPLICIT_GO_PUBLISH");
  assert.equal(value.exactPrompt, "GO PUBLISH EXAMPLE SA");
  assert.equal(value.publicationAuthorized, false);
  assert.equal(value.mutationPerformed, false);
});
test("I3-B failure prevents the publication prompt", () => {
  const s = candidate();
  s.stages.INTEGRATION!.gates.I3B_PASS = false;
  blocked(s, "GATE_NOT_ADMITTED_I3B_PASS");
});
test("missing canonical candidate prevents publication prompt", () => {
  const s = candidate();
  s.stages.INTEGRATION!.artifacts = s.stages.INTEGRATION!.artifacts.map(a =>
    a.artifactType === "CANONICAL_SNAPSHOT_CANDIDATE" ? { ...a, available: false } : a);
  blocked(s, "ARTIFACT_MISSING_UNVERIFIED_OR_AMBIGUOUS_CANONICAL_SNAPSHOT_CANDIDATE");
});
test("PUBLISHED is terminal and cannot republish", () => {
  const s = candidate();
  s.run.status = "PUBLISHED";
  const d = planPilotageV31(s);
  assert.equal(d.outcome, "DONE");
  assert.equal(d.exactPrompt, null);
});
test("same state produces same fingerprint; manifest is invalid after reopen", () => {
  const s = candidate();
  assert.equal(planPilotageV31(s).stateFingerprintSha256,
    planPilotageV31(s).stateFingerprintSha256);
  s.stages.INTEGRATION!.revision++;
  blocked(s, "MANIFEST_REVISION_STALE");
});
