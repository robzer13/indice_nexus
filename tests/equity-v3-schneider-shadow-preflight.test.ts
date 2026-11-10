import assert from "node:assert/strict";
import test from "node:test";
import { RESEARCH_V2_FINAL_OUTPUTS } from "../lib/orotitan-equity/v1/research-required-outputs";
import type { PilotageSnapshot, ArtifactProof } from "../lib/orotitan-equity/v3/pilotage-adapter-v3-1";
import {
  SCHNEIDER_SHADOW_RUN_ID, loadTrustedPilotageSnapshot,
  loadTrustedSchneiderShadowSnapshot, TrustedLoaderError,
  type TrustedPilotagePort,
} from "../lib/orotitan-equity/v3/pilotage-trusted-loader-v3-1";
import {
  SCHNEIDER_FINAL_RESEARCH_TARGET,
  assessVerifiedSchneiderShadowSnapshot,
  SchneiderShadowPreflightError,
} from "../lib/orotitan-equity/v3/schneider-shadow-research-preflight-v3-1";
import pins from "../contracts/orotitan-equity/v3/contract-pin-pack-v3/OROTITAN_CONTRACT_PIN_PACK_V3_0_2.json";

const SHADOW = {
  projectRef: "awgsurdyvsyolcgpnygh", projectName: "orotitan-vnext-shadow",
  region: "eu-west-3", status: "ACTIVE_HEALTHY",
};
const ID = SCHNEIDER_FINAL_RESEARCH_TARGET;

function artifact(type: string, index: number): ArtifactProof {
  return {
    runId: ID.runId, stageCode: "RESEARCH", artifactType: type,
    artifactId: "71000000-0000-4000-8000-" + String(index).padStart(12, "0"),
    version: 2, sha256: "a".repeat(64), available: true, bytesVerified: true,
  };
}

/** Synthetic verified-proof fixture, never a substitute for the trusted byte reader. */
function snapshot(): PilotageSnapshot {
  const researchPin = pins.contract_pins.research_stage;
  return {
    environment: { ...SHADOW },
    run: {
      runId: ID.runId, company: "Schneider Electric", companyCommandName: "SCHNEIDER ELECTRIC",
      issuerId: ID.issuerId, securityId: ID.securityId,
      dossierId: "44000000-0000-4000-8000-000000000001",
      dataCutoff: ID.dataCutoff, processVersion: "3.1", pilotageVersion: "3.1",
      contractSetSha256: pins.contract_set_sha256,
      stateVersion: 5, status: "ACTIVE", currentStage: "RESEARCH",
    },
    stages: {
      RESEARCH: {
        code: "RESEARCH", lifecycle: "COMPLETE", stateVersion: 3, revision: 1,
        contractName: researchPin.name, contractVersion: researchPin.version,
        contractSha256: researchPin.content_sha256, consumedPredecessorRevision: null,
        manifest: {
          ...artifact("RESEARCH_STAGE_MANIFEST", 99),
          artifactId: ID.manifestArtifactId, version: ID.manifestVersion,
          sha256: ID.manifestSha256, kind: "FINAL", stageRevision: 1,
          gatesVerifiedAgainstBytes: true,
        },
        artifacts: RESEARCH_V2_FINAL_OUTPUTS.map((type, i) => artifact(type, i+1)),
        gates: { READY_FOR_DEEP_DIVE: true, FINAL_OUTPUTS_VERIFIED: true },
      },
    },
    expectedRunStateVersion: 5,
    expectedCurrentStageStateVersion: 3,
  };
}

function rejects(snapshot: PilotageSnapshot, code: string) {
  assert.throws(() => assessVerifiedSchneiderShadowSnapshot(snapshot),
    (error: unknown) => error instanceof SchneiderShadowPreflightError &&
      error.code === code, code);
}

test("Schneider SHADOW pre-start: exact Research FINAL V2 only, no mutation/start/publication", () => {
  const v = assessVerifiedSchneiderShadowSnapshot(snapshot());
  assert.equal(v.outcome, "PRESTART_EVIDENCE_VERIFIED");
  assert.equal(v.runId, SCHNEIDER_SHADOW_RUN_ID);
  assert.equal(v.verifiedResearchOutputs, 7);
  assert.equal(v.manifest.sha256, ID.manifestSha256);
  assert.equal(v.expectedRunStateVersion, 5);
  assert.equal(v.expectedResearchStageStateVersion, 3);
  assert.equal(v.startAuthorized, false);
  assert.equal(v.startPerformed, false);
  assert.equal(v.mutationPerformed, false);
  assert.equal(v.publicationAuthorized, false);
});

test("SHADOW identity cannot be swapped for production", () => {
  const s = snapshot();
  s.environment.projectRef = "cugpgtzygqqlxetyeven";
  rejects(s, "SHADOW_ENVIRONMENT_MISMATCH");
});

test("another run cannot inherit Schneider's pre-start verdict", () => {
  const s = snapshot();
  s.run.runId = "11111111-1111-4111-8111-111111111111";
  rejects(s, "SCHNEIDER_RUN_IDENTITY_MISMATCH");
});

test("Research closure alone cannot bypass the exact sealed FINAL manifest", () => {
  const s = snapshot();
  s.stages.RESEARCH!.manifest!.sha256 = "b".repeat(64);
  rejects(s, "SCHNEIDER_FINAL_MANIFEST_NOT_EXACT");
});

test("Research V2 never substitutes historical artifact names", () => {
  const s = snapshot();
  s.stages.RESEARCH!.artifacts = s.stages.RESEARCH!.artifacts.map(a =>
    a.artifactType === "CLASSIFICATION_INPUT_RECORD"
      ? { ...a, artifactType: "ANALYSIS_INPUT_LOCK" } : a);
  rejects(s, "SCHNEIDER_OUTPUT_NOT_VERIFIED_CLASSIFICATION_INPUT_RECORD");
});

test("missing byte proof prevents even an advisory pre-start result", () => {
  const s = snapshot();
  s.stages.RESEARCH!.artifacts = s.stages.RESEARCH!.artifacts.map(a =>
    a.artifactType === "EVIDENCE_LEDGER" ? { ...a, bytesVerified: false } : a);
  rejects(s, "SCHNEIDER_OUTPUT_NOT_VERIFIED_EVIDENCE_LEDGER");
});

test("no pre-start admission once a downstream stage exists", () => {
  const s = snapshot();
  s.stages.DEEP_DIVE = {
    code: "DEEP_DIVE", lifecycle: "NOT_STARTED", stateVersion: 1, revision: 1,
    contractName: pins.contract_pins.deep_dive_stage.name,
    contractVersion: pins.contract_pins.deep_dive_stage.version,
    contractSha256: pins.contract_pins.deep_dive_stage.content_sha256,
    consumedPredecessorRevision: null, manifest: null, artifacts: [], gates: {},
  };
  rejects(s, "SCHNEIDER_DOWNSTREAM_STAGE_ALREADY_EXISTS");
});

test("changed Registry state versions fail closed before an explicit start", () => {
  const s = snapshot();
  s.expectedRunStateVersion--;
  rejects(s, "SCHNEIDER_PREFLIGHT_STALE_CAS");
});

test("Research must be COMPLETE with the persisted gate YES", () => {
  const s = snapshot();
  s.stages.RESEARCH!.lifecycle = "IN_PROGRESS";
  rejects(s, "SCHNEIDER_RESEARCH_NOT_COMPLETE");
});

test("production-only loader continues to refuse SHADOW before reading rows", async () => {
  const neverRead: TrustedPilotagePort = {
    getRun: async () => { throw Error("unexpected getRun"); },
    getStages: async () => { throw Error("unexpected getStages"); },
    getArtifacts: async () => { throw Error("unexpected getArtifacts"); },
    getIssuer: async () => { throw Error("unexpected getIssuer"); },
    readImmutableBytes: async () => { throw Error("unexpected readImmutableBytes"); },
  };
  await assert.rejects(loadTrustedPilotageSnapshot({
    port: neverRead, environment: SHADOW, runId: SCHNEIDER_SHADOW_RUN_ID,
    expectedRunStateVersion: 5, expectedCurrentStageStateVersion: 3,
  }), /WRONG_ENVIRONMENT/);
});

test("Schneider SHADOW loader refuses any other run before Registry access", async () => {
  const neverRead: TrustedPilotagePort = {
    getRun: async () => { throw Error("unexpected getRun"); },
    getStages: async () => { throw Error("unexpected getStages"); },
    getArtifacts: async () => { throw Error("unexpected getArtifacts"); },
    getIssuer: async () => { throw Error("unexpected getIssuer"); },
    readImmutableBytes: async () => { throw Error("unexpected readImmutableBytes"); },
  };
  await assert.rejects(loadTrustedSchneiderShadowSnapshot({
    port: neverRead, environment: SHADOW,
    runId: "11111111-1111-4111-8111-111111111111",
    expectedRunStateVersion: 5, expectedCurrentStageStateVersion: 3,
  }), (error: unknown) => error instanceof TrustedLoaderError &&
    error.code === "SHADOW_RUN_NOT_ALLOWLISTED");
});

test("Schneider SHADOW loader rejects spoofed project name before Registry access", async () => {
  const neverRead: TrustedPilotagePort = {
    getRun: async () => { throw Error("unexpected getRun"); },
    getStages: async () => { throw Error("unexpected getStages"); },
    getArtifacts: async () => { throw Error("unexpected getArtifacts"); },
    getIssuer: async () => { throw Error("unexpected getIssuer"); },
    readImmutableBytes: async () => { throw Error("unexpected readImmutableBytes"); },
  };
  await assert.rejects(loadTrustedSchneiderShadowSnapshot({
    port: neverRead, environment: { ...SHADOW, projectName: "orotitan-screener" },
    runId: SCHNEIDER_SHADOW_RUN_ID,
    expectedRunStateVersion: 5, expectedCurrentStageStateVersion: 3,
  }), (error: unknown) => error instanceof TrustedLoaderError &&
    error.code === "WRONG_SHADOW_ENVIRONMENT");
});
