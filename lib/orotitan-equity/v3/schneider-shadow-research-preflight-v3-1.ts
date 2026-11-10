/**
 * Schneider J2 Research V2 FINAL: SHADOW-only, read-only pre-start attestation.
 *
 * This is NOT Pilotage production runtime admission, nor a stage-transition RPC.
 * The trusted loader verifies immutable Git bytes + Registry rows before this
 * advisory result can be returned. A separate human GO and database CAS are
 * still required before start_orotitan_stage.
 */
import { activeContractPinPack } from "./runtime-bootstrap-v3-0-3";
import { RESEARCH_V2_FINAL_OUTPUTS } from "../v1/research-required-outputs";
import {
  loadTrustedSchneiderShadowSnapshot,
  SCHNEIDER_SHADOW_RUN_ID,
  type TrustedPilotageSnapshotInput,
} from "./pilotage-trusted-loader-v3-1";
import type { PilotageSnapshot } from "./pilotage-adapter-v3-1";

export const SCHNEIDER_FINAL_RESEARCH_TARGET = Object.freeze({
  runId: SCHNEIDER_SHADOW_RUN_ID,
  issuerId: "81736c80-9d6a-4f4e-bbc1-1b418912d94a",
  securityId: "e60d26ba-925c-4e02-95b0-6ffd53a73d42",
  dataCutoff: "2026-10-30",
  manifestArtifactId: "7f000000-0000-4000-8000-000000000001",
  manifestVersion: 2,
  manifestSha256: "2ebe9e54ac4126680f6cc81dcadb919a3c4b2056773eaedfc99485a859717395",
});

export class SchneiderShadowPreflightError extends Error {
  constructor(readonly code: string) { super(code); }
}

function need(condition: unknown, code: string): asserts condition {
  if (condition !== true) throw new SchneiderShadowPreflightError(code);
}

/** Pure, advisory assessment. Never accept an untrusted caller's snapshot as verified evidence. */
export function assessVerifiedSchneiderShadowSnapshot(snapshot: PilotageSnapshot) {
  const r = snapshot.run;
  const environment = snapshot.environment;
  need(environment.projectRef === "awgsurdyvsyolcgpnygh" &&
    environment.projectName === "orotitan-vnext-shadow" &&
    environment.region === "eu-west-3" &&
    environment.status === "ACTIVE_HEALTHY", "SHADOW_ENVIRONMENT_MISMATCH");
  need(r.runId === SCHNEIDER_FINAL_RESEARCH_TARGET.runId &&
    r.issuerId === SCHNEIDER_FINAL_RESEARCH_TARGET.issuerId &&
    r.securityId === SCHNEIDER_FINAL_RESEARCH_TARGET.securityId &&
    r.dataCutoff === SCHNEIDER_FINAL_RESEARCH_TARGET.dataCutoff,
    "SCHNEIDER_RUN_IDENTITY_MISMATCH");
  need(r.processVersion === "3.1" && r.pilotageVersion === "3.1" &&
    r.contractSetSha256 === activeContractPinPack.contract_set_sha256,
    "SCHNEIDER_CONTRACT_SET_MISMATCH");
  need(r.status === "ACTIVE" && r.currentStage === "RESEARCH",
    "SCHNEIDER_NOT_AT_RESEARCH_PRESTART");
  need(Object.keys(snapshot.stages).length === 1 &&
    snapshot.stages.DEEP_DIVE === undefined &&
    snapshot.stages.INTEGRATION === undefined,
    "SCHNEIDER_DOWNSTREAM_STAGE_ALREADY_EXISTS");
  const research = snapshot.stages.RESEARCH;
  need(research !== undefined && research.code === "RESEARCH" &&
    research.lifecycle === "COMPLETE" && research.revision === 1,
    "SCHNEIDER_RESEARCH_NOT_COMPLETE");
  const expectedPin = activeContractPinPack.contract_pins.research_stage;
  need(research.contractName === expectedPin.name &&
    research.contractVersion === expectedPin.version &&
    research.contractSha256 === expectedPin.content_sha256,
    "SCHNEIDER_RESEARCH_PIN_MISMATCH");
  const manifest = research.manifest;
  need(manifest !== null && manifest.kind === "FINAL" &&
    manifest.artifactType === "RESEARCH_STAGE_MANIFEST" &&
    manifest.artifactId === SCHNEIDER_FINAL_RESEARCH_TARGET.manifestArtifactId &&
    manifest.version === SCHNEIDER_FINAL_RESEARCH_TARGET.manifestVersion &&
    manifest.sha256 === SCHNEIDER_FINAL_RESEARCH_TARGET.manifestSha256 &&
    manifest.stageRevision === research.revision &&
    manifest.bytesVerified === true &&
    manifest.gatesVerifiedAgainstBytes === true,
    "SCHNEIDER_FINAL_MANIFEST_NOT_EXACT");
  need(research.gates.READY_FOR_DEEP_DIVE === true &&
    research.gates.FINAL_OUTPUTS_VERIFIED === true,
    "SCHNEIDER_RESEARCH_HANDOFF_GATE_MISSING");
  need(research.artifacts.length === RESEARCH_V2_FINAL_OUTPUTS.length &&
    new Set(research.artifacts.map(a => a.artifactType)).size === RESEARCH_V2_FINAL_OUTPUTS.length,
    "SCHNEIDER_OUTPUT_SET_NOT_EXACT");
  for (const name of RESEARCH_V2_FINAL_OUTPUTS) {
    const matching = research.artifacts.filter(a => a.artifactType === name &&
      a.runId === r.runId && a.stageCode === "RESEARCH" &&
      a.version === 2 && a.available === true && a.bytesVerified === true);
    need(matching.length === 1, "SCHNEIDER_OUTPUT_NOT_VERIFIED_" + name);
  }
  need(r.stateVersion === snapshot.expectedRunStateVersion &&
    research.stateVersion === snapshot.expectedCurrentStageStateVersion,
    "SCHNEIDER_PREFLIGHT_STALE_CAS");

  return {
    outcome: "PRESTART_EVIDENCE_VERIFIED" as const,
    environment: "SHADOW" as const,
    runId: r.runId,
    currentStage: "RESEARCH" as const,
    nextStage: "DEEP_DIVE" as const,
    manifest: {
      artifactId: manifest.artifactId,
      version: manifest.version,
      sha256: manifest.sha256,
    },
    verifiedResearchOutputs: RESEARCH_V2_FINAL_OUTPUTS.length,
    expectedRunStateVersion: r.stateVersion,
    expectedResearchStageStateVersion: research.stateVersion,
    readyForSeparateStartAuthorization: true as const,
    startAuthorized: false as const,
    startPerformed: false as const,
    publicationAuthorized: false as const,
    mutationPerformed: false as const,
  };
}

/**
 * Server-owned port only. Re-reads Registry state after byte verification.
 * Never invokes start_orotitan_stage, deploys, writes or publishes.
 */
export async function preflightSchneiderShadowResearchReadOnly(input: TrustedPilotageSnapshotInput) {
  const verifiedSnapshot = await loadTrustedSchneiderShadowSnapshot(input);
  return assessVerifiedSchneiderShadowSnapshot(verifiedSnapshot);
}
