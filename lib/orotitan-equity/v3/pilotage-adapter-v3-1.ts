/**
 * OroTitan Pilotage V3.1: read-only admission planner (D1).
 * Inputs MUST originate from a trusted Registry reader that independently
 * verifies artifact bytes, hashes, manifest gates and complete lineage.
 * This pure function performs no RPC, mutation, dispatch or publication.
 */
import { createHash } from "node:crypto";
import pins from "../../../contracts/orotitan-equity/v3/contract-pin-pack-v3/OROTITAN_CONTRACT_PIN_PACK_V3_0_2.json";
import bootstrap from "../../../contracts/orotitan-equity/v3/runtime/OROTITAN_RUNTIME_BOOTSTRAP_V3.0.3.json";

export type StageCode = "RESEARCH" | "DEEP_DIVE" | "INTEGRATION";
export type Phase = "RESEARCH" | "FUNDAMENTALS" | "VALUATION" | "CERTIFICATION_RECONCILIATION" | "INTEGRATION";
export type Lifecycle = "NOT_STARTED" | "IN_PROGRESS" | "PAUSED" | "BLOCKED" | "COMPLETE";
export type Gate =
  | "READY_FOR_DEEP_DIVE" | "READY_FOR_VALUATION" | "FUNDAMENTAL_RED_TEAM_COMPLETE"
  | "DENOMINATOR_ADMISSION_VALID" | "VALUATION_DATE_ALIGNMENT_VALID"
  | "DCF_BASIS_ADMITTED" | "EV_BRIDGE_GOVERNED" | "READY_FOR_CERTIFICATION"
  | "READY_FOR_INTEGRATION" | "READY_TO_PUBLISH" | "CANONICAL_CANDIDATE_ADMITTED"
  | "I2_PASS" | "I3B_PASS" | "FINAL_OUTPUTS_VERIFIED";

export interface ArtifactProof {
  runId: string;
  stageCode: StageCode;
  artifactType: string;
  artifactId: string;
  version: number;
  sha256: string;
  available: boolean;
  bytesVerified: boolean;
}
export interface ManifestProof extends ArtifactProof {
  kind: "CHECKPOINT" | "FINAL";
  stageRevision: number;
  gatesVerifiedAgainstBytes: boolean;
}
export interface StageState {
  code: StageCode;
  lifecycle: Lifecycle;
  stateVersion: number;
  revision: number;
  contractName: string;
  contractVersion: string;
  contractSha256: string;
  consumedPredecessorRevision: number | null;
  manifest: ManifestProof | null;
  artifacts: readonly ArtifactProof[];
  gates: Partial<Record<Gate, boolean>>;
}
export interface PilotageSnapshot {
  environment: { projectRef: string; projectName: string; region: string; status: string };
  run: {
    runId: string;
    company: string;
    companyCommandName: string;
    issuerId: string;
    securityId: string;
    dossierId: string;
    dataCutoff: string;
    processVersion: string;
    pilotageVersion: string;
    contractSetSha256: string;
    stateVersion: number;
    status: "ACTIVE" | "PAUSED" | "BLOCKED" | "READY_TO_PUBLISH" | "PUBLISHED" | "CANCELLED";
    currentStage: StageCode;
  };
  stages: Partial<Record<StageCode, StageState>>;
  expectedRunStateVersion: number;
  expectedCurrentStageStateVersion: number;
}
export interface PilotageDecision {
  outcome: "CONTINUE" | "HANDOFF" | "AWAIT_EXPLICIT_GO_PUBLISH" | "DONE" | "BLOCKED";
  phase: Phase | null;
  reasonCode: string;
  exactPrompt: string | null;
  expectedRunStateVersion: number;
  expectedStageStateVersion: number;
  stateFingerprintSha256: string;
  publicationAuthorized: false;
  mutationPerformed: false;
}

const SHA = /^[0-9a-f]{64}$/;
const UUID = /^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i;
const STAGE_PINS = {
  RESEARCH: pins.contract_pins.research_stage,
  DEEP_DIVE: pins.contract_pins.deep_dive_stage,
  INTEGRATION: pins.contract_pins.integration_stage,
} as const;

class AdmissionError extends Error {
  constructor(readonly code: string) { super(code); }
}
function need(condition: unknown, code: string): asserts condition {
  if (condition !== true) throw new AdmissionError(code);
}
function stageFor(s: PilotageSnapshot, code: StageCode): StageState {
  const x = s.stages[code];
  need(x !== undefined && x.code === code, "STAGE_MISSING_OR_MISMATCH");
  need(Number.isSafeInteger(x.stateVersion) && x.stateVersion >= 1 &&
    Number.isSafeInteger(x.revision) && x.revision >= 1, "STAGE_VERSION_INVALID");
  const pin = STAGE_PINS[code];
  need(x.contractName === pin.name && x.contractVersion === pin.version &&
    x.contractSha256 === pin.content_sha256, "STAGE_CONTRACT_PIN_MISMATCH");
  return x;
}
function artifactValid(s: PilotageSnapshot, x: StageState, a: ArtifactProof): boolean {
  return a.runId === s.run.runId && a.stageCode === x.code && UUID.test(a.artifactId) &&
    Number.isSafeInteger(a.version) && a.version >= 1 && SHA.test(a.sha256) &&
    a.available === true && a.bytesVerified === true;
}
function requireArtifact(s: PilotageSnapshot, x: StageState, name: string): void {
  const matches = x.artifacts.filter(a => a.artifactType === name && artifactValid(s, x, a));
  need(matches.length === 1, "ARTIFACT_MISSING_UNVERIFIED_OR_AMBIGUOUS_" + name);
}
function requireArtifacts(s: PilotageSnapshot, x: StageState, names: readonly string[]): void {
  for (const name of names) requireArtifact(s, x, name);
}
function requireManifest(s: PilotageSnapshot, x: StageState, kind: "CHECKPOINT" | "FINAL"): void {
  const m = x.manifest;
  need(m !== null && m.kind === kind, "MANIFEST_KIND_MISMATCH");
  need(m.stageRevision === x.revision, "MANIFEST_REVISION_STALE");
  need(artifactValid(s, x, m) && m.gatesVerifiedAgainstBytes === true, "MANIFEST_NOT_BYTE_VERIFIED");
  need(m.artifactType.startsWith(x.code + "_STAGE_MANIFEST"), "MANIFEST_TYPE_MISMATCH");
}
function gate(x: StageState, key: Gate): void {
  need(x.gates[key] === true, "GATE_NOT_ADMITTED_" + key);
}
function requireFinal(s: PilotageSnapshot, x: StageState, gates: readonly Gate[], artifacts: readonly string[]): void {
  need(x.lifecycle === "COMPLETE", "UPSTREAM_STAGE_NOT_COMPLETE");
  requireManifest(s, x, "FINAL");
  for (const key of gates) gate(x, key);
  requireArtifacts(s, x, artifacts);
}
function requirePrevious(s: PilotageSnapshot, current: StageState, previous: StageCode): StageState {
  const p = stageFor(s, previous);
  need(current.consumedPredecessorRevision === p.revision, "UPSTREAM_REOPENED_OR_STALE_LINEAGE");
  return p;
}
function validate(s: PilotageSnapshot): StageState {
  const e = bootstrap.production_environment;
  need(s.environment.projectRef === e.project_ref && s.environment.projectName === e.project_name &&
    s.environment.region === e.region && s.environment.status === e.required_status, "WRONG_ENVIRONMENT");
  need(pins.version === "3.0.2" && bootstrap.version === "3.0.3" && bootstrap.orotitan_version === "3.1" &&
    pins.contract_set_sha256 === bootstrap.authority_boundary.contract_set_sha256, "ACTIVE_PIN_PACK_INCONSISTENT");
  const r = s.run;
  need(UUID.test(r.runId) && UUID.test(r.issuerId) && UUID.test(r.securityId) &&
    UUID.test(r.dossierId), "RUN_IDENTITY_INVALID");
  need(r.company.trim().length > 0 && r.companyCommandName.trim().length > 0 &&
    r.companyCommandName.length <= 256 && !/[\x00-\x1f\x7f]/.test(r.companyCommandName), "COMPANY_COMMAND_INVALID");
  need(/^\d{4}-\d{2}-\d{2}$/.test(r.dataCutoff), "DATA_CUTOFF_INVALID");
  need(r.processVersion === pins.contract_pins.process.version &&
    r.pilotageVersion === pins.contract_pins.pilotage.version &&
    r.contractSetSha256 === pins.contract_set_sha256, "RUN_CONTRACT_SET_MISMATCH");
  need(Number.isSafeInteger(r.stateVersion) && r.stateVersion >= 1, "RUN_STATE_VERSION_INVALID");
  need(r.stateVersion === s.expectedRunStateVersion, "STALE_RUN_CAS");
  const current = stageFor(s, r.currentStage);
  need(current.stateVersion === s.expectedCurrentStageStateVersion, "STALE_STAGE_CAS");
  return current;
}
function prompt(s: PilotageSnapshot, phase: Phase, x: StageState): string {
  const p = STAGE_PINS[x.code];
  // Handoffs consume the PREDECESSOR FINAL proof set, not the empty next stage.
  const evidenceStage = phase === "FUNDAMENTALS" ? s.stages.RESEARCH ?? x :
    phase === "INTEGRATION" ? s.stages.DEEP_DIVE ?? x : x;
  const refs = [evidenceStage.manifest, ...evidenceStage.artifacts]
    .filter((a): a is ArtifactProof => a !== null && artifactValid(s, evidenceStage, a))
    .map(a => ({
    artifact_id: a.artifactId, version: a.version, sha256: a.sha256, artifact_type: a.artifactType,
  }));
  return [
    "OROTITAN V3.1 — PILOTAGE HANDOFF",
    "COMPANY = " + s.run.company,
    "RUN_ID = " + s.run.runId,
    "DATA_CUTOFF = " + s.run.dataCutoff,
    "CONTRACT_SET_SHA256 = " + pins.contract_set_sha256,
    "REGISTRY_STAGE = " + x.code,
    "EXECUTION_PHASE = " + phase,
    "EXPECTED_STAGE_CONTRACT = " + p.name,
    "EXPECTED_STAGE_CONTRACT_VERSION = " + p.version,
    "EXPECTED_STAGE_CONTRACT_SHA256 = " + p.content_sha256,
    "EXPECTED_RUN_STATE_VERSION = " + s.run.stateVersion,
    "EXPECTED_STAGE_STATE_VERSION = " + x.stateVersion,
    "EXPECTED_INPUT_ARTIFACTS = " + JSON.stringify(refs),
    "PUBLICATION_AUTHORIZED = NO",
    "LOAD AND VERIFY AUTHORITATIVE REGISTRY STATE AND EXACT ARTIFACT BYTES BEFORE WORK.",
    "DO NOT USE THIS BOOTSTRAP AS ANALYTICAL EVIDENCE.",
    "FAIL CLOSED ON VERSION / ID / STATUS / HASH / LINEAGE MISMATCH.",
  ].join("\n");
}

/** Advisory only. Never performs an RPC, state transition, analytical computation or publication. */
export function planPilotageV31(s: PilotageSnapshot): PilotageDecision {
  const r = s.run;
  const x = s.stages[r.currentStage];
  const sv = x?.stateVersion ?? 0;
  const fp = createHash("sha256").update([
    "OROTITAN_PILOTAGE_V3_1_READONLY", r.runId, r.currentStage,
    String(r.stateVersion), String(sv), String(x?.revision ?? 0), x?.manifest?.sha256 ?? "NULL",
  ].join("\n"), "utf8").digest("hex");
  const result = (outcome: PilotageDecision["outcome"], phase: Phase | null, reasonCode: string, exactPrompt: string | null): PilotageDecision =>
    ({ outcome, phase, reasonCode, exactPrompt, expectedRunStateVersion: r.stateVersion,
       expectedStageStateVersion: sv, stateFingerprintSha256: fp,
       publicationAuthorized: false, mutationPerformed: false });
  try {
    const current = validate(s);
    if (r.status === "PUBLISHED") return result("DONE", null, "ALREADY_PUBLISHED", null);
    need(r.status === "ACTIVE" || r.status === "READY_TO_PUBLISH", "RUN_NOT_ACTIVE");
    need(current.lifecycle !== "BLOCKED" && current.lifecycle !== "PAUSED", "STAGE_BLOCKED_OR_PAUSED");
    if (r.currentStage === "RESEARCH") {
      need(r.status === "ACTIVE", "RUN_STATUS_STAGE_MISMATCH");
      need(current.lifecycle === "NOT_STARTED" || current.lifecycle === "IN_PROGRESS", "RESEARCH_STAGE_NOT_EXECUTABLE");
      need(current.manifest === null || current.lifecycle === "IN_PROGRESS", "RESEARCH_MANIFEST_INCONSISTENT");
      if (current.manifest) requireManifest(s, current, "CHECKPOINT");
      return result("CONTINUE", "RESEARCH", "RESEARCH_ADMITTED", prompt(s, "RESEARCH", current));
    }
    if (r.currentStage === "DEEP_DIVE") {
      need(r.status === "ACTIVE", "RUN_STATUS_STAGE_MISMATCH");
      const research = requirePrevious(s, current, "RESEARCH");
      requireFinal(s, research, ["READY_FOR_DEEP_DIVE", "FINAL_OUTPUTS_VERIFIED"], [
        "RESEARCH_SOURCE_MANIFEST", "EVIDENCE_LEDGER", "MATERIAL_RESEARCH_HYPOTHESIS_REGISTER",
        "RESEARCH_GAP_REGISTER", "INPUT_SUFFICIENCY_ASSESSMENT", "CLASSIFICATION_INPUT_RECORD",
      ]);
      need(current.lifecycle === "NOT_STARTED" || current.lifecycle === "IN_PROGRESS", "DEEP_DIVE_NOT_EXECUTABLE");
      if (current.lifecycle === "NOT_STARTED") {
        need(current.manifest === null, "NOT_STARTED_WITH_MANIFEST");
        return result("HANDOFF", "FUNDAMENTALS", "RESEARCH_FINAL_ADMITTED", prompt(s, "FUNDAMENTALS", current));
      }
      if (current.manifest === null) {
        return result("CONTINUE", "FUNDAMENTALS", "FUNDAMENTALS_IN_PROGRESS", prompt(s, "FUNDAMENTALS", current));
      }
      requireManifest(s, current, "CHECKPOINT");
      if (current.gates.READY_FOR_CERTIFICATION === true) {
        for (const g of ["READY_FOR_VALUATION", "FUNDAMENTAL_RED_TEAM_COMPLETE", "DENOMINATOR_ADMISSION_VALID",
          "VALUATION_DATE_ALIGNMENT_VALID", "DCF_BASIS_ADMITTED", "EV_BRIDGE_GOVERNED"] as const) gate(current, g);
        requireArtifacts(s, current, ["FUNDAMENTALS_LOCK", "VALUATION_LOCK", "FUNDAMENTAL_RED_TEAM_RECORD"]);
        return result("HANDOFF", "CERTIFICATION_RECONCILIATION", "VALUATION_LOCK_ADMITTED", prompt(s, "CERTIFICATION_RECONCILIATION", current));
      }
      if (current.gates.READY_FOR_VALUATION === true) {
        for (const g of ["FUNDAMENTAL_RED_TEAM_COMPLETE", "DENOMINATOR_ADMISSION_VALID",
          "VALUATION_DATE_ALIGNMENT_VALID", "DCF_BASIS_ADMITTED", "EV_BRIDGE_GOVERNED"] as const) gate(current, g);
        requireArtifacts(s, current, ["FUNDAMENTALS_LOCK", "FUNDAMENTAL_RED_TEAM_RECORD"]);
        return result("HANDOFF", "VALUATION", "FUNDAMENTALS_LOCK_ADMITTED", prompt(s, "VALUATION", current));
      }
      return result("BLOCKED", null, "DEEP_DIVE_CHECKPOINT_GATE_NOT_ADMITTED", null);
    }
    const dd = requirePrevious(s, current, "DEEP_DIVE");
    requireFinal(s, dd, ["READY_FOR_INTEGRATION", "FINAL_OUTPUTS_VERIFIED"], [
      "FUNDAMENTALS_LOCK", "VALUATION_LOCK", "CERTIFICATION_ARTIFACT", "DEEP_DIVE_REPORT",
    ]);
    if (current.lifecycle === "NOT_STARTED" || current.lifecycle === "IN_PROGRESS") {
      need(r.status === "ACTIVE", "RUN_STATUS_STAGE_MISMATCH");
      if (current.manifest) requireManifest(s, current, "CHECKPOINT");
      return result(current.lifecycle === "NOT_STARTED" ? "HANDOFF" : "CONTINUE",
        "INTEGRATION", "DEEP_DIVE_FINAL_ADMITTED", prompt(s, "INTEGRATION", current));
    }
    requireFinal(s, current, ["READY_TO_PUBLISH", "CANONICAL_CANDIDATE_ADMITTED",
      "I2_PASS", "I3B_PASS", "FINAL_OUTPUTS_VERIFIED"], [
      "CANONICAL_SNAPSHOT_CANDIDATE", "INTEGRATION_MAPPING_RECORD", "SCHEMA_VALIDATION_REPORT",
      "I2_RECONCILIATION_REPORT", "HISTORY_TRANSITION_VALIDATION_REPORT",
      "I3B_ADMISSION_REPORT", "PRE_PUBLICATION_CONTROL_CARD",
    ]);
    need(r.status === "READY_TO_PUBLISH", "RUN_NOT_READY_TO_PUBLISH");
    return result("AWAIT_EXPLICIT_GO_PUBLISH", null, "EXPLICIT_PUBLICATION_REQUIRED",
      "GO PUBLISH " + r.companyCommandName);
  } catch (error) {
    if (error instanceof AdmissionError) return result("BLOCKED", null, error.code, null);
    throw error;
  }
}
