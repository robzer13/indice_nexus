/**
 * OroTitan V3.1 trusted Registry reader / verifier (D2, no side effects).
 *
 * The port is server-owned. All rows are re-read after byte verification to
 * detect stale multi-query state. Any subsequent write still requires DB CAS.
 * Never pass raw client input as verified evidence.
 */
import { createHash } from "node:crypto";
import { isDeepStrictEqual } from "node:util";
import {
  type ArtifactProof, type ManifestProof, type PilotageSnapshot,
  type StageCode, type StageState, planPilotageV31,
} from "./pilotage-adapter-v3-1";
import { activeContractPinPack, assertProductionEnvironment, assertRuntimeBootstrapIntegrity,
  type RuntimeEnvironmentObservation } from "./runtime-bootstrap-v3-0-3";
import { validateStageManifest, type StageManifestValidationContext } from "../v1/stage-manifest";

export type RunRow = {
  run_id: string; issuer_id: string | null; security_id: string | null;
  dossier_id: string | null; baseline_snapshot_id: string | null;
  canonical_mode: string; run_type: "INITIAL" | "REFRESH" | null;
  run_status: PilotageSnapshot["run"]["status"]; current_stage: StageCode | null;
  data_cutoff: string; process_version: string; pilotage_contract_version: string;
  contract_pins: unknown; contract_set_sha256: string; state_version: number;
};
export type StageRow = {
  run_id: string; stage_code: StageCode; stage_revision: number;
  stage_contract_name: string; stage_contract_version: string; stage_contract_sha256: string;
  lifecycle_status: StageState["lifecycle"]; handoff_gate_name: string;
  handoff_gate_state: "NOT_EVALUATED" | "YES" | "NO";
  active_manifest_artifact_id: string | null; active_manifest_version: number | null;
  active_manifest_kind: "CHECKPOINT" | "FINAL" | null;
  blocker_summary: unknown; state_version: number;
};
export type ArtifactRow = {
  artifact_id: string; version: number; run_id: string; stage_code: StageCode;
  artifact_type: string; authority_class: string; artifact_status: string;
  authority_state: string; availability_state: string; media_type: string;
  size_bytes: number; hash_algorithm: string; content_sha256: string;
  storage_backend: string; storage_uri: string; github_repository: string | null;
  github_path: string | null; github_commit_sha: string | null; github_blob_sha: string | null;
  manifest_artifact_id: string | null; manifest_version: number | null;
};
export type IdentityRow = { issuer_id: string; display_name: string; legal_name: string | null };
export type TrustedPilotagePort = {
  getRun(runId: string): Promise<RunRow | null>;
  getStages(runId: string): Promise<StageRow[]>;
  getArtifacts(runId: string): Promise<ArtifactRow[]>;
  getIssuer(issuerId: string): Promise<IdentityRow | null>;
  readImmutableBytes(row: ArtifactRow): Promise<Uint8Array>;
};

export class TrustedLoaderError extends Error {
  constructor(readonly code: string, details?: string) { super(details ? code + ": " + details : code); }
}
const UUID = /^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i;
const SHA1 = /^[a-f0-9]{40}$/;
const SHA256 = /^[a-f0-9]{64}$/;
const MAX_ROWS = 3000;
const MAX_ARTIFACT_BYTES = 35 * 1024 * 1024;
const STAGES: readonly StageCode[] = ["RESEARCH", "DEEP_DIVE", "INTEGRATION"];
const GATE: Record<StageCode, string> = {
  RESEARCH: "READY_FOR_DEEP_DIVE",
  DEEP_DIVE: "READY_FOR_INTEGRATION",
  INTEGRATION: "READY_TO_PUBLISH",
};
const REQUIRED: Record<StageCode, readonly string[]> = {
  RESEARCH: ["RESEARCH_SOURCE_MANIFEST", "EVIDENCE_LEDGER", "CONFLICT_LEDGER",
    "MATERIAL_RESEARCH_HYPOTHESIS_REGISTER", "RESEARCH_GAP_REGISTER",
    "DD_INPUT_SUFFICIENCY_RECORD", "ANALYSIS_INPUT_LOCK"],
  DEEP_DIVE: ["DEEP_DIVE_REPORT", "EVIDENCE_LEDGER", "CONFLICT_LEDGER",
    "CALCULATION_LEDGER", "MATERIAL_ASSUMPTION_REGISTER", "ANALYTICAL_BLOCK_OUTPUTS",
    "CROSS_BLOCK_RECONCILIATION_RECORD", "RED_TEAM_PREMORTEM_RECORD",
    "VALUATION_ARTIFACT", "CERTIFICATION_ARTIFACT",
    "OROTITAN_TERMINAL_GATE_ARTIFACT", "READINESS_NEXT_ACTION_ARTIFACT"],
  INTEGRATION: ["CANONICAL_SNAPSHOT_CANDIDATE", "INTEGRATION_MAPPING_RECORD",
    "SCHEMA_VALIDATION_REPORT", "I2_RECONCILIATION_REPORT",
    "HISTORY_TRANSITION_VALIDATION_REPORT", "I3B_ADMISSION_REPORT",
    "PRE_PUBLICATION_CONTROL_CARD"],
};

function check(value: unknown, code: string): asserts value {
  if (!value) throw new TrustedLoaderError(code);
}
function isObject(v: unknown): v is Record<string, unknown> {
  return v !== null && typeof v === "object" && !Array.isArray(v);
}
function requireExactRows<T>(rows: T[], label: string): void {
  check(Array.isArray(rows) && rows.length < MAX_ROWS, "REGISTRY_ROWS_UNBOUNDED_" + label);
}
function key(a: Pick<ArtifactRow, "artifact_id" | "version">): string {
  return a.artifact_id + ":" + a.version;
}
function expectedAuthority(kind: "FINAL" | "CHECKPOINT"): [string, string] {
  return kind === "FINAL" ? ["AUTHORITATIVE_STAGE_OUTPUT", "AUTHORITATIVE"] :
    ["CHECKPOINT_STAGE_OUTPUT", "CHECKPOINT"];
}
function bytesProof(row: ArtifactRow, bytes: Uint8Array): ArtifactProof {
  check(row.storage_backend === "PRIVATE_GITHUB", "IMMUTABLE_BACKEND_REQUIRED");
  check(row.storage_uri.startsWith("github://"), "IMMUTABLE_STORAGE_URI_INVALID");
  check(row.github_repository !== null && row.github_path !== null &&
    row.github_commit_sha !== null && row.github_blob_sha !== null, "IMMUTABLE_COORDINATES_MISSING");
  check(SHA1.test(row.github_commit_sha) && SHA1.test(row.github_blob_sha) &&
    SHA256.test(row.content_sha256), "ARTIFACT_HASH_FORMAT_INVALID");
  check(row.storage_uri === "github://" + row.github_repository + "@" +
    row.github_commit_sha + "/" + row.github_path, "IMMUTABLE_URI_MISMATCH");
  check(row.hash_algorithm === "SHA-256" && row.artifact_status === "SEALED" &&
    row.availability_state === "AVAILABLE", "ARTIFACT_INACTIVE");
  check(bytes.byteLength <= MAX_ARTIFACT_BYTES, "ARTIFACT_EXCEEDS_MAX_BYTES");
  check(Number.isSafeInteger(row.size_bytes) && row.size_bytes >= 0 &&
    row.size_bytes === bytes.byteLength, "ARTIFACT_SIZE_MISMATCH");
  check(createHash("sha256").update(bytes).digest("hex") === row.content_sha256,
    "ARTIFACT_SHA256_MISMATCH");
  const gitHeader = Buffer.from("blob " + bytes.length + "\0", "utf8");
  const gitBlob = createHash("sha1").update(gitHeader).update(bytes).digest("hex");
  check(gitBlob === row.github_blob_sha, "ARTIFACT_GIT_BLOB_SHA_MISMATCH");
  return {
    runId: row.run_id, stageCode: row.stage_code, artifactType: row.artifact_type,
    artifactId: row.artifact_id, version: row.version, sha256: row.content_sha256,
    available: true, bytesVerified: true,
  };
}
function readJson(bytes: Uint8Array): Record<string, unknown> {
  let value: unknown;
  try {
    value = JSON.parse(new TextDecoder("utf-8", { fatal: true }).decode(bytes));
  } catch {
    throw new TrustedLoaderError("MANIFEST_BYTES_NOT_VALID_JSON");
  }
  check(isObject(value), "MANIFEST_NOT_JSON_OBJECT");
  return value;
}
function matchingOutput(ref: unknown, row: ArtifactRow): boolean {
  if (!isObject(ref) || !isObject(ref.storage_ref)) return false;
  const store = ref.storage_ref;
  return ref.artifact_id === row.artifact_id &&
    ref.version === row.version && ref.artifact_type === row.artifact_type &&
    ref.content_sha256 === row.content_sha256 && ref.authority_class === row.authority_class &&
    ref.media_type === row.media_type && ref.size_bytes === row.size_bytes &&
    store.backend === row.storage_backend &&
    store.repository === row.github_repository && store.path === row.github_path &&
    // Post-V1.8 locations may include commit/blob URI; if present they must match.
    (store.commit_sha === undefined || store.commit_sha === row.github_commit_sha) &&
    (store.blob_sha === undefined || store.blob_sha === row.github_blob_sha) &&
    (store.storage_uri === undefined || store.storage_uri === row.storage_uri);
}
function expectedStageContext(run: RunRow, st: StageRow): StageManifestValidationContext {
  return {
    runId: run.run_id, stage: st.stage_code, stageRevision: st.stage_revision,
    issuerId: run.issuer_id, securityId: run.security_id, dossierId: run.dossier_id,
    canonicalMode: run.canonical_mode, runType: run.run_type, dataCutoff: run.data_cutoff,
    baselineSnapshotId: run.baseline_snapshot_id, processVersion: run.process_version,
    pilotageContractVersion: run.pilotage_contract_version,
    contractSetSha256: run.contract_set_sha256, stageContractName: st.stage_contract_name,
    stageContractVersion: st.stage_contract_version, stageContractSha256: st.stage_contract_sha256,
  };
}
function locateManifestRow(stage: StageRow, rows: ArtifactRow[]): ArtifactRow | null {
  if (stage.active_manifest_artifact_id === null) {
    check(stage.active_manifest_version === null && stage.active_manifest_kind === null,
      "ACTIVE_MANIFEST_TUPLE_INCOMPLETE");
    return null;
  }
  check(Number.isSafeInteger(stage.active_manifest_version) &&
    stage.active_manifest_kind !== null, "ACTIVE_MANIFEST_TUPLE_INCOMPLETE");
  const matches = rows.filter(x => x.artifact_id === stage.active_manifest_artifact_id &&
    x.version === stage.active_manifest_version && x.run_id === stage.run_id &&
    x.stage_code === stage.stage_code);
  check(matches.length === 1, "ACTIVE_MANIFEST_REGISTRATION_NOT_EXACT");
  return matches[0];
}

type VerifiedStage = { state: StageState; manifest: Record<string, unknown> | null; row: StageRow };

async function verifyStage(
  port: TrustedPilotagePort, run: RunRow, row: StageRow, artifacts: ArtifactRow[],
): Promise<VerifiedStage> {
  check(row.run_id === run.run_id && STAGES.includes(row.stage_code),
    "STAGE_RUN_ID_MISMATCH");
  const matching = artifacts.filter(a => a.run_id === run.run_id && a.stage_code === row.stage_code);
  const manifestRow = locateManifestRow(row, matching);
  const state: StageState = {
    code: row.stage_code, lifecycle: row.lifecycle_status,
    stateVersion: row.state_version, revision: row.stage_revision,
    contractName: row.stage_contract_name, contractVersion: row.stage_contract_version,
    contractSha256: row.stage_contract_sha256, consumedPredecessorRevision: null,
    manifest: null, artifacts: [], gates: {},
  };
  if (!manifestRow) {
    check(row.lifecycle_status === "NOT_STARTED" || row.lifecycle_status === "IN_PROGRESS" ||
      row.lifecycle_status === "BLOCKED" || row.lifecycle_status === "PAUSED", "COMPLETE_WITHOUT_FINAL_MANIFEST");
    return { state, manifest: null, row };
  }
  const kind = row.active_manifest_kind!;
  const [authorityClass, authorityState] = expectedAuthority(kind);
  check(manifestRow.artifact_type === row.stage_code + "_STAGE_MANIFEST" &&
    manifestRow.authority_class === authorityClass && manifestRow.authority_state === authorityState &&
    manifestRow.media_type === "application/json", "MANIFEST_AUTHORITY_INVALID");
  const manifestBytes = await port.readImmutableBytes(manifestRow);
  const mp = bytesProof(manifestRow, manifestBytes);
  const json = readJson(manifestBytes);
  check(json.manifest_id === manifestRow.artifact_id &&
    json.manifest_kind === kind && json.stage === row.stage_code &&
    json.run_id === row.run_id, "MANIFEST_IDENTITY_MISMATCH");
  const schema = validateStageManifest(json, expectedStageContext(run, row));
  check(schema.ok, "MANIFEST_SCHEMA_OR_CONTRACT_INVALID");
  check(isDeepStrictEqual(json.contract_pins, run.contract_pins), "MANIFEST_CONTRACT_PINS_MISMATCH");
  const handoff = json.handoff_gate;
  check(isObject(handoff) && handoff.name === GATE[row.stage_code] &&
    handoff.state === row.handoff_gate_state, "MANIFEST_HANDOFF_GATE_MISMATCH");
  check(Array.isArray(json.critical_blockers) && Array.isArray(row.blocker_summary) &&
    isDeepStrictEqual(json.critical_blockers, row.blocker_summary), "MANIFEST_BLOCKER_MISMATCH");
  check(row.lifecycle_status === json.stage_status, "MANIFEST_STAGE_STATUS_MISMATCH");
  const outputs = json.output_artifacts;
  check(Array.isArray(outputs), "MANIFEST_OUTPUTS_MALFORMED");
  const registrations = matching.filter(a => a.manifest_artifact_id === manifestRow.artifact_id &&
    a.manifest_version === manifestRow.version);
  check(registrations.length === outputs.length, "MANIFEST_OUTPUT_COUNT_MISMATCH");
  const expectedByKey = new Map<string, unknown>();
  for (const value of outputs) {
    check(isObject(value) && typeof value.artifact_id === "string" &&
      Number.isSafeInteger(value.version), "MANIFEST_OUTPUT_REF_MALFORMED");
    const k = String(value.artifact_id) + ":" + value.version;
    check(!expectedByKey.has(k), "MANIFEST_OUTPUT_REF_DUPLICATE");
    expectedByKey.set(k, value);
  }
  const verified: ArtifactProof[] = [];
  for (const artifact of registrations) {
    const reference = expectedByKey.get(key(artifact));
    check(matchingOutput(reference, artifact), "OUTPUT_ARTIFACT_MANIFEST_MISMATCH");
    check(artifact.authority_class === authorityClass &&
      artifact.authority_state === authorityState, "OUTPUT_ARTIFACT_AUTHORITY_MISMATCH");
    verified.push(bytesProof(artifact, await port.readImmutableBytes(artifact)));
  }
  const m: ManifestProof = {
    ...mp, kind, stageRevision: row.stage_revision, gatesVerifiedAgainstBytes: true,
  };
  state.manifest = m;
  state.artifacts = verified;
  if (kind === "FINAL" && row.lifecycle_status === "COMPLETE" &&
    row.handoff_gate_state === "YES" && json.critical_blockers.length === 0 &&
    REQUIRED[row.stage_code].every(type => verified.some(a => a.artifactType === type))) {
    state.gates.FINAL_OUTPUTS_VERIFIED = true;
    if (row.stage_code === "RESEARCH") state.gates.READY_FOR_DEEP_DIVE = true;
    if (row.stage_code === "DEEP_DIVE") state.gates.READY_FOR_INTEGRATION = true;
    if (row.stage_code === "INTEGRATION") state.gates.READY_TO_PUBLISH = true;
  }
  // CHECKPOINT phase gates and Integration I2/I3-B PASS are deliberately NOT
  // inferred from the existence of output files. They require dedicated
  // authoritative validators of the actual typed artifact content.
  return { state, manifest: json, row };
}

function validateParentProof(previous: VerifiedStage, current: VerifiedStage): void {
  check(previous.state.lifecycle === "COMPLETE" && previous.state.manifest?.kind === "FINAL",
    "PREDECESSOR_NOT_FINAL");
  const ref = previous.state.manifest;
  if (current.manifest !== null) {
    const parents = current.manifest.parent_manifests;
    check(Array.isArray(parents) && parents.some(p => isObject(p) &&
      p.manifest_id === ref.artifactId && p.version === ref.version &&
      p.content_sha256 === ref.sha256), "PARENT_MANIFEST_LINEAGE_NOT_VERIFIED");
  } else {
    // Pre-start stage has no persisted consumption yet. Only current, read-verified
    // upstream FINAL is acceptable as a *proposed* handoff. DB CAS still governs start.
    check(current.state.lifecycle === "NOT_STARTED", "ACTIVE_STAGE_WITHOUT_PARENT_MANIFEST");
  }
  current.state.consumedPredecessorRevision = previous.state.revision;
}

function requireRun(r: RunRow, expectedRunId: string): void {
  check(r.run_id === expectedRunId && UUID.test(r.run_id) && UUID.test(r.issuer_id ?? "") &&
    UUID.test(r.security_id ?? "") && UUID.test(r.dossier_id ?? ""), "RUN_IDENTITY_MISMATCH");
  check(r.contract_set_sha256 === activeContractPinPack.contract_set_sha256 &&
    isDeepStrictEqual(r.contract_pins, activeContractPinPack.contract_pins), "PIN_PACK_NOT_EXACT");
  check(r.current_stage !== null && STAGES.includes(r.current_stage), "CURRENT_STAGE_INVALID");
  check(r.process_version === "3.1" && r.pilotage_contract_version === "3.1", "RUN_VERSION_MISMATCH");
}
function extractCompany(issuer: IdentityRow): string {
  check(issuer.display_name.length > 0 && issuer.display_name.length <= 256,
    "ISSUER_NAME_INVALID");
  return issuer.display_name;
}

/**
 * Authenticated caller only; port may NOT be supplied by a user-controlled client.
 * Throws on any incomplete/non-authoritative evidence. No SQL writes occur.
 */
export async function loadTrustedPilotageSnapshot(input: {
  port: TrustedPilotagePort;
  environment: RuntimeEnvironmentObservation;
  runId: string;
  expectedRunStateVersion: number;
  expectedCurrentStageStateVersion: number;
}): Promise<PilotageSnapshot> {
  assertRuntimeBootstrapIntegrity();
  assertProductionEnvironment(input.environment);
  check(UUID.test(input.runId), "RUN_ID_INVALID");
  const [run, stages, artifacts] = await Promise.all([
    input.port.getRun(input.runId), input.port.getStages(input.runId),
    input.port.getArtifacts(input.runId),
  ]);
  check(run !== null, "RUN_NOT_FOUND");
  requireRun(run, input.runId);
  requireExactRows(stages, "STAGES");
  requireExactRows(artifacts, "ARTIFACTS");
  check(stages.length > 0 && stages.length <= 3, "STAGE_COUNT_INVALID");
  check(new Set(stages.map(s => s.stage_code)).size === stages.length, "DUPLICATE_STAGES");
  check(artifacts.every(a => a.run_id === run.run_id && STAGES.includes(a.stage_code)), "ARTIFACT_SCOPE_INVALID");
  check(new Set(artifacts.map(a => key(a))).size === artifacts.length, "DUPLICATE_ARTIFACT_IDENTITIES");
  check(run.state_version === input.expectedRunStateVersion, "STALE_RUN_STATE");
  const current = stages.find(x => x.stage_code === run.current_stage);
  check(current !== undefined, "CURRENT_STAGE_MISSING");
  check(current.state_version === input.expectedCurrentStageStateVersion, "STALE_STAGE_STATE");
  const issuer = await input.port.getIssuer(run.issuer_id!);
  check(issuer !== null && issuer.issuer_id === run.issuer_id, "ISSUER_IDENTITY_NOT_FOUND");
  const verified = await Promise.all(stages.map(st => verifyStage(input.port, run, st, artifacts)));
  const index = new Map(verified.map(v => [v.state.code, v]));
  const stageStates: PilotageSnapshot["stages"] = {};
  for (const v of verified) stageStates[v.state.code] = v.state;
  if (run.current_stage === "DEEP_DIVE") {
    check(index.has("RESEARCH"), "RESEARCH_STAGE_NOT_FOUND");
    validateParentProof(index.get("RESEARCH")!, index.get("DEEP_DIVE")!);
  }
  if (run.current_stage === "INTEGRATION") {
    check(index.has("DEEP_DIVE"), "DEEP_DIVE_STAGE_NOT_FOUND");
    validateParentProof(index.get("DEEP_DIVE")!, index.get("INTEGRATION")!);
  }
  const [afterRun, afterStages, afterArtifacts, afterIssuer] = await Promise.all([
    input.port.getRun(input.runId), input.port.getStages(input.runId),
    input.port.getArtifacts(input.runId), input.port.getIssuer(run.issuer_id!),
  ]);
  check(isDeepStrictEqual(run, afterRun) && isDeepStrictEqual(stages, afterStages) &&
    isDeepStrictEqual(artifacts, afterArtifacts) && isDeepStrictEqual(issuer, afterIssuer),
    "REGISTRY_CHANGED_DURING_VERIFICATION");
  const company = extractCompany(issuer);
  return {
    environment: input.environment,
    run: {
      runId: run.run_id, company, companyCommandName: company,
      issuerId: run.issuer_id!, securityId: run.security_id!, dossierId: run.dossier_id!,
      dataCutoff: run.data_cutoff, processVersion: run.process_version,
      pilotageVersion: run.pilotage_contract_version, contractSetSha256: run.contract_set_sha256,
      stateVersion: run.state_version, status: run.run_status, currentStage: run.current_stage!,
    },
    stages: stageStates,
    expectedRunStateVersion: input.expectedRunStateVersion,
    expectedCurrentStageStateVersion: input.expectedCurrentStageStateVersion,
  };
}

export async function loadTrustedPilotageDecision(
  input: Parameters<typeof loadTrustedPilotageSnapshot>[0],
) {
  return planPilotageV31(await loadTrustedPilotageSnapshot(input));
}
