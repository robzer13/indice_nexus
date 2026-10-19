/**
 * Read-only, schema-only compatibility projection for sealed legacy-shaped
 * V3.1 CHECKPOINT manifests. Raw Registry/GitHub manifest bytes stay immutable.
 * Never apply to FINAL; never use projected blocker codes for business gates.
 */
import type { ArtifactRow, RunRow, StageRow } from "./pilotage-trusted-loader-v3-1";

type JsonObject = Record<string, unknown>;
export type VerifiedCheckpointParent = {
  artifactId: string;
  version: number;
  sha256: string;
  runId: string;
  stageCode: string;
};

export class CheckpointCompatError extends Error {
  constructor(readonly code: string) { super(code); }
}
function guard(value: unknown, code: string): asserts value {
  if (!value) throw new CheckpointCompatError(code);
}
function object(value: unknown): value is JsonObject {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}
function exactKeys(value: JsonObject, allowed: readonly string[]): boolean {
  return Object.keys(value).every(k => allowed.includes(k));
}
function projectIssues(value: unknown, kind: "BLOCKER" | "LIMITATION"): unknown {
  guard(Array.isArray(value) && value.length <= 100, "CHECKPOINT_" + kind + "_LIST_INVALID");
  return value.map((issue, i) => {
    if (object(issue)) return issue;
    guard(typeof issue === "string" && issue.trim() !== "" && issue.length <= 4096,
      "CHECKPOINT_" + kind + "_STRING_INVALID");
    return {
      code: "LEGACY_CHECKPOINT_" + kind + "_" + (i + 1),
      summary: issue,
    };
  });
}
/**
 * Returns a *schema-validation* view only. Parent proofs are checked
 * cryptographically by the caller against registered immutable bytes.
 * The unmodified manifest is the ONLY source for decisions and blocker text.
 */
export function projectV31CheckpointForSchema(input: {
  manifest: JsonObject;
  run: RunRow;
  stage: StageRow;
  activeManifest: ArtifactRow;
  verifiedParents: readonly VerifiedCheckpointParent[];
}): JsonObject {
  const {manifest, run, stage, activeManifest, verifiedParents} = input;
  guard(run.process_version === "3.1" && run.pilotage_contract_version === "3.1" &&
    manifest.manifest_schema_version === "1.0.0" &&
    manifest.manifest_kind === "CHECKPOINT" &&
    stage.active_manifest_kind === "CHECKPOINT" &&
    stage.stage_revision === manifest.stage_revision &&
    manifest.stage === stage.stage_code && manifest.run_id === run.run_id &&
    manifest.manifest_id === activeManifest.artifact_id &&
    stage.active_manifest_artifact_id === activeManifest.artifact_id &&
    stage.active_manifest_version === activeManifest.version,
    "CHECKPOINT_PROJECTION_SCOPE_INVALID");

  const key = stage.stage_code === "RESEARCH" ? "research_stage" :
    stage.stage_code === "DEEP_DIVE" ? "deep_dive_stage" : "integration_stage";
  guard(object(manifest.contract_pins) && object(manifest.contract_pins[key]),
    "CHECKPOINT_STAGE_PIN_MISSING");
  const pin = manifest.contract_pins[key] as JsonObject;
  const rawStageContract = manifest.stage_contract;
  guard(object(rawStageContract) && exactKeys(rawStageContract,
    ["name", "version", "content_sha256", "locator"]) &&
    rawStageContract.name === pin.name &&
    rawStageContract.version === pin.version &&
    rawStageContract.content_sha256 === pin.content_sha256 &&
    rawStageContract.name === stage.stage_contract_name &&
    rawStageContract.version === stage.stage_contract_version &&
    rawStageContract.content_sha256 === stage.stage_contract_sha256,
    "CHECKPOINT_STAGE_CONTRACT_NOT_PINNED");
  guard(object(pin.locator), "CHECKPOINT_STAGE_PIN_LOCATOR_MISSING");
  if (Object.prototype.hasOwnProperty.call(rawStageContract, "locator")) {
    guard(JSON.stringify(rawStageContract.locator) === JSON.stringify(pin.locator),
      "CHECKPOINT_STAGE_CONTRACT_LOCATOR_MISMATCH");
  }

  const rawParents = manifest.parent_manifests;
  guard(Array.isArray(rawParents) && rawParents.length <= 1,
    "CHECKPOINT_PARENT_SHAPE_INVALID");
  const projectedParents = rawParents.map(parent => {
    if (object(parent) && Object.prototype.hasOwnProperty.call(parent, "manifest_id")) {
      guard(exactKeys(parent, ["manifest_id", "version", "content_sha256"]) &&
        verifiedParents.some(p => p.artifactId === parent.manifest_id &&
          p.version === parent.version && p.sha256 === parent.content_sha256 &&
          p.runId === run.run_id && p.stageCode === stage.stage_code),
        "CHECKPOINT_PARENT_REFERENCE_UNVERIFIED");
      return parent;
    }
    guard(object(parent) && exactKeys(parent,
      ["artifact_id", "version", "relation_type"]) &&
      parent.relation_type === "SUPERSEDES_PRIOR_CHECKPOINT" &&
      typeof parent.artifact_id === "string" &&
      Number.isSafeInteger(parent.version) && Number(parent.version) > 0 &&
      parent.version === activeManifest.version - 1,
      "CHECKPOINT_PARENT_SHAPE_INVALID");
    const exact = verifiedParents.filter(p =>
      p.artifactId === parent.artifact_id &&
      p.version === parent.version &&
      p.runId === run.run_id && p.stageCode === stage.stage_code);
    guard(exact.length === 1, "CHECKPOINT_PARENT_REFERENCE_UNVERIFIED");
    return {
      manifest_id: exact[0].artifactId,
      version: exact[0].version,
      content_sha256: exact[0].sha256,
    };
  });
  guard(verifiedParents.length === rawParents.length,
    "CHECKPOINT_PARENT_PROOF_COUNT_MISMATCH");

  return {
    ...manifest,
    stage_contract: {...rawStageContract, locator: pin.locator},
    critical_blockers: projectIssues(manifest.critical_blockers, "BLOCKER"),
    open_material_limitations: projectIssues(manifest.open_material_limitations, "LIMITATION"),
    parent_manifests: projectedParents,
  };
}
