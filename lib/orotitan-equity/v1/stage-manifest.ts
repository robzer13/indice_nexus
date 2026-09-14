import Ajv2020, { type ErrorObject } from "ajv/dist/2020";
import addFormats from "ajv-formats";
import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";

type JsonObject = Record<string, unknown>;

export type StageCode = "RESEARCH" | "DEEP_DIVE" | "INTEGRATION";
export type ManifestKind = "CHECKPOINT" | "FINAL";
export type HandoffState = "NOT_EVALUATED" | "YES" | "NO";

export type StageManifestValidationContext = {
  runId: string;
  stage: StageCode;
  stageRevision: number;
  issuerId: string | null;
  securityId: string | null;
  dossierId: string | null;
  canonicalMode: string;
  runType: "INITIAL" | "REFRESH" | null;
  dataCutoff: string;
  baselineSnapshotId: string | null;
  processVersion: string;
  pilotageContractVersion: string;
  contractSetSha256: string;
  stageContractName: string;
  stageContractVersion: string;
  stageContractSha256: string;
};

export type StageManifestValidationResult =
  | { ok: true; manifest: JsonObject; downstreamAdmission: boolean }
  | { ok: false; stage: "schema" | "contract"; errors: string[] };

const schema = JSON.parse(
  readFileSync(new URL("../../../contracts/orotitan-equity/v1/OROTITAN_STAGE_MANIFEST_SCHEMA_V1.json", import.meta.url), "utf8"),
) as JsonObject;

const ajv = new Ajv2020({ allErrors: true, strict: true, strictRequired: false, strictTypes: false });
addFormats(ajv as unknown as Parameters<typeof addFormats>[0]);
const validateSchema = ajv.compile(schema);

function formatAjvErrors(errors: ErrorObject[] | null | undefined): string[] {
  return (errors ?? []).map((error) => `${error.instancePath || "/"} ${error.message ?? "is invalid"}`);
}

function isObject(value: unknown): value is JsonObject {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function stringAt(obj: JsonObject, key: string): string | null {
  const value = obj[key];
  return typeof value === "string" ? value : value === null ? null : "__INVALID__";
}

function artifactKey(value: unknown): string | null {
  if (!isObject(value) || typeof value.artifact_id !== "string" || typeof value.version !== "number") return null;
  return `${value.artifact_id}:${value.version}`;
}

function expectedGate(stage: StageCode): string {
  switch (stage) {
    case "RESEARCH": return "READY_FOR_DEEP_DIVE";
    case "DEEP_DIVE": return "READY_FOR_INTEGRATION";
    case "INTEGRATION": return "READY_TO_PUBLISH";
  }
}

export function computeContractSetSha256(contractPins: unknown): string {
  if (!isObject(contractPins)) throw new Error("contract_pins must be an object");
  const lines = Object.entries(contractPins)
    .sort(([left], [right]) => left.localeCompare(right))
    .map(([logicalName, value]) => {
      if (!isObject(value) || typeof value.version !== "string" || typeof value.content_sha256 !== "string") {
        throw new Error(`contract_pins.${logicalName} is malformed`);
      }
      return `${logicalName}|${value.version}|${value.content_sha256}`;
    });
  return sha256Hex(Buffer.from(`${lines.join("\n")}\n`, "utf8"));
}

function contractErrors(manifest: JsonObject, context: StageManifestValidationContext): string[] {
  const errors: string[] = [];
  const exact: Array<[string, unknown, unknown]> = [
    ["run_id", manifest.run_id, context.runId],
    ["stage", manifest.stage, context.stage],
    ["stage_revision", manifest.stage_revision, context.stageRevision],
    ["issuer_id", manifest.issuer_id, context.issuerId],
    ["security_id", manifest.security_id, context.securityId],
    ["dossier_id", manifest.dossier_id, context.dossierId],
    ["canonical_mode", manifest.canonical_mode, context.canonicalMode],
    ["run_type", manifest.run_type, context.runType],
    ["data_cutoff", manifest.data_cutoff, context.dataCutoff],
    ["baseline_snapshot_id", manifest.baseline_snapshot_id, context.baselineSnapshotId],
    ["process_version", manifest.process_version, context.processVersion],
    ["pilotage_contract_version", manifest.pilotage_contract_version, context.pilotageContractVersion],
    ["contract_set_sha256", manifest.contract_set_sha256, context.contractSetSha256],
  ];
  for (const [name, actual, expected] of exact) {
    if (actual !== expected) errors.push(`${name} does not match authoritative run state`);
  }

  try {
    const computedContractSetSha256 = computeContractSetSha256(manifest.contract_pins);
    if (computedContractSetSha256 !== manifest.contract_set_sha256) {
      errors.push("contract_pins do not reconcile to contract_set_sha256");
    }
  } catch (error) {
    errors.push(error instanceof Error ? error.message : "contract_pins are malformed");
  }

  const stageContract = isObject(manifest.stage_contract) ? manifest.stage_contract : {};
  if (stageContract.name !== context.stageContractName) errors.push("stage_contract.name mismatch");
  if (stageContract.version !== context.stageContractVersion) errors.push("stage_contract.version mismatch");
  if (stageContract.content_sha256 !== context.stageContractSha256) errors.push("stage_contract.content_sha256 mismatch");

  const handoff = isObject(manifest.handoff_gate) ? manifest.handoff_gate : {};
  if (handoff.name !== expectedGate(context.stage)) errors.push("handoff gate name does not match stage");

  const kind = manifest.manifest_kind as ManifestKind;
  const status = manifest.stage_status;
  const gateState = handoff.state as HandoffState;
  const blockers = Array.isArray(manifest.critical_blockers) ? manifest.critical_blockers : [];

  if (kind === "CHECKPOINT") {
    if (status === "COMPLETE") errors.push("CHECKPOINT manifest cannot declare COMPLETE stage status");
    if (gateState === "YES") errors.push("CHECKPOINT manifest cannot admit a downstream stage");
    if (manifest.completed_at !== null) errors.push("CHECKPOINT manifest completed_at must be null");
  }

  if (kind === "FINAL" && status === "COMPLETE" && manifest.completed_at === null) {
    errors.push("FINAL COMPLETE manifest requires completed_at");
  }

  if (gateState === "YES" && blockers.length > 0) {
    errors.push("handoff gate YES is incompatible with critical blockers");
  }

  const inputArtifacts = Array.isArray(manifest.input_artifacts) ? manifest.input_artifacts : [];
  const outputArtifacts = Array.isArray(manifest.output_artifacts) ? manifest.output_artifacts : [];
  const inputKeys = inputArtifacts.map(artifactKey);
  const outputKeys = outputArtifacts.map(artifactKey);
  if (inputKeys.some((key) => key === null) || new Set(inputKeys).size !== inputKeys.length) {
    errors.push("input_artifacts contains malformed or duplicate artifact identities");
  }
  if (outputKeys.some((key) => key === null) || new Set(outputKeys).size !== outputKeys.length) {
    errors.push("output_artifacts contains malformed or duplicate artifact identities");
  }

  const manifestId = stringAt(manifest, "manifest_id");
  if (outputArtifacts.some((item) => isObject(item) && item.artifact_id === manifestId)) {
    errors.push("Stage Manifest must not list itself in output_artifacts");
  }

  for (const output of outputArtifacts) {
    if (!isObject(output)) continue;
    const authorityClass = output.authority_class;
    if (kind === "FINAL" && authorityClass === "CHECKPOINT_STAGE_OUTPUT") {
      errors.push("FINAL manifest cannot emit CHECKPOINT_STAGE_OUTPUT as current stage output");
    }
    if (kind === "CHECKPOINT" && authorityClass === "AUTHORITATIVE_STAGE_OUTPUT") {
      errors.push("CHECKPOINT manifest cannot promote output to AUTHORITATIVE_STAGE_OUTPUT");
    }
  }

  return errors;
}

export function validateStageManifest(
  input: unknown,
  context: StageManifestValidationContext,
): StageManifestValidationResult {
  if (!validateSchema(input)) {
    return { ok: false, stage: "schema", errors: formatAjvErrors(validateSchema.errors) };
  }
  const manifest = input as JsonObject;
  const errors = contractErrors(manifest, context);
  if (errors.length > 0) return { ok: false, stage: "contract", errors };

  const handoff = manifest.handoff_gate as JsonObject;
  const downstreamAdmission = manifest.manifest_kind === "FINAL"
    && manifest.stage_status === "COMPLETE"
    && handoff.state === "YES"
    && Array.isArray(manifest.critical_blockers)
    && manifest.critical_blockers.length === 0;

  return { ok: true, manifest, downstreamAdmission };
}

export function sha256Hex(bytes: Uint8Array): string {
  return createHash("sha256").update(bytes).digest("hex");
}

export function verifyArtifactBytes(
  bytes: Uint8Array,
  expectedSha256: string,
  expectedSizeBytes?: number,
): { ok: true } | { ok: false; error: string } {
  if (expectedSizeBytes !== undefined && bytes.byteLength !== expectedSizeBytes) {
    return { ok: false, error: `size mismatch: expected ${expectedSizeBytes}, got ${bytes.byteLength}` };
  }
  const actual = sha256Hex(bytes);
  if (actual !== expectedSha256) return { ok: false, error: `sha256 mismatch: expected ${expectedSha256}, got ${actual}` };
  return { ok: true };
}

export { validateSchema };
