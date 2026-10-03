import { createHash } from "node:crypto";
import Ajv2020, { type ErrorObject } from "ajv/dist/2020";
import addFormats from "ajv-formats";
import controlledOperationSchema from "../../../schemas/vnext/chatgpt-supabase/controlled-operation.schema.v0.1.json";
import type { SaveDisposition } from "./process-engine-v2";

export type StageCode = "RESEARCH" | "DEEP_DIVE" | "INTEGRATION";
export type StageLifecycle = "NOT_STARTED" | "IN_PROGRESS" | "PAUSED" | "BLOCKED" | "COMPLETE";

export type IssuerRow = {
  issuer_id: string;
  display_name: string;
  legal_name: string | null;
};

export type SecurityRow = {
  security_id: string;
  issuer_id: string;
  ticker: string;
  market_data_symbol: string | null;
  primary_listing: boolean | null;
  listing_status: string | null;
};

export type DossierRow = {
  dossier_id: string;
  issuer_id: string;
  current_snapshot_id: string | null;
  active: boolean;
};

export type RunRow = {
  run_id: string;
  issuer_id: string | null;
  security_id: string | null;
  dossier_id: string | null;
  run_status: string;
  current_stage: StageCode | null;
  run_type: string | null;
  canonical_mode: string;
  data_cutoff: string;
  contract_set_sha256: string;
  state_version: number;
  updated_at: string;
};

export type StageRow = {
  run_id: string;
  stage_code: StageCode;
  stage_revision: number;
  lifecycle_status: StageLifecycle;
  handoff_gate_state: "NOT_EVALUATED" | "YES" | "NO";
  active_manifest_artifact_id: string | null;
  active_manifest_version: number | null;
  active_manifest_kind: "CHECKPOINT" | "FINAL" | null;
  blocker_summary: unknown;
  state_version: number;
};

export type ArtifactRow = {
  artifact_id: string;
  version: number;
  run_id: string;
  stage_code: StageCode;
  artifact_type: string;
  logical_name: string;
  authority_class: string;
  authority_state: string;
  artifact_status: string;
  availability_state: string;
  content_sha256: string;
  size_bytes: number;
  media_type: string;
  storage_backend: string;
  storage_uri: string;
  github_repository: string | null;
  github_path: string | null;
  github_commit_sha: string | null;
  github_blob_sha: string | null;
  supabase_bucket: string | null;
  supabase_object_path: string | null;
  manifest_artifact_id: string | null;
  manifest_version: number | null;
};

export type RpcResult = Record<string, unknown>;

export type ControlledBridgePort = {
  listIssuers(): Promise<IssuerRow[]>;
  listSecurities(): Promise<SecurityRow[]>;
  listDossiers(issuerId: string): Promise<DossierRow[]>;
  listRuns(issuerId: string): Promise<RunRow[]>;
  getRun(runId: string): Promise<RunRow | null>;
  getStage(runId: string, stage: StageCode): Promise<StageRow | null>;
  listArtifacts(runId: string): Promise<ArtifactRow[]>;
  readSupabaseObject(bucket: string, objectPath: string): Promise<Uint8Array>;
  resolveArtifact(args: {
    p_run_id: string;
    p_artifact_id: string;
    p_version: number;
    p_expected_sha256: string | null;
    p_required_authority_class: string | null;
  }): Promise<RpcResult>;
  checkpointStage(args: Record<string, unknown>): Promise<RpcResult>;
  finalizeStage(args: Record<string, unknown>): Promise<RpcResult>;
  reopenStage(args: Record<string, unknown>): Promise<RpcResult>;
};

type ArtifactRef = {
  artifact_id: string;
  version: number;
  content_sha256?: string | null;
  required_authority_class?: string | null;
};

export type LoadRequest = {
  contract_version: "0.1.0";
  operation: "LOAD";
  issuer_query: string;
  run_id?: string | null;
  requested_context_tiers?: Array<"L0" | "L1" | "L2" | "L3">;
};

export type LoadResult = {
  contract_version: "0.1.0";
  operation: "LOAD_RESULT";
  mutation_allowed: false;
  issuer_id: string;
  security_id: string | null;
  dossier_id: string;
  run_id: string | null;
  run_status: string | null;
  run_state_version: number | null;
  run_type: string | null;
  canonical_mode: string | null;
  data_cutoff: string | null;
  contract_set_sha256: string | null;
  current_stage: StageCode | null;
  stage: {
    stage_code: StageCode;
    stage_revision: number;
    lifecycle_status: StageLifecycle;
    stage_state_version: number;
    handoff_gate_state: "NOT_EVALUATED" | "NO" | "YES";
    active_manifest: ArtifactRef | null;
  } | null;
  blockers: Record<string, unknown>[];
  artifact_index: ArtifactRef[];
  process_state_artifact?: ArtifactRef | null;
  context_plan: {
    l0: ArtifactRef[];
    l1: ArtifactRef[];
    l2: ArtifactRef[];
    l3: ArtifactRef[];
  };
};

export type MutationReceipt = {
  contract_version: "0.1.0";
  operation: "MUTATION_RECEIPT";
  status: "SUCCESS" | "IDEMPOTENT_REPLAY";
  run_id: string;
  stage_code: StageCode;
  stage_state_version: number;
  manifest_id: string | null;
  event_id: string | null;
  idempotent_replay: boolean;
  verification: {
    durable_state_reloaded: true;
    state_matches_intent: true;
    artifact_integrity_verified: true;
  };
};

export type OperationFailure = {
  contract_version: "0.1.0";
  operation: "OPERATION_FAILURE";
  error_class:
    | "STALE_STATE"
    | "NOT_FOUND"
    | "INVALID_STATE"
    | "CONTRACT_VIOLATION"
    | "PERSISTENCE_INTEGRITY"
    | "AUTHORITY_VIOLATION"
    | "INFRASTRUCTURE";
  message: string;
  retry_without_reload_allowed: false;
};

export type CheckpointRequest = {
  contract_version: "0.1.0";
  operation: "CHECKPOINT_STAGE";
  run_id: string;
  stage_code: StageCode;
  expected_run_state_version: number;
  expected_stage_state_version: number;
  save_disposition: "CHECKPOINT" | "BLOCK";
  target_lifecycle: "IN_PROGRESS" | "PAUSED" | "BLOCKED";
  bundle: Bundle;
  idempotency_key: string;
  request_fingerprint_sha256: string;
  actor_type: string;
  publish_authorized: false;
};

export type FinalizeRequest = {
  contract_version: "0.1.0";
  operation: "FINALIZE_STAGE";
  run_id: string;
  stage_code: StageCode;
  expected_run_state_version: number;
  expected_stage_state_version: number;
  save_disposition: "FINALIZE";
  bundle: Bundle;
  idempotency_key: string;
  request_fingerprint_sha256: string;
  actor_type: string;
  publish_authorized: false;
};

export type ReopenRequest = {
  contract_version: "0.1.0";
  operation: "REOPEN_STAGE";
  run_id: string;
  stage_code: StageCode;
  expected_run_state_version: number;
  expected_stage_state_version: number;
  target_lifecycle: "IN_PROGRESS" | "BLOCKED";
  reason: { code: string; summary: string; [key: string]: unknown };
  idempotency_key: string;
  request_fingerprint_sha256: string;
  publish_authorized: false;
};

export type NoopRequest = {
  contract_version: "0.1.0";
  operation: "NOOP";
  run_id: string;
  stage_code: StageCode;
  save_disposition: "NOOP";
  reason: string;
  publish_authorized: false;
};

export type Bundle = {
  manifest: Record<string, unknown>;
  manifest_registration: Record<string, unknown>;
  output_artifacts: Record<string, unknown>[];
  edges: Record<string, unknown>[];
  persistence_receipts_verified: true;
};

export type ControlledRequest =
  | LoadRequest
  | CheckpointRequest
  | FinalizeRequest
  | ReopenRequest
  | NoopRequest;

export type ControlledResult =
  | LoadResult
  | MutationReceipt
  | NoopRequest
  | OperationFailure;

export class ControlledBridgeError extends Error {
  readonly errorClass: OperationFailure["error_class"];

  constructor(errorClass: OperationFailure["error_class"], message: string) {
    super(message);
    this.errorClass = errorClass;
    this.name = "ControlledBridgeError";
  }
}

const ajv = new Ajv2020({ allErrors: true, strict: false });
addFormats(ajv as unknown as Parameters<typeof addFormats>[0]);
const validateContract = ajv.compile(controlledOperationSchema);

function formatAjvErrors(errors: ErrorObject[] | null | undefined): string {
  return (errors ?? [])
    .map((error) => `${error.instancePath || "/"} ${error.message ?? "is invalid"}`)
    .join("; ");
}

function canonicalJson(value: unknown): string {
  if (value === null || typeof value !== "object") return JSON.stringify(value);
  if (Array.isArray(value)) return `[${value.map(canonicalJson).join(",")}]`;
  const entries = Object.entries(value as Record<string, unknown>)
    .sort(([left], [right]) => left.localeCompare(right));
  return `{${entries.map(([key, child]) => `${JSON.stringify(key)}:${canonicalJson(child)}`).join(",")}}`;
}

function sha256(bytes: Uint8Array | string): string {
  return createHash("sha256").update(bytes).digest("hex");
}

function gitBlobSha1(bytes: Uint8Array): string {
  const prefix = Buffer.from(`blob ${bytes.byteLength}\0`, "utf8");
  return createHash("sha1").update(Buffer.concat([prefix, Buffer.from(bytes)])).digest("hex");
}

function normalized(value: string): string {
  return value.trim().toLocaleLowerCase("en-US");
}

function assertObject(value: unknown, label: string): asserts value is Record<string, unknown> {
  if (typeof value !== "object" || value === null || Array.isArray(value)) {
    throw new ControlledBridgeError("CONTRACT_VIOLATION", `${label} must be an object`);
  }
}

function asString(value: unknown, label: string): string {
  if (typeof value !== "string" || value.length === 0) {
    throw new ControlledBridgeError("CONTRACT_VIOLATION", `${label} must be a non-empty string`);
  }
  return value;
}

function asPositiveInteger(value: unknown, label: string): number {
  if (!Number.isInteger(value) || (value as number) < 1) {
    throw new ControlledBridgeError("CONTRACT_VIOLATION", `${label} must be a positive integer`);
  }
  return value as number;
}

function validateOrThrow(value: unknown): void {
  if (!validateContract(value)) {
    throw new ControlledBridgeError("CONTRACT_VIOLATION", formatAjvErrors(validateContract.errors));
  }
}

function failure(errorClass: OperationFailure["error_class"], message: string): OperationFailure {
  const result: OperationFailure = {
    contract_version: "0.1.0",
    operation: "OPERATION_FAILURE",
    error_class: errorClass,
    message: message.slice(0, 256) || errorClass,
    retry_without_reload_allowed: false,
  };
  validateOrThrow(result);
  return result;
}

function classifyError(error: unknown): OperationFailure {
  if (error instanceof ControlledBridgeError) return failure(error.errorClass, error.message);
  const message = error instanceof Error ? error.message : String(error);
  if (/RUN_STATE_VERSION_MISMATCH|STAGE_STATE_VERSION_MISMATCH/i.test(message)) {
    return failure("STALE_STATE", message);
  }
  if (/NOT_FOUND|P0002/i.test(message)) return failure("NOT_FOUND", message);
  if (/PERSISTENCE|VERIFIED_(SHA256|SIZE|GIT_BLOB|BYTES)|storage/i.test(message)) {
    return failure("PERSISTENCE_INTEGRITY", message);
  }
  if (/AUTHORITY|ARTIFACT_HASH_MISMATCH|ARTIFACT_STATUS_INVALID/i.test(message)) {
    return failure("AUTHORITY_VIOLATION", message);
  }
  if (/INVALID|MISMATCH|ALREADY_COMPLETE|23514|22023/i.test(message)) {
    return failure("INVALID_STATE", message);
  }
  return failure("INFRASTRUCTURE", message);
}

function exactRef(row: ArtifactRow): ArtifactRef {
  return {
    artifact_id: row.artifact_id,
    version: row.version,
    content_sha256: row.content_sha256,
    required_authority_class: row.authority_class,
  };
}

function selectSecurity(rows: SecurityRow[], issuerId: string): SecurityRow | null {
  const issuerRows = rows.filter((row) => row.issuer_id === issuerId);
  const primary = issuerRows.filter((row) => row.primary_listing === true);
  if (primary.length === 1) return primary[0];
  if (primary.length > 1) {
    throw new ControlledBridgeError("INVALID_STATE", "multiple primary securities resolve to the same issuer");
  }
  if (issuerRows.length === 1) return issuerRows[0];
  if (issuerRows.length === 0) return null;
  throw new ControlledBridgeError("INVALID_STATE", "issuer has multiple securities and no unique primary listing");
}

function resolveIssuer(
  query: string,
  issuers: IssuerRow[],
  securities: SecurityRow[],
): IssuerRow {
  const needle = normalized(query);
  if (!needle) throw new ControlledBridgeError("CONTRACT_VIOLATION", "issuer_query is required");

  const ids = new Set<string>();
  for (const issuer of issuers) {
    if (
      normalized(issuer.display_name) === needle ||
      (issuer.legal_name !== null && normalized(issuer.legal_name) === needle)
    ) {
      ids.add(issuer.issuer_id);
    }
  }
  for (const security of securities) {
    if (
      normalized(security.ticker) === needle ||
      (security.market_data_symbol !== null && normalized(security.market_data_symbol) === needle)
    ) {
      ids.add(security.issuer_id);
    }
  }

  if (ids.size === 0) throw new ControlledBridgeError("NOT_FOUND", `issuer not found for query: ${query}`);
  if (ids.size > 1) throw new ControlledBridgeError("INVALID_STATE", `issuer query is ambiguous: ${query}`);
  const issuerId = [...ids][0];
  const issuer = issuers.find((row) => row.issuer_id === issuerId);
  if (!issuer) throw new ControlledBridgeError("NOT_FOUND", "resolved issuer row is missing");
  return issuer;
}

function activeArtifacts(rows: ArtifactRow[]): ArtifactRow[] {
  return rows.filter(
    (row) =>
      row.artifact_status === "SEALED" &&
      row.availability_state === "AVAILABLE" &&
      (row.authority_state === "AUTHORITATIVE" || row.authority_state === "CHECKPOINT"),
  );
}

function uniqueRefs(refs: ArtifactRef[]): ArtifactRef[] {
  const seen = new Set<string>();
  return refs.filter((ref) => {
    const key = `${ref.artifact_id}:${ref.version}`;
    if (seen.has(key)) return false;
    seen.add(key);
    return true;
  });
}

export async function loadControlledState(
  port: ControlledBridgePort,
  request: LoadRequest,
): Promise<LoadResult> {
  validateOrThrow(request);
  const [issuers, securities] = await Promise.all([port.listIssuers(), port.listSecurities()]);
  const issuer = resolveIssuer(request.issuer_query, issuers, securities);
  const security = selectSecurity(securities, issuer.issuer_id);

  const [dossiers, runs] = await Promise.all([
    port.listDossiers(issuer.issuer_id),
    port.listRuns(issuer.issuer_id),
  ]);
  const activeDossiers = dossiers.filter((row) => row.active);
  if (activeDossiers.length !== 1) {
    throw new ControlledBridgeError(
      "INVALID_STATE",
      `expected exactly one active dossier for issuer, found ${activeDossiers.length}`,
    );
  }
  const dossier = activeDossiers[0];

  const nonTerminalRuns = runs.filter(
    (row) => row.run_status !== "PUBLISHED" && row.run_status !== "CANCELLED",
  );
  let run: RunRow | null = null;
  if (request.run_id) {
    run = nonTerminalRuns.find((row) => row.run_id === request.run_id) ?? null;
    if (!run) throw new ControlledBridgeError("NOT_FOUND", "requested active run was not found for issuer");
  } else if (nonTerminalRuns.length === 1) {
    run = nonTerminalRuns[0];
  } else if (nonTerminalRuns.length > 1) {
    throw new ControlledBridgeError(
      "INVALID_STATE",
      "multiple non-terminal runs exist; LOAD must specify run_id",
    );
  }

  if (run === null) {
    const result: LoadResult = {
      contract_version: "0.1.0",
      operation: "LOAD_RESULT",
      mutation_allowed: false,
      issuer_id: issuer.issuer_id,
      security_id: security?.security_id ?? null,
      dossier_id: dossier.dossier_id,
      run_id: null,
      run_status: null,
      run_state_version: null,
      run_type: null,
      canonical_mode: null,
      data_cutoff: null,
      contract_set_sha256: null,
      current_stage: null,
      stage: null,
      blockers: [],
      artifact_index: [],
      process_state_artifact: null,
      context_plan: { l0: [], l1: [], l2: [], l3: [] },
    };
    validateOrThrow(result);
    return result;
  }

  if (run.issuer_id !== issuer.issuer_id) {
    throw new ControlledBridgeError("AUTHORITY_VIOLATION", "run issuer does not match resolved issuer");
  }
  if (run.dossier_id !== dossier.dossier_id) {
    throw new ControlledBridgeError("AUTHORITY_VIOLATION", "run dossier does not match active dossier");
  }
  if (!run.security_id) {
    throw new ControlledBridgeError("INVALID_STATE", "active run is missing security_id");
  }
  if (security && run.security_id !== security.security_id) {
    throw new ControlledBridgeError("AUTHORITY_VIOLATION", "run security does not match resolved primary security");
  }
  if (!run.current_stage) {
    throw new ControlledBridgeError(
      "INVALID_STATE",
      "active run has no current_stage and cannot be represented by frozen LOAD_RESULT",
    );
  }

  const [stage, artifacts] = await Promise.all([
    port.getStage(run.run_id, run.current_stage),
    port.listArtifacts(run.run_id),
  ]);
  if (!stage) throw new ControlledBridgeError("NOT_FOUND", "current stage row is missing");
  if (stage.stage_code !== run.current_stage) {
    throw new ControlledBridgeError("AUTHORITY_VIOLATION", "current_stage does not match stage row");
  }

  const currentArtifacts = activeArtifacts(artifacts);
  const artifactIndex = currentArtifacts.map(exactRef);

  let activeManifest: ArtifactRef | null = null;
  if (stage.active_manifest_artifact_id !== null || stage.active_manifest_version !== null) {
    if (stage.active_manifest_artifact_id === null || stage.active_manifest_version === null) {
      throw new ControlledBridgeError("AUTHORITY_VIOLATION", "active manifest identity is partially populated");
    }
    const manifestRow = currentArtifacts.find(
      (row) =>
        row.artifact_id === stage.active_manifest_artifact_id &&
        row.version === stage.active_manifest_version &&
        row.stage_code === stage.stage_code,
    );
    if (!manifestRow) {
      throw new ControlledBridgeError("AUTHORITY_VIOLATION", "active manifest has no current authoritative artifact row");
    }
    activeManifest = exactRef(manifestRow);
  }

  const processCandidates = currentArtifacts
    .filter(
      (row) =>
        row.stage_code === stage.stage_code &&
        (row.artifact_type === "PROCESS_ENGINE_STATE" || row.logical_name === "process_engine_state"),
    )
    .sort((left, right) => right.version - left.version);
  const processStateArtifact = processCandidates.length > 0 ? exactRef(processCandidates[0]) : null;

  const l0 = uniqueRefs(
    [activeManifest, processStateArtifact].filter((ref): ref is ArtifactRef => ref !== null),
  );
  const l1 = uniqueRefs(
    currentArtifacts
      .filter((row) => row.stage_code === stage.stage_code)
      .map(exactRef),
  );

  const blockers = Array.isArray(stage.blocker_summary)
    ? stage.blocker_summary.filter(
        (value): value is Record<string, unknown> =>
          typeof value === "object" && value !== null && !Array.isArray(value),
      )
    : [];

  const result: LoadResult = {
    contract_version: "0.1.0",
    operation: "LOAD_RESULT",
    mutation_allowed: false,
    issuer_id: issuer.issuer_id,
    security_id: run.security_id,
    dossier_id: run.dossier_id,
    run_id: run.run_id,
    run_status: run.run_status,
    run_state_version: run.state_version,
    run_type: run.run_type,
    canonical_mode: run.canonical_mode,
    data_cutoff: run.data_cutoff,
    contract_set_sha256: run.contract_set_sha256,
    current_stage: run.current_stage,
    stage: {
      stage_code: stage.stage_code,
      stage_revision: stage.stage_revision,
      lifecycle_status: stage.lifecycle_status,
      stage_state_version: stage.state_version,
      handoff_gate_state: stage.handoff_gate_state,
      active_manifest: activeManifest,
    },
    blockers,
    artifact_index: artifactIndex,
    process_state_artifact: processStateArtifact,
    context_plan: {
      l0,
      l1,
      l2: [],
      l3: [],
    },
  };
  validateOrThrow(result);
  return result;
}

function buildFingerprint(value: Record<string, unknown>): string {
  return sha256(canonicalJson(value));
}

function mutationIdentity(
  operation: "CHECKPOINT_STAGE" | "FINALIZE_STAGE" | "REOPEN_STAGE",
  descriptor: Record<string, unknown>,
): { idempotency_key: string; request_fingerprint_sha256: string } {
  const requestFingerprint = buildFingerprint(descriptor);
  return {
    idempotency_key: `orotitan:bridge:v1:${operation.toLocaleLowerCase("en-US")}:${requestFingerprint}`,
    request_fingerprint_sha256: requestFingerprint,
  };
}

function mutationDescriptor(
  request: CheckpointRequest | FinalizeRequest | ReopenRequest,
): Record<string, unknown> {
  if (request.operation === "CHECKPOINT_STAGE") {
    return {
      operation: request.operation,
      run_id: request.run_id,
      stage_code: request.stage_code,
      expected_run_state_version: request.expected_run_state_version,
      expected_stage_state_version: request.expected_stage_state_version,
      save_disposition: request.save_disposition,
      target_lifecycle: request.target_lifecycle,
      bundle: request.bundle,
      actor_type: request.actor_type,
    };
  }
  if (request.operation === "FINALIZE_STAGE") {
    return {
      operation: request.operation,
      run_id: request.run_id,
      stage_code: request.stage_code,
      expected_run_state_version: request.expected_run_state_version,
      expected_stage_state_version: request.expected_stage_state_version,
      bundle: request.bundle,
      actor_type: request.actor_type,
    };
  }
  return {
    operation: request.operation,
    run_id: request.run_id,
    stage_code: request.stage_code,
    expected_run_state_version: request.expected_run_state_version,
    expected_stage_state_version: request.expected_stage_state_version,
    target_lifecycle: request.target_lifecycle,
    reason: request.reason,
  };
}

function assertMutationIdentity(
  request: CheckpointRequest | FinalizeRequest | ReopenRequest,
): void {
  if (request.operation === "REOPEN_STAGE") {
    asPositiveInteger(
      request.reason.expected_stage_revision,
      "reason.expected_stage_revision",
    );
  }
  const expected = mutationIdentity(request.operation, mutationDescriptor(request));
  if (
    request.request_fingerprint_sha256 !== expected.request_fingerprint_sha256 ||
    request.idempotency_key !== expected.idempotency_key
  ) {
    throw new ControlledBridgeError(
      "CONTRACT_VIOLATION",
      "mutation idempotency identity does not match the canonical request payload",
    );
  }
}

export function buildSaveOperation(input: {
  load: LoadResult;
  disposition: SaveDisposition;
  bundle?: Bundle;
  actorType: string;
  checkpointLifecycle?: "IN_PROGRESS" | "PAUSED";
}): CheckpointRequest | FinalizeRequest | NoopRequest {
  if (!input.load.run_id || !input.load.current_stage || !input.load.stage || !input.load.run_state_version) {
    throw new ControlledBridgeError("INVALID_STATE", "SAVE requires an active loaded run and current stage");
  }
  if (input.load.stage.stage_code !== input.load.current_stage) {
    throw new ControlledBridgeError("AUTHORITY_VIOLATION", "loaded current stage is inconsistent");
  }

  const common = {
    run_id: input.load.run_id,
    stage_code: input.load.current_stage,
    expected_run_state_version: input.load.run_state_version,
    expected_stage_state_version: input.load.stage.stage_state_version,
  };

  if (input.disposition.action === "NOOP") {
    const request: NoopRequest = {
      contract_version: "0.1.0",
      operation: "NOOP",
      run_id: common.run_id,
      stage_code: common.stage_code,
      save_disposition: "NOOP",
      reason: input.disposition.reason,
      publish_authorized: false,
    };
    validateOrThrow(request);
    return request;
  }

  if (!input.bundle) {
    throw new ControlledBridgeError("CONTRACT_VIOLATION", "artifact bundle is required for mutating SAVE disposition");
  }

  if (input.disposition.action === "FINALIZE") {
    const descriptor = {
      operation: "FINALIZE_STAGE",
      ...common,
      bundle: input.bundle,
      actor_type: input.actorType,
    };
    const identity = mutationIdentity("FINALIZE_STAGE", descriptor);
    const request: FinalizeRequest = {
      contract_version: "0.1.0",
      operation: "FINALIZE_STAGE",
      ...common,
      save_disposition: "FINALIZE",
      bundle: input.bundle,
      ...identity,
      actor_type: input.actorType,
      publish_authorized: false,
    };
    validateOrThrow(request);
    return request;
  }

  const saveDisposition = input.disposition.action === "BLOCK" ? "BLOCK" : "CHECKPOINT";
  const targetLifecycle =
    input.disposition.action === "BLOCK"
      ? "BLOCKED"
      : input.checkpointLifecycle ?? "IN_PROGRESS";
  const descriptor = {
    operation: "CHECKPOINT_STAGE",
    ...common,
    save_disposition: saveDisposition,
    target_lifecycle: targetLifecycle,
    bundle: input.bundle,
    actor_type: input.actorType,
  };
  const identity = mutationIdentity("CHECKPOINT_STAGE", descriptor);
  const request: CheckpointRequest = {
    contract_version: "0.1.0",
    operation: "CHECKPOINT_STAGE",
    ...common,
    save_disposition: saveDisposition,
    target_lifecycle: targetLifecycle,
    bundle: input.bundle,
    ...identity,
    actor_type: input.actorType,
    publish_authorized: false,
  };
  validateOrThrow(request);
  return request;
}

export function buildReopenOperation(input: {
  load: LoadResult;
  targetLifecycle: "IN_PROGRESS" | "BLOCKED";
  reason: { code: string; summary: string; [key: string]: unknown };
}): ReopenRequest {
  if (!input.load.run_id || !input.load.current_stage || !input.load.stage || !input.load.run_state_version) {
    throw new ControlledBridgeError("INVALID_STATE", "REOPEN requires an active loaded run and current stage");
  }
  const reason = {
    ...input.reason,
    expected_stage_revision: input.load.stage.stage_revision + 1,
  };
  const descriptor = {
    operation: "REOPEN_STAGE",
    run_id: input.load.run_id,
    stage_code: input.load.current_stage,
    expected_run_state_version: input.load.run_state_version,
    expected_stage_state_version: input.load.stage.stage_state_version,
    target_lifecycle: input.targetLifecycle,
    reason,
  };
  const identity = mutationIdentity("REOPEN_STAGE", descriptor);
  const request: ReopenRequest = {
    contract_version: "0.1.0",
    operation: "REOPEN_STAGE",
    run_id: input.load.run_id,
    stage_code: input.load.current_stage,
    expected_run_state_version: input.load.run_state_version,
    expected_stage_state_version: input.load.stage.stage_state_version,
    target_lifecycle: input.targetLifecycle,
    reason,
    ...identity,
    publish_authorized: false,
  };
  validateOrThrow(request);
  return request;
}

function receiptBytes(registration: Record<string, unknown>): Uint8Array | null {
  const receipt = registration.persistence_receipt;
  if (typeof receipt !== "object" || receipt === null || Array.isArray(receipt)) return null;
  const base64 = (receipt as Record<string, unknown>).verified_content_base64;
  if (typeof base64 !== "string" || base64.length === 0) return null;
  try {
    return Buffer.from(base64, "base64");
  } catch {
    return null;
  }
}

function verifyRegistrationBytes(
  registration: Record<string, unknown>,
  bytes: Uint8Array,
): void {
  const expectedSha = asString(registration.content_sha256, "content_sha256");
  const expectedSize = asPositiveInteger(registration.size_bytes, "size_bytes");
  if (bytes.byteLength !== expectedSize) {
    throw new ControlledBridgeError("PERSISTENCE_INTEGRITY", "persisted artifact size does not match registration");
  }
  if (sha256(bytes) !== expectedSha) {
    throw new ControlledBridgeError("PERSISTENCE_INTEGRITY", "persisted artifact SHA-256 does not match registration");
  }
}

function verifyPrivateGithubReceipt(
  registration: Record<string, unknown>,
  runId: string,
  stageCode: StageCode,
  expectedJson?: Record<string, unknown>,
): void {
  if (
    registration.storage_backend !== "PRIVATE_GITHUB" ||
    registration.github_repository !== "robzer13/real-orotitan"
  ) {
    throw new ControlledBridgeError(
      "PERSISTENCE_INTEGRITY",
      "deployed Registry requires canonical PRIVATE_GITHUB persistence",
    );
  }
  assertObject(registration.persistence_receipt, "persistence_receipt");
  const receipt = registration.persistence_receipt;
  if (
    receipt.receipt_schema_version !== "1.1" ||
    receipt.verification_method !== "PRIVATE_GITHUB_ATTESTED_REREAD_EXACT_BYTES_V1" ||
    receipt.storage_backend !== "PRIVATE_GITHUB" ||
    receipt.commit_path_resolved !== true
  ) {
    throw new ControlledBridgeError(
      "PERSISTENCE_INTEGRITY",
      "attested private GitHub persistence receipt discriminator mismatch",
    );
  }

  const artifactId = asString(registration.artifact_id, "artifact_id");
  const version = asPositiveInteger(registration.version, "version");
  const artifactType = asString(registration.artifact_type, "artifact_type");
  const identityChecks: Array<[unknown, unknown, string]> = [
    [receipt.run_id, runId, "run_id"],
    [receipt.stage_code, stageCode, "stage_code"],
    [receipt.artifact_id, artifactId, "artifact_id"],
    [receipt.version, version, "version"],
    [receipt.artifact_type, artifactType, "artifact_type"],
  ];
  for (const [actual, expected, label] of identityChecks) {
    if (actual !== expected) {
      throw new ControlledBridgeError(
        "PERSISTENCE_INTEGRITY",
        `persistence receipt ${label} mismatch`,
      );
    }
  }

  for (const key of ["github_repository", "github_path", "github_commit_sha", "github_blob_sha"] as const) {
    if (receipt[key] !== registration[key]) {
      throw new ControlledBridgeError("PERSISTENCE_INTEGRITY", `persistence receipt ${key} mismatch`);
    }
  }
  asString(receipt.attestation_event_id, "attestation_event_id");
  asString(receipt.verified_at, "verified_at");

  const bytes = receiptBytes(registration);
  if (!bytes) throw new ControlledBridgeError("PERSISTENCE_INTEGRITY", "verified artifact bytes are missing");
  verifyRegistrationBytes(registration, bytes);

  const expectedBlob = asString(registration.github_blob_sha, "github_blob_sha");
  if (gitBlobSha1(bytes) !== expectedBlob) {
    throw new ControlledBridgeError("PERSISTENCE_INTEGRITY", "persisted Git blob SHA does not match registration");
  }

  const repository = asString(registration.github_repository, "github_repository");
  const commit = asString(registration.github_commit_sha, "github_commit_sha");
  const path = asString(registration.github_path, "github_path");
  const expectedUri = `github://${repository}@${commit}/${path}`;
  if (registration.storage_uri !== expectedUri) {
    throw new ControlledBridgeError("PERSISTENCE_INTEGRITY", "private GitHub storage URI mismatch");
  }

  if (registration.media_type === "application/json") {
    if (!Object.prototype.hasOwnProperty.call(registration, "canonical_json_content")) {
      throw new ControlledBridgeError("PERSISTENCE_INTEGRITY", "canonical_json_content is required for JSON artifacts");
    }
    let parsed: unknown;
    try {
      parsed = JSON.parse(Buffer.from(bytes).toString("utf8"));
    } catch {
      throw new ControlledBridgeError("PERSISTENCE_INTEGRITY", "persisted artifact bytes are not valid UTF-8 JSON");
    }
    if (canonicalJson(parsed) !== canonicalJson(registration.canonical_json_content)) {
      throw new ControlledBridgeError("PERSISTENCE_INTEGRITY", "persisted JSON differs from canonical_json_content");
    }
    if (
      expectedJson &&
      canonicalJson(registration.canonical_json_content) !== canonicalJson(expectedJson)
    ) {
      throw new ControlledBridgeError("PERSISTENCE_INTEGRITY", "persisted manifest payload differs from submitted manifest");
    }
  } else if (expectedJson) {
    throw new ControlledBridgeError(
      "PERSISTENCE_INTEGRITY",
      "submitted JSON manifest requires application/json media type",
    );
  }
}

async function verifyBundlePersistence(
  bundle: Bundle,
  runId: string,
  stageCode: StageCode,
): Promise<void> {
  if (bundle.persistence_receipts_verified !== true) {
    throw new ControlledBridgeError("PERSISTENCE_INTEGRITY", "bundle persistence receipts are not verified");
  }
  assertObject(bundle.manifest, "bundle.manifest");
  assertObject(bundle.manifest_registration, "bundle.manifest_registration");
  verifyPrivateGithubReceipt(
    bundle.manifest_registration,
    runId,
    stageCode,
    bundle.manifest,
  );

  for (const registration of bundle.output_artifacts) {
    assertObject(registration, "output artifact registration");
    verifyPrivateGithubReceipt(registration, runId, stageCode);
  }
}

async function verifyRegisteredArtifacts(
  port: ControlledBridgePort,
  runId: string,
  bundle: Bundle,
): Promise<void> {
  const registrations = [bundle.manifest_registration, ...bundle.output_artifacts];
  for (const registration of registrations) {
    assertObject(registration, "artifact registration");
    await port.resolveArtifact({
      p_run_id: runId,
      p_artifact_id: asString(registration.artifact_id, "artifact_id"),
      p_version: asPositiveInteger(registration.version, "version"),
      p_expected_sha256: asString(registration.content_sha256, "content_sha256"),
      p_required_authority_class: asString(registration.authority_class, "authority_class"),
    });
  }
}

async function postWriteReceipt(input: {
  port: ControlledBridgePort;
  request: CheckpointRequest | FinalizeRequest | ReopenRequest;
  rpcResult: RpcResult;
}): Promise<MutationReceipt> {
  const run = await input.port.getRun(input.request.run_id);
  if (!run) throw new ControlledBridgeError("NOT_FOUND", "run disappeared after mutation");
  if (run.current_stage !== input.request.stage_code) {
    throw new ControlledBridgeError("AUTHORITY_VIOLATION", "post-write current_stage does not match mutation stage");
  }
  const stage = await input.port.getStage(input.request.run_id, input.request.stage_code);
  if (!stage) throw new ControlledBridgeError("NOT_FOUND", "stage disappeared after mutation");

  let manifestId: string | null = null;
  if (input.request.operation === "CHECKPOINT_STAGE" || input.request.operation === "FINALIZE_STAGE") {
    manifestId = asString(input.request.bundle.manifest_registration.artifact_id, "manifest artifact_id");
    const manifestVersion = asPositiveInteger(input.request.bundle.manifest_registration.version, "manifest version");
    if (
      stage.active_manifest_artifact_id !== manifestId ||
      stage.active_manifest_version !== manifestVersion
    ) {
      throw new ControlledBridgeError("AUTHORITY_VIOLATION", "post-write active manifest does not match mutation bundle");
    }
    if (
      input.request.operation === "CHECKPOINT_STAGE" &&
      stage.lifecycle_status !== input.request.target_lifecycle
    ) {
      throw new ControlledBridgeError("INVALID_STATE", "post-write checkpoint lifecycle mismatch");
    }
    if (input.request.operation === "FINALIZE_STAGE" && stage.lifecycle_status !== "COMPLETE") {
      throw new ControlledBridgeError("INVALID_STATE", "post-write finalize did not complete stage");
    }
    await verifyRegisteredArtifacts(input.port, input.request.run_id, input.request.bundle);
  } else {
    if (stage.lifecycle_status !== input.request.target_lifecycle) {
      throw new ControlledBridgeError("INVALID_STATE", "post-write reopen lifecycle mismatch");
    }
    if (
      stage.active_manifest_artifact_id !== null ||
      stage.active_manifest_version !== null ||
      stage.active_manifest_kind !== null
    ) {
      throw new ControlledBridgeError("INVALID_STATE", "reopened stage retained active manifest");
    }
    const expectedStageRevision = asPositiveInteger(
      input.request.reason.expected_stage_revision,
      "reason.expected_stage_revision",
    );
    const rpcStageRevision = input.rpcResult.stage_revision;
    if (
      stage.stage_revision !== expectedStageRevision ||
      typeof rpcStageRevision !== "number" ||
      rpcStageRevision !== expectedStageRevision
    ) {
      throw new ControlledBridgeError(
        "STALE_STATE",
        "post-write reopen stage revision does not match the intended successor revision",
      );
    }
  }

  const idempotentReplay = input.rpcResult.idempotent_replay === true;
  const rpcStageVersion = input.rpcResult.stage_state_version;
  if (typeof rpcStageVersion === "number" && rpcStageVersion !== stage.state_version) {
    throw new ControlledBridgeError("STALE_STATE", "RPC stage_state_version differs from durable post-write state");
  }

  const receipt: MutationReceipt = {
    contract_version: "0.1.0",
    operation: "MUTATION_RECEIPT",
    status: idempotentReplay ? "IDEMPOTENT_REPLAY" : "SUCCESS",
    run_id: input.request.run_id,
    stage_code: input.request.stage_code,
    stage_state_version: stage.state_version,
    manifest_id: manifestId,
    event_id: typeof input.rpcResult.event_id === "string" ? input.rpcResult.event_id : null,
    idempotent_replay: idempotentReplay,
    verification: {
      durable_state_reloaded: true,
      state_matches_intent: true,
      artifact_integrity_verified: true,
    },
  };
  validateOrThrow(receipt);
  return receipt;
}

export async function executeControlledOperation(
  port: ControlledBridgePort,
  request: ControlledRequest,
): Promise<ControlledResult> {
  try {
    validateOrThrow(request);

    if (request.operation === "LOAD") {
      return await loadControlledState(port, request);
    }
    if (request.operation === "NOOP") {
      return request;
    }

    assertMutationIdentity(request);

    let rpcResult: RpcResult;
    if (request.operation === "CHECKPOINT_STAGE") {
      await verifyBundlePersistence(request.bundle, request.run_id, request.stage_code);
      rpcResult = await port.checkpointStage({
        p_run_id: request.run_id,
        p_stage_code: request.stage_code,
        p_expected_run_state_version: request.expected_run_state_version,
        p_expected_stage_state_version: request.expected_stage_state_version,
        p_manifest: request.bundle.manifest,
        p_manifest_registration: request.bundle.manifest_registration,
        p_output_artifacts: request.bundle.output_artifacts,
        p_edges: request.bundle.edges,
        p_target_lifecycle: request.target_lifecycle,
        p_idempotency_key: request.idempotency_key,
        p_request_fingerprint_sha256: request.request_fingerprint_sha256,
        p_actor_type: request.actor_type,
      });
    } else if (request.operation === "FINALIZE_STAGE") {
      await verifyBundlePersistence(request.bundle, request.run_id, request.stage_code);
      rpcResult = await port.finalizeStage({
        p_run_id: request.run_id,
        p_stage_code: request.stage_code,
        p_expected_run_state_version: request.expected_run_state_version,
        p_expected_stage_state_version: request.expected_stage_state_version,
        p_manifest: request.bundle.manifest,
        p_manifest_registration: request.bundle.manifest_registration,
        p_output_artifacts: request.bundle.output_artifacts,
        p_edges: request.bundle.edges,
        p_idempotency_key: request.idempotency_key,
        p_request_fingerprint_sha256: request.request_fingerprint_sha256,
        p_actor_type: request.actor_type,
      });
    } else {
      rpcResult = await port.reopenStage({
        p_run_id: request.run_id,
        p_stage_code: request.stage_code,
        p_expected_run_state_version: request.expected_run_state_version,
        p_expected_stage_state_version: request.expected_stage_state_version,
        p_target_lifecycle: request.target_lifecycle,
        p_reason: request.reason,
        p_idempotency_key: request.idempotency_key,
        p_request_fingerprint_sha256: request.request_fingerprint_sha256,
      });
    }

    return await postWriteReceipt({ port, request, rpcResult });
  } catch (error) {
    return classifyError(error);
  }
}

export function assertControlledOperationContract(value: unknown): void {
  validateOrThrow(value);
}
