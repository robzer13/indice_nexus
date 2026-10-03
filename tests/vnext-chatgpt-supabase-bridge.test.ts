import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import test from "node:test";
import {
  buildReopenOperation,
  buildSaveOperation,
  executeControlledOperation,
  loadControlledState,
  type ArtifactRow,
  type Bundle,
  type ControlledBridgePort,
  type DossierRow,
  type IssuerRow,
  type RpcResult,
  type RunRow,
  type SecurityRow,
  type StageCode,
  type StageRow,
} from "../lib/orotitan-equity/post-c7/chatgpt-supabase-bridge";

const RUN_ID = "10000000-0000-4000-8000-000000000001";
const ISSUER_ID = "20000000-0000-4000-8000-000000000002";
const SECURITY_ID = "30000000-0000-4000-8000-000000000003";
const DOSSIER_ID = "40000000-0000-4000-8000-000000000004";
const MANIFEST_ID = "50000000-0000-4000-8000-000000000005";
const sha = "a".repeat(64);

function stableJson(value: unknown): string {
  if (value === null || typeof value !== "object") return JSON.stringify(value);
  if (Array.isArray(value)) return `[${value.map(stableJson).join(",")}]`;
  const entries = Object.entries(value as Record<string, unknown>)
    .sort(([left], [right]) => left.localeCompare(right));
  return `{${entries.map(([key, child]) => `${JSON.stringify(key)}:${stableJson(child)}`).join(",")}}`;
}

function manifestRegistration(
  manifest: Record<string, unknown>,
  authorityClass: "CHECKPOINT_STAGE_OUTPUT" | "AUTHORITATIVE_STAGE_OUTPUT" = "CHECKPOINT_STAGE_OUTPUT",
): Record<string, unknown> {
  const bytes = Buffer.from(JSON.stringify(manifest), "utf8");
  const commit = "c".repeat(40);
  const repository = "robzer13/real-orotitan";
  const path = "runs/test/manifest.json";
  const blob = createHash("sha1")
    .update(Buffer.concat([Buffer.from(`blob ${bytes.byteLength}\0`, "utf8"), bytes]))
    .digest("hex");
  return {
    artifact_id: MANIFEST_ID,
    version: 1,
    artifact_type: "DEEP_DIVE_STAGE_MANIFEST",
    logical_name: "deep_dive_stage_manifest",
    authority_class: authorityClass,
    artifact_status: "SEALED",
    authority_state: authorityClass === "AUTHORITATIVE_STAGE_OUTPUT" ? "AUTHORITATIVE" : "CHECKPOINT",
    availability_state: "AVAILABLE",
    media_type: "application/json",
    size_bytes: bytes.byteLength,
    content_sha256: createHash("sha256").update(bytes).digest("hex"),
    storage_backend: "PRIVATE_GITHUB",
    storage_uri: `github://${repository}@${commit}/${path}`,
    github_repository: repository,
    github_path: path,
    github_commit_sha: commit,
    github_blob_sha: blob,
    canonical_json_content: manifest,
    persistence_receipt: {
      receipt_schema_version: "1.1",
      verification_method: "PRIVATE_GITHUB_ATTESTED_REREAD_EXACT_BYTES_V1",
      storage_backend: "PRIVATE_GITHUB",
      run_id: RUN_ID,
      stage_code: "DEEP_DIVE",
      artifact_id: MANIFEST_ID,
      version: 1,
      artifact_type: "DEEP_DIVE_STAGE_MANIFEST",
      github_repository: repository,
      github_path: path,
      github_commit_sha: commit,
      github_blob_sha: blob,
      attestation_event_id: "70000000-0000-4000-8000-000000000007",
      commit_path_resolved: true,
      verified_content_base64: bytes.toString("base64"),
      verified_at: "2026-10-02T08:30:00Z",
    },
  };
}

function baseRows() {
  const issuers: IssuerRow[] = [
    { issuer_id: ISSUER_ID, display_name: "ASML Holding N.V.", legal_name: "ASML Holding N.V." },
  ];
  const securities: SecurityRow[] = [
    {
      security_id: SECURITY_ID,
      issuer_id: ISSUER_ID,
      ticker: "ASML",
      market_data_symbol: "ASML.AS",
      primary_listing: true,
      listing_status: "ACTIVE",
    },
  ];
  const dossiers: DossierRow[] = [
    {
      dossier_id: DOSSIER_ID,
      issuer_id: ISSUER_ID,
      current_snapshot_id: null,
      active: true,
    },
  ];
  const runs: RunRow[] = [
    {
      run_id: RUN_ID,
      issuer_id: ISSUER_ID,
      security_id: SECURITY_ID,
      dossier_id: DOSSIER_ID,
      run_status: "ACTIVE",
      current_stage: "DEEP_DIVE",
      run_type: "INITIAL",
      canonical_mode: "ANALYZE",
      data_cutoff: "2026-10-02",
      contract_set_sha256: sha,
      state_version: 7,
      updated_at: "2026-10-02T08:00:00Z",
    },
  ];
  const stages: StageRow[] = [
    {
      run_id: RUN_ID,
      stage_code: "DEEP_DIVE",
      stage_revision: 1,
      lifecycle_status: "IN_PROGRESS",
      handoff_gate_state: "NOT_EVALUATED",
      active_manifest_artifact_id: null,
      active_manifest_version: null,
      active_manifest_kind: null,
      blocker_summary: [],
      state_version: 4,
    },
  ];
  const artifacts: ArtifactRow[] = [];
  return { issuers, securities, dossiers, runs, stages, artifacts };
}

class FakePort implements ControlledBridgePort {
  readonly state = baseRows();
  checkpointCalls = 0;
  finalizeCalls = 0;
  reopenCalls = 0;
  resolveCalls = 0;
  checkpointError: Error | null = null;
  reopenReplay = false;

  async listIssuers() { return this.state.issuers; }
  async listSecurities() { return this.state.securities; }
  async listDossiers(issuerId: string) {
    return this.state.dossiers.filter((row) => row.issuer_id === issuerId);
  }
  async listRuns(issuerId: string) {
    return this.state.runs.filter((row) => row.issuer_id === issuerId);
  }
  async getRun(runId: string) {
    return this.state.runs.find((row) => row.run_id === runId) ?? null;
  }
  async getStage(runId: string, stage: StageCode) {
    return this.state.stages.find((row) => row.run_id === runId && row.stage_code === stage) ?? null;
  }
  async listArtifacts(runId: string) {
    return this.state.artifacts.filter((row) => row.run_id === runId);
  }
  async readSupabaseObject(): Promise<Uint8Array> {
    throw new Error("unexpected Supabase object read");
  }
  async resolveArtifact(): Promise<RpcResult> {
    this.resolveCalls += 1;
    return { artifact_id: MANIFEST_ID };
  }
  async checkpointStage(args: Record<string, unknown>): Promise<RpcResult> {
    this.checkpointCalls += 1;
    if (this.checkpointError) throw this.checkpointError;
    const stage = this.state.stages[0];
    const registration = (args.p_manifest_registration ?? {}) as Record<string, unknown>;
    stage.lifecycle_status = args.p_target_lifecycle as StageRow["lifecycle_status"];
    stage.active_manifest_artifact_id = registration.artifact_id as string;
    stage.active_manifest_version = registration.version as number;
    stage.active_manifest_kind = "CHECKPOINT";
    stage.state_version += 1;
    this.state.runs[0].state_version += 1;
    return {
      stage_state_version: stage.state_version,
      manifest_id: registration.artifact_id,
      event_id: "60000000-0000-4000-8000-000000000006",
      idempotent_replay: false,
    };
  }
  async finalizeStage(args: Record<string, unknown>): Promise<RpcResult> {
    this.finalizeCalls += 1;
    const stage = this.state.stages[0];
    const registration = (args.p_manifest_registration ?? {}) as Record<string, unknown>;
    stage.lifecycle_status = "COMPLETE";
    stage.active_manifest_artifact_id = registration.artifact_id as string;
    stage.active_manifest_version = registration.version as number;
    stage.active_manifest_kind = "FINAL";
    stage.state_version += 1;
    this.state.runs[0].state_version += 1;
    return {
      stage_state_version: stage.state_version,
      manifest_id: registration.artifact_id,
      event_id: "60000000-0000-4000-8000-000000000006",
      idempotent_replay: false,
    };
  }
  async reopenStage(args: Record<string, unknown>): Promise<RpcResult> {
    this.reopenCalls += 1;
    const stage = this.state.stages[0];
    if (this.reopenReplay) {
      return {
        stage_revision: stage.stage_revision,
        stage_state_version: stage.state_version,
        idempotent_replay: true,
      };
    }
    stage.lifecycle_status = args.p_target_lifecycle as StageRow["lifecycle_status"];
    stage.active_manifest_artifact_id = null;
    stage.active_manifest_version = null;
    stage.active_manifest_kind = null;
    stage.stage_revision += 1;
    stage.state_version += 1;
    this.state.runs[0].state_version += 1;
    return {
      stage_revision: stage.stage_revision,
      stage_state_version: stage.state_version,
      event_id: "60000000-0000-4000-8000-000000000006",
      idempotent_replay: false,
    };
  }
}

async function loaded(port: FakePort) {
  return loadControlledState(port, {
    contract_version: "0.1.0",
    operation: "LOAD",
    issuer_query: "ASML",
    requested_context_tiers: ["L0", "L1"],
  });
}

function checkpointBundle(kind: "CHECKPOINT" | "FINAL" = "CHECKPOINT"): Bundle {
  const manifest = {
    manifest_id: MANIFEST_ID,
    run_id: RUN_ID,
    stage: "DEEP_DIVE",
    stage_revision: 1,
    manifest_kind: kind,
    output_artifacts: [],
  };
  return {
    manifest,
    manifest_registration: manifestRegistration(
      manifest,
      kind === "FINAL" ? "AUTHORITATIVE_STAGE_OUTPUT" : "CHECKPOINT_STAGE_OUTPUT",
    ),
    output_artifacts: [],
    edges: [],
    persistence_receipts_verified: true,
  };
}

test("LOAD resolves exact issuer by ticker and returns mutation-free current control state", async () => {
  const port = new FakePort();
  const result = await loaded(port);
  assert.equal(result.mutation_allowed, false);
  assert.equal(result.issuer_id, ISSUER_ID);
  assert.equal(result.security_id, SECURITY_ID);
  assert.equal(result.dossier_id, DOSSIER_ID);
  assert.equal(result.run_id, RUN_ID);
  assert.equal(result.current_stage, "DEEP_DIVE");
  assert.equal(result.stage?.stage_code, result.current_stage);
  assert.equal(result.run_state_version, 7);
  assert.equal(result.stage?.stage_state_version, 4);
});

test("LOAD fails closed when multiple non-terminal runs exist without exact run_id", async () => {
  const port = new FakePort();
  port.state.runs.push({ ...port.state.runs[0], run_id: "10000000-0000-4000-8000-000000000099" });
  const result = await executeControlledOperation(port, {
    contract_version: "0.1.0",
    operation: "LOAD",
    issuer_query: "ASML",
  });
  assert.equal(result.operation, "OPERATION_FAILURE");
  if (result.operation === "OPERATION_FAILURE") {
    assert.equal(result.error_class, "INVALID_STATE");
    assert.equal(result.retry_without_reload_allowed, false);
  }
});

test("Process Engine SAVE dispositions map deterministically to frozen bridge operations", async () => {
  const port = new FakePort();
  const load = await loaded(port);

  const checkpoint = buildSaveOperation({
    load,
    disposition: { action: "CHECKPOINT", reason: "working state", publishAuthorized: false },
    bundle: checkpointBundle(),
    actorType: "DEEP_DIVE_WORKER",
  });
  assert.equal(checkpoint.operation, "CHECKPOINT_STAGE");
  if (checkpoint.operation === "CHECKPOINT_STAGE") {
    assert.equal(checkpoint.save_disposition, "CHECKPOINT");
    assert.equal(checkpoint.target_lifecycle, "IN_PROGRESS");
    assert.equal(checkpoint.publish_authorized, false);
  }

  const blocked = buildSaveOperation({
    load,
    disposition: { action: "BLOCK", reason: "critical blocker", publishAuthorized: false },
    bundle: checkpointBundle(),
    actorType: "DEEP_DIVE_WORKER",
  });
  assert.equal(blocked.operation, "CHECKPOINT_STAGE");
  if (blocked.operation === "CHECKPOINT_STAGE") {
    assert.equal(blocked.save_disposition, "BLOCK");
    assert.equal(blocked.target_lifecycle, "BLOCKED");
  }

  const noop = buildSaveOperation({
    load,
    disposition: { action: "NOOP", reason: "nothing durable", publishAuthorized: false },
    actorType: "DEEP_DIVE_WORKER",
  });
  assert.equal(noop.operation, "NOOP");
});

test("CHECKPOINT verifies persisted manifest, calls guarded RPC and reloads durable state", async () => {
  const port = new FakePort();
  const load = await loaded(port);
  const request = buildSaveOperation({
    load,
    disposition: { action: "CHECKPOINT", reason: "working state", publishAuthorized: false },
    bundle: checkpointBundle(),
    actorType: "DEEP_DIVE_WORKER",
  });
  assert.equal(request.operation, "CHECKPOINT_STAGE");
  const result = await executeControlledOperation(port, request);
  assert.equal(result.operation, "MUTATION_RECEIPT");
  if (result.operation === "MUTATION_RECEIPT") {
    assert.equal(result.status, "SUCCESS");
    assert.equal(result.idempotent_replay, false);
    assert.equal(result.verification.durable_state_reloaded, true);
    assert.equal(result.verification.state_matches_intent, true);
    assert.equal(result.verification.artifact_integrity_verified, true);
  }
  assert.equal(port.checkpointCalls, 1);
  assert.equal(port.resolveCalls, 1);
});

test("FINALIZE remains distinct from publication and verifies durable completion", async () => {
  const port = new FakePort();
  const load = await loaded(port);
  const request = buildSaveOperation({
    load,
    disposition: { action: "FINALIZE", reason: "gate passed", publishAuthorized: false },
    bundle: checkpointBundle("FINAL"),
    actorType: "DEEP_DIVE_WORKER",
  });
  assert.equal(request.operation, "FINALIZE_STAGE");
  if (request.operation !== "FINALIZE_STAGE") throw new Error("expected FINALIZE_STAGE");
  assert.equal(request.publish_authorized, false);
  const result = await executeControlledOperation(port, request);
  assert.equal(result.operation, "MUTATION_RECEIPT");
  assert.equal(port.finalizeCalls, 1);
});

test("stale CAS failure is normalized and never blind-retried", async () => {
  const port = new FakePort();
  const load = await loaded(port);
  port.checkpointError = new Error("RUN_STATE_VERSION_MISMATCH");
  const request = buildSaveOperation({
    load,
    disposition: { action: "CHECKPOINT", reason: "working state", publishAuthorized: false },
    bundle: checkpointBundle(),
    actorType: "DEEP_DIVE_WORKER",
  });
  const result = await executeControlledOperation(port, request);
  assert.equal(result.operation, "OPERATION_FAILURE");
  if (result.operation === "OPERATION_FAILURE") {
    assert.equal(result.error_class, "STALE_STATE");
    assert.equal(result.retry_without_reload_allowed, false);
  }
  assert.equal(port.checkpointCalls, 1);
});

test("REOPEN uses loaded CAS state and verifies active manifest is cleared", async () => {
  const port = new FakePort();
  port.state.stages[0].lifecycle_status = "COMPLETE";
  port.state.stages[0].active_manifest_artifact_id = MANIFEST_ID;
  port.state.stages[0].active_manifest_version = 1;
  port.state.stages[0].active_manifest_kind = "FINAL";
  port.state.artifacts.push({
    artifact_id: MANIFEST_ID,
    version: 1,
    run_id: RUN_ID,
    stage_code: "DEEP_DIVE",
    artifact_type: "DEEP_DIVE_STAGE_MANIFEST",
    logical_name: "deep_dive_stage_manifest",
    authority_class: "AUTHORITATIVE_STAGE_OUTPUT",
    authority_state: "AUTHORITATIVE",
    artifact_status: "SEALED",
    availability_state: "AVAILABLE",
    content_sha256: sha,
    size_bytes: 2,
    media_type: "application/json",
    storage_backend: "SUPABASE_STORAGE",
    storage_uri: "supabase://bucket/path",
    github_repository: null,
    github_path: null,
    github_commit_sha: null,
    github_blob_sha: null,
    supabase_bucket: "bucket",
    supabase_object_path: "path",
    manifest_artifact_id: null,
    manifest_version: null,
  });
  const load = await loaded(port);
  const request = buildReopenOperation({
    load,
    targetLifecycle: "IN_PROGRESS",
    reason: { code: "NEW_MATERIAL_EVIDENCE", summary: "Material evidence requires revalidation" },
  });
  assert.equal(request.reason.expected_stage_revision, 2);
  const result = await executeControlledOperation(port, request);
  assert.equal(result.operation, "MUTATION_RECEIPT");
  assert.equal(port.reopenCalls, 1);
  assert.equal(port.state.stages[0].active_manifest_artifact_id, null);
});

test("obsolete pre-attestation receipt is rejected before Registry mutation", async () => {
  const port = new FakePort();
  const load = await loaded(port);
  const bundle = checkpointBundle();
  const receipt = bundle.manifest_registration.persistence_receipt as Record<string, unknown>;
  receipt.receipt_schema_version = "1.0";
  receipt.verification_method = "PRIVATE_GITHUB_REREAD_EXACT_BYTES_V1";
  const request = buildSaveOperation({
    load,
    disposition: { action: "CHECKPOINT", reason: "working state", publishAuthorized: false },
    bundle,
    actorType: "DEEP_DIVE_WORKER",
  });
  const result = await executeControlledOperation(port, request);
  assert.equal(result.operation, "OPERATION_FAILURE");
  if (result.operation === "OPERATION_FAILURE") {
    assert.equal(result.error_class, "PERSISTENCE_INTEGRITY");
  }
  assert.equal(port.checkpointCalls, 0);
});

test("manifest persistence receipt is checked locally before Registry mutation", async () => {
  const port = new FakePort();
  const load = await loaded(port);
  const bundle = checkpointBundle();
  bundle.manifest_registration.content_sha256 = "0".repeat(64);
  const request = buildSaveOperation({
    load,
    disposition: { action: "CHECKPOINT", reason: "working state", publishAuthorized: false },
    bundle,
    actorType: "DEEP_DIVE_WORKER",
  });
  const result = await executeControlledOperation(port, request);
  assert.equal(result.operation, "OPERATION_FAILURE");
  if (result.operation === "OPERATION_FAILURE") {
    assert.equal(result.error_class, "PERSISTENCE_INTEGRITY");
  }
  assert.equal(port.checkpointCalls, 0);
});

test("generated idempotency identity is stable for the same loaded state and bundle", async () => {
  const port = new FakePort();
  const load = await loaded(port);
  const bundle = checkpointBundle();
  const left = buildSaveOperation({
    load,
    disposition: { action: "CHECKPOINT", reason: "working state", publishAuthorized: false },
    bundle,
    actorType: "DEEP_DIVE_WORKER",
  });
  const right = buildSaveOperation({
    load,
    disposition: { action: "CHECKPOINT", reason: "working state", publishAuthorized: false },
    bundle,
    actorType: "DEEP_DIVE_WORKER",
  });
  if (left.operation !== "CHECKPOINT_STAGE" || right.operation !== "CHECKPOINT_STAGE") {
    throw new Error("expected checkpoint requests");
  }
  assert.equal(left.request_fingerprint_sha256, right.request_fingerprint_sha256);
  assert.equal(left.idempotency_key, right.idempotency_key);
  assert.equal(stableJson(left.bundle), stableJson(right.bundle));
});

test("dispatch rejects a tampered mutation identity before guarded RPC invocation", async () => {
  const port = new FakePort();
  const load = await loaded(port);
  const request = buildSaveOperation({
    load,
    disposition: { action: "CHECKPOINT", reason: "working state", publishAuthorized: false },
    bundle: checkpointBundle(),
    actorType: "DEEP_DIVE_WORKER",
  });
  if (request.operation !== "CHECKPOINT_STAGE") throw new Error("expected checkpoint request");
  request.bundle.manifest = { ...request.bundle.manifest, stage_revision: 99 };
  const result = await executeControlledOperation(port, request);
  assert.equal(result.operation, "OPERATION_FAILURE");
  if (result.operation === "OPERATION_FAILURE") {
    assert.equal(result.error_class, "CONTRACT_VIOLATION");
  }
  assert.equal(port.checkpointCalls, 0);
});

test("replayed REOPEN cannot validate against a later durable stage revision", async () => {
  const port = new FakePort();
  port.state.stages[0].lifecycle_status = "COMPLETE";
  const load = await loaded(port);
  const request = buildReopenOperation({
    load,
    targetLifecycle: "IN_PROGRESS",
    reason: { code: "REVALIDATE", summary: "Reopen for revalidation" },
  });
  assert.equal(request.reason.expected_stage_revision, 2);

  port.reopenReplay = true;
  port.state.stages[0].stage_revision = 3;
  port.state.stages[0].state_version = 12;
  port.state.stages[0].lifecycle_status = "IN_PROGRESS";
  port.state.stages[0].active_manifest_artifact_id = null;
  port.state.stages[0].active_manifest_version = null;
  port.state.stages[0].active_manifest_kind = null;
  port.state.runs[0].state_version = 15;

  const result = await executeControlledOperation(port, request);
  assert.equal(result.operation, "OPERATION_FAILURE");
  if (result.operation === "OPERATION_FAILURE") {
    assert.equal(result.error_class, "STALE_STATE");
  }
  assert.equal(port.reopenCalls, 1);
});
