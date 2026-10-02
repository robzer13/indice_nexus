import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import Ajv2020 from "ajv/dist/2020";
import addFormats from "ajv-formats";

const schema = JSON.parse(
  readFileSync(
    new URL(
      "../schemas/vnext/chatgpt-supabase/controlled-operation.schema.v0.1.json",
      import.meta.url,
    ),
    "utf8",
  ),
);

const ajv = new Ajv2020({ allErrors: true, strict: false });
addFormats(ajv as unknown as Parameters<typeof addFormats>[0]);
const validate = ajv.compile(schema);

const sha = "a".repeat(64);

function expectValid(value: unknown) {
  assert.equal(validate(value), true, JSON.stringify(validate.errors, null, 2));
}

function expectInvalid(value: unknown) {
  assert.equal(validate(value), false);
}

const bundle = {
  manifest: { manifest_id: "manifest-1" },
  manifest_registration: { version: 1 },
  output_artifacts: [],
  edges: [],
  persistence_receipts_verified: true,
};

test("LOAD request is explicitly read-only", () => {
  expectValid({
    contract_version: "0.1.0",
    operation: "LOAD",
    issuer_query: "Constellation Software",
    requested_context_tiers: ["L0", "L1"],
  });

  expectInvalid({
    contract_version: "0.1.0",
    operation: "LOAD",
    issuer_query: "Constellation Software",
    mutation_allowed: true,
  });
});

test("LOAD_RESULT pins control state and exact artifact refs", () => {
  expectValid({
    contract_version: "0.1.0",
    operation: "LOAD_RESULT",
    mutation_allowed: false,
    issuer_id: "issuer-1",
    security_id: "security-1",
    dossier_id: "dossier-1",
    run_id: "run-1",
    run_status: "ACTIVE",
    run_state_version: 9,
    run_type: "INITIAL",
    canonical_mode: "STANDARD",
    data_cutoff: "2026-10-02",
    contract_set_sha256: sha,
    current_stage: "DEEP_DIVE",
    stage: {
      stage_code: "DEEP_DIVE",
      stage_revision: 1,
      lifecycle_status: "IN_PROGRESS",
      stage_state_version: 4,
      handoff_gate_state: "NOT_EVALUATED",
      active_manifest: {
        artifact_id: "manifest-1",
        version: 2,
        content_sha256: sha,
        required_authority_class: "STAGE_MANIFEST",
      },
    },
    blockers: [],
    artifact_index: [
      { artifact_id: "artifact-1", version: 3, content_sha256: sha },
    ],
    process_state_artifact: {
      artifact_id: "process-state-1",
      version: 1,
      content_sha256: sha,
    },
    context_plan: { l0: [], l1: [], l2: [], l3: [] },
  });
});

test("CHECKPOINT requires CAS, idempotency, verified persistence and no publication", () => {
  expectValid({
    contract_version: "0.1.0",
    operation: "CHECKPOINT_STAGE",
    run_id: "run-1",
    stage_code: "DEEP_DIVE",
    expected_run_state_version: 9,
    expected_stage_state_version: 4,
    save_disposition: "CHECKPOINT",
    target_lifecycle: "IN_PROGRESS",
    bundle,
    idempotency_key: "checkpoint:run-1:dd:4",
    request_fingerprint_sha256: sha,
    actor_type: "DEEP_DIVE_WORKER",
    publish_authorized: false,
  });

  expectInvalid({
    contract_version: "0.1.0",
    operation: "CHECKPOINT_STAGE",
    run_id: "run-1",
    stage_code: "DEEP_DIVE",
    expected_stage_state_version: 4,
    save_disposition: "CHECKPOINT",
    target_lifecycle: "IN_PROGRESS",
    bundle,
    idempotency_key: "checkpoint:run-1:dd:4",
    request_fingerprint_sha256: sha,
    actor_type: "DEEP_DIVE_WORKER",
    publish_authorized: false,
  });
});

test("BLOCK maps to checkpoint lifecycle BLOCKED", () => {
  expectValid({
    contract_version: "0.1.0",
    operation: "CHECKPOINT_STAGE",
    run_id: "run-1",
    stage_code: "RESEARCH",
    expected_run_state_version: 2,
    expected_stage_state_version: 2,
    save_disposition: "BLOCK",
    target_lifecycle: "BLOCKED",
    bundle,
    idempotency_key: "block:run-1:research:2",
    request_fingerprint_sha256: sha,
    actor_type: "RESEARCH_WORKER",
    publish_authorized: false,
  });
});

test("FINALIZE cannot authorize publication", () => {
  const final = {
    contract_version: "0.1.0",
    operation: "FINALIZE_STAGE",
    run_id: "run-1",
    stage_code: "INTEGRATION",
    expected_run_state_version: 12,
    expected_stage_state_version: 7,
    save_disposition: "FINALIZE",
    bundle,
    idempotency_key: "finalize:run-1:integration:7",
    request_fingerprint_sha256: sha,
    actor_type: "INTEGRATION_WORKER",
    publish_authorized: false,
  };
  expectValid(final);
  expectInvalid({ ...final, publish_authorized: true });
});

test("REOPEN requires exact state versions and a structured reason", () => {
  expectValid({
    contract_version: "0.1.0",
    operation: "REOPEN_STAGE",
    run_id: "run-1",
    stage_code: "DEEP_DIVE",
    expected_run_state_version: 15,
    expected_stage_state_version: 8,
    target_lifecycle: "IN_PROGRESS",
    reason: {
      code: "NEW_MATERIAL_EVIDENCE",
      summary: "Material evidence invalidated a prior locked conclusion",
    },
    idempotency_key: "reopen:run-1:dd:8",
    request_fingerprint_sha256: sha,
    publish_authorized: false,
  });
});

test("NOOP carries no mutation controls", () => {
  expectValid({
    contract_version: "0.1.0",
    operation: "NOOP",
    run_id: "run-1",
    stage_code: "DEEP_DIVE",
    save_disposition: "NOOP",
    reason: "No durable transition is permitted",
    publish_authorized: false,
  });
});

test("mutation receipt requires post-write durable verification", () => {
  const receipt = {
    contract_version: "0.1.0",
    operation: "MUTATION_RECEIPT",
    status: "SUCCESS",
    run_id: "run-1",
    stage_code: "DEEP_DIVE",
    stage_state_version: 5,
    manifest_id: "manifest-2",
    event_id: "event-1",
    idempotent_replay: false,
    verification: {
      durable_state_reloaded: true,
      state_matches_intent: true,
      artifact_integrity_verified: true,
    },
  };
  expectValid(receipt);
  expectInvalid({
    ...receipt,
    verification: {
      ...receipt.verification,
      durable_state_reloaded: false,
    },
  });
});

test("operation failure forbids blind retry without reload", () => {
  expectValid({
    contract_version: "0.1.0",
    operation: "OPERATION_FAILURE",
    error_class: "STALE_STATE",
    message: "RUN_STATE_VERSION_MISMATCH",
    retry_without_reload_allowed: false,
  });

  expectInvalid({
    contract_version: "0.1.0",
    operation: "OPERATION_FAILURE",
    error_class: "STALE_STATE",
    message: "RUN_STATE_VERSION_MISMATCH",
    retry_without_reload_allowed: true,
  });
});

test("artifact refs require exact version", () => {
  const value = {
    contract_version: "0.1.0",
    operation: "LOAD_RESULT",
    mutation_allowed: false,
    issuer_id: "issuer-1",
    security_id: null,
    dossier_id: "dossier-1",
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
    artifact_index: [{ artifact_id: "artifact-without-version" }],
    context_plan: { l0: [], l1: [], l2: [], l3: [] },
  };
  expectInvalid(value);
});

test("LOAD_RESULT fails closed when current_stage and stage.stage_code disagree", () => {
  const value = {
    contract_version: "0.1.0",
    operation: "LOAD_RESULT",
    mutation_allowed: false,
    issuer_id: "issuer-1",
    security_id: "security-1",
    dossier_id: "dossier-1",
    run_id: "run-1",
    run_status: "ACTIVE",
    run_state_version: 9,
    run_type: "INITIAL",
    canonical_mode: "STANDARD",
    data_cutoff: "2026-10-02",
    contract_set_sha256: sha,
    current_stage: "RESEARCH",
    stage: {
      stage_code: "DEEP_DIVE",
      stage_revision: 1,
      lifecycle_status: "IN_PROGRESS",
      stage_state_version: 4,
      handoff_gate_state: "NOT_EVALUATED",
      active_manifest: null,
    },
    blockers: [],
    artifact_index: [],
    process_state_artifact: null,
    context_plan: { l0: [], l1: [], l2: [], l3: [] },
  };
  expectInvalid(value);

  expectInvalid({
    ...value,
    run_id: null,
    run_status: null,
    run_state_version: null,
    run_type: null,
    canonical_mode: null,
    data_cutoff: null,
    contract_set_sha256: null,
    current_stage: null,
  });
});

test("CHECKPOINT and BLOCK dispositions are coupled to lifecycle", () => {
  const common = {
    contract_version: "0.1.0",
    operation: "CHECKPOINT_STAGE",
    run_id: "run-1",
    stage_code: "RESEARCH",
    expected_run_state_version: 2,
    expected_stage_state_version: 2,
    bundle,
    idempotency_key: "checkpoint:run-1:research:2",
    request_fingerprint_sha256: sha,
    actor_type: "RESEARCH_WORKER",
    publish_authorized: false,
  };

  expectInvalid({
    ...common,
    save_disposition: "BLOCK",
    target_lifecycle: "IN_PROGRESS",
  });

  expectInvalid({
    ...common,
    save_disposition: "CHECKPOINT",
    target_lifecycle: "BLOCKED",
  });

  expectValid({
    ...common,
    save_disposition: "CHECKPOINT",
    target_lifecycle: "PAUSED",
  });
});

test("mutation receipt status is coupled to replay flag", () => {
  const receipt = {
    contract_version: "0.1.0",
    operation: "MUTATION_RECEIPT",
    run_id: "run-1",
    stage_code: "DEEP_DIVE",
    stage_state_version: 5,
    manifest_id: "manifest-2",
    event_id: "event-1",
    verification: {
      durable_state_reloaded: true,
      state_matches_intent: true,
      artifact_integrity_verified: true,
    },
  };

  expectInvalid({
    ...receipt,
    status: "SUCCESS",
    idempotent_replay: true,
  });

  expectInvalid({
    ...receipt,
    status: "IDEMPOTENT_REPLAY",
    idempotent_replay: false,
  });

  expectValid({
    ...receipt,
    status: "IDEMPOTENT_REPLAY",
    idempotent_replay: true,
  });
});
