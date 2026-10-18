import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import test from "node:test";
import {
  loadTrustedPilotageDecision, TrustedLoaderError,
  type ArtifactRow, type IdentityRow, type RunRow, type StageRow, type TrustedPilotagePort,
} from "../lib/orotitan-equity/v3/pilotage-trusted-loader-v3-1";
import pins from "../contracts/orotitan-equity/v3/contract-pin-pack-v3/OROTITAN_CONTRACT_PIN_PACK_V3_0_2.json";

const IDS = {
  run: "11111111-1111-4111-8111-111111111111",
  issuer: "22222222-2222-4222-8222-222222222222",
  security: "33333333-3333-4333-8333-333333333333",
  dossier: "44444444-4444-4444-8444-444444444444",
};
const COMMIT = "b".repeat(40);
const REPOSITORY = "example/immutable-private-store";
const ENV = { projectRef: "cugpgtzygqqlxetyeven", projectName: "orotitan-screener",
  region: "eu-west-3", status: "ACTIVE_HEALTHY" };
const RESEARCH_OUTPUTS = [
  "RESEARCH_SOURCE_MANIFEST", "EVIDENCE_LEDGER", "CONFLICT_LEDGER",
  "MATERIAL_RESEARCH_HYPOTHESIS_REGISTER", "RESEARCH_GAP_REGISTER",
  "DD_INPUT_SUFFICIENCY_RECORD", "ANALYSIS_INPUT_LOCK",
];

function digest(bytes: Uint8Array, kind: "sha256" | "sha1"): string {
  if (kind === "sha256") return createHash(kind).update(bytes).digest("hex");
  return createHash(kind).update(Buffer.from("blob " + bytes.length + "\0"))
    .update(bytes).digest("hex");
}
function artifact(code: "RESEARCH" | "DEEP_DIVE", type: string, idNum: number,
  bytes: Uint8Array, manifestId: string | null): ArtifactRow {
  const id = "77777777-7777-4777-8777-" + idNum.toString(16).padStart(12, "0");
  const path = "runs/" + IDS.run + "/" + id + ".json";
  return {
    artifact_id: id, version: 1, run_id: IDS.run, stage_code: code,
    artifact_type: type,
    authority_class: "AUTHORITATIVE_STAGE_OUTPUT", authority_state: "AUTHORITATIVE",
    artifact_status: "SEALED", availability_state: "AVAILABLE",
    media_type: type === "RESEARCH_STAGE_MANIFEST" ? "application/json" : "text/plain",
    size_bytes: bytes.length, hash_algorithm: "SHA-256", content_sha256: digest(bytes, "sha256"),
    storage_backend: "PRIVATE_GITHUB", storage_uri: "github://" + REPOSITORY + "@" + COMMIT + "/" + path,
    github_repository: REPOSITORY, github_path: path, github_commit_sha: COMMIT,
    github_blob_sha: digest(bytes, "sha1"),
    manifest_artifact_id: manifestId, manifest_version: manifestId === null ? null : 1,
  };
}
function stage(code: "RESEARCH" | "DEEP_DIVE", lifecycle: StageRow["lifecycle_status"]): StageRow {
  const p = code === "RESEARCH" ? pins.contract_pins.research_stage : pins.contract_pins.deep_dive_stage;
  return {
    run_id: IDS.run, stage_code: code, stage_revision: 1,
    stage_contract_name: p.name, stage_contract_version: p.version,
    stage_contract_sha256: p.content_sha256, lifecycle_status: lifecycle,
    handoff_gate_name: code === "RESEARCH" ? "READY_FOR_DEEP_DIVE" : "READY_FOR_INTEGRATION",
    handoff_gate_state: "NOT_EVALUATED", active_manifest_artifact_id: null,
    active_manifest_version: null, active_manifest_kind: null, blocker_summary: [],
    state_version: 2,
  };
}

type Fixture = {
  port: TrustedPilotagePort;
  run: RunRow;
  stages: StageRow[];
  artifacts: ArtifactRow[];
  bytes: Map<string, Uint8Array>;
  mutateOnSecondRead: boolean;
};

function fixture(deepDive = false): Fixture {
  const run: RunRow = {
    run_id: IDS.run, issuer_id: IDS.issuer, security_id: IDS.security,
    dossier_id: IDS.dossier, baseline_snapshot_id: null, canonical_mode: "ANALYZE", run_type: "INITIAL",
    run_status: "ACTIVE", current_stage: deepDive ? "DEEP_DIVE" : "RESEARCH",
    data_cutoff: "2026-10-16", process_version: "3.1", pilotage_contract_version: "3.1",
    contract_pins: pins.contract_pins, contract_set_sha256: pins.contract_set_sha256,
    state_version: 4,
  };
  const stages = [stage("RESEARCH", deepDive ? "COMPLETE" : "NOT_STARTED")];
  if (deepDive) stages.push(stage("DEEP_DIVE", "NOT_STARTED"));
  const artifacts: ArtifactRow[] = [];
  const bytes = new Map<string, Uint8Array>();
  if (deepDive) {
    const manifestId = "77777777-7777-4777-8777-00000000008f";
    RESEARCH_OUTPUTS.forEach((name, i) => {
      const buffer = Buffer.from("Fixture Research " + name, "utf8");
      const row = artifact("RESEARCH", name, 80 + i, buffer, manifestId);
      artifacts.push(row);
      bytes.set(row.artifact_id, buffer);
    });
    const outputRefs = artifacts.map(a => ({
      artifact_id: a.artifact_id, version: a.version,
      artifact_type: a.artifact_type, content_sha256: a.content_sha256,
      authority_class: a.authority_class, media_type: a.media_type, size_bytes: a.size_bytes,
      storage_ref: { backend: a.storage_backend, repository: a.github_repository, path: a.github_path,
        commit_sha: a.github_commit_sha, blob_sha: a.github_blob_sha, storage_uri: a.storage_uri },
    }));
    const manifestJson = {
      manifest_schema_version: "1.0.0",
      manifest_id: manifestId, manifest_kind: "FINAL",
      run_id: IDS.run, stage: "RESEARCH", stage_revision: 1,
      issuer_id: IDS.issuer, security_id: IDS.security, dossier_id: IDS.dossier,
      canonical_mode: "ANALYZE", run_type: "INITIAL", data_cutoff: "2026-10-16",
      baseline_snapshot_id: null, process_version: "3.1", pilotage_contract_version: "3.1",
      contract_pins: pins.contract_pins,
      stage_contract: pins.contract_pins.research_stage,
      contract_set_sha256: pins.contract_set_sha256,
      input_artifacts: [], output_artifacts: outputRefs,
      stage_status: "COMPLETE", contract_status_code: null,
      handoff_gate: { name: "READY_FOR_DEEP_DIVE", state: "YES" },
      critical_blockers: [], open_material_limitations: [], parent_manifests: [],
      started_at: "2026-10-16T11:00:00Z", completed_at: "2026-10-16T12:00:00Z",
    };
    const b = Buffer.from(JSON.stringify(manifestJson));
    const manifestRow = artifact("RESEARCH", "RESEARCH_STAGE_MANIFEST", 143, b, null);
    artifacts.push(manifestRow);
    bytes.set(manifestRow.artifact_id, b);
    stages[0].handoff_gate_state = "YES";
    stages[0].active_manifest_artifact_id = manifestRow.artifact_id;
    stages[0].active_manifest_version = 1;
    stages[0].active_manifest_kind = "FINAL";
  }
  const issuer: IdentityRow = { issuer_id: IDS.issuer, display_name: "Example SA", legal_name: "Example SA" };
  let reads = 0;
  const output: Fixture = {
    port: {
      async getRun() {
        reads++;
        // Alter the persisted version only during the final confirmation round.
        if (output.mutateOnSecondRead && reads > 1) return { ...run, state_version: 5 };
        return { ...run };
      },
      async getStages() { return stages.map(x => ({ ...x })); },
      async getArtifacts() { return artifacts.map(x => ({ ...x })); },
      async getIssuer() { return { ...issuer }; },
      async readImmutableBytes(row) {
        const buffer = bytes.get(row.artifact_id);
        if (!buffer) throw new Error("unexpected test artifact request");
        return buffer;
      },
    },
    run, stages, artifacts, bytes, mutateOnSecondRead: false,
  };
  return output;
}

async function load(f: Fixture) {
  return loadTrustedPilotageDecision({
    port: f.port, environment: ENV, runId: IDS.run,
    expectedRunStateVersion: 4, expectedCurrentStageStateVersion: 2,
  });
}
async function fails(f: Fixture, code: string) {
  await assert.rejects(load(f), (e: unknown) => e instanceof TrustedLoaderError &&
    e.code === code, code);
}
test("empty initial Research is read without writes", async () => {
  const decision = await load(fixture());
  assert.equal(decision.outcome, "CONTINUE");
  assert.equal(decision.phase, "RESEARCH");
  assert.equal(decision.mutationPerformed, false);
});
test("byte-verified FINAL manifest and canonical outputs admit Fundamentals", async () => {
  const decision = await load(fixture(true));
  assert.equal(decision.outcome, "HANDOFF");
  assert.equal(decision.phase, "FUNDAMENTALS");
  assert.match(decision.exactPrompt ?? "", /RESEARCH_STAGE_MANIFEST/);
  assert.match(decision.exactPrompt ?? "", /ANALYSIS_INPUT_LOCK/);
  assert.equal(decision.publicationAuthorized, false);
});
test("tampered output bytes fail SHA-256", async () => {
  const f = fixture(true);
  f.bytes.set(f.artifacts[0].artifact_id, Buffer.from("tampered output"));
  await fails(f, "ARTIFACT_SIZE_MISMATCH");
});
test("same-length byte tampering fails SHA-256", async () => {
  const f = fixture(true);
  const id = f.artifacts[0].artifact_id;
  const b = Buffer.from(f.bytes.get(id)!);
  b[0] ^= 1;
  f.bytes.set(id, b);
  await fails(f, "ARTIFACT_SHA256_MISMATCH");
});
test("wrong immutable URI fails before handoff", async () => {
  const f = fixture(true);
  f.artifacts[0].storage_uri = "github://forged/repo@commit/path";
  // Manifest and Registry immutable locators disagree before the byte check.
  await fails(f, "OUTPUT_ARTIFACT_MANIFEST_MISMATCH");
});
test("missing authoritative output fails exact bundle count", async () => {
  const f = fixture(true);
  f.artifacts.shift();
  await fails(f, "MANIFEST_OUTPUT_COUNT_MISMATCH");
});
test("Registry YES cannot override byte-verified manifest NO", async () => {
  const f = fixture(true);
  f.stages[0].handoff_gate_state = "NO";
  await fails(f, "MANIFEST_HANDOFF_GATE_MISMATCH");
});
test("optimistic CAS mismatch fails before reading content", async () => {
  const f = fixture();
  f.run.state_version = 5;
  await fails(f, "STALE_RUN_STATE");
});
test("revalidation catches Registry version changing during verification", async () => {
  const f = fixture(true);
  f.mutateOnSecondRead = true;
  await fails(f, "REGISTRY_CHANGED_DURING_VERIFICATION");
});
test("cannot load another environment", async () => {
  const f = fixture();
  await assert.rejects(loadTrustedPilotageDecision({
    port: f.port, environment: { ...ENV, projectRef: "awgsurdyvsyolcgpnygh" },
    runId: IDS.run, expectedRunStateVersion: 4,
    expectedCurrentStageStateVersion: 2,
  }), /WRONG_ENVIRONMENT/);
});
test("wrong or incomplete contract pin pack fails before any handoff", async () => {
  const f = fixture(true);
  f.run.contract_pins = { ...pins.contract_pins, pilotage: null };
  await fails(f, "PIN_PACK_NOT_EXACT");
});
