import assert from "node:assert/strict";
import test from "node:test";
import {
  computeContractSetSha256,
  validateStageManifest,
  verifyArtifactBytes,
  type StageManifestValidationContext,
} from "../lib/orotitan-equity/v1/stage-manifest";

const ids = {
  manifest: "11111111-1111-4111-8111-111111111111",
  run: "22222222-2222-4222-8222-222222222222",
  issuer: "33333333-3333-4333-8333-333333333333",
  security: "44444444-4444-4444-8444-444444444444",
  dossier: "55555555-5555-4555-8555-555555555555",
  output: "66666666-6666-4666-8666-666666666666",
};

const stageHash = "a".repeat(64);
const processHash = "b".repeat(64);
const contractPins = {
  process: {
    name: "OROTITAN_RESEARCH_EXECUTION_PROCESS_V1_FREEZE_V1.0",
    version: "1.0",
    content_sha256: processHash,
    locator: {
      backend: "GITHUB_IMMUTABLE",
      repository: "robzer13/indice_nexus",
      path: "contracts/orotitan-equity/v1/execution/OROTITAN_RESEARCH_EXECUTION_PROCESS_V1_FREEZE_V1.0.md",
      commit_sha: "c".repeat(40),
      blob_sha: "d".repeat(40),
    },
  },
  research_stage: {
    name: "OROTITAN_RESEARCH_STAGE_CONTRACT_V1_FREEZE_V1.0",
    version: "1.0",
    content_sha256: stageHash,
    locator: {
      backend: "GITHUB_IMMUTABLE",
      repository: "robzer13/indice_nexus",
      path: "contracts/orotitan-equity/v1/execution/OROTITAN_RESEARCH_STAGE_CONTRACT_V1_FREEZE_V1.0.md",
      commit_sha: "e".repeat(40),
      blob_sha: "f".repeat(40),
    },
  },
};
const contractSetSha256 = computeContractSetSha256(contractPins);

const context: StageManifestValidationContext = {
  runId: ids.run,
  stage: "RESEARCH",
  stageRevision: 1,
  issuerId: ids.issuer,
  securityId: ids.security,
  dossierId: ids.dossier,
  canonicalMode: "ANALYZE",
  runType: "INITIAL",
  dataCutoff: "2026-09-14",
  baselineSnapshotId: null,
  processVersion: "1.0",
  pilotageContractVersion: "1.0.1",
  contractSetSha256,
  stageContractName: "OROTITAN_RESEARCH_STAGE_CONTRACT_V1_FREEZE_V1.0",
  stageContractVersion: "1.0",
  stageContractSha256: stageHash,
};

const validFinal = (): Record<string, unknown> => {
  const pins = structuredClone(contractPins);
  return ({
  manifest_schema_version: "1.0.0",
  manifest_id: ids.manifest,
  manifest_kind: "FINAL",
  run_id: ids.run,
  stage: "RESEARCH",
  stage_revision: 1,
  issuer_id: ids.issuer,
  security_id: ids.security,
  dossier_id: ids.dossier,
  canonical_mode: "ANALYZE",
  run_type: "INITIAL",
  data_cutoff: "2026-09-14",
  baseline_snapshot_id: null,
  process_version: "1.0",
  pilotage_contract_version: "1.0.1",
  contract_pins: pins,
  stage_contract: pins.research_stage,
  contract_set_sha256: contractSetSha256,
  input_artifacts: [],
  output_artifacts: [
    {
      artifact_id: ids.output,
      version: 1,
      artifact_type: "ANALYSIS_INPUT_LOCK",
      content_sha256: "1".repeat(64),
      authority_class: "AUTHORITATIVE_STAGE_OUTPUT",
      media_type: "application/json",
      size_bytes: 123,
      storage_ref: {
        backend: "PRIVATE_GITHUB",
        repository: "robzer13/orotitan-artifacts",
        path: `artifacts/${ids.run}/research/ANALYSIS_INPUT_LOCK__v001.json`,
      },
    },
  ],
  stage_status: "COMPLETE",
  contract_status_code: "COMPLETE",
  handoff_gate: { name: "READY_FOR_DEEP_DIVE", state: "YES" },
  critical_blockers: [],
  open_material_limitations: [],
  parent_manifests: [],
  started_at: "2026-09-14T14:00:00Z",
  completed_at: "2026-09-14T15:00:00Z",
  });
};

test("M01 valid FINAL manifest admits downstream stage", () => {
  const result = validateStageManifest(validFinal(), context);
  assert.equal(result.ok, true);
  if (result.ok) assert.equal(result.downstreamAdmission, true);
});

test("M02 CHECKPOINT cannot declare gate YES", () => {
  const manifest = validFinal();
  manifest.manifest_kind = "CHECKPOINT";
  manifest.stage_status = "PAUSED";
  manifest.completed_at = null;
  const handoff = manifest.handoff_gate as { name: string; state: string };
  handoff.state = "YES";
  const outputs = manifest.output_artifacts as Array<Record<string, unknown>>;
  outputs[0].authority_class = "CHECKPOINT_STAGE_OUTPUT";
  const result = validateStageManifest(manifest, context);
  assert.equal(result.ok, false);
  if (!result.ok) assert.match(result.errors.join("\n"), /CHECKPOINT manifest cannot admit/);
});

test("M03 Stage Manifest cannot self-reference in output_artifacts", () => {
  const manifest = validFinal();
  (manifest.output_artifacts as Array<Record<string, unknown>>)[0].artifact_id = ids.manifest;
  const result = validateStageManifest(manifest, context);
  assert.equal(result.ok, false);
  if (!result.ok) assert.match(result.errors.join("\n"), /must not list itself/);
});

test("M04 data cutoff mismatch fails contract validation", () => {
  const manifest = validFinal();
  manifest.data_cutoff = "2026-09-15";
  const result = validateStageManifest(manifest, context);
  assert.equal(result.ok, false);
  if (!result.ok) assert.match(result.errors.join("\n"), /data_cutoff/);
});

test("M05 duplicate output artifact identity rejected", () => {
  const manifest = validFinal();
  const outputs = manifest.output_artifacts as Array<Record<string, unknown>>;
  outputs.push({ ...outputs[0] });
  const result = validateStageManifest(manifest, context);
  assert.equal(result.ok, false);
  if (!result.ok) assert.match(result.errors.join("\n"), /duplicate artifact identities/);
});

test("M06 wrong stage handoff gate rejected", () => {
  const manifest = validFinal();
  (manifest.handoff_gate as { name: string; state: string }).name = "READY_FOR_INTEGRATION";
  const result = validateStageManifest(manifest, context);
  assert.equal(result.ok, false);
  if (!result.ok) assert.match(result.errors.join("\n"), /handoff gate name/);
});

test("M07 gate YES with critical blocker rejected", () => {
  const manifest = validFinal();
  manifest.critical_blockers = [{ code: "BLOCK", summary: "critical" }];
  const result = validateStageManifest(manifest, context);
  assert.equal(result.ok, false);
  if (!result.ok) assert.match(result.errors.join("\n"), /critical blockers/);
});

test("M08 contract pins must reconcile to contract_set_sha256", () => {
  const manifest = validFinal();
  const pins = manifest.contract_pins as typeof contractPins;
  pins.research_stage.content_sha256 = "9".repeat(64);
  const result = validateStageManifest(manifest, context);
  assert.equal(result.ok, false);
  if (!result.ok) assert.match(result.errors.join("\n"), /contract_pins do not reconcile/);
});

test("M09 durable contract locator is schema-required", () => {
  const manifest = validFinal();
  const pins = manifest.contract_pins as typeof contractPins;
  const locator = pins.research_stage.locator as Record<string, unknown>;
  delete locator.commit_sha;
  const result = validateStageManifest(manifest, context);
  assert.equal(result.ok, false);
  if (!result.ok) assert.equal(result.stage, "schema");
});

test("M10 checkpoint cannot promote authoritative stage output", () => {
  const manifest = validFinal();
  manifest.manifest_kind = "CHECKPOINT";
  manifest.stage_status = "PAUSED";
  manifest.completed_at = null;
  (manifest.handoff_gate as { name: string; state: string }).state = "NOT_EVALUATED";
  const result = validateStageManifest(manifest, context);
  assert.equal(result.ok, false);
  if (!result.ok) assert.match(result.errors.join("\n"), /CHECKPOINT manifest cannot promote/);
});

test("M11 SHA-256 byte verifier accepts exact bytes and rejects mismatch", () => {
  const bytes = Buffer.from("orotitan", "utf8");
  const expected = "0c8f8664a9371a4f4c301498efbcc7669ea1d2237e424acda41fcd8756567605";
  assert.deepEqual(verifyArtifactBytes(bytes, expected, bytes.byteLength), { ok: true });
  const invalid = verifyArtifactBytes(bytes, "0".repeat(64), bytes.byteLength);
  assert.equal(invalid.ok, false);
});
