import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

test("v1.1 shadow replay authorization is diagnostic-only under standing authority", () => {
  const auth = JSON.parse(
    readFileSync(
      "calibration/vnext/OROTITAN_GATE18_V1_1_SHADOW_REPLAY_AUTH_001.json",
      "utf8",
    ),
  );

  assert.equal(
    auth.status,
    "AUTHORIZED_NO_INFERENCE_SHADOW_REPLAY_IMPLEMENTATION_AND_EXECUTION",
  );
  assert.equal(
    auth.authorization_source.authorization_id,
    "OROTITAN-STANDING-TECHNICAL-AUTH-001",
  );
  assert.equal(auth.authorization_source.user_reprompt_required, false);
  assert.equal(auth.scope.implement_shadow_replay_tool, true);
  assert.equal(auth.scope.execute_against_existing_private_artifacts, true);
  assert.equal(auth.scope.new_model_inference, false);
  assert.equal(auth.scope.source_artifact_mutation, false);
  assert.equal(auth.scope.historical_v1_0_result_mutation, false);
  assert.equal(auth.scope.v1_1_contract_change, false);
});

test("v1.1 shadow replay manifest includes Phi-4, Qwen3 4B failures, and one positive control", () => {
  const manifest = JSON.parse(
    readFileSync(
      "calibration/vnext/OROTITAN_GATE18_V1_1_SHADOW_REPLAY_MANIFEST_001.json",
      "utf8",
    ),
  );

  assert.equal(manifest.status, "PREPARED_NO_INFERENCE");
  assert.equal(manifest.cases.length, 6);
  assert.equal(
    manifest.cases.some(
      (item: { case_id: string }) => item.case_id === "PHI4_STMICRO_FULL_001",
    ),
    true,
  );
  assert.equal(
    manifest.cases.some(
      (item: { case_id: string }) => item.case_id === "QWEN3_4B_RATIONAL_FULL_001",
    ),
    true,
  );
  assert.equal(
    manifest.cases.some(
      (item: { case_id: string }) =>
        item.case_id === "QWEN3_4B_STMICRO_FULL_CONTROL_001",
    ),
    true,
  );
  assert.equal(manifest.boundaries.source_artifact_mutation, false);
  assert.equal(manifest.boundaries.new_inference, false);
  assert.equal(manifest.boundaries.historical_result_reclassification, false);
});

test("v1.1 shadow replay runner performs punctuation-only normalization and no model inference", () => {
  const raw = readFileSync(
    "scripts/vnext-gate18-v1-1-shadow-replay-existing-artifacts.ts",
    "utf8",
  );

  assert.match(raw, /SAFE_NARRATIVE_MAX_EXCLUSIVE = 178/);
  assert.match(raw, /terminalPunctuation/);
  assert.match(raw, /field\.set\(\`\$\{trimmed\}\.\`\)/);
  assert.match(raw, /assertGate18PhaseBV10Semantics/);
  assert.match(raw, /assertGate18V10TargetedProbeSemantics/);
  assert.match(raw, /RAW_FAIL_TO_SHADOW_SEMANTIC_PASS_AFTER_PRESENTATION_ONLY_NORMALIZATION/);
  assert.match(raw, /RAW_PRESENTATION_FAILURE_MASKED_DOWNSTREAM_SUBSTANTIVE_FAILURE/);
  assert.match(raw, /CONTROL_PASS_STABLE/);
  assert.match(raw, /historicalV10ResultsChanged: false/);
  assert.match(raw, /retroactivePassAllowed: false/);
  assert.match(raw, /sourceArtifactsMutated: false/);
  assert.match(raw, /inferenceExecuted: false/);
  assert.match(raw, /v11ContractImplemented: false/);
  assert.doesNotMatch(raw, /\/api\/generate|\/api\/chat|11434/);
  assert.doesNotMatch(raw, /fetch\(/);
});

test("contract review advances only to shadow replay, not v1.1 implementation or another model", () => {
  const review = JSON.parse(
    readFileSync(
      "calibration/vnext/OROTITAN_GATE18_V1_1_CONTRACT_ARCHITECTURE_REVIEW_PREP_001.json",
      "utf8",
    ),
  );
  const entry = JSON.parse(
    readFileSync(
      "calibration/vnext/OROTITAN_GATE18_PHASE_C_ENTRY_V0.1.json",
      "utf8",
    ),
  );

  assert.equal(
    review.status,
    "SHADOW_REPLAY_AUTHORIZED_PENDING_NO_INFERENCE",
  );
  assert.equal(review.authority.shadow_validator_implementation_authorized, true);
  assert.equal(review.authority.shadow_replay_execution_authorized, true);
  assert.equal(review.authority.v1_1_contract_change_authorized, false);
  assert.equal(entry.v1_1_shadow_replay_execution_authorized, true);
  assert.equal(entry.v1_1_contract_change_authorized, false);
  assert.equal(entry.phi4_mini_second_c4_cell_authorized, false);
  assert.equal(entry.qwen3_5_download_authorized, false);
  assert.equal(entry.model_switch_authorized, false);
  assert.equal(
    entry.next_action,
    "RUN_V1_1_SHADOW_REPLAY_ON_EXISTING_PRIVATE_ARTIFACTS_NO_INFERENCE",
  );
});
