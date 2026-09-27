import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

test("Qwen3 8B Constellation engineering result is a clean engineering pass", () => {
  const r = JSON.parse(
    readFileSync(
      "calibration/vnext/OROTITAN_GATE18_PHASE_C_C4_CONSTELLATION_QWEN3_8B_V1_1_RESULT_001.json",
      "utf8",
    ),
  );
  assert.equal(r.status, "ENGINEERING_PASS_HUMAN_ADJUDICATION_REQUIRED");
  assert.equal(r.execution.runtime_error, null);
  assert.equal(r.execution.schema_valid, true);
  assert.equal(r.execution.semantic_valid, true);
  assert.equal(r.validation_v1_1.raw_presentation_compliant, true);
  assert.equal(r.validation_v1_1.normalized_path_count, 0);
  assert.equal(r.validation_v1_1.substantive_status, "PASS");
  assert.equal(r.execution.eval_count, 860);
  assert.equal(r.execution.output_token_margin, 164);
  assert.equal(r.interpretation.human_adjudication_required, true);
  assert.equal(r.interpretation.hardware_production_fit_established, false);
});

test("Qwen3 8B private human-adjudication prep does not authorize another inference", () => {
  const p = JSON.parse(
    readFileSync(
      "calibration/vnext/OROTITAN_GATE18_PHASE_C_C4_CONSTELLATION_QWEN3_8B_HUMAN_ADJUDICATION_PREP_001.json",
      "utf8",
    ),
  );
  assert.equal(p.status, "READY_PRIVATE_LOCAL_BUNDLE_GENERATION");
  assert.equal(p.authority.new_model_inference_authorized, false);
  assert.equal(p.authority.retry_authorized, false);
  assert.equal(p.privacy.public_repo_generated_content_forbidden, true);
  assert.equal(p.known_same_packet_checks.length, 6);
});

test("Qwen3 8B human-adjudication bundle builder is private and non-inferential", () => {
  const raw = readFileSync(
    "scripts/vnext-gate18-phase-c-qwen3-8b-human-adjudication-bundle.ts",
    "utf8",
  );
  assert.match(raw, /PRIVATE_HUMAN_ADJUDICATION_BUNDLE/);
  assert.match(raw, /buildVerifiedGate18V10MoatPacket/);
  assert.match(raw, /EXPECTED_PACKET_SHA256/);
  assert.match(raw, /EXPECTED_PROMPT_SHA256/);
  assert.match(raw, /privateArtifact: true/);
  assert.match(raw, /publication: false/);
  assert.match(raw, /inferenceExecuted: false/);
  assert.doesNotMatch(raw, /\/api\/generate/);
  assert.doesNotMatch(raw, /\/api\/chat/);
});
