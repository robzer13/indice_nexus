import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

test("Granite 4 Constellation result is engineering PASS with presentation carry", () => {
  const r = JSON.parse(
    readFileSync(
      "calibration/vnext/OROTITAN_GATE18_PHASE_C_C4_CONSTELLATION_GRANITE4_3B_V1_1_RESULT_001.json",
      "utf8",
    ),
  );

  assert.equal(r.status, "ENGINEERING_PASS_HUMAN_ADJUDICATION_REQUIRED");
  assert.equal(r.execution.done_reason, "stop");
  assert.equal(r.execution.eval_count, 536);
  assert.equal(r.execution.schema_valid, true);
  assert.equal(r.execution.semantic_valid, true);
  assert.equal(r.validation_v1_1.raw_presentation_compliant, false);
  assert.equal(r.validation_v1_1.normalized_path_count, 5);
  assert.equal(r.validation_v1_1.substantive_status, "PASS");
});

test("Granite 4 human adjudication is completed with critical quality failure", () => {
  const r = JSON.parse(
    readFileSync(
      "calibration/vnext/OROTITAN_GATE18_PHASE_C_C4_CONSTELLATION_GRANITE4_3B_V1_1_HUMAN_ADJUDICATION_RESULT_001.json",
      "utf8",
    ),
  );

  assert.equal(
    r.status,
    "COMPLETED_WITH_ENGINEERING_PASS_AND_HUMAN_QUALITY_CRITICAL_FAILURE",
  );
  assert.equal(r.human_quality.exact_evidence_grounding, "FAIL");
  assert.equal(r.human_quality.conflict_handling, "PASS");
  assert.equal(r.human_quality.weak_link_usefulness, "PASS");
  assert.equal(r.human_quality.unresolved_point_usefulness, "FAIL");
  assert.equal(r.human_quality.priority_selection_usefulness, "FAIL");
  assert.equal(r.critical_failures.evidence_id_invention, false);
  assert.equal(r.critical_failures.conflict_id_invention, false);
  assert.equal(r.critical_failures.exact_evidence_grounding_failure, true);
  assert.equal(r.critical_failures.priority_selection_critical_failure, true);
  assert.equal(r.disposition.critical_human_quality_failure, true);
  assert.equal(r.authority.retry_authorized, false);
  assert.equal(r.authority.new_granite_inference_authorized, false);
});

test("Granite 4 post-Constellation disposition stops expansion without global family failure", () => {
  const d = JSON.parse(
    readFileSync(
      "calibration/vnext/OROTITAN_GATE18_GRANITE4_POST_CONSTELLATION_STRATEGY_DISPOSITION_001.json",
      "utf8",
    ),
  );

  assert.equal(d.status, "STOP_GRANITE4_C4_EXPANSION_RETAIN_CALIBRATION_EVIDENCE");
  assert.equal(d.boundaries.granite4_candidate_admitted, false);
  assert.equal(d.boundaries.granite4_family_global_failure_concluded, false);
  assert.equal(d.boundaries.retry_authorized, false);
  assert.equal(d.next_action, "EXECUTE_QWEN3_8B_PINNED_DOWNLOAD_AND_IDENTITY_VERIFY");
});

test("Qwen3 8B pinned download authorization is exact, zero-cost, and non-inferential", () => {
  const a = JSON.parse(
    readFileSync(
      "calibration/vnext/OROTITAN_GATE18_PHASE_C_QWEN3_8B_DOWNLOAD_AUTH_001.json",
      "utf8",
    ),
  );

  assert.equal(a.status, "AUTHORIZED_SINGLE_DOWNLOAD_ONLY_UNCONSUMED");
  assert.equal(a.model.ollama_model_name, "qwen3:8b-q4_K_M");
  assert.equal(a.model.expected_digest_prefix, "500a1f067a9f");
  assert.equal(a.model.expected_quantization, "Q4_K_M");
  assert.equal(a.model.expected_artifact_class, "~5.2GB");
  assert.equal(a.public_verification.parameter_size, "8.19B");
  assert.equal(a.authorization_source.cost_usd, 0);
  assert.equal(a.authority.qwen3_8b_download_authorized, true);
  assert.equal(a.authority.qwen3_8b_load_smoke_authorized, false);
  assert.equal(a.authority.qwen3_8b_inference_authorized, false);
  assert.equal(a.constraints.automatic_retry, false);
});

test("Qwen3 8B download verifier pulls exact Q4_K_M tag and performs no inference", () => {
  const raw = readFileSync(
    "scripts/vnext-gate18-phase-c-qwen3-8b-download-verify.ts",
    "utf8",
  );

  assert.match(raw, /const MODEL = "qwen3:8b-q4_K_M"/);
  assert.match(raw, /const EXPECTED_DIGEST_PREFIX = "500a1f067a9f"/);
  assert.match(raw, /const EXPECTED_QUANTIZATION = "Q4_K_M"/);
  assert.match(raw, /ollama", \["pull", MODEL\]/);
  assert.match(raw, /\/api\/tags/);
  assert.match(raw, /\/api\/show/);
  assert.doesNotMatch(raw, /\/api\/generate/);
  assert.doesNotMatch(raw, /\/api\/chat/);
  assert.doesNotMatch(raw, /prompt\s*:/);
  assert.match(raw, /modelInferenceExecuted: false/);
  assert.match(raw, /loadSmokeExecuted: false/);
});
