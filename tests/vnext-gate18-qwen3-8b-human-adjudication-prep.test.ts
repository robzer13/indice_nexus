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
  assert.equal(p.status, "COMPLETED_PRIVATE_BUNDLE_GENERATED_AND_REVIEWED");
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


test("Qwen3 8B human adjudication is a critical quality failure despite clean engineering", () => {
  const r = JSON.parse(
    readFileSync(
      "calibration/vnext/OROTITAN_GATE18_PHASE_C_C4_CONSTELLATION_QWEN3_8B_V1_1_HUMAN_ADJUDICATION_RESULT_001.json",
      "utf8",
    ),
  );
  assert.equal(
    r.status,
    "COMPLETED_WITH_ENGINEERING_PASS_AND_HUMAN_QUALITY_CRITICAL_FAILURE",
  );
  assert.equal(r.engineering_disposition.disposition, "CLEAN_ENGINEERING_PASS");
  assert.equal(r.human_quality.exact_evidence_grounding, "FAIL");
  assert.equal(r.human_quality.conflict_handling, "PASS");
  assert.equal(r.human_quality.unresolved_point_usefulness, "FAIL");
  assert.equal(r.human_quality.priority_selection_usefulness, "FAIL");
  assert.equal(r.critical_failures.exact_evidence_grounding_failure, true);
  assert.equal(r.critical_failures.priority_selection_critical_failure, true);
  assert.equal(r.disposition.critical_human_quality_failure, true);
  assert.equal(r.authority.retry_authorized, false);
  assert.equal(r.authority.new_qwen3_8b_inference_authorized, false);
});

test("Qwen3 8B post-Constellation disposition stops expansion without global family failure", () => {
  const d = JSON.parse(
    readFileSync(
      "calibration/vnext/OROTITAN_GATE18_QWEN3_8B_POST_CONSTELLATION_STRATEGY_DISPOSITION_001.json",
      "utf8",
    ),
  );
  assert.equal(d.status, "STOP_QWEN3_8B_C4_EXPANSION_RETAIN_CALIBRATION_EVIDENCE");
  assert.equal(d.boundaries.qwen3_8b_candidate_admitted, false);
  assert.equal(d.boundaries.qwen3_family_global_failure_concluded, false);
  assert.equal(d.boundaries.retry_authorized, false);
  assert.equal(d.next_candidate, "MINISTRAL3_3B_INSTRUCT_2512_Q4_K_M");
  assert.equal(d.next_action, "EXECUTE_MINISTRAL3_3B_PINNED_DOWNLOAD_AND_IDENTITY_VERIFY");
});

test("Ministral 3 3B download authorization is exact and non-inferential", () => {
  const a = JSON.parse(
    readFileSync(
      "calibration/vnext/OROTITAN_GATE18_PHASE_C_MINISTRAL3_3B_DOWNLOAD_AUTH_001.json",
      "utf8",
    ),
  );
  assert.equal(a.status, "AUTHORIZED_SINGLE_DOWNLOAD_ONLY_UNCONSUMED");
  assert.equal(a.model.ollama_model_name, "ministral-3:3b-instruct-2512-q4_K_M");
  assert.equal(a.model.expected_digest_prefix, "f04aa1c738f6");
  assert.equal(a.model.expected_quantization, "Q4_K_M");
  assert.equal(a.model.expected_artifact_class, "~3.0GB");
  assert.equal(a.public_verification.parameter_size, "3.85B");
  assert.equal(a.authority.ministral3_download_authorized, true);
  assert.equal(a.authority.ministral3_load_smoke_authorized, false);
  assert.equal(a.authority.ministral3_inference_authorized, false);
  assert.equal(a.constraints.automatic_retry, false);
});

test("Ministral 3 3B download verifier pulls exact tag and performs no inference", () => {
  const raw = readFileSync(
    "scripts/vnext-gate18-phase-c-ministral3-3b-download-verify.ts",
    "utf8",
  );
  assert.match(raw, /const MODEL = "ministral-3:3b-instruct-2512-q4_K_M"/);
  assert.match(raw, /const EXPECTED_DIGEST_PREFIX = "f04aa1c738f6"/);
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
