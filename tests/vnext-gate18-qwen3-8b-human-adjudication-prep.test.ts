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
  assert.equal(a.status, "CONSUMED_SINGLE_DOWNLOAD_ONLY");
  assert.equal(a.model.ollama_model_name, "ministral-3:3b-instruct-2512-q4_K_M");
  assert.equal(a.model.expected_digest_prefix, "f04aa1c738f6");
  assert.equal(a.model.expected_quantization, "Q4_K_M");
  assert.equal(a.model.expected_artifact_class, "~3.0GB");
  assert.equal(a.public_verification.parameter_size, "3.85B");
  assert.equal(a.authority.ministral3_download_authorized, false);
  assert.equal(a.authority.ministral3_load_smoke_authorized, false);
  assert.equal(a.authority.ministral3_inference_authorized, false);
  assert.equal(a.constraints.automatic_retry, false);
  assert.equal(a.consumption.consumed, true);
  assert.equal(a.consumption.result_status, "PASS_PINNED_DOWNLOAD_ONLY");
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


test("Ministral 3 3B pinned download result captures exact local identity", () => {
  const r = JSON.parse(
    readFileSync(
      "calibration/vnext/OROTITAN_GATE18_PHASE_C_MINISTRAL3_3B_DOWNLOAD_RESULT_001.json",
      "utf8",
    ),
  );
  assert.equal(r.status, "PASS_PINNED_DOWNLOAD_ONLY");
  assert.equal(r.model.ollama_model_name, "ministral-3:3b-instruct-2512-q4_K_M");
  assert.equal(
    r.model.digest,
    "f04aa1c738f64e13c625b82ae92504fc0260fa6723b509ed1ece0fa188179b1d",
  );
  assert.equal(r.model.size_bytes, 2953840808);
  assert.equal(r.model.family, "mistral3");
  assert.equal(r.model.ollama_reported_parameter_size, "3.8B");
  assert.equal(r.model.quantization, "Q4_K_M");
  assert.equal(r.safety.load_smoke_executed, false);
  assert.equal(r.safety.model_inference_executed, false);
});

test("Ministral 3 3B context4096 load-only authorization is exact and single-use", () => {
  const a = JSON.parse(
    readFileSync(
      "calibration/vnext/OROTITAN_GATE18_PHASE_C_MINISTRAL3_3B_CONTEXT4096_LOAD_SMOKE_AUTH_001.json",
      "utf8",
    ),
  );
  assert.equal(a.status, "CONSUMED_SINGLE_LOAD_ONLY_COMPLETE");
  assert.equal(a.planned_execution.context_tokens, 4096);
  assert.equal(a.authority.load_smoke_authorized, false);
  assert.equal(a.authority.inference_authorized, false);
  assert.equal(a.authority.authorized_run_count, 0);
  assert.equal(
    a.execution_result,
    "G18-PHASEC-MINISTRAL3-3B-CONTEXT4096-LOAD-SMOKE-RESULT-001",
  );
  assert.equal(a.constraints.automatic_retry_authorized, false);
  assert.equal(a.constraints.context_change_authorized, false);
});

test("Ministral 3 3B context4096 runner is exact-digest load-only", () => {
  const raw = readFileSync(
    "scripts/vnext-gate18-phase-c-ministral3-3b-context4096-load-smoke.ts",
    "utf8",
  );
  assert.match(raw, /ministral-3:3b-instruct-2512-q4_K_M/);
  assert.match(raw, /f04aa1c738f64e13c625b82ae92504fc0260fa6723b509ed1ece0fa188179b1d/);
  assert.match(raw, /CONTEXT_TOKENS = 4096/);
  assert.match(raw, /keep_alive:\s*"2m"/);
  assert.match(raw, /keep_alive:\s*0/);
  assert.doesNotMatch(raw, /\/api\/chat/);
  assert.doesNotMatch(raw, /prompt\s*:/);
});


test("Ministral 3 3B context4096 result records critical RAM pressure and justifies only diagnostic context8192", () => {
  const r = JSON.parse(
    readFileSync(
      "calibration/vnext/OROTITAN_GATE18_PHASE_C_MINISTRAL3_3B_CONTEXT4096_LOAD_SMOKE_RESULT_001.json",
      "utf8",
    ),
  );
  const a = JSON.parse(
    readFileSync(
      "calibration/vnext/OROTITAN_GATE18_PHASE_C_MINISTRAL3_3B_CONTEXT8192_LOAD_SMOKE_AUTH_001.json",
      "utf8",
    ),
  );

  assert.equal(r.status, "PASS_LOAD_ONLY_MEASURED");
  assert.equal(r.target.context_tokens, 4096);
  assert.equal(r.measured.loaded_free_ram_gib, 0.35);
  assert.equal(r.measured.loaded_vram_used_mib, 2397);
  assert.equal(r.measured.loaded_vram_free_mib, 1566);
  assert.equal(r.measured.processor_split, "48%/52% CPU/GPU");
  assert.equal(r.interpretation.hardware_fit_at_4096, "PASS_WITH_CRITICAL_RAM_PRESSURE");
  assert.equal(r.interpretation.inference_fit_proven, false);
  assert.equal(r.interpretation.context8192_load_only_justified, true);
  assert.equal(r.interpretation.direct_inference_authorized, false);

  assert.equal(a.status, "CONSUMED_SINGLE_LOAD_ONLY_COMPLETE");
  assert.equal(a.planned_execution.context_tokens, 8192);
  assert.equal(a.authority.load_smoke_authorized, false);
  assert.equal(a.authority.inference_authorized, false);
  assert.equal(a.authority.authorized_run_count, 0);
  assert.equal(
    a.execution_result,
    "G18-PHASEC-MINISTRAL3-3B-CONTEXT8192-LOAD-SMOKE-RESULT-001",
  );
  assert.equal(a.safety_rationale.risk_class, "CRITICAL_RAM_PRESSURE_DIAGNOSTIC_ONLY");
  assert.equal(a.safety_rationale.context16384_not_pre_authorized, true);
});

test("Ministral 3 3B context8192 runner is exact-digest load-only", () => {
  const raw = readFileSync(
    "scripts/vnext-gate18-phase-c-ministral3-3b-context8192-load-smoke.ts",
    "utf8",
  );

  assert.match(raw, /ministral-3:3b-instruct-2512-q4_K_M/);
  assert.match(raw, /CONTEXT_TOKENS = 8192/);
  assert.match(raw, /f04aa1c738f64e13c625b82ae92504fc0260fa6723b509ed1ece0fa188179b1d/);
  assert.match(raw, /AUTHORIZED_SINGLE_LOAD_ONLY_UNCONSUMED/);
  assert.match(raw, /keep_alive:\s*"2m"/);
  assert.match(raw, /keep_alive:\s*0/);
  assert.doesNotMatch(raw, /\/api\/chat/);
  assert.doesNotMatch(raw, /prompt\s*:/);
});


test("Ministral 3 3B context8192 result remains critically RAM-constrained but justifies final context16384 load-only", () => {
  const r = JSON.parse(
    readFileSync(
      "calibration/vnext/OROTITAN_GATE18_PHASE_C_MINISTRAL3_3B_CONTEXT8192_LOAD_SMOKE_RESULT_001.json",
      "utf8",
    ),
  );
  const a = JSON.parse(
    readFileSync(
      "calibration/vnext/OROTITAN_GATE18_PHASE_C_MINISTRAL3_3B_CONTEXT16384_LOAD_SMOKE_AUTH_001.json",
      "utf8",
    ),
  );

  assert.equal(r.status, "PASS_LOAD_ONLY_MEASURED");
  assert.equal(r.target.context_tokens, 8192);
  assert.equal(r.measured.loaded_free_ram_gib, 0.38);
  assert.equal(r.measured.loaded_vram_used_mib, 2407);
  assert.equal(r.measured.loaded_vram_free_mib, 1556);
  assert.equal(r.measured.processor_split, "54%/46% CPU/GPU");
  assert.equal(r.interpretation.hardware_fit_at_8192, "PASS_WITH_CRITICAL_RAM_PRESSURE");
  assert.equal(r.interpretation.inference_fit_proven, false);
  assert.equal(r.interpretation.context16384_load_only_justified, true);
  assert.equal(r.interpretation.direct_inference_authorized, false);

  assert.equal(a.status, "CONSUMED_SINGLE_LOAD_ONLY_COMPLETE");
  assert.equal(a.planned_execution.context_tokens, 16384);
  assert.equal(a.authority.load_smoke_authorized, false);
  assert.equal(a.authority.inference_authorized, false);
  assert.equal(a.authority.authorized_run_count, 0);
  assert.equal(
    a.execution_result,
    "G18-PHASEC-MINISTRAL3-3B-CONTEXT16384-LOAD-SMOKE-RESULT-001",
  );
  assert.equal(a.safety_rationale.risk_class, "CRITICAL_RAM_PRESSURE_FINAL_DIAGNOSTIC");
  assert.equal(a.safety_rationale.no_further_context_growth, true);
});

test("Ministral 3 3B context16384 runner is exact-digest load-only", () => {
  const raw = readFileSync(
    "scripts/vnext-gate18-phase-c-ministral3-3b-context16384-load-smoke.ts",
    "utf8",
  );

  assert.match(raw, /ministral-3:3b-instruct-2512-q4_K_M/);
  assert.match(raw, /CONTEXT_TOKENS = 16384/);
  assert.match(raw, /f04aa1c738f64e13c625b82ae92504fc0260fa6723b509ed1ece0fa188179b1d/);
  assert.match(raw, /AUTHORIZED_SINGLE_LOAD_ONLY_UNCONSUMED/);
  assert.match(raw, /keep_alive:\s*"2m"/);
  assert.match(raw, /keep_alive:\s*0/);
  assert.doesNotMatch(raw, /\/api\/chat/);
  assert.doesNotMatch(raw, /prompt\s*:/);
});


test("Ministral 3 3B context16384 result supports one bounded same-packet C4 inference", () => {
  const r = JSON.parse(
    readFileSync(
      "calibration/vnext/OROTITAN_GATE18_PHASE_C_MINISTRAL3_3B_CONTEXT16384_LOAD_SMOKE_RESULT_001.json",
      "utf8",
    ),
  );
  const d = JSON.parse(
    readFileSync(
      "calibration/vnext/OROTITAN_GATE18_MINISTRAL3_FIRST_C4_DISCRIMINATOR_DECISION_001.json",
      "utf8",
    ),
  );
  const a = JSON.parse(
    readFileSync(
      "calibration/vnext/OROTITAN_GATE18_PHASE_C_C4_CONSTELLATION_MINISTRAL3_3B_V1_1_AUTH_001.json",
      "utf8",
    ),
  );

  assert.equal(r.status, "PASS_LOAD_ONLY_MEASURED");
  assert.equal(r.target.context_tokens, 16384);
  assert.equal(r.measured.loaded_free_ram_gib, 1.21);
  assert.equal(r.measured.loaded_vram_used_mib, 2399);
  assert.equal(r.measured.loaded_vram_free_mib, 1564);
  assert.equal(r.measured.processor_split, "63%/37% CPU/GPU");
  assert.equal(r.interpretation.hardware_fit_at_16384, "PASS_WITH_USEFUL_HEADROOM");
  assert.equal(r.interpretation.inference_fit_proven, false);
  assert.equal(r.interpretation.first_c4_constellation_same_packet_justified, true);

  assert.equal(d.status, "CONSTELLATION_SAME_PACKET_SELECTED");
  assert.equal(d.selected_cell.company, "Constellation Software");
  assert.equal(d.safeguards.pre_inference_minimum_free_ram_gib, 1.0);

  assert.equal(a.status, "CONSUMED_SINGLE_LOCAL_INFERENCE");
  assert.equal(a.c4_inference.authorized, false);
  assert.equal(a.c4_inference.context_tokens, 16384);
  assert.equal(a.c4_inference.max_output_tokens, 1024);
  assert.equal(a.c4_inference.temperature, 0);
  assert.equal(a.c4_inference.client_timeout_ms, 600000);
  assert.equal(a.c4_inference.pre_inference_minimum_free_ram_gib, 1.0);
  assert.equal(a.constraints.authorized_run_count, 0);
  assert.equal(a.constraints.automatic_retry_authorized, false);
  assert.equal(a.constraints.context_change_authorized, false);
  assert.equal(a.constraints.model_switch_authorized, false);
});

test("Ministral 3 3B bounded Constellation runner preserves exact protocol and RAM guard", () => {
  const raw = readFileSync(
    "scripts/vnext-gate18-phase-c-c4-constellation-ministral3-3b-v1-1-context16384-output1024-timeout600-loopback-guarded.ts",
    "utf8",
  );

  assert.match(raw, /ministral-3:3b-instruct-2512-q4_K_M/);
  assert.match(raw, /f04aa1c738f64e13c625b82ae92504fc0260fa6723b509ed1ece0fa188179b1d/);
  assert.match(raw, /CONTEXT_TOKENS = 16384/);
  assert.match(raw, /MAX_OUTPUT_TOKENS = 1024/);
  assert.match(raw, /CLIENT_TIMEOUT_MS = 600_000/);
  assert.match(raw, /pre_inference_minimum_free_ram_gib !== 1\.0/);
  assert.match(raw, /baselineFreeRamGiB < minimumFreeRamGiB/);
  assert.match(raw, /keep_alive: "0s"/);
  assert.match(raw, /temperature: 0/);
  assert.match(raw, /NODE_HTTP_REQUEST_LOOPBACK/);
  assert.doesNotMatch(raw, /think:\s*false/);
});


test("Ministral 3 3B Constellation engineering result is a clean pass with near-saturation carry", () => {
  const r = JSON.parse(
    readFileSync(
      "calibration/vnext/OROTITAN_GATE18_PHASE_C_C4_CONSTELLATION_MINISTRAL3_3B_V1_1_RESULT_001.json",
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
  assert.equal(r.execution.eval_count, 991);
  assert.equal(r.execution.output_token_margin, 33);
  assert.equal(r.execution.output_budget_near_saturation, true);
  assert.equal(r.interpretation.human_adjudication_required, true);
  assert.equal(r.interpretation.hardware_production_fit_established, false);
});

test("Ministral 3 3B private human-adjudication prep authorizes no new inference", () => {
  const p = JSON.parse(
    readFileSync(
      "calibration/vnext/OROTITAN_GATE18_PHASE_C_C4_CONSTELLATION_MINISTRAL3_3B_HUMAN_ADJUDICATION_PREP_001.json",
      "utf8",
    ),
  );

  assert.equal(p.status, "PRIVATE_BUNDLE_GENERATION_READY");
  assert.equal(p.authority.new_model_inference_authorized, false);
  assert.equal(p.authority.retry_authorized, false);
  assert.equal(p.privacy.public_repo_generated_content_forbidden, true);
  assert.equal(p.known_same_packet_checks.length, 6);
  assert.equal(p.engineering_carry.output_token_margin, 33);
});

test("Ministral 3 3B human-adjudication bundle builder is private and non-inferential", () => {
  const raw = readFileSync(
    "scripts/vnext-gate18-phase-c-ministral3-3b-human-adjudication-bundle.ts",
    "utf8",
  );

  assert.match(raw, /MINISTRAL3_3B_PRIVATE_HUMAN_ADJUDICATION_BUNDLE/);
  assert.match(raw, /buildVerifiedGate18V10MoatPacket/);
  assert.match(raw, /EXPECTED_PACKET_SHA256/);
  assert.match(raw, /EXPECTED_PROMPT_SHA256/);
  assert.match(raw, /2026-09-27T195934460Z__C4_CONSTELLATION_MOAT_EVIDENCE_AUDIT_MINISTRAL3_3B_V1_1_CONTEXT16384_001\.json/);
  assert.match(raw, /privateArtifact: true/);
  assert.match(raw, /publication: false/);
  assert.match(raw, /inferenceExecuted: false/);
  assert.doesNotMatch(raw, /\/api\/generate/);
  assert.doesNotMatch(raw, /\/api\/chat/);
});
