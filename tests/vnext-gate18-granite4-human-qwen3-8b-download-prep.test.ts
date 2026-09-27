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

test("Qwen3 8B pinned download authorization is consumed after exact identity pass", () => {
  const a = JSON.parse(
    readFileSync(
      "calibration/vnext/OROTITAN_GATE18_PHASE_C_QWEN3_8B_DOWNLOAD_AUTH_001.json",
      "utf8",
    ),
  );

  assert.equal(a.status, "CONSUMED_SINGLE_DOWNLOAD_ONLY");
  assert.equal(a.model.ollama_model_name, "qwen3:8b-q4_K_M");
  assert.equal(a.model.expected_digest_prefix, "500a1f067a9f");
  assert.equal(a.model.expected_quantization, "Q4_K_M");
  assert.equal(a.model.expected_artifact_class, "~5.2GB");
  assert.equal(a.public_verification.parameter_size, "8.19B");
  assert.equal(a.authorization_source.cost_usd, 0);
  assert.equal(a.authority.qwen3_8b_download_authorized, false);
  assert.equal(a.authority.qwen3_8b_load_smoke_authorized, false);
  assert.equal(a.authority.qwen3_8b_inference_authorized, false);
  assert.equal(a.constraints.automatic_retry, false);
  assert.equal(a.consumption.consumed, true);
  assert.equal(a.consumption.result_status, "PASS_PINNED_DOWNLOAD_ONLY");
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


test("Qwen3 8B pinned download result is exact and non-inferential", () => {
  const r = JSON.parse(
    readFileSync(
      "calibration/vnext/OROTITAN_GATE18_PHASE_C_QWEN3_8B_DOWNLOAD_RESULT_001.json",
      "utf8",
    ),
  );

  assert.equal(r.status, "PASS_PINNED_DOWNLOAD_ONLY");
  assert.equal(r.model.ollama_model_name, "qwen3:8b-q4_K_M");
  assert.equal(
    r.model.digest,
    "500a1f067a9f782620b40bee6f7b0c89e17ae61f686b92c24933e4ca4b2b8b41",
  );
  assert.equal(r.model.size_bytes, 5225388164);
  assert.equal(r.model.ollama_reported_parameter_size, "8.2B");
  assert.equal(r.model.quantization, "Q4_K_M");
  assert.equal(r.safety.load_smoke_executed, false);
  assert.equal(r.safety.model_inference_executed, false);
});

test("Qwen3 8B context4096 load-only authorization is consumed after measured pass", () => {
  const a = JSON.parse(
    readFileSync(
      "calibration/vnext/OROTITAN_GATE18_PHASE_C_QWEN3_8B_CONTEXT4096_LOAD_SMOKE_AUTH_001.json",
      "utf8",
    ),
  );

  assert.equal(a.status, "CONSUMED_SINGLE_LOAD_ONLY_COMPLETE");
  assert.equal(a.planned_execution.context_tokens, 4096);
  assert.equal(a.authority.load_smoke_authorized, false);
  assert.equal(a.authority.inference_authorized, false);
  assert.equal(a.authority.authorized_run_count, 0);
  assert.equal(a.execution_result, "G18-PHASEC-QWEN3-8B-CONTEXT4096-LOAD-SMOKE-RESULT-001");
  assert.equal(a.hardware_boundary.risk_class, "HIGH_MEMORY_PRESSURE_EXPECTED");
  assert.equal(a.hardware_boundary.direct_inference_forbidden, true);
});

test("Qwen3 8B context4096 runner is exact-digest load-only", () => {
  const raw = readFileSync(
    "scripts/vnext-gate18-phase-c-qwen3-8b-context4096-load-smoke.ts",
    "utf8",
  );

  assert.match(raw, /qwen3:8b-q4_K_M/);
  assert.match(
    raw,
    /500a1f067a9f782620b40bee6f7b0c89e17ae61f686b92c24933e4ca4b2b8b41/,
  );
  assert.match(raw, /CONTEXT_TOKENS = 4096/);
  assert.match(raw, /num_ctx:\s*CONTEXT_TOKENS/);
  assert.match(raw, /AUTHORIZED_SINGLE_LOAD_ONLY_UNCONSUMED/);
  assert.match(raw, /keep_alive:\s*"2m"/);
  assert.match(raw, /keep_alive:\s*0/);
  assert.match(raw, /semanticInferenceExecuted:\s*false/);
  assert.doesNotMatch(raw, /\/api\/chat/);
  assert.doesNotMatch(raw, /prompt\s*:/);
});


test("Qwen3 8B context4096 result records critical RAM pressure and only justifies diagnostic 8192 load", () => {
  const r = JSON.parse(
    readFileSync(
      "calibration/vnext/OROTITAN_GATE18_PHASE_C_QWEN3_8B_CONTEXT4096_LOAD_SMOKE_RESULT_001.json",
      "utf8",
    ),
  );
  const a = JSON.parse(
    readFileSync(
      "calibration/vnext/OROTITAN_GATE18_PHASE_C_QWEN3_8B_CONTEXT8192_LOAD_SMOKE_AUTH_001.json",
      "utf8",
    ),
  );

  assert.equal(r.status, "PASS_LOAD_ONLY_MEASURED");
  assert.equal(r.target.context_tokens, 4096);
  assert.equal(r.measured.loaded_free_ram_gib, 0.21);
  assert.equal(r.measured.loaded_vram_used_mib, 2279);
  assert.equal(r.measured.loaded_vram_free_mib, 1684);
  assert.equal(r.measured.processor_split, "61%/39% CPU/GPU");
  assert.equal(r.interpretation.hardware_fit_at_4096, "PASS_WITH_CRITICAL_RAM_PRESSURE");
  assert.equal(r.interpretation.inference_fit_proven, false);
  assert.equal(r.interpretation.context8192_load_only_justified, true);
  assert.equal(r.interpretation.context16384_authorized, false);
  assert.equal(r.interpretation.direct_inference_authorized, false);

  assert.equal(a.status, "CONSUMED_SINGLE_LOAD_ONLY_COMPLETE");
  assert.equal(a.planned_execution.context_tokens, 8192);
  assert.equal(a.authority.load_smoke_authorized, false);
  assert.equal(a.authority.inference_authorized, false);
  assert.equal(a.authority.authorized_run_count, 0);
  assert.equal(a.execution_result, "G18-PHASEC-QWEN3-8B-CONTEXT8192-LOAD-SMOKE-RESULT-001");
  assert.equal(a.safety_rationale.risk_class, "CRITICAL_RAM_PRESSURE_DIAGNOSTIC_ONLY");
  assert.equal(a.safety_rationale.context16384_not_pre_authorized, true);
});

test("Qwen3 8B context8192 runner is exact-digest load-only", () => {
  const raw = readFileSync(
    "scripts/vnext-gate18-phase-c-qwen3-8b-context8192-load-smoke.ts",
    "utf8",
  );

  assert.match(raw, /qwen3:8b-q4_K_M/);
  assert.match(raw, /CONTEXT_TOKENS = 8192/);
  assert.match(raw, /500a1f067a9f782620b40bee6f7b0c89e17ae61f686b92c24933e4ca4b2b8b41/);
  assert.match(raw, /AUTHORIZED_SINGLE_LOAD_ONLY_UNCONSUMED/);
  assert.match(raw, /keep_alive:\s*"2m"/);
  assert.match(raw, /keep_alive:\s*0/);
  assert.doesNotMatch(raw, /\/api\/chat/);
  assert.doesNotMatch(raw, /prompt\s*:/);
});


test("Qwen3 8B context8192 result remains critically RAM-constrained but justifies final context16384 load-only", () => {
  const r = JSON.parse(
    readFileSync(
      "calibration/vnext/OROTITAN_GATE18_PHASE_C_QWEN3_8B_CONTEXT8192_LOAD_SMOKE_RESULT_001.json",
      "utf8",
    ),
  );
  const a = JSON.parse(
    readFileSync(
      "calibration/vnext/OROTITAN_GATE18_PHASE_C_QWEN3_8B_CONTEXT16384_LOAD_SMOKE_AUTH_001.json",
      "utf8",
    ),
  );

  assert.equal(r.status, "PASS_LOAD_ONLY_MEASURED");
  assert.equal(r.target.context_tokens, 8192);
  assert.equal(r.measured.loaded_free_ram_gib, 0.22);
  assert.equal(r.measured.loaded_vram_used_mib, 2323);
  assert.equal(r.measured.loaded_vram_free_mib, 1640);
  assert.equal(r.measured.processor_split, "64%/36% CPU/GPU");
  assert.equal(r.interpretation.hardware_fit_at_8192, "PASS_WITH_CRITICAL_RAM_PRESSURE");
  assert.equal(r.interpretation.inference_fit_proven, false);
  assert.equal(r.interpretation.context16384_load_only_justified, true);
  assert.equal(r.interpretation.direct_inference_authorized, false);

  assert.equal(a.status, "AUTHORIZED_SINGLE_LOAD_ONLY_UNCONSUMED");
  assert.equal(a.planned_execution.context_tokens, 16384);
  assert.equal(a.authority.load_smoke_authorized, true);
  assert.equal(a.authority.inference_authorized, false);
  assert.equal(a.authority.authorized_run_count, 1);
  assert.equal(a.safety_rationale.risk_class, "CRITICAL_RAM_PRESSURE_FINAL_DIAGNOSTIC");
  assert.equal(a.safety_rationale.no_further_context_growth, true);
});

test("Qwen3 8B context16384 runner is exact-digest load-only", () => {
  const raw = readFileSync(
    "scripts/vnext-gate18-phase-c-qwen3-8b-context16384-load-smoke.ts",
    "utf8",
  );

  assert.match(raw, /qwen3:8b-q4_K_M/);
  assert.match(raw, /CONTEXT_TOKENS = 16384/);
  assert.match(raw, /500a1f067a9f782620b40bee6f7b0c89e17ae61f686b92c24933e4ca4b2b8b41/);
  assert.match(raw, /AUTHORIZED_SINGLE_LOAD_ONLY_UNCONSUMED/);
  assert.match(raw, /keep_alive:\s*"2m"/);
  assert.match(raw, /keep_alive:\s*0/);
  assert.doesNotMatch(raw, /\/api\/chat/);
  assert.doesNotMatch(raw, /prompt\s*:/);
});
