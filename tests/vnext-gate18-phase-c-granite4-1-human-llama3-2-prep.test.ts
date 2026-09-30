import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

test("Granite 4.1 human adjudication preserves engineering pass but stops expansion", () => {
  const r = JSON.parse(
    readFileSync(
      "calibration/vnext/OROTITAN_GATE18_PHASE_C_C4_CONSTELLATION_GRANITE4_1_3B_V1_1_HUMAN_ADJUDICATION_RESULT_001.json",
      "utf8",
    ),
  );

  assert.equal(
    r.status,
    "COMPLETED_WITH_ENGINEERING_PASS_AND_HUMAN_QUALITY_CRITICAL_FAILURE",
  );
  assert.equal(r.engineering_disposition.runtime_pass, true);
  assert.equal(r.engineering_disposition.schema_valid, true);
  assert.equal(r.engineering_disposition.v1_1_raw_presentation_compliant, true);
  assert.equal(r.engineering_disposition.v1_1_safe_normalization_count, 0);
  assert.equal(r.engineering_disposition.output_token_margin, 589);
  assert.equal(r.human_quality.unresolved_point_usefulness, "FAIL");
  assert.equal(r.human_quality.priority_selection_usefulness, "FAIL");
  assert.equal(r.critical_failures.duplicate_priority_finding_critical_failure, true);
  assert.equal(r.critical_failures.unresolved_point_usefulness_critical_failure, true);
  assert.equal(r.critical_failures.priority_selection_critical_failure, true);
  assert.equal(r.authority.retry_authorized, false);
  assert.equal(r.authority.new_granite4_1_inference_authorized, false);
});

test("Granite 4.1 post-Constellation disposition activates Llama 3.2", () => {
  const d = JSON.parse(
    readFileSync(
      "calibration/vnext/OROTITAN_GATE18_PHASE_C_GRANITE4_1_POST_CONSTELLATION_DISPOSITION_001.json",
      "utf8",
    ),
  );

  assert.equal(
    d.status,
    "STOP_GRANITE4_1_C4_EXPANSION_RETAIN_CALIBRATION_EVIDENCE",
  );
  assert.equal(d.interpretation.granite4_1_candidate_admitted, false);
  assert.equal(d.interpretation.retry_authorized, false);
  assert.equal(d.next_candidate.candidate_id, "LLAMA3_2_3B_OLLAMA_Q4_K_M");
  assert.equal(d.next_candidate.model_id, "llama3.2:3b-instruct-q4_K_M");
});

test("Llama 3.2 download authorization is consumed after exact pinned download", () => {
  const a = JSON.parse(
    readFileSync(
      "calibration/vnext/OROTITAN_GATE18_PHASE_C_LLAMA3_2_3B_DOWNLOAD_AUTH_001.json",
      "utf8",
    ),
  );

  assert.equal(a.status, "CONSUMED_SINGLE_DOWNLOAD_ONLY");
  assert.equal(a.model.ollama_model_name, "llama3.2:3b-instruct-q4_K_M");
  assert.equal(a.model.expected_digest_prefix, "a80c4f17acd5");
  assert.equal(a.model.expected_quantization, "Q4_K_M");
  assert.equal(a.execution.action_count_authorized, 1);
  assert.equal(a.authority.llama3_2_download_authorized, false);
  assert.equal(a.authority.llama3_2_load_smoke_authorized, false);
  assert.equal(a.authority.llama3_2_inference_authorized, false);
  assert.equal(a.constraints.automatic_retry, false);
  assert.equal(a.constraints.automatic_model_switch, false);
});

test("Llama 3.2 download verifier contains no load or inference path", () => {
  const raw = readFileSync(
    "scripts/vnext-gate18-phase-c-llama3-2-3b-download-verify.ts",
    "utf8",
  );

  assert.match(raw, /const MODEL = "llama3\.2:3b-instruct-q4_K_M";/);
  assert.match(raw, /const EXPECTED_DIGEST_PREFIX = "a80c4f17acd5";/);
  assert.match(raw, /ollama", \["pull", MODEL\]/);
  assert.doesNotMatch(raw, /\/api\/generate|\/api\/chat/);
  assert.match(raw, /modelInferenceExecuted: false/);
  assert.match(raw, /loadSmokeExecuted: false/);
});


test("Llama 3.2 pinned download result records exact local identity", () => {
  const r = JSON.parse(
    readFileSync(
      "calibration/vnext/OROTITAN_GATE18_PHASE_C_LLAMA3_2_3B_DOWNLOAD_RESULT_001.json",
      "utf8",
    ),
  );

  assert.equal(r.status, "PASS_PINNED_DOWNLOAD_ONLY");
  assert.equal(r.model.ollama_model_name, "llama3.2:3b-instruct-q4_K_M");
  assert.equal(
    r.model.digest,
    "a80c4f17acd55265feec403c7aef86be0c25983ab279d83f3bcd3abbcb5b8b72",
  );
  assert.equal(r.model.size_bytes, 2019393189);
  assert.equal(r.model.family, "llama");
  assert.equal(r.model.ollama_reported_parameter_size, "3.2B");
  assert.equal(r.model.quantization, "Q4_K_M");
  assert.equal(r.verification.expected_digest_prefix_matched, true);
  assert.equal(r.verification.expected_quantization_matched, true);
  assert.equal(r.safety.load_smoke_executed, false);
  assert.equal(r.safety.model_inference_executed, false);
});

test("Llama 3.2 context4096 load-only authorization is single-use and non-inferential", () => {
  const a = JSON.parse(
    readFileSync(
      "calibration/vnext/OROTITAN_GATE18_PHASE_C_LLAMA3_2_3B_LOAD_SMOKE_AUTH_001.json",
      "utf8",
    ),
  );

  assert.equal(a.status, "CONSUMED_SINGLE_LOAD_ONLY_COMPLETE");
  assert.equal(a.planned_execution.context_tokens, 4096);
  assert.equal(a.authority.load_smoke_authorized, false);
  assert.equal(a.authority.inference_authorized, false);
  assert.equal(a.authority.download_authorized, false);
  assert.equal(a.authority.authorized_run_count, 0);
  assert.equal(a.constraints.automatic_retry_authorized, false);
  assert.equal(a.constraints.context_change_authorized, false);
  assert.equal(
    a.execution_result,
    "G18-PHASEC-LLAMA3_2-3B-CONTEXT4096-LOAD-SMOKE-RESULT-001",
  );
});

test("Llama 3.2 context4096 protocol contains no prompt or semantic inference", () => {
  const p = JSON.parse(
    readFileSync(
      "calibration/vnext/OROTITAN_GATE18_PHASE_C_LLAMA3_2_3B_CONTEXT4096_LOAD_ONLY_PROTOCOL_001.json",
      "utf8",
    ),
  );

  assert.equal(p.status, "AUTHORIZED_READY_FOR_MANUAL_POWERSHELL_EXECUTION");
  assert.equal(p.target.context_tokens, 4096);
  assert.equal(p.target.model, "llama3.2:3b-instruct-q4_K_M");
  assert.equal(
    p.target.digest,
    "a80c4f17acd55265feec403c7aef86be0c25983ab279d83f3bcd3abbcb5b8b72",
  );
  assert.equal(p.execution_contract.load_request.prompt_field_present, false);
  assert.equal(p.execution_contract.unload_request.prompt_field_present, false);
  assert.equal(p.authority.semantic_inference_authorized, false);
  assert.equal(p.authority.retry_authorized, false);
  assert.equal(p.authority.context_change_authorized, false);
});


test("Llama 3.2 runtime precondition block preserves historical non-consumption before the later successful run", () => {
  const r = JSON.parse(
    readFileSync(
      "calibration/vnext/OROTITAN_GATE18_PHASE_C_LLAMA3_2_3B_CONTEXT4096_PRECONDITION_BLOCK_001.json",
      "utf8",
    ),
  );
  const a = JSON.parse(
    readFileSync(
      "calibration/vnext/OROTITAN_GATE18_PHASE_C_LLAMA3_2_3B_LOAD_SMOKE_AUTH_001.json",
      "utf8",
    ),
  );

  assert.equal(r.status, "BLOCKED_PRE_EXECUTION_OLLAMA_RUNTIME_UNREACHABLE");
  assert.equal(r.execution_boundary.model_load_attempted, false);
  assert.equal(r.execution_boundary.prompt_provided, false);
  assert.equal(r.execution_boundary.semantic_inference_executed, false);
  assert.equal(r.authorization_consumption.consumed, false);
  assert.equal(r.authorization_consumption.authorized_run_count_remaining, 1);
  assert.equal(a.status, "CONSUMED_SINGLE_LOAD_ONLY_COMPLETE");
  assert.equal(a.authority.load_smoke_authorized, false);
  assert.equal(a.authority.authorized_run_count, 0);
  assert.equal(a.authority.inference_authorized, false);
  assert.equal(
    a.execution_result,
    "G18-PHASEC-LLAMA3_2-3B-CONTEXT4096-LOAD-SMOKE-RESULT-001",
  );
});

test("Llama 3.2 context4096 result records measured pass with critical RAM pressure", () => {
  const r = JSON.parse(
    readFileSync(
      "calibration/vnext/OROTITAN_GATE18_PHASE_C_LLAMA3_2_3B_CONTEXT4096_LOAD_SMOKE_RESULT_001.json",
      "utf8",
    ),
  );

  assert.equal(r.status, "PASS_LOAD_ONLY_MEASURED");
  assert.equal(r.target.context_tokens, 4096);
  assert.equal(r.target.quantization, "Q4_K_M");
  assert.equal(r.measured.loaded_free_ram_gib, 0.34);
  assert.equal(r.measured.loaded_vram_used_mib, 2683);
  assert.equal(r.measured.loaded_vram_free_mib, 1280);
  assert.equal(r.measured.processor_split, "20%/80% CPU/GPU");
  assert.equal(r.measured.load_only_confirmed, true);
  assert.equal(r.measured.explicit_unload_complete, true);
  assert.equal(r.derived.ram_snapshot_non_monotonic, true);
  assert.equal(
    r.interpretation.hardware_fit_at_4096,
    "PASS_WITH_CRITICAL_RAM_PRESSURE",
  );
  assert.equal(r.interpretation.context8192_load_only_justified, true);
  assert.equal(r.interpretation.context16384_authorized, false);
  assert.equal(r.safety.semantic_inference_executed, false);
});

test("Llama 3.2 context8192 authorization is consumed after the measured diagnostic", () => {
  const a = JSON.parse(
    readFileSync(
      "calibration/vnext/OROTITAN_GATE18_PHASE_C_LLAMA3_2_3B_CONTEXT8192_LOAD_SMOKE_AUTH_001.json",
      "utf8",
    ),
  );

  assert.equal(a.status, "CONSUMED_SINGLE_LOAD_ONLY_COMPLETE");
  assert.equal(a.planned_execution.context_tokens, 8192);
  assert.equal(a.authority.load_smoke_authorized, false);
  assert.equal(a.authority.inference_authorized, false);
  assert.equal(a.authority.authorized_run_count, 0);
  assert.equal(a.constraints.automatic_retry_authorized, false);
  assert.equal(a.constraints.context_change_beyond_8192_authorized, false);
  assert.equal(
    a.execution_result,
    "G18-PHASEC-LLAMA3_2-3B-CONTEXT8192-LOAD-SMOKE-RESULT-001",
  );
});

test("Llama 3.2 context8192 protocol and runner preserve load-only boundaries", () => {
  const p = JSON.parse(
    readFileSync(
      "calibration/vnext/OROTITAN_GATE18_PHASE_C_LLAMA3_2_3B_CONTEXT8192_LOAD_ONLY_PROTOCOL_001.json",
      "utf8",
    ),
  );
  const raw = readFileSync(
    "scripts/vnext-gate18-phase-c-llama3-2-3b-context8192-load-smoke.ts",
    "utf8",
  );

  assert.equal(p.target.context_tokens, 8192);
  assert.equal(p.execution_contract.load_request.prompt_field_present, false);
  assert.equal(p.execution_contract.unload_request.prompt_field_present, false);
  assert.equal(p.authority.semantic_inference_authorized, false);
  assert.equal(p.authority.retry_authorized, false);
  assert.equal(p.authority.context_change_beyond_8192_authorized, false);

  assert.match(raw, /const CONTEXT_TOKENS = 8192;/);
  assert.match(
    raw,
    /OROTITAN_GATE18_PHASE_C_LLAMA3_2_3B_CONTEXT8192_LOAD_SMOKE_AUTH_001\.json/,
  );
  assert.doesNotMatch(
    raw,
    /readFile\(\s*"calibration\/vnext\/OROTITAN_GATE18_PHASE_C_LLAMA3_2_3B_LOAD_SMOKE_AUTH_001\.json"/,
  );

  assert.match(raw, /a80c4f17acd55265feec403c7aef86be0c25983ab279d83f3bcd3abbcb5b8b72/);
  assert.match(raw, /OTHER_OLLAMA_MODEL_ALREADY_LOADED/);
  assert.doesNotMatch(raw, /prompt\s*:/);
  assert.match(raw, /semanticInferenceExecuted: false/);
  assert.match(raw, /retryExecuted: false/);
});

test("Llama 3.2 context8192 result records measured pass with critical RAM pressure", () => {
  const r = JSON.parse(
    readFileSync(
      "calibration/vnext/OROTITAN_GATE18_PHASE_C_LLAMA3_2_3B_CONTEXT8192_LOAD_SMOKE_RESULT_001.json",
      "utf8",
    ),
  );

  assert.equal(r.status, "PASS_LOAD_ONLY_MEASURED");
  assert.equal(r.target.context_tokens, 8192);
  assert.equal(r.target.quantization, "Q4_K_M");
  assert.equal(r.measured.loaded_free_ram_gib, 0.47);
  assert.equal(r.measured.loaded_vram_used_mib, 2751);
  assert.equal(r.measured.loaded_vram_free_mib, 1212);
  assert.equal(r.measured.processor_split, "32%/68% CPU/GPU");
  assert.equal(r.measured.load_only_confirmed, true);
  assert.equal(r.measured.explicit_unload_complete, true);
  assert.equal(r.derived.ram_snapshot_non_monotonic, true);
  assert.equal(
    r.interpretation.hardware_fit_at_8192,
    "PASS_WITH_CRITICAL_RAM_PRESSURE",
  );
  assert.equal(r.interpretation.context16384_load_only_justified, true);
  assert.equal(r.safety.semantic_inference_executed, false);
});

test("Llama 3.2 context16384 authorization is one final load-only diagnostic", () => {
  const a = JSON.parse(
    readFileSync(
      "calibration/vnext/OROTITAN_GATE18_PHASE_C_LLAMA3_2_3B_CONTEXT16384_LOAD_SMOKE_AUTH_001.json",
      "utf8",
    ),
  );

  assert.equal(a.status, "AUTHORIZED_SINGLE_LOAD_ONLY_UNCONSUMED");
  assert.equal(a.planned_execution.context_tokens, 16384);
  assert.equal(a.authority.load_smoke_authorized, true);
  assert.equal(a.authority.inference_authorized, false);
  assert.equal(a.authority.authorized_run_count, 1);
  assert.equal(a.constraints.automatic_retry_authorized, false);
  assert.equal(a.constraints.context_change_beyond_16384_authorized, false);
  assert.equal(
    a.safety_rationale.risk_class,
    "CRITICAL_RAM_PRESSURE_FINAL_DIAGNOSTIC",
  );
  assert.equal(a.safety_rationale.no_further_context_growth, true);
});

test("Llama 3.2 context16384 protocol and runner preserve final load-only boundaries", () => {
  const p = JSON.parse(
    readFileSync(
      "calibration/vnext/OROTITAN_GATE18_PHASE_C_LLAMA3_2_3B_CONTEXT16384_LOAD_ONLY_PROTOCOL_001.json",
      "utf8",
    ),
  );
  const raw = readFileSync(
    "scripts/vnext-gate18-phase-c-llama3-2-3b-context16384-load-smoke.ts",
    "utf8",
  );

  assert.equal(p.target.context_tokens, 16384);
  assert.equal(p.execution_contract.load_request.prompt_field_present, false);
  assert.equal(p.execution_contract.unload_request.prompt_field_present, false);
  assert.equal(p.authority.semantic_inference_authorized, false);
  assert.equal(p.authority.retry_authorized, false);
  assert.equal(p.authority.context_change_beyond_16384_authorized, false);

  assert.match(raw, /const CONTEXT_TOKENS = 16384;/);
  assert.match(
    raw,
    /OROTITAN_GATE18_PHASE_C_LLAMA3_2_3B_CONTEXT16384_LOAD_SMOKE_AUTH_001\.json/,
  );
  assert.match(
    raw,
    /OROTITAN_GATE18_PHASE_C_LLAMA3_2_3B_CONTEXT16384_LOAD_ONLY_PROTOCOL_001\.json/,
  );
  assert.doesNotMatch(raw, /prompt\s*:/);
  assert.match(raw, /OTHER_OLLAMA_MODEL_ALREADY_LOADED/);
  assert.match(raw, /semanticInferenceExecuted: false/);
  assert.match(raw, /retryExecuted: false/);
});

