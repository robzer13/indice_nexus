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

test("Llama 3.2 context16384 authorization is consumed after the final load-only diagnostic", () => {
  const a = JSON.parse(
    readFileSync(
      "calibration/vnext/OROTITAN_GATE18_PHASE_C_LLAMA3_2_3B_CONTEXT16384_LOAD_SMOKE_AUTH_001.json",
      "utf8",
    ),
  );

  assert.equal(a.status, "CONSUMED_SINGLE_LOAD_ONLY_COMPLETE");
  assert.equal(a.planned_execution.context_tokens, 16384);
  assert.equal(a.authority.load_smoke_authorized, false);
  assert.equal(a.authority.inference_authorized, false);
  assert.equal(a.authority.authorized_run_count, 0);
  assert.equal(a.constraints.automatic_retry_authorized, false);
  assert.equal(a.constraints.context_change_beyond_16384_authorized, false);
  assert.equal(
    a.execution_result,
    "G18-PHASEC-LLAMA3_2-3B-CONTEXT16384-LOAD-SMOKE-RESULT-001",
  );
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

test("Llama 3.2 context16384 result records measured pass with high RAM pressure", () => {
  const r = JSON.parse(
    readFileSync(
      "calibration/vnext/OROTITAN_GATE18_PHASE_C_LLAMA3_2_3B_CONTEXT16384_LOAD_SMOKE_RESULT_001.json",
      "utf8",
    ),
  );

  assert.equal(r.status, "PASS_LOAD_ONLY_MEASURED");
  assert.equal(r.target.context_tokens, 16384);
  assert.equal(r.target.quantization, "Q4_K_M");
  assert.equal(r.measured.loaded_free_ram_gib, 0.52);
  assert.equal(r.measured.loaded_vram_used_mib, 2746);
  assert.equal(r.measured.loaded_vram_free_mib, 1217);
  assert.equal(r.measured.processor_split, "46%/54% CPU/GPU");
  assert.equal(r.measured.ollama_reported_size, "4.4 GB");
  assert.equal(r.measured.load_only_confirmed, true);
  assert.equal(r.measured.explicit_unload_complete, true);
  assert.equal(r.derived.ram_snapshot_non_monotonic, true);
  assert.equal(
    r.interpretation.hardware_fit_at_16384,
    "PASS_WITH_HIGH_RAM_PRESSURE",
  );
  assert.equal(r.interpretation.context_growth_beyond_16384_authorized, false);
  assert.equal(r.interpretation.same_packet_constellation_c4_justified, true);
  assert.equal(r.interpretation.pre_inference_minimum_free_ram_gib, 1);
  assert.equal(r.safety.semantic_inference_executed, false);
});

test("Llama 3.2 first Constellation C4 authorization is single-use and fixed", () => {
  const a = JSON.parse(
    readFileSync(
      "calibration/vnext/OROTITAN_GATE18_PHASE_C_C4_CONSTELLATION_LLAMA3_2_3B_V1_1_AUTH_001.json",
      "utf8",
    ),
  );

  assert.equal(a.status, "AUTHORIZED_SINGLE_LOCAL_INFERENCE");
  assert.equal(a.c4_inference.authorized, true);
  assert.equal(a.c4_inference.company, "Constellation Software");
  assert.equal(a.c4_inference.model_name, "llama3.2:3b-instruct-q4_K_M");
  assert.equal(
    a.c4_inference.model_digest,
    "a80c4f17acd55265feec403c7aef86be0c25983ab279d83f3bcd3abbcb5b8b72",
  );
  assert.equal(a.c4_inference.context_tokens, 16384);
  assert.equal(a.c4_inference.max_output_tokens, 1024);
  assert.equal(a.c4_inference.temperature, 0);
  assert.equal(a.c4_inference.client_timeout_ms, 600000);
  assert.equal(a.c4_inference.pre_inference_minimum_free_ram_gib, 1);
  assert.equal(a.constraints.authorized_run_count, 1);
  assert.equal(a.constraints.automatic_retry_authorized, false);
  assert.equal(a.constraints.prompt_change_authorized, false);
  assert.equal(a.constraints.context_change_authorized, false);
  assert.equal(a.constraints.timeout_change_authorized, false);
  assert.equal(a.constraints.production_mutation, false);
  assert.equal(a.output_policy.public_repo_generated_content_forbidden, true);
  assert.equal(a.output_policy.human_adjudication_required_if_engineering_pass, true);
});

test("Llama 3.2 first Constellation C4 prep and guarded runner pin the same packet and memory guard", () => {
  const p = JSON.parse(
    readFileSync(
      "calibration/vnext/OROTITAN_GATE18_PHASE_C_C4_CONSTELLATION_LLAMA3_2_3B_V1_1_PREP_001.json",
      "utf8",
    ),
  );
  const raw = readFileSync(
    "scripts/vnext-gate18-phase-c-c4-constellation-llama3-2-3b-v1-1-context16384-output1024-timeout600-loopback-guarded.ts",
    "utf8",
  );

  assert.equal(p.status, "AUTHORIZED_READY_TO_EXECUTE");
  assert.equal(p.cell.company, "Constellation Software");
  assert.equal(
    p.cell.packet_sha256,
    "9a47bcf15d0c90da55349cbb3fbb2da8e9645b859a535dc8909501e89a20b6d8",
  );
  assert.equal(
    p.cell.prompt_sha256,
    "0891fa34d02a47c83e8342d5da5e653a3566d90f1c63c1bfc662f4e6097c10b8",
  );
  assert.equal(p.parameters.context_tokens, 16384);
  assert.equal(p.parameters.max_output_tokens, 1024);
  assert.equal(p.parameters.temperature, 0);
  assert.equal(p.parameters.client_timeout_ms, 600000);
  assert.equal(p.parameters.pre_inference_minimum_free_ram_gib, 1);
  assert.equal(p.runtime_basis.context16384_load_fit, "PASS_WITH_HIGH_RAM_PRESSURE");

  assert.match(raw, /llama3\.2:3b-instruct-q4_K_M/);
  assert.match(raw, /const CONTEXT_TOKENS = 16384;/);
  assert.match(raw, /const MAX_OUTPUT_TOKENS = 1024;/);
  assert.match(raw, /const CLIENT_TIMEOUT_MS = 600_000;/);
  assert.match(raw, /LLAMA3_2_3B_V1_1_AUTH_001\.json/);
  assert.match(raw, /pre_inference_minimum_free_ram_gib/);
  assert.match(raw, /INSUFFICIENT_BASELINE_FREE_RAM/);
  assert.match(raw, /NODE_HTTP_REQUEST_LOOPBACK/);
  assert.match(raw, /private-runs/);
  assert.doesNotMatch(raw, /think:\s*false/);
});

test("Llama 3.2 first C4 baseline RAM block preserves the single inference authorization", () => {
  const b = JSON.parse(
    readFileSync(
      "calibration/vnext/OROTITAN_GATE18_PHASE_C_C4_CONSTELLATION_LLAMA3_2_3B_BASELINE_RAM_PRECONDITION_BLOCK_001.json",
      "utf8",
    ),
  );
  const a = JSON.parse(
    readFileSync(
      "calibration/vnext/OROTITAN_GATE18_PHASE_C_C4_CONSTELLATION_LLAMA3_2_3B_V1_1_AUTH_001.json",
      "utf8",
    ),
  );

  assert.equal(
    b.status,
    "BLOCKED_BEFORE_INFERENCE_INSUFFICIENT_BASELINE_FREE_RAM",
  );
  assert.equal(b.precondition.required_baseline_free_ram_gib, 1);
  assert.equal(b.precondition.observed_baseline_free_ram_gib, 0.75);
  assert.equal(b.execution_boundary.provider_generate_request_reached, false);
  assert.equal(b.execution_boundary.semantic_inference_executed, false);
  assert.equal(b.authorization_consumption.consumed, false);
  assert.equal(b.authorization_consumption.authorized_run_count_remaining, 1);

  assert.equal(a.status, "AUTHORIZED_SINGLE_LOCAL_INFERENCE");
  assert.equal(a.c4_inference.authorized, true);
  assert.equal(a.constraints.authorized_run_count, 1);
  assert.equal(
    a.precondition_block,
    "G18-PHASEC-C4-CONSTELLATION-LLAMA3_2-3B-BASELINE-RAM-PRECONDITION-BLOCK-001",
  );
});

test("Llama 3.2 second C4 RAM block preserves the same single inference authorization", () => {
  const b = JSON.parse(
    readFileSync(
      "calibration/vnext/OROTITAN_GATE18_PHASE_C_C4_CONSTELLATION_LLAMA3_2_3B_BASELINE_RAM_PRECONDITION_BLOCK_002.json",
      "utf8",
    ),
  );
  const a = JSON.parse(
    readFileSync(
      "calibration/vnext/OROTITAN_GATE18_PHASE_C_C4_CONSTELLATION_LLAMA3_2_3B_V1_1_AUTH_001.json",
      "utf8",
    ),
  );

  assert.equal(
    b.status,
    "BLOCKED_BEFORE_INFERENCE_INSUFFICIENT_BASELINE_FREE_RAM",
  );
  assert.equal(b.observed_external_prelaunch_free_ram_gib, 1.09);
  assert.equal(b.precondition.observed_runner_baseline_free_ram_gib, 0.7);
  assert.equal(b.precondition.external_to_runner_free_ram_delta_gib, -0.39);
  assert.equal(b.execution_boundary.provider_generate_request_reached, false);
  assert.equal(b.execution_boundary.semantic_inference_executed, false);
  assert.equal(b.authorization_consumption.consumed, false);
  assert.equal(b.authorization_consumption.authorized_run_count_remaining, 1);
  assert.equal(b.operational_interpretation.fixed_protocol_threshold_changed, false);
  assert.equal(b.operational_interpretation.prelaunch_headroom_target_recommended_gib, 1.5);

  assert.equal(a.status, "AUTHORIZED_SINGLE_LOCAL_INFERENCE");
  assert.equal(a.c4_inference.authorized, true);
  assert.equal(a.constraints.authorized_run_count, 1);
  assert.equal(
    a.latest_precondition_block,
    "G18-PHASEC-C4-CONSTELLATION-LLAMA3_2-3B-BASELINE-RAM-PRECONDITION-BLOCK-002",
  );
});

test("Llama 3.2 Ollama ps precondition block preserves authorization and the runner uses loopback api ps", () => {
  const b = JSON.parse(
    readFileSync(
      "calibration/vnext/OROTITAN_GATE18_PHASE_C_C4_CONSTELLATION_LLAMA3_2_3B_OLLAMA_PS_PRECONDITION_BLOCK_001.json",
      "utf8",
    ),
  );
  const a = JSON.parse(
    readFileSync(
      "calibration/vnext/OROTITAN_GATE18_PHASE_C_C4_CONSTELLATION_LLAMA3_2_3B_V1_1_AUTH_001.json",
      "utf8",
    ),
  );
  const raw = readFileSync(
    "scripts/vnext-gate18-phase-c-c4-constellation-llama3-2-3b-v1-1-context16384-output1024-timeout600-loopback-guarded.ts",
    "utf8",
  );

  assert.equal(b.status, "BLOCKED_BEFORE_INFERENCE_OLLAMA_PS_CHECK_FAILED");
  assert.equal(b.execution_boundary.provider_generate_request_reached, false);
  assert.equal(b.execution_boundary.semantic_inference_executed, false);
  assert.equal(b.authorization_consumption.consumed, false);
  assert.equal(b.authorization_consumption.authorized_run_count_remaining, 1);

  assert.equal(a.status, "AUTHORIZED_SINGLE_LOCAL_INFERENCE");
  assert.equal(a.c4_inference.authorized, true);
  assert.equal(a.constraints.authorized_run_count, 1);
  assert.equal(
    a.latest_precondition_block,
    "G18-PHASEC-C4-CONSTELLATION-LLAMA3_2-3B-OLLAMA-PS-PRECONDITION-BLOCK-001",
  );

  assert.equal(raw.includes('"/api/ps"'), true);
  assert.match(raw, /await assertNoLoadedModels\(\);/);
  assert.doesNotMatch(raw, /execFileSync\("ollama", \["ps"\]/);
});

