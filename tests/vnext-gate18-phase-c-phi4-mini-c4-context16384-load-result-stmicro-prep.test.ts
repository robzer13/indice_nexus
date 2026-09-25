import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

test("Phi-4 context16384 load result records measured hardware fit and complete unload", () => {
  const result = JSON.parse(
    readFileSync(
      "calibration/vnext/OROTITAN_GATE18_PHASE_C_C4_PHI4_MINI_CONTEXT16384_LOAD_SMOKE_RESULT_001.json",
      "utf8",
    ),
  );

  assert.equal(result.status, "PASS_LOAD_ONLY_MEASURED");
  assert.equal(result.target.model, "phi4-mini:3.8b-q4_K_M");
  assert.equal(result.target.context_tokens, 16384);
  assert.equal(result.loaded.vram_used_mib, 2283);
  assert.equal(result.loaded.vram_free_mib, 1680);
  assert.equal(result.loaded.free_ram_gib, 0.84);
  assert.equal(result.loaded.processor_split, "56%/44% CPU/GPU");
  assert.equal(result.after.vram_used_mib, 0);
  assert.equal(result.after.ollama_loaded_model_count, 0);
  assert.equal(result.interpretation.context16384_load_fit, "PASS");
  assert.equal(result.interpretation.context16384_inference_fit, "NOT_YET_PROVEN");
  assert.equal(result.interpretation.system_ram_pressure, "HIGH");
  assert.equal(result.interpretation.first_c4_cell_selected, "STMicroelectronics");
});

test("First Phi-4 C4 cell prep selects the smallest full packet and a bounded 1024-token output budget", () => {
  const prep = JSON.parse(
    readFileSync(
      "calibration/vnext/OROTITAN_GATE18_PHASE_C_C4_STMICRO_PHI4_MINI_CONTEXT16384_INFERENCE_PREP_001.json",
      "utf8",
    ),
  );

  assert.equal(prep.status, "EXECUTED_SEMANTIC_FAIL_AUTHORIZATION_CONSUMED_FORENSICS_REQUIRED");
  assert.equal(prep.matrix_cell.company, "STMicroelectronics");
  assert.equal(prep.matrix_cell.packet_mode, "FULL_PINNED_PACKET");
  assert.equal(prep.matrix_cell.evidence_count, 9);
  assert.equal(prep.matrix_cell.conflict_count, 2);
  assert.equal(prep.matrix_cell.prompt_bytes, 7558);
  assert.equal(
    prep.matrix_cell.prompt_sha256,
    "3f06d6e22dbbc88abbbc64fd470811f438a595c71154f7fcb8c004a6d4f5bc91",
  );
  assert.equal(prep.model.context_tokens, 16384);
  assert.equal(prep.model.max_output_tokens, 1024);
  assert.equal(prep.model.temperature, 0);
  assert.equal(prep.model.client_timeout_ms, 420000);
  assert.equal(prep.model.transport, "NODE_HTTP_REQUEST_LOOPBACK");
  assert.equal(prep.model.sleep_guard_required, true);
  assert.equal(prep.rationale.first_c4_cell_selection, "SMALLEST_FULL_PACKET_BY_STATIC_PREFLIGHT");
  assert.equal(prep.constraints.automatic_retry_authorized, false);
  assert.equal(prep.evaluation.auto_repair_forbidden, true);
});

test("First Phi-4 C4 STMicro authorization is one-run, loopback-only, and derived from standing authority", () => {
  const auth = JSON.parse(
    readFileSync(
      "calibration/vnext/OROTITAN_GATE18_PHASE_C_C4_STMICRO_PHI4_MINI_CONTEXT16384_INFERENCE_AUTH_001.json",
      "utf8",
    ),
  );

  assert.equal(auth.status, "CONSUMED_SINGLE_LOCAL_INFERENCE");
  assert.equal(auth.authorization_source.type, "STANDING_USER_AUTHORIZATION");
  assert.equal(auth.authorization_source.authorization_id, "OROTITAN-STANDING-TECHNICAL-AUTH-001");
  assert.equal(auth.authorization_source.separate_user_reprompt_required, false);
  assert.equal(auth.c4_inference.authorized, false);
  assert.equal(auth.c4_inference.company, "STMicroelectronics");
  assert.equal(auth.c4_inference.model_name, "phi4-mini:3.8b-q4_K_M");
  assert.equal(
    auth.c4_inference.model_digest,
    "78fad5d182a7c33065e153a5f8ba210754207ba9d91973f57dffa7f487363753",
  );
  assert.equal(auth.c4_inference.context_tokens, 16384);
  assert.equal(auth.c4_inference.max_output_tokens, 1024);
  assert.equal(auth.c4_inference.client_timeout_ms, 420000);
  assert.equal(auth.c4_inference.transport, "NODE_HTTP_REQUEST_LOOPBACK");
  assert.equal(auth.constraints.authorized_run_count, 0);
  assert.equal(auth.constraints.automatic_retry_authorized, false);
  assert.equal(auth.constraints.context_change_authorized, false);
  assert.equal(auth.constraints.production_mutation, false);
  assert.equal(auth.constraints.publication_authority, false);
});

test("Phi-4 STMicro C4 runner is identity-pinned, sleep-guarded, loopback, private-output, and human-adjudication bounded", () => {
  const raw = readFileSync(
    "scripts/vnext-gate18-phase-c-c4-stmicro-phi4-mini-context16384-output1024-timeout420-loopback-guarded.ts",
    "utf8",
  );

  assert.match(raw, /phi4-mini:3\.8b-q4_K_M/);
  assert.match(raw, /78fad5d182a7c33065e153a5f8ba210754207ba9d91973f57dffa7f487363753/);
  assert.match(raw, /CONTEXT_TOKENS = 16384/);
  assert.match(raw, /MAX_OUTPUT_TOKENS = 1024/);
  assert.match(raw, /CLIENT_TIMEOUT_MS = 420_000/);
  assert.match(raw, /NODE_HTTP_REQUEST_LOOPBACK/);
  assert.match(raw, /acquireWindowsSystemRequiredGuard/);
  assert.match(raw, /assertNoLoadedModels/);
  assert.match(raw, /AUTHORIZED_SINGLE_LOCAL_INFERENCE/);
  assert.match(raw, /C4_STMICRO_MOAT_EVIDENCE_AUDIT_PHI4MINI_CONTEXT16384_001/);
  assert.match(raw, /3f06d6e22dbbc88abbbc64fd470811f438a595c71154f7fcb8c004a6d4f5bc91/);
  assert.match(raw, /gate18PhaseBV10OutputSchema\.safeParse/);
  assert.match(raw, /assertGate18PhaseBV10Semantics/);
  assert.match(raw, /humanAdjudicationRequired: true/);
  assert.match(raw, /calibration\/vnext\/private-runs/);
  assert.match(raw, /productionMutation: false/);
  assert.match(raw, /publicationAuthority: false/);
});

test("Phase C preserves the first Phi-4 STMicro C4 execution history after semantic FAIL", () => {
  const entry = JSON.parse(
    readFileSync(
      "calibration/vnext/OROTITAN_GATE18_PHASE_C_ENTRY_V0.1.json",
      "utf8",
    ),
  );

  assert.equal(entry.phi4_mini_c4_context16384_load_fit, "PASS");
  assert.equal(entry.phi4_mini_c4_context16384_inference_fit, "PASS_FOR_STMICRO_RUNTIME_EXECUTION");
  assert.equal(entry.phi4_mini_c4_first_cell, "STMicroelectronics");
  assert.equal(entry.phi4_mini_c4_stmicro_authorization_status, "CONSUMED_SINGLE_LOCAL_INFERENCE");
  assert.equal(entry.phi4_mini_c4_stmicro_authorized_run_count, 0);
  assert.equal(entry.phi4_mini_c4_stmicro_context_tokens, 16384);
  assert.equal(entry.phi4_mini_c4_stmicro_max_output_tokens, 1024);
  assert.equal(entry.phi4_mini_c4_stmicro_transport, "NODE_HTTP_REQUEST_LOOPBACK");
  assert.equal(entry.phi4_mini_c4_inference_authorized, false);
  assert.equal(entry.phi4_mini_inference_authorized, false);
  assert.equal(entry.phi4_mini_automatic_retry_authorized, false);
  assert.equal(entry.model_switch_authorized, false);
  assert.equal(entry.phi4_mini_c4_stmicro_result_status, "SEMANTIC_FAIL_FORENSICS_REQUIRED");
  assert.equal(entry.phi4_mini_c4_stmicro_forensics_required, true);
  assert.equal(
    entry.next_action,
    "RUN_LOCAL_READ_ONLY_STMICRO_FINDING_CLAIM_PUNCTUATION_FORENSIC",
  );
});
