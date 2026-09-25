import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

test("Phi-4 mini pinned download result preserves exact identity and no-execution boundaries", () => {
  const result = JSON.parse(
    readFileSync(
      "calibration/vnext/OROTITAN_GATE18_PHASE_C_PHI4_MINI_DOWNLOAD_RESULT_001.json",
      "utf8",
    ),
  );

  assert.equal(result.status, "PASS_PINNED_DOWNLOAD_ONLY");
  assert.equal(result.model.ollama_model_name, "phi4-mini:3.8b-q4_K_M");
  assert.equal(
    result.model.digest,
    "78fad5d182a7c33065e153a5f8ba210754207ba9d91973f57dffa7f487363753",
  );
  assert.equal(result.model.size_bytes, 2491876774);
  assert.equal(result.model.parameter_size, "3.8B");
  assert.equal(result.model.quantization, "Q4_K_M");
  assert.equal(result.model.ollama_reported_family, "phi3");
  assert.equal(result.verification.expected_digest_prefix_matched, true);
  assert.equal(result.verification.expected_quantization_matched, true);
  assert.equal(result.safety.load_smoke_executed, false);
  assert.equal(result.safety.model_inference_executed, false);
  assert.equal(result.authority.phi4_mini_load_smoke_authorized, false);
  assert.equal(result.authority.phi4_mini_inference_authorized, false);
});

test("Phi-4 mini load-smoke prep records completed 4096 load-only execution", () => {
  const prep = JSON.parse(
    readFileSync(
      "calibration/vnext/OROTITAN_GATE18_PHASE_C_PHI4_MINI_LOAD_SMOKE_PREP_001.json",
      "utf8",
    ),
  );

  assert.equal(
    prep.status,
    "EXECUTED_RESULT_RECORDED_AUTHORIZATION_CONSUMED",
  );
  assert.equal(prep.planned_execution.context_tokens, 4096);
  assert.equal(prep.planned_execution.prompt_provided, false);
  assert.equal(prep.planned_execution.messages_provided, false);
  assert.equal(prep.planned_execution.semantic_inference_requested, false);
  assert.equal(prep.authority.phi4_mini_load_smoke_authorized, false);
  assert.equal(prep.authority.phi4_mini_inference_authorized, false);
  assert.equal(prep.authority.automatic_retry_authorized, false);
  assert.equal(
    prep.future_authorization_id,
    "G18-PHASEC-PHI4-MINI-LOAD-SMOKE-AUTH-001",
  );
});

test("Phi-4 mini load-smoke runner requires the exact separate authorization artifact", () => {
  const raw = readFileSync(
    "scripts/vnext-gate18-phase-c-phi4-mini-load-smoke.ts",
    "utf8",
  );

  assert.match(raw, /PHI4_MINI_LOAD_SMOKE_AUTHORIZATION_ARTIFACT_MISSING/);
  assert.match(raw, /AUTHORIZED_SINGLE_LOAD_ONLY_UNCONSUMED/);
  assert.match(raw, /phi4_mini_load_smoke_authorized !== true/);
  assert.match(raw, /phi4_mini_inference_authorized !== false/);
  assert.match(raw, /78fad5d182a7c33065e153a5f8ba210754207ba9d91973f57dffa7f487363753/);
  assert.match(raw, /num_ctx: CONTEXT_TOKENS/);
  assert.match(raw, /CONTEXT_TOKENS = 4096/);
  assert.match(raw, /keep_alive: "2m"/);
  assert.match(raw, /keep_alive: 0/);
  assert.match(raw, /loadResponse\.response === ""/);
  assert.match(raw, /loadResponse\.eval_count === undefined \|\| loadResponse\.eval_count === 0/);
  assert.match(raw, /semanticInferenceExecuted: false/);
  assert.doesNotMatch(raw, /\/api\/chat/);
});

test("Phase C preserves pinned download and measured load fit across later semantic-probe authorization", () => {
  const entry = JSON.parse(
    readFileSync(
      "calibration/vnext/OROTITAN_GATE18_PHASE_C_ENTRY_V0.1.json",
      "utf8",
    ),
  );

  assert.equal(entry.phi4_mini_download_executed, true);
  assert.equal(entry.phi4_mini_identity_pin_status, "PASS");
  assert.equal(entry.phi4_mini_load_smoke_context_tokens, 4096);
  assert.equal(entry.phi4_mini_load_smoke_authorized, false);
  assert.equal(
    entry.phi4_mini_load_smoke_authorization_status,
    "CONSUMED_SINGLE_LOAD_ONLY",
  );
  assert.equal(entry.model_switch_authorized, false);
  assert.equal(entry.phi4_mini_load_fit_at_4096, "PASS");
  assert.equal(entry.phi4_mini_loaded_vram_free_mib, 1668);
});
