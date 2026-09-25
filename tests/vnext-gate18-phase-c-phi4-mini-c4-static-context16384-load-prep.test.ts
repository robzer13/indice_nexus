import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

test("Phi-4 C4 static request result preserves all five measured request identities", () => {
  const result = JSON.parse(
    readFileSync(
      "calibration/vnext/OROTITAN_GATE18_PHASE_C_C4_PHI4_MINI_STATIC_REQUEST_PREFLIGHT_RESULT_001.json",
      "utf8",
    ),
  );

  assert.equal(result.status, "PASS_STATIC_REQUESTS_MEASURED");
  assert.equal(result.mode, "NO_INFERENCE_NO_OLLAMA_REQUEST");
  assert.equal(result.rows.length, 5);
  assert.equal(result.summary.min_prompt_bytes, 7558);
  assert.equal(result.summary.max_prompt_bytes, 35756);
  assert.equal(result.rows[0].company, "RATIONAL AG");
  assert.equal(
    result.rows[0].prompt_sha256,
    "70265015372600e619010150a72e72dad5df32973f61c01a79658994ea02c0a5",
  );
  assert.equal(result.rows[4].company, "Adyen");
  assert.equal(
    result.rows[4].request_sha256,
    "701f5caf3dce3265821e9890059fe82b3ca615f13e186ceacffd8ef5eb725278",
  );
  assert.equal(result.safety.model_inference_executed, false);
});

test("Phi-4 C4 context-risk screen rejects 4096 and selects 16384 only for load-smoke preparation", () => {
  const result = JSON.parse(
    readFileSync(
      "calibration/vnext/OROTITAN_GATE18_PHASE_C_C4_PHI4_MINI_CONTEXT_FIT_DIAGNOSTIC_001.json",
      "utf8",
    ),
  );

  assert.equal(
    result.status,
    "CONTEXT_4096_NOT_ADMISSIBLE_CONTEXT16384_LOAD_SMOKE_REQUIRED",
  );
  assert.equal(result.basis.empirical_local_relation.exact_tokenizer, false);
  assert.equal(result.rows[0].company, "RATIONAL AG");
  assert.equal(result.rows[0].estimated_total_tokens_at_max_output_768, 10984);
  assert.equal(result.rows[0].headroom_vs_12288, 1304);
  assert.equal(result.rows[0].headroom_vs_16384, 5400);
  assert.equal(result.decision.common_context_4096_admissible, false);
  assert.equal(result.decision.common_context_8192_admissible, false);
  assert.equal(result.decision.common_context_12288_selected, false);
  assert.equal(result.decision.proposed_common_context_tokens, 16384);
  assert.equal(result.hardware_boundary.context_16384_load_fit_proven, false);
  assert.equal(result.authority.context16384_inference_authorized, false);
  assert.equal(result.authority.c4_inference_authorized, false);
});

test("Phi-4 context16384 load-smoke authorization derives from standing authority and remains non-inference", () => {
  const auth = JSON.parse(
    readFileSync(
      "calibration/vnext/OROTITAN_GATE18_PHASE_C_C4_PHI4_MINI_CONTEXT16384_LOAD_SMOKE_AUTH_001.json",
      "utf8",
    ),
  );

  assert.equal(auth.status, "AUTHORIZED_SINGLE_LOAD_ONLY_UNCONSUMED");
  assert.equal(auth.authorization_source.type, "STANDING_USER_AUTHORIZATION");
  assert.equal(
    auth.authorization_source.authorization_id,
    "OROTITAN-STANDING-TECHNICAL-AUTH-001",
  );
  assert.equal(auth.authorization_source.separate_user_reprompt_required, false);
  assert.equal(auth.model.ollama_model_name, "phi4-mini:3.8b-q4_K_M");
  assert.equal(
    auth.model.digest,
    "78fad5d182a7c33065e153a5f8ba210754207ba9d91973f57dffa7f487363753",
  );
  assert.equal(auth.model.context_tokens, 16384);
  assert.equal(auth.execution.action_count_authorized, 1);
  assert.equal(auth.execution.semantic_inference_authorized, false);
  assert.equal(auth.constraints.c4_inference_authorized, false);
  assert.equal(auth.constraints.automatic_retry_authorized, false);
  assert.equal(auth.constraints.further_context_change_authorized, false);
  assert.equal(auth.authority.context16384_load_smoke_authorized, true);
  assert.equal(auth.authority.context16384_inference_authorized, false);
});

test("Phi-4 context16384 load-smoke runner is authorization-gated and generation-free", () => {
  const source = readFileSync(
    "scripts/vnext-gate18-phase-c-c4-phi4-mini-context16384-load-smoke.ts",
    "utf8",
  );

  assert.match(source, /phi4-mini:3\.8b-q4_K_M/);
  assert.match(
    source,
    /78fad5d182a7c33065e153a5f8ba210754207ba9d91973f57dffa7f487363753/,
  );
  assert.match(source, /CONTEXT_TOKENS = 16384/);
  assert.match(
    source,
    /G18-PHASEC-C4-PHI4-MINI-CONTEXT16384-LOAD-SMOKE-AUTH-001/,
  );
  assert.match(source, /AUTHORIZED_SINGLE_LOAD_ONLY_UNCONSUMED/);
  assert.match(source, /context16384_load_smoke_authorized !== true/);
  assert.match(source, /context16384_inference_authorized !== false/);
  assert.match(source, /promptProvided: false/);
  assert.match(source, /semanticInferenceExecuted: false/);
  assert.match(source, /num_ctx: CONTEXT_TOKENS/);
  assert.doesNotMatch(source, /prompt:/);
  assert.doesNotMatch(source, /messages:/);
});

test("Phase C current state authorizes only the context16384 load smoke, not C4 inference", () => {
  const entry = JSON.parse(
    readFileSync(
      "calibration/vnext/OROTITAN_GATE18_PHASE_C_ENTRY_V0.1.json",
      "utf8",
    ),
  );

  assert.equal(entry.phi4_mini_c4_static_request_preflight_status, "PASS_STATIC_REQUESTS_MEASURED");
  assert.equal(entry.phi4_mini_c4_context4096_admissible, false);
  assert.equal(entry.phi4_mini_c4_context8192_admissible, false);
  assert.equal(entry.phi4_mini_c4_context12288_selected, false);
  assert.equal(entry.phi4_mini_c4_proposed_common_context_tokens, 16384);
  assert.equal(entry.phi4_mini_c4_context16384_load_smoke_authorized, true);
  assert.equal(entry.phi4_mini_c4_context16384_load_fit, "NOT_YET_PROVEN");
  assert.equal(entry.phi4_mini_c4_inference_authorized, false);
  assert.equal(entry.phi4_mini_inference_authorized, false);
  assert.equal(entry.phi4_mini_automatic_retry_authorized, false);
  assert.equal(entry.model_switch_authorized, false);
  assert.equal(
    entry.next_action,
    "RUN_ONE_PHI4_MINI_CONTEXT16384_LOAD_ONLY_SMOKE_NO_INFERENCE",
  );
});
