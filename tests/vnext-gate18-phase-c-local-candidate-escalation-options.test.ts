import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

test("post-C4 escalation options prefer same-size family diversity before Qwen3 8B and authorize no execution", () => {
  const options = JSON.parse(
    readFileSync(
      "calibration/vnext/OROTITAN_GATE18_PHASE_C_LOCAL_CANDIDATE_ESCALATION_OPTIONS_001.json",
      "utf8",
    ),
  );

  assert.equal(
    options.status,
    "STATIC_OPTIONS_COMPLETE_REGISTRY_EXPANSION_JUSTIFIED_NO_DOWNLOAD_NO_INFERENCE",
  );
  assert.equal(options.current_registered_escalation.state, "HOLD_POOR_CURRENT_HARDWARE_FIT");
  assert.equal(options.static_evaluation_sequence[0].candidate_id, "PHI4_MINI_3_8B_OLLAMA_Q4_K_M");
  assert.equal(options.static_evaluation_sequence[1].candidate_id, "QWEN3_5_4B_OLLAMA_Q4_K_M");
  assert.equal(options.static_evaluation_sequence[2].candidate_id, "GEMMA3_4B_OLLAMA_Q4_K_M");
  assert.equal(options.static_evaluation_sequence[3].candidate_id, "QWEN3_8B_LOCAL");
  assert.equal(options.registry_decision.candidate_registry_expansion_justified, true);
  assert.equal(options.authority.candidate_registry_static_expansion_authorized, true);
  assert.equal(options.authority.model_download_authorized, false);
  assert.equal(options.authority.model_inference_authorized, false);
  assert.equal(options.authority.model_switch_authorized, false);
  assert.equal(options.authority.paid_benchmark_authorized, false);
});

test("expanded local candidate registry remains static and selects no winner", () => {
  const registry = JSON.parse(
    readFileSync(
      "calibration/vnext/OROTITAN_GATE18_PHASE_C_LOCAL_MODEL_CANDIDATES_V0.1.json",
      "utf8",
    ),
  );
  const ids = new Set(registry.candidates.map((candidate: { candidate_id: string }) => candidate.candidate_id));

  assert.equal(registry.status, "POST_C4_STATIC_EXPANSION_RECORDED_NO_DOWNLOAD");
  assert.equal(registry.selection_policy.no_model_winner_selected, true);
  assert.equal(registry.selection_policy.next_static_candidate, "PHI4_MINI_3_8B_OLLAMA_Q4_K_M");
  assert.equal(ids.has("PHI4_MINI_3_8B_OLLAMA_Q4_K_M"), true);
  assert.equal(ids.has("QWEN3_5_4B_OLLAMA_Q4_K_M"), true);
  assert.equal(ids.has("GEMMA3_4B_OLLAMA_Q4_K_M"), true);
});

test("Phi-4 mini preflight prep requires explicit authorization before download and forbids inference", () => {
  const prep = JSON.parse(
    readFileSync(
      "calibration/vnext/OROTITAN_GATE18_PHASE_C_PHI4_MINI_DOWNLOAD_MEMORY_PREFLIGHT_PREP_001.json",
      "utf8",
    ),
  );

  assert.equal(
    prep.status,
    "PREPARED_STATIC_NO_DOWNLOAD_NO_INFERENCE_AWAITING_EXPLICIT_USER_AUTHORIZATION",
  );
  assert.equal(prep.candidate.model_id, "phi4-mini:3.8b-q4_K_M");
  assert.equal(prep.candidate.expected_ollama_artifact_size_gb, 2.5);
  assert.equal(prep.hardware_basis.static_artifact_fit, "PLAUSIBLE");
  assert.equal(prep.hardware_basis.runtime_memory_fit, "NOT_YET_PROVEN");
  assert.equal(prep.authority.download_authorized, false);
  assert.equal(prep.authority.inference_authorized, false);
  assert.equal(prep.authority.automatic_retry_authorized, false);
  assert.equal(prep.authority.automatic_model_switch_authorized, false);
  assert.equal(
    prep.next_action,
    "AWAIT_EXPLICIT_USER_AUTHORIZATION_TO_DOWNLOAD_PHI4_MINI_3_8B_Q4_K_M",
  );
});

test("Phase C entry stops at the explicit Phi-4 mini download authorization boundary", () => {
  const entry = JSON.parse(
    readFileSync(
      "calibration/vnext/OROTITAN_GATE18_PHASE_C_ENTRY_V0.1.json",
      "utf8",
    ),
  );

  assert.equal(entry.candidate_registry_static_expansion_completed, true);
  assert.equal(entry.candidate_registry_static_expansion_count, 3);
  assert.equal(entry.candidate_escalation_first_static_candidate, "PHI4_MINI_3_8B_OLLAMA_Q4_K_M");
  assert.equal(entry.phi4_mini_download_authorized, false);
  assert.equal(entry.phi4_mini_inference_authorized, false);
  assert.equal(entry.model_switch_authorized, false);
  assert.equal(
    entry.next_action,
    "AWAIT_EXPLICIT_USER_AUTHORIZATION_TO_DOWNLOAD_PHI4_MINI_3_8B_Q4_K_M",
  );
});
