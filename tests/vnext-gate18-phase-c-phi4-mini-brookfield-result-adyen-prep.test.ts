import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

test("Phi-4 mini Brookfield compact result is a bounded engineering PASS only", () => {
  const result = JSON.parse(
    readFileSync(
      "calibration/vnext/OROTITAN_GATE18_PHASE_C_PHI4_MINI_BROOKFIELD_COMPACT4096_LOCAL_RUN_RESULT_001.json",
      "utf8",
    ),
  );

  assert.equal(result.status, "PASS");
  assert.equal(
    result.classification,
    "ENGINEERING_PASS_SCHEMA_AND_TARGETED_SEMANTICS_PASS",
  );
  assert.equal(result.execution.authorized_run_consumed, true);
  assert.equal(result.execution.wall_clock_ms, 72703);
  assert.equal(result.execution.done_reason, "stop");
  assert.equal(result.execution.prompt_eval_count, 2020);
  assert.equal(result.execution.eval_count, 401);
  assert.equal(result.execution.runtime_error, null);
  assert.equal(result.execution.schema_valid, true);
  assert.equal(result.execution.semantic_valid, true);
  assert.equal(result.interpretation.inference_fit_at_4096_context, "PASS_FOR_THIS_BOUNDED_PROBE");
  assert.equal(result.interpretation.c4_quality_assessed, false);
  assert.equal(result.interpretation.c5_repeatability_assessed, false);
  assert.equal(result.authority.model_ranking_authority, false);
  assert.equal(result.authority.production_candidate_decision_authority, false);
});

test("Phi-4 mini Adyen prep records executed semantic FAIL and consumed authorization", () => {
  const prep = JSON.parse(
    readFileSync(
      "calibration/vnext/OROTITAN_GATE18_PHASE_C_PHI4_MINI_ADYEN_COMPACT4096_INFERENCE_PREP_001.json",
      "utf8",
    ),
  );

  assert.equal(prep.status, "EXECUTED_FAIL_AUTHORIZATION_CONSUMED_FORENSICS_REQUIRED");
  assert.equal(prep.probe.semantic_probe_id, "ADYEN_CLAIM_TARGET_CORE_001");
  assert.equal(
    prep.probe.execution_id,
    "ADYEN_CLAIM_TARGET_CORE_001_LOCAL_COMPACT_PHI4MINI_4096",
  );
  assert.equal(prep.execution_plan.context_tokens, 4096);
  assert.equal(prep.execution_plan.max_output_tokens, 768);
  assert.equal(prep.execution_plan.temperature, 0);
  assert.equal(prep.execution_plan.client_timeout_ms, 180000);
  assert.deepEqual(prep.packet_compaction.required_conflict_ids, ["C-005", "C-010"]);
  assert.equal(prep.expected_regression_map.finding_1.support_state, "MIXED");
  assert.equal(prep.expected_regression_map.finding_2.support_state, "SUPPORTED");
  assert.equal(prep.expected_regression_map.finding_3.support_state, "SUPPORTED");
  assert.equal(prep.guard.authorization_artifact_present, true);
  assert.equal(prep.guard.inference_currently_authorized, false);
  assert.equal(prep.authority.phi4_mini_inference_authorized, false);
  assert.equal(prep.authority.automatic_retry_authorized, false);
  assert.equal(prep.authority.context_growth_authorized, false);
});

test("Phi-4 mini Adyen runner is digest-pinned and cannot execute without separate authorization", () => {
  const raw = readFileSync(
    "scripts/vnext-gate18-phase-c-phi4-mini-adyen-compact4096-local-run.ts",
    "utf8",
  );

  assert.match(raw, /phi4-mini:3\.8b-q4_K_M/);
  assert.match(raw, /78fad5d182a7c33065e153a5f8ba210754207ba9d91973f57dffa7f487363753/);
  assert.match(raw, /CONTEXT_TOKENS = 4096/);
  assert.match(raw, /MAX_OUTPUT_TOKENS = 768/);
  assert.match(raw, /G18-PHASEC-PHI4-MINI-ADYEN-COMPACT4096-INFERENCE-AUTH-001/);
  assert.match(raw, /phi4_mini_inference\?\.authorized !== true/);
  assert.match(raw, /options\.execute/);
  assert.match(raw, /authorizationId !== AUTHORIZATION_ID/);
  assert.match(raw, /assertNoLoadedModels/);
  assert.match(raw, /temperature: 0/);
  assert.match(raw, /num_ctx: CONTEXT_TOKENS/);
  assert.match(raw, /num_predict: MAX_OUTPUT_TOKENS/);
  assert.match(raw, /keep_alive: "0s"/);
  assert.match(raw, /ADYEN_CLAIM_TARGET_CORE_001_LOCAL_COMPACT_PHI4MINI_4096/);
  assert.match(raw, /findingCount: 3/);
  assert.match(raw, /supportState: "MIXED"/);
  assert.match(raw, /"C-005": "UNRESOLVED_IN_PACKET"/);
  assert.match(raw, /"C-010": "RESOLVED_IN_PACKET"/);
  assert.match(raw, /calibration\/vnext\/private-runs/);
  assert.match(raw, /productionMutation: false/);
  assert.match(raw, /publicationAuthority: false/);
});

test("Phase C preserves Brookfield PASS and the original Adyen semantic FAIL across later forensic advances", () => {
  const entry = JSON.parse(
    readFileSync(
      "calibration/vnext/OROTITAN_GATE18_PHASE_C_ENTRY_V0.1.json",
      "utf8",
    ),
  );

  assert.equal(entry.phi4_mini_brookfield_compact4096_result_status, "PASS");
  assert.equal(entry.phi4_mini_inference_authorized, false);
  assert.equal(entry.phi4_mini_adyen_compact4096_inference_authorized, false);
  assert.equal(entry.phi4_mini_adyen_compact4096_context_tokens, 4096);
  assert.equal(entry.phi4_mini_adyen_compact4096_max_output_tokens, 768);
  assert.equal(entry.phi4_mini_adyen_compact4096_temperature, 0);
  assert.equal(entry.phi4_mini_context_growth_authorized, false);
  assert.equal(entry.phi4_mini_automatic_retry_authorized, false);
  assert.equal(entry.model_switch_authorized, false);
  assert.equal(entry.phi4_mini_adyen_compact4096_authorized_run_count, 0);
  assert.equal(entry.phi4_mini_adyen_compact4096_result_status, "SEMANTIC_FAIL_FORENSICS_REQUIRED");
  assert.equal(
    entry.phi4_mini_adyen_compact4096_result_status,
    "SEMANTIC_FAIL_FORENSICS_REQUIRED",
  );
});
