import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

test("Phi-4 mini Adyen result preserves the original semantic FAIL", () => {
  const result = JSON.parse(
    readFileSync(
      "calibration/vnext/OROTITAN_GATE18_PHASE_C_PHI4_MINI_ADYEN_COMPACT4096_LOCAL_RUN_RESULT_001.json",
      "utf8",
    ),
  );

  assert.equal(result.status, "FAIL");
  assert.equal(
    result.classification,
    "SEMANTIC_FAIL_CONFLICT_IMPLICATION_INCOMPLETE_FORENSICS_REQUIRED",
  );
  assert.equal(result.execution.authorized_run_consumed, true);
  assert.equal(result.execution.wall_clock_ms, 108752);
  assert.equal(result.execution.done_reason, "stop");
  assert.equal(result.execution.prompt_eval_count, 2872);
  assert.equal(result.execution.eval_count, 706);
  assert.equal(result.execution.output_token_margin, 62);
  assert.equal(result.execution.output_budget_fully_consumed, false);
  assert.equal(result.execution.runtime_error, null);
  assert.equal(result.execution.schema_valid, true);
  assert.equal(result.execution.semantic_valid, false);
  assert.equal(
    result.execution.semantic_error,
    "VNEXT_GATE18_V10_CONFLICT_IMPLICATION_INCOMPLETE",
  );
  assert.equal(result.validator_interpretation.exact_conflict_id_not_yet_determined, true);
  assert.equal(result.validator_interpretation.same_underlying_defect_as_qwen3_4b_concluded, false);
  assert.equal(result.interpretation.model_capability_failure_concluded, false);
  assert.equal(result.authority.retry_authorized, false);
  assert.equal(result.authority.model_switch_authorized, false);
});

test("Phi-4 mini Adyen forensic prep is read-only, provider-agnostic, and non-inference", () => {
  const prep = JSON.parse(
    readFileSync(
      "calibration/vnext/OROTITAN_GATE18_PHASE_C_PHI4_MINI_ADYEN_PUNCTUATION_FORENSIC_PREP_001.json",
      "utf8",
    ),
  );

  assert.equal(prep.status, "PREPARED_NO_INFERENCE_REQUIRED");
  assert.equal(
    prep.source_run_result,
    "G18-PHASEC-PHI4-MINI-ADYEN-COMPACT4096-LOCAL-RUN-001",
  );
  assert.equal(
    prep.observed_first_error,
    "VNEXT_GATE18_V10_CONFLICT_IMPLICATION_INCOMPLETE",
  );
  assert.equal(
    prep.forensic.script,
    "scripts/vnext-gate18-phase-c-c3-adyen-punctuation-normalization-forensic.ts",
  );
  assert.equal(prep.forensic.source_artifact_mutated, false);
  assert.equal(prep.forensic.inference_executed, false);
  assert.equal(prep.forensic.network_request_executed, false);
  assert.equal(prep.interpretation_boundary.original_run_remains_fail, true);
  assert.equal(prep.interpretation_boundary.retroactive_pass_allowed, false);
  assert.equal(prep.interpretation_boundary.same_underlying_defect_as_qwen3_4b_not_assumed, true);
  assert.equal(prep.authority.inference_authorized, false);
  assert.equal(prep.authority.retry_authorized, false);
  assert.equal(prep.authority.artifact_mutation_authorized, false);
});

test("Reused Adyen punctuation forensic performs no model inference and full targeted revalidation", () => {
  const source = readFileSync(
    "scripts/vnext-gate18-phase-c-c3-adyen-punctuation-normalization-forensic.ts",
    "utf8",
  );

  assert.match(source, /IN_MEMORY_DIAGNOSTIC_ONLY_NO_ARTIFACT_MUTATION_NO_INFERENCE/);
  assert.doesNotMatch(source, /fetch\(/);
  assert.doesNotMatch(source, /127\.0\.0\.1:11434|\/api\/generate|\/api\/chat/);
  assert.match(source, /assertGate18V10TargetedProbeSemantics/);
  assert.match(source, /sourceArtifactMutated: false/);
  assert.match(source, /retroactivePassAllowed: false/);
  assert.match(source, /originalRunStatusChanged: false/);
});

test("Phase C has no active Phi-4 inference while the Adyen forensic is pending", () => {
  const entry = JSON.parse(
    readFileSync(
      "calibration/vnext/OROTITAN_GATE18_PHASE_C_ENTRY_V0.1.json",
      "utf8",
    ),
  );

  assert.equal(entry.standing_technical_authorization_status, "ACTIVE");
  assert.equal(entry.standing_in_scope_reprompt_required, false);
  assert.equal(entry.phi4_mini_inference_authorized, false);
  assert.equal(entry.phi4_mini_adyen_compact4096_inference_authorized, false);
  assert.equal(
    entry.phi4_mini_adyen_compact4096_inference_authorization_status,
    "CONSUMED_SINGLE_LOCAL_INFERENCE",
  );
  assert.equal(entry.phi4_mini_adyen_compact4096_authorized_run_count, 0);
  assert.equal(
    entry.phi4_mini_adyen_compact4096_result_status,
    "SEMANTIC_FAIL_FORENSICS_REQUIRED",
  );
  assert.equal(entry.phi4_mini_adyen_forensics_required, true);
  assert.equal(entry.phi4_mini_automatic_retry_authorized, false);
  assert.equal(entry.phi4_mini_context_growth_authorized, false);
  assert.equal(entry.model_switch_authorized, false);
  assert.equal(
    entry.next_action,
    "RUN_LOCAL_READ_ONLY_PHI4_MINI_ADYEN_PUNCTUATION_FORENSIC",
  );
});
