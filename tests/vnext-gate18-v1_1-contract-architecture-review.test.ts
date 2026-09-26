import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

test("Phi-4 global narrative forensic isolates presentation compliance from deterministic semantics", () => {
  const result = JSON.parse(
    readFileSync(
      "calibration/vnext/OROTITAN_GATE18_PHI4_GLOBAL_NARRATIVE_FORENSIC_RESULT_001.json",
      "utf8",
    ),
  );

  assert.equal(
    result.status,
    "FORENSIC_COMPLETE_CONTRACT_ARCHITECTURE_REVIEW_SIGNAL",
  );
  assert.equal(result.inventory.field_count, 13);
  assert.equal(result.inventory.missing_terminal_punctuation_count, 13);
  assert.equal(result.inventory.saturation_boundary_count, 0);
  assert.equal(result.diagnostic_normalization.normalized_count, 13);
  assert.equal(result.downstream_validation.schema_pass, true);
  assert.equal(result.downstream_validation.semantic_pass, true);
  assert.equal(result.downstream_validation.semantic_error, null);
  assert.equal(result.interpretation.original_run_status_changed, false);
  assert.equal(result.interpretation.retroactive_pass_allowed, false);
  assert.equal(result.interpretation.systematic_instruction_following_defect_observed, true);
  assert.equal(result.interpretation.human_quality_assessed, false);
  assert.equal(result.strategic_discriminator.contract_architecture_review_signal, true);
  assert.equal(result.strategic_discriminator.model_pivot_signal, false);
  assert.equal(result.strategic_discriminator.second_phi4_c4_cell_authorized, false);
});

test("v1.1 architecture review separates raw compliance, safe normalization, semantics, and human quality", () => {
  const review = JSON.parse(
    readFileSync(
      "calibration/vnext/OROTITAN_GATE18_V1_1_CONTRACT_ARCHITECTURE_REVIEW_PREP_001.json",
      "utf8",
    ),
  );

  assert.equal(review.status, "SHADOW_REPLAY_AUTHORIZED_PENDING_NO_INFERENCE");
  assert.equal(review.current_v1_0_architecture.validator_boundary.function, "assertCompleteNarrative");
  assert.deepEqual(
    review.current_v1_0_architecture.validator_boundary.checks,
    ["TERMINAL_PUNCTUATION", "LENGTH_LT_178"],
  );
  assert.equal(review.current_v1_0_architecture.validator_boundary.semantic_completeness_actually_proven, false);
  assert.equal(review.current_v1_0_architecture.architecture_problem.length >= 3, true);
  assert.equal(review.preserved_truths.phi4_raw_v1_0_stmicro_status, "FAIL");
  assert.equal(review.preserved_truths.phi4_instruction_following_defect_real, true);
  assert.equal(review.preserved_truths.retroactive_pass_forbidden, true);
  assert.equal(review.preserved_truths.historical_v1_0_results_immutable, true);
  assert.equal(
    review.preferred_review_hypothesis.architecture_id,
    "A_TWO_LAYER_VALIDATION_WITH_SAFE_NORMALIZATION",
  );
  assert.equal(review.preferred_review_hypothesis.status, "HYPOTHESIS_NOT_IMPLEMENTED");
  assert.equal(review.shadow_replay_requirements.no_new_model_inference, true);
  assert.equal(review.shadow_replay_requirements.no_source_artifact_mutation, true);
  assert.equal(review.authority.v1_1_contract_change_authorized, false);
  assert.equal(review.authority.shadow_validator_implementation_authorized, true);
  assert.equal(review.authority.shadow_replay_execution_authorized, true);
  assert.equal(review.authority.model_inference_authorized, false);
  assert.equal(review.authority.second_phi4_c4_cell_authorized, false);
  assert.equal(review.authority.qwen3_5_download_authorized, false);
});

test("Phase C freezes model execution while v1.1 contract architecture review is open", () => {
  const entry = JSON.parse(
    readFileSync(
      "calibration/vnext/OROTITAN_GATE18_PHASE_C_ENTRY_V0.1.json",
      "utf8",
    ),
  );

  assert.equal(
    entry.phi4_mini_c4_strategy_checkpoint_status,
    "COMPLETE_CONTRACT_ARCHITECTURE_REVIEW_SELECTED",
  );
  assert.equal(
    entry.phi4_mini_global_narrative_forensic_result,
    "G18-PHASEC-PHI4-GLOBAL-NARRATIVE-FORENSIC-RESULT-001",
  );
  assert.equal(entry.phi4_mini_global_narrative_field_count, 13);
  assert.equal(entry.phi4_mini_global_narrative_missing_punctuation_count, 13);
  assert.equal(entry.phi4_mini_global_narrative_semantic_pass_after_normalization, true);
  assert.equal(entry.phi4_mini_raw_v1_0_stmicro_status_remains, "FAIL");
  assert.equal(entry.phi4_mini_second_c4_cell_authorized, false);
  assert.equal(entry.phi4_mini_c4_inference_authorized, false);
  assert.equal(entry.phi4_mini_inference_authorized, false);
  assert.equal(entry.qwen3_5_download_authorized, false);
  assert.equal(entry.v1_1_contract_change_authorized, false);
  assert.equal(entry.v1_1_shadow_validator_implementation_authorized, true);
  assert.equal(entry.v1_1_shadow_replay_execution_authorized, true);
  assert.equal(
    entry.next_action,
    "RUN_V1_1_SHADOW_REPLAY_ON_EXISTING_PRIVATE_ARTIFACTS_NO_INFERENCE",
  );
});
