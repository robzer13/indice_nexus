import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

test("Corrected STMicro forensic records recurrent narrative-completeness failure", () => {
  const result = JSON.parse(
    readFileSync(
      "calibration/vnext/OROTITAN_GATE18_PHASE_C_C4_STMICRO_PHI4_MINI_FINDING_CLAIM_FORENSIC_RESULT_001.json",
      "utf8",
    ),
  );

  assert.equal(
    result.status,
    "FORENSIC_COMPLETE_ADDITIONAL_NARRATIVE_COMPLETENESS_DEFECT_FOUND",
  );
  assert.equal(result.original.priority_findings.length, 3);
  assert.deepEqual(
    result.original.priority_findings.map(
      (item: { ends_with_terminal_punctuation: boolean }) =>
        item.ends_with_terminal_punctuation,
    ),
    [false, false, false],
  );
  assert.deepEqual(
    result.diagnostic_normalization.normalized_finding_indexes,
    [1, 2, 3],
  );
  assert.equal(result.downstream_validation.schema_pass, true);
  assert.equal(result.downstream_validation.semantic_pass, false);
  assert.equal(
    result.downstream_validation.semantic_error,
    "VNEXT_GATE18_V10_CAUSAL_LINK_INCOMPLETE",
  );
  assert.equal(result.interpretation_boundary.original_run_status_changed, false);
  assert.equal(result.interpretation_boundary.retroactive_pass_allowed, false);
  assert.equal(result.interpretation_boundary.inference_executed, false);
});

test("Strategy checkpoint selects one zero-inference discriminator before any second Phi-4 C4 cell", () => {
  const decision = JSON.parse(
    readFileSync(
      "calibration/vnext/OROTITAN_GATE18_PHI4_STRATEGY_CHECKPOINT_DECISION_001.json",
      "utf8",
    ),
  );

  assert.equal(
    decision.status,
    "CHECKPOINT_COMPLETE_CONTRACT_ARCHITECTURE_REVIEW_SELECTED",
  );
  assert.equal(decision.diagnosis.same_defect_family_recurrent, true);
  assert.equal(decision.diagnosis.across_multiple_cases, true);
  assert.equal(decision.diagnosis.across_multiple_output_fields, true);
  assert.equal(decision.diagnosis.sunk_cost_continuation_rejected, true);
  assert.equal(
    decision.selected_next_experiment.type,
    "ZERO_INFERENCE_GLOBAL_NARRATIVE_COMPLETENESS_FORENSIC",
  );
  assert.equal(decision.selected_next_experiment.inference, false);
  assert.equal(decision.constraints.second_phi4_c4_cell_authorized, false);
  assert.equal(decision.constraints.phi4_retry_authorized, false);
  assert.equal(decision.constraints.model_switch_executed, false);
  assert.equal(
    decision.decision_tree.if_non_punctuation_or_non_narrative_semantic_error_remains,
    "STOP_PHI4_C4_EXPANSION_AND_ADVANCE_TO_QWEN3_5_4B_HARDWARE_PREFLIGHT_UNDER_SAME_V1_0_CONTRACT",
  );
});

test("Global narrative forensic covers every assertCompleteNarrative field and performs no inference", () => {
  const raw = readFileSync(
    "scripts/vnext-gate18-phase-c-phi4-global-narrative-completeness-forensic.ts",
    "utf8",
  );

  for (const pattern of [
    /priority_findings/,
    /causal_link/,
    /evidence_qualifications/,
    /counterevidence_link/,
    /material_conflicts/,
    /weak_link_candidates/,
    /why_uncertain/,
    /unresolved_points/,
    /question/,
  ]) {
    assert.match(raw, pattern);
  }

  assert.match(raw, /assertGate18PhaseBV10Semantics/);
  assert.match(raw, /gate18PhaseBV10OutputSchema\.safeParse/);
  assert.match(raw, /sourceArtifactMutated: false/);
  assert.match(raw, /inferenceExecuted: false/);
  assert.match(raw, /secondPhi4C4CellAuthorized: false/);
  assert.doesNotMatch(raw, /fetch\(/);
  assert.doesNotMatch(raw, /\/api\/generate|\/api\/chat|127\.0\.0\.1:11434/);
});

test("Phase C keeps second Phi-4 C4 cell blocked after the strategy discriminator completes", () => {
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
    entry.phi4_mini_c4_strategy_checkpoint_decision,
    "G18-PHASEC-PHI4-STRATEGY-CHECKPOINT-001",
  );
  assert.equal(entry.phi4_mini_second_c4_cell_authorized, false);
  assert.equal(entry.phi4_mini_inference_authorized, false);
  assert.equal(entry.phi4_mini_c4_inference_authorized, false);
  assert.equal(entry.model_switch_authorized, false);
  assert.equal(entry.v1_1_contract_change_authorized, false);
  assert.equal(entry.v1_1_shadow_replay_execution_authorized, true);
});
