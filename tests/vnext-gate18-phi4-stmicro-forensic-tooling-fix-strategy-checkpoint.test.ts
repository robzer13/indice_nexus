import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

test("STMicro forensic tooling defect is classified as tooling-only and non-inference", () => {
  const result = JSON.parse(
    readFileSync(
      "calibration/vnext/OROTITAN_GATE18_PHASE_C_C4_STMICRO_PHI4_MINI_FORENSIC_TOOLING_DEFECT_001.json",
      "utf8",
    ),
  );

  assert.equal(result.status, "TOOLING_DEFECT_IDENTIFIED_AND_FIXED_RERUN_REQUIRED");
  assert.equal(result.root_cause.canonical_schema_field, "priority_findings");
  assert.equal(result.root_cause.erroneous_tooling_field, "findings");
  assert.equal(result.root_cause.classification, "FORENSIC_TOOLING_FIELD_NAME_BUG");
  assert.equal(result.root_cause.model_defect, false);
  assert.equal(result.impact.forensic_completed, false);
  assert.equal(result.impact.inference_executed, false);
  assert.equal(result.impact.source_artifact_mutated, false);
  assert.equal(result.remediation.rerun_required, true);
});

test("OroTitan strategy checkpoint policy prevents sunk-cost continuation", () => {
  const policy = JSON.parse(
    readFileSync(
      "calibration/vnext/OROTITAN_EXECUTION_STRATEGY_CHECKPOINT_POLICY_001.json",
      "utf8",
    ),
  );

  assert.equal(policy.status, "ACTIVE");
  assert.equal(policy.decision_rules.sunk_cost_is_not_a_reason_to_continue, true);
  assert.equal(policy.decision_rules.prior_investment_does_not_raise_candidate_merit, true);
  assert.equal(policy.decision_rules.prefer_information_gain_per_execution_cost, true);
  assert.equal(
    policy.current_application.checkpoint_timing,
    "AFTER_CORRECTED_STMICRO_FINDING_CLAIM_FORENSIC_BEFORE_ANY_SECOND_C4_MATRIX_CELL",
  );
  assert.equal(policy.authority.automatic_model_switch, false);
  assert.equal(policy.authority.strategy_review_required_at_trigger, true);
});

test("Phase C requires the strategy checkpoint after the corrected STMicro forensic", () => {
  const entry = JSON.parse(
    readFileSync(
      "calibration/vnext/OROTITAN_GATE18_PHASE_C_ENTRY_V0.1.json",
      "utf8",
    ),
  );

  assert.equal(
    entry.strategy_checkpoint_policy,
    "OROTITAN-EXECUTION-STRATEGY-CHECKPOINT-POLICY-001",
  );
  assert.equal(entry.phi4_mini_c4_strategy_checkpoint_required_after_stmicro_forensic, true);
  assert.equal(entry.phi4_mini_c4_strategy_checkpoint_before_second_matrix_cell, true);
  assert.equal(
    entry.phi4_mini_c4_strategy_checkpoint_status,
    "TRIGGERED_DISCRIMINATING_FORENSIC_PENDING",
  );
});
