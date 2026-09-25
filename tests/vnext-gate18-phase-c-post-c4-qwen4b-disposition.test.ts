import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

test("post-C4 disposition does not admit Qwen3 4B to C5 production path", () => {
  const result = JSON.parse(
    readFileSync(
      "calibration/vnext/OROTITAN_GATE18_PHASE_C_POST_C4_QWEN3_4B_DISPOSITION_001.json",
      "utf8",
    ),
  );

  assert.equal(
    result.status,
    "QWEN3_4B_NOT_ADMITTED_TO_C5_PRODUCTION_PATH_ESCALATION_OPTIONS_REQUIRED",
  );
  assert.equal(result.evidence_basis.c4_cells_completed, 5);
  assert.equal(result.evidence_basis.c4_engineering_pass_count, 2);
  assert.equal(result.evidence_basis.c4_engineering_fail_count, 3);
  assert.equal(result.evidence_basis.c4_human_quality_clean_pass_count, 0);
  assert.equal(result.evidence_basis.c4_human_quality_pass_with_carry_count, 2);
  assert.equal(result.evidence_basis.c4_human_quality_critical_failure_count, 3);
  assert.equal(result.evidence_basis.recurrent_critical_errors_observed, true);
  assert.equal(result.reasoning.c5_repeatability_cannot_retroactively_repair_failed_c4_outputs, true);
  assert.equal(result.disposition.qwen3_4b_candidate_capability_failure_concluded, true);
  assert.equal(result.disposition.qwen3_family_global_failure_concluded, false);
  assert.equal(result.disposition.c5_repeatability_as_production_admission_path, "NOT_ADMITTED");
  assert.equal(result.disposition.production_candidate_status, "NOT_QUALIFIED");
  assert.equal(result.escalation_boundary.model_switch_authorized, false);
  assert.equal(result.escalation_boundary.model_download_authorized, false);
  assert.equal(result.escalation_boundary.new_inference_authorized, false);
  assert.equal(result.escalation_boundary.paid_benchmark_authorized, false);
  assert.equal(result.next_action, "PREPARE_LOCAL_CANDIDATE_ESCALATION_OPTIONS_STATIC_ONLY");
});

test("Phase C entry records Qwen3 4B post-C4 disposition without authorizing a model switch", () => {
  const entry = JSON.parse(
    readFileSync(
      "calibration/vnext/OROTITAN_GATE18_PHASE_C_ENTRY_V0.1.json",
      "utf8",
    ),
  );

  assert.equal(
    entry.current_state.c5_repeatability,
    "QWEN3_4B_NOT_ADMITTED_POST_C4_RECURRENT_CRITICAL_FAILURES",
  );
  assert.equal(entry.c4_qwen3_4b_candidate_capability_failure_concluded, true);
  assert.equal(entry.c4_qwen3_family_global_failure_concluded, false);
  assert.equal(entry.c5_qwen3_4b_production_admission_path, "NOT_ADMITTED");
  assert.equal(entry.candidate_escalation_static_options_required, true);
  assert.equal(entry.model_switch_authorized, false);
  assert.equal(
    entry.next_action,
    "PREPARE_LOCAL_CANDIDATE_ESCALATION_OPTIONS_STATIC_ONLY",
  );
});
