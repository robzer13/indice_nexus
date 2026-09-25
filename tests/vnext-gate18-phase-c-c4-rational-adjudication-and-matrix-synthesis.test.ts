import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

test("RATIONAL human adjudication records critical engineering and human-quality failure without persisting private packet content", () => {
  const resultPath =
    "calibration/vnext/OROTITAN_GATE18_PHASE_C_C4_RATIONAL_HUMAN_ADJUDICATION_RESULT_001.json";
  const raw = readFileSync(resultPath, "utf8");
  const result = JSON.parse(raw);

  assert.equal(
    result.status,
    "COMPLETED_WITH_ENGINEERING_FAIL_AND_HUMAN_QUALITY_CRITICAL_FAILURE",
  );
  assert.equal(result.engineering_disposition.runtime_pass, true);
  assert.equal(result.engineering_disposition.schema_valid, true);
  assert.equal(result.engineering_disposition.semantic_validator_pass, false);
  assert.equal(result.engineering_disposition.deterministic_defect_class_count, 4);
  assert.equal(result.disposition.critical_human_quality_failure, true);
  assert.equal(result.disposition.clean_pass, false);
  assert.equal(result.disposition.pass_with_carry, false);
  assert.equal(result.disposition.matrix_cell_completed, true);
  assert.equal(result.human_quality.exact_evidence_grounding, "FAIL");
  assert.equal(result.human_quality.claim_atomicity, "FAIL");
  assert.equal(result.human_quality.support_counterevidence_polarity, "FAIL");
  assert.equal(result.human_quality.weak_link_usefulness, "PASS");
  assert.equal(result.human_quality.unresolved_point_usefulness, "PASS");
  assert.equal(result.human_quality.judgment_boundary_compliance, "PASS");
  assert.equal(result.critical_failures.evidence_id_invention, false);
  assert.equal(result.critical_failures.conflict_id_invention, false);
  assert.equal(result.critical_failures.deterministic_semantic_failure, true);
  assert.equal(result.source.private_packet_content_persisted_publicly, false);
  assert.doesNotMatch(raw, /"modelOutput"\s*:/);
  assert.doesNotMatch(raw, /"evidencePacket"\s*:/);
});

test("RATIONAL adjudication export result preserves the public-private boundary", () => {
  const result = JSON.parse(
    readFileSync(
      "calibration/vnext/OROTITAN_GATE18_PHASE_C_C4_RATIONAL_HUMAN_ADJUDICATION_EXPORT_RESULT_001.json",
      "utf8",
    ),
  );

  assert.equal(result.status, "PASS_REVIEW_PACKET_READY_ENGINEERING_FAIL");
  assert.deepEqual(result.counts, {
    evidence: 56,
    conflicts: 9,
    priority_findings: 2,
    material_conflicts: 2,
    weak_links: 2,
    unresolved_points: 3,
  });
  assert.equal(result.policy.private_artifact, true);
  assert.equal(result.policy.generated_content_persisted_publicly, false);
  assert.equal(result.policy.original_output_only, true);
  assert.equal(result.policy.diagnostic_normalizations_applied, false);
  assert.equal(result.policy.no_inference, true);
  assert.equal(result.policy.publication_authority, false);
});

test("C4 five-company synthesis closes all five matrix cells without authorizing C5 or model switching", () => {
  const synthesis = JSON.parse(
    readFileSync(
      "calibration/vnext/OROTITAN_GATE18_PHASE_C_C4_FIVE_COMPANY_SYNTHESIS_001.json",
      "utf8",
    ),
  );

  assert.equal(
    synthesis.status,
    "C4_MATRIX_COMPLETE_RECURRENT_CRITICAL_FAILURES_OBSERVED",
  );
  assert.equal(synthesis.matrix.planned_cells, 5);
  assert.equal(synthesis.matrix.completed_cells, 5);
  assert.equal(synthesis.matrix.remaining_cells, 0);
  assert.equal(synthesis.aggregate.automated_engineering_pass_count, 2);
  assert.equal(synthesis.aggregate.deterministic_engineering_fail_count, 3);
  assert.equal(synthesis.aggregate.human_quality_clean_pass_count, 0);
  assert.equal(synthesis.aggregate.human_quality_pass_with_carry_count, 2);
  assert.equal(synthesis.aggregate.human_quality_critical_failure_count, 3);
  assert.equal(synthesis.interpretation_boundary.recurrent_critical_errors_observed, true);
  assert.equal(synthesis.interpretation_boundary.global_model_capability_failure_concluded, false);
  assert.equal(synthesis.interpretation_boundary.c5_repeatability_authorized, false);
  assert.equal(synthesis.interpretation_boundary.model_switch_authorized, false);
  assert.equal(synthesis.interpretation_boundary.production_candidate_decision_authority, false);
});

test("Phase C entry preserves completed RATIONAL adjudication and closed C4 matrix across later stage advances", () => {
  const entry = JSON.parse(
    readFileSync(
      "calibration/vnext/OROTITAN_GATE18_PHASE_C_ENTRY_V0.1.json",
      "utf8",
    ),
  );

  assert.equal(entry.c4_matrix_cells_completed, 5);
  assert.equal(entry.c4_matrix_cells_remaining, 0);
  assert.equal(entry.c4_rational_human_adjudication_completed, true);
  assert.equal(entry.c4_rational_human_quality_critical_failure, true);
  assert.equal(
    entry.c4_five_company_synthesis_status,
    "C4_MATRIX_COMPLETE_RECURRENT_CRITICAL_FAILURES_OBSERVED",
  );
  assert.equal(entry.c4_human_quality_critical_failure_count, 3);
  assert.equal(entry.c4_recurrent_critical_errors_observed, true);
  assert.equal(entry.c5_repeatability_authorized, false);
  assert.equal(
    entry.current_state.c4_diversified_matrix,
    "COMPLETE_5_OF_5_SYNTHESIS_RECORDED_RECURRENT_CRITICAL_FAILURES",
  );
  assert.equal(entry.c4_matrix_cells_completed, 5);
  assert.equal(entry.c4_matrix_cells_remaining, 0);
});
