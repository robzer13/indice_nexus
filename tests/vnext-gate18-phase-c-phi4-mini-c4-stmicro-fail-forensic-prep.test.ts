import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

test("Phi-4 STMicro C4 result preserves original deterministic semantic FAIL", () => {
  const result = JSON.parse(
    readFileSync(
      "calibration/vnext/OROTITAN_GATE18_PHASE_C_C4_STMICRO_PHI4_MINI_CONTEXT16384_RUN_RESULT_001.json",
      "utf8",
    ),
  );

  assert.equal(result.status, "FAIL");
  assert.equal(
    result.classification,
    "SEMANTIC_FAIL_FINDING_CLAIM_INCOMPLETE_FORENSICS_REQUIRED",
  );
  assert.equal(result.execution.authorized_run_consumed, true);
  assert.equal(result.execution.wall_clock_ms, 192409);
  assert.equal(result.execution.done_reason, "stop");
  assert.equal(result.execution.prompt_eval_count, 2845);
  assert.equal(result.execution.eval_count, 714);
  assert.equal(result.execution.runtime_error, null);
  assert.equal(result.execution.schema_valid, true);
  assert.equal(result.execution.semantic_valid, false);
  assert.equal(
    result.execution.semantic_error,
    "VNEXT_GATE18_V10_FINDING_CLAIM_INCOMPLETE",
  );
  assert.equal(
    result.interpretation.context16384_inference_runtime_fit,
    "PASS_FOR_THIS_CELL_EXECUTION",
  );
  assert.equal(result.interpretation.c4_human_quality_assessed, false);
  assert.equal(result.authority.retry_authorized, false);
  assert.equal(result.c4_effect.matrix_cells_completed, 0);
});

test("Phi-4 STMicro finding-claim forensic prep is read-only and non-inference", () => {
  const prep = JSON.parse(
    readFileSync(
      "calibration/vnext/OROTITAN_GATE18_PHASE_C_C4_STMICRO_PHI4_MINI_FINDING_CLAIM_FORENSIC_PREP_001.json",
      "utf8",
    ),
  );

  assert.equal(prep.status, "PREPARED_NO_INFERENCE_REQUIRED");
  assert.equal(
    prep.source_run_result,
    "G18-PHASEC-C4-STM-PHI4MINI-CONTEXT16384-RUN-001",
  );
  assert.equal(
    prep.observed_first_error,
    "VNEXT_GATE18_V10_FINDING_CLAIM_INCOMPLETE",
  );
  assert.equal(prep.forensic.source_artifact_mutated, false);
  assert.equal(prep.forensic.inference_executed, false);
  assert.equal(prep.forensic.network_request_executed, false);
  assert.equal(prep.interpretation_boundary.original_run_remains_fail, true);
  assert.equal(prep.interpretation_boundary.retroactive_pass_allowed, false);
  assert.equal(prep.authority.inference_authorized, false);
  assert.equal(prep.authority.retry_authorized, false);
});

test("Phi-4 STMicro finding-claim forensic normalizes only in-memory claims and reruns full validators", () => {
  const raw = readFileSync(
    "scripts/vnext-gate18-phase-c-c4-stmicro-phi4-mini-finding-claim-punctuation-forensic.ts",
    "utf8",
  );

  assert.match(raw, /STMicroelectronics/);
  assert.match(raw, /IN_MEMORY_DIAGNOSTIC_ONLY_NO_ARTIFACT_MUTATION_NO_INFERENCE/);
  assert.match(raw, /normalizedFindingIndexes/);
  assert.match(raw, /Append one period only when finding\.claim terminal punctuation is absent/);
  assert.match(raw, /gate18PhaseBV10OutputSchema\.safeParse/);
  assert.match(raw, /assertGate18PhaseBV10Semantics/);
  assert.doesNotMatch(raw, /fetch\(/);
  assert.doesNotMatch(raw, /\/api\/generate|\/api\/chat|127\.0\.0\.1:11434/);
  assert.match(raw, /sourceArtifactMutated: false/);
  assert.match(raw, /retroactivePassAllowed: false/);
  assert.match(raw, /originalRunStatusChanged: false/);
  assert.match(raw, /inferenceExecuted: false/);
});

test("Phase C current state blocks broader Phi-4 C4 execution pending read-only STMicro forensic", () => {
  const entry = JSON.parse(
    readFileSync(
      "calibration/vnext/OROTITAN_GATE18_PHASE_C_ENTRY_V0.1.json",
      "utf8",
    ),
  );

  assert.equal(entry.phi4_mini_c4_stmicro_result_status, "SEMANTIC_FAIL_FORENSICS_REQUIRED");
  assert.equal(entry.phi4_mini_c4_stmicro_schema_valid, true);
  assert.equal(entry.phi4_mini_c4_stmicro_semantic_valid, false);
  assert.equal(
    entry.phi4_mini_c4_stmicro_semantic_error,
    "VNEXT_GATE18_V10_FINDING_CLAIM_INCOMPLETE",
  );
  assert.equal(entry.phi4_mini_c4_stmicro_forensics_required, true);
  assert.equal(entry.phi4_mini_c4_inference_authorized, false);
  assert.equal(entry.phi4_mini_inference_authorized, false);
  assert.equal(entry.phi4_mini_c4_matrix_cells_completed, 0);
  assert.equal(entry.phi4_mini_c4_matrix_cells_remaining, 5);
  assert.equal(entry.phi4_mini_automatic_retry_authorized, false);
  assert.equal(entry.model_switch_authorized, false);
  assert.equal(
    entry.next_action,
    "RUN_LOCAL_READ_ONLY_STMICRO_FINDING_CLAIM_PUNCTUATION_FORENSIC",
  );
});
