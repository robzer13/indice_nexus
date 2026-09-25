import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

test("Phi-4 Adyen punctuation forensic isolates a reliability-only defect", () => {
  const result = JSON.parse(
    readFileSync(
      "calibration/vnext/OROTITAN_GATE18_PHASE_C_PHI4_MINI_ADYEN_PUNCTUATION_FORENSIC_RESULT_001.json",
      "utf8",
    ),
  );

  assert.equal(
    result.status,
    "FORENSIC_COMPLETE_PUNCTUATION_ONLY_DEFECT_DOWNSTREAM_TARGETED_VALIDATOR_PASS",
  );
  assert.deepEqual(
    result.diagnostic_normalization.normalized_conflict_ids,
    ["C-005", "C-010"],
  );
  assert.equal(result.downstream_validation.pass, true);
  assert.equal(result.downstream_validation.error, null);
  assert.equal(result.conclusion_boundary.original_run_status_changed, false);
  assert.equal(result.conclusion_boundary.retroactive_pass_allowed, false);
  assert.equal(result.conclusion_boundary.inference_executed, false);
  assert.equal(
    result.conclusion_boundary.only_observed_defect_for_this_targeted_probe,
    "MISSING_TERMINAL_PUNCTUATION_ON_C005_AND_C010_IMPLICATIONS",
  );
});

test("Phi-4 targeted regression conclusion admits C4 conditionally with reliability carry", () => {
  const result = JSON.parse(
    readFileSync(
      "calibration/vnext/OROTITAN_GATE18_PHASE_C_PHI4_MINI_C3_FORENSIC_CONCLUSION_001.json",
      "utf8",
    ),
  );

  assert.equal(
    result.status,
    "PASS_TARGETED_SEMANTIC_REGRESSIONS_WITH_OUTPUT_CONTRACT_RELIABILITY_DEFECT",
  );
  assert.equal(result.mandatory_regressions.total, 5);
  assert.equal(result.mandatory_regressions.satisfied, 5);
  assert.equal(result.adyen_original_run_integrity.original_run_status, "FAIL");
  assert.equal(result.adyen_original_run_integrity.original_run_status_changed, false);
  assert.equal(result.protocol_interpretation.c4_entry_allowed, true);
  assert.equal(result.protocol_interpretation.c5_repeatability_allowed_now, false);
  assert.equal(result.candidate_disposition.broader_phase_c_admission, true);
  assert.equal(
    result.candidate_disposition.admission_mode,
    "CONDITIONAL_WITH_RELIABILITY_CARRY",
  );
  assert.equal(result.reliability_carry.auto_repair_for_c4_forbidden, true);
  assert.equal(result.authority.c4_execution_authorized, false);
});

test("Phi-4 C4 matrix prep preserves the five archetypes and original-output adjudication", () => {
  const prep = JSON.parse(
    readFileSync(
      "calibration/vnext/OROTITAN_GATE18_PHASE_C_C4_PHI4_MINI_MATRIX_PREP_001.json",
      "utf8",
    ),
  );

  assert.equal(prep.status, "CONTEXT16384_LOAD_PASS_FIRST_CELL_STMICRO_AUTHORIZED");
  assert.equal(prep.matrix.length, 5);
  assert.deepEqual(
    prep.matrix.map((row: { company: string }) => row.company),
    [
      "RATIONAL AG",
      "Constellation Software",
      "STMicroelectronics",
      "Brookfield Corporation",
      "Adyen",
    ],
  );
  assert.equal(prep.candidate.model, "phi4-mini:3.8b-q4_K_M");
  assert.equal(prep.candidate.context_tokens, 4096);
  assert.equal(prep.candidate.max_output_tokens, 768);
  assert.equal(prep.reliability_carry.c4_auto_repair_forbidden, true);
  assert.equal(prep.reliability_carry.original_output_must_be_adjudicated_as_emitted, true);
  assert.equal(prep.execution_policy.initial_mode, "STATIC_REQUEST_PREFLIGHT_ONLY");
});

test("Phi-4 C4 static preflight performs no Ollama or network inference", () => {
  const source = readFileSync(
    "scripts/vnext-gate18-phase-c-c4-phi4-mini-static-request-preflight.ts",
    "utf8",
  );

  assert.match(source, /phi4-mini:3\.8b-q4_K_M/);
  assert.match(
    source,
    /78fad5d182a7c33065e153a5f8ba210754207ba9d91973f57dffa7f487363753/,
  );
  assert.match(source, /CONTEXT_TOKENS = 4096/);
  assert.match(source, /MAX_OUTPUT_TOKENS = 768/);
  assert.match(source, /NO_INFERENCE_NO_OLLAMA_REQUEST/);
  assert.doesNotMatch(source, /fetch\(/);
  assert.doesNotMatch(source, /127\.0\.0\.1:11434|\/api\/generate|\/api\/chat/);
  assert.match(source, /compactBrookfieldPromptEvalCount: 2020/);
  assert.match(source, /compactAdyenPromptEvalCount: 2872/);
  assert.match(source, /c4InferenceAuthorized: false/);
});

test("Phase C preserves Phi-4 C4 admission across later first-cell execution advances", () => {
  const entry = JSON.parse(
    readFileSync(
      "calibration/vnext/OROTITAN_GATE18_PHASE_C_ENTRY_V0.1.json",
      "utf8",
    ),
  );

  assert.equal(
    entry.phi4_mini_targeted_regression_disposition,
    "PASS_5_OF_5_WITH_OUTPUT_CONTRACT_RELIABILITY_CARRY",
  );
  assert.equal(entry.phi4_mini_broader_phase_c_admission, true);
  assert.equal(
    entry.phi4_mini_broader_phase_c_admission_mode,
    "CONDITIONAL_WITH_RELIABILITY_CARRY",
  );
  assert.equal(entry.phi4_mini_structured_output_reliability_acceptable, false);
  assert.equal(entry.phi4_mini_reliability_retest_required, true);
  assert.equal(entry.phi4_mini_context_growth_authorized, false);
  assert.equal(entry.phi4_mini_c4_proposed_common_context_tokens, 16384);
  assert.equal(entry.phi4_mini_c4_context16384_load_fit, "PASS");
});
