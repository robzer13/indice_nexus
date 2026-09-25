import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

test("RATIONAL timeout2700 result is a semantic-contract failure after healthy runtime", () => {
  const result = JSON.parse(
    readFileSync(
      "calibration/vnext/OROTITAN_GATE18_PHASE_C_C4_RATIONAL_OUTPUT1280_TIMEOUT2700_RESULT_001.json",
      "utf8",
    ),
  );

  assert.equal(result.status, "FAIL_DETERMINISTIC_SEMANTIC_CONTRACT");
  assert.equal(result.execution.done_reason, "stop");
  assert.equal(result.execution.runtime_error, null);
  assert.equal(result.execution.schema_valid, true);
  assert.equal(result.execution.semantic_valid, false);
  assert.equal(
    result.execution.semantic_error,
    "VNEXT_GATE18_V10_CONFLICT_NOT_GROUNDED_IN_FINDING_REFS",
  );
  assert.equal(result.execution.eval_count, 1098);
  assert.equal(result.execution.output_token_margin, 182);
  assert.equal(result.classification.sole_defect_concluded, false);
  assert.equal(result.authority.retry_authorized, false);
});

test("RATIONAL conflict-grounding forensic V1 is read-only and non-inference", () => {
  const prep = JSON.parse(
    readFileSync(
      "calibration/vnext/OROTITAN_GATE18_PHASE_C_C4_RATIONAL_CONFLICT_GROUNDING_FORENSIC_V1_PREP_001.json",
      "utf8",
    ),
  );
  const source = readFileSync(
    "scripts/vnext-gate18-phase-c-c4-rational-conflict-grounding-forensic-v1.ts",
    "utf8",
  );

  assert.equal(prep.status, "PREPARED_LOCAL_READ_ONLY_NO_INFERENCE");
  assert.equal(prep.observed_deterministic_defect_classes.length, 1);
  assert.equal(prep.diagnostic_normalization.in_memory_only, true);
  assert.equal(prep.diagnostic_normalization.source_artifact_mutation, false);
  assert.equal(prep.diagnostic_normalization.retroactive_pass_allowed, false);
  assert.equal(prep.authority.inference_authorized, false);
  assert.equal(prep.authority.retry_authorized, false);

  assert.match(
    source,
    /C4_RATIONAL_MOAT_EVIDENCE_AUDIT_QWEN4B_CONTEXT16384_OUTPUT1280_TIMEOUT2700_LOOPBACK_GUARDED_001/,
  );
  assert.match(
    source,
    /VNEXT_GATE18_V10_CONFLICT_NOT_GROUNDED_IN_FINDING_REFS/,
  );
  assert.match(source, /originalFindingConflictGrounding/);
  assert.match(source, /removedConflictRefs/);
  assert.match(source, /assertGate18PhaseBV10Semantics/);
  assert.match(source, /rawNarrativeTextIncludedInConsole: false/);
  assert.match(source, /sourceArtifactMutated: false/);
  assert.match(source, /inferenceExecuted: false/);
  assert.doesNotMatch(source, /requestLoopbackJson/);
  assert.doesNotMatch(source, /\/api\/generate/);
  assert.doesNotMatch(source, /\bfetch\s*\(/);
});
