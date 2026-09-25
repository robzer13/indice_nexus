import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

test("RATIONAL V3R records the fourth deterministic semantic defect without changing the run", () => {
  const result = JSON.parse(
    readFileSync(
      "calibration/vnext/OROTITAN_GATE18_PHASE_C_C4_RATIONAL_ATOMIC_CLAIM_FORENSIC_V3R_RESULT_001.json",
      "utf8",
    ),
  );

  assert.equal(
    result.status,
    "FORENSIC_COMPLETE_ADDITIONAL_SEMANTIC_DEFECT_FOUND",
  );
  assert.equal(
    result.fourth_deterministic_defect.class,
    "COUNTEREVIDENCE_LINK_INCOMPLETE",
  );
  assert.equal(
    result.observed_console.downstream_full_validator_error,
    "VNEXT_GATE18_V10_COUNTEREVIDENCE_LINK_INCOMPLETE",
  );
  assert.equal(
    result.interpretation_boundary.all_known_deterministic_defects_exhausted,
    false,
  );
  assert.equal(result.interpretation_boundary.inference_executed, false);
  assert.equal(result.interpretation_boundary.retry_authorized, false);
});

test("RATIONAL V4 is cumulative, narrow, private-safe, and no-inference", () => {
  const prep = JSON.parse(
    readFileSync(
      "calibration/vnext/OROTITAN_GATE18_PHASE_C_C4_RATIONAL_COUNTEREVIDENCE_LINK_FORENSIC_V4_PREP_001.json",
      "utf8",
    ),
  );
  const source = readFileSync(
    "scripts/vnext-gate18-phase-c-c4-rational-counterevidence-link-forensic-v4.ts",
    "utf8",
  );

  assert.equal(prep.status, "PREPARED_LOCAL_READ_ONLY_NO_INFERENCE");
  assert.equal(prep.diagnostic_normalization.cumulative, true);
  assert.equal(prep.diagnostic_normalization.in_memory_only, true);
  assert.equal(prep.diagnostic_normalization.source_artifact_mutation, false);
  assert.equal(prep.authority.inference_authorized, false);
  assert.equal(prep.authority.retry_authorized, false);

  assert.match(source, /const SATURATION_BOUNDARY = 178/);
  assert.match(source, /incompleteCounterevidenceLinks/);
  assert.match(source, /MISSING_TERMINAL_PUNCTUATION/);
  assert.match(source, /NARRATIVE_BOUNDARY_SATURATION/);
  assert.match(source, /Diagnostic counterevidence link\./);
  assert.match(source, /rawCounterevidenceTextIncludedInConsole: false/);
  assert.match(source, /removedConflictRefs/);
  assert.match(source, /removedSupportOverlapRefs/);
  assert.match(source, /nonAtomicFindings/);
  assert.match(source, /assertGate18PhaseBV10Semantics/);
  assert.doesNotMatch(source, /requestLoopbackJson/);
  assert.doesNotMatch(source, /\/api\/generate/);
  assert.doesNotMatch(source, /\bfetch\s*\(/);
});
