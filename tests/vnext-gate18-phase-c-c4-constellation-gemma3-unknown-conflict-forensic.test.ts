import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

test("Gemma Constellation first forensic records a second deterministic defect", () => {
  const r = JSON.parse(
    readFileSync(
      "calibration/vnext/OROTITAN_GATE18_PHASE_C_C4_CONSTELLATION_GEMMA3_COUNTEREVIDENCE_FORENSIC_RESULT_001.json",
      "utf8",
    ),
  );

  assert.equal(
    r.status,
    "FORENSIC_COMPLETE_ADDITIONAL_SEMANTIC_DEFECT_FOUND",
  );
  assert.deepEqual(r.observed.violating_finding_indexes, [1, 2, 3]);
  assert.equal(r.observed.violating_finding_count, 3);
  assert.equal(r.downstream_validation.pass, false);
  assert.equal(
    r.downstream_validation.error,
    "VNEXT_GATE18_V10_UNKNOWN_CONFLICT_REF",
  );
  assert.equal(r.interpretation.first_defect_isolated, false);
  assert.equal(r.interpretation.retry_authorized, false);
});

test("Gemma unknown-conflict forensic is cumulative, read-only, and non-inferential", () => {
  const prep = JSON.parse(
    readFileSync(
      "calibration/vnext/OROTITAN_GATE18_PHASE_C_C4_CONSTELLATION_GEMMA3_UNKNOWN_CONFLICT_REF_FORENSIC_PREP_001.json",
      "utf8",
    ),
  );
  const source = readFileSync(
    "scripts/vnext-gate18-phase-c-c4-constellation-gemma3-unknown-conflict-ref-forensic.ts",
    "utf8",
  );

  assert.equal(prep.status, "PREPARED_LOCAL_READ_ONLY_NO_INFERENCE");
  assert.equal(prep.diagnostic_normalization.cumulative, true);
  assert.equal(prep.diagnostic_normalization.in_memory_only, true);
  assert.equal(prep.diagnostic_normalization.source_artifact_mutation, false);
  assert.equal(prep.diagnostic_normalization.retroactive_pass_allowed, false);
  assert.equal(prep.authority.inference_authorized, false);
  assert.equal(prep.authority.retry_authorized, false);

  assert.match(source, /VNEXT_GATE18_V10_UNKNOWN_CONFLICT_REF/);
  assert.match(source, /knownConflictIds/);
  assert.match(source, /priorityFindingConflictRefs/);
  assert.match(source, /materialConflictEntries/);
  assert.match(source, /weakLinkConflictRefs/);
  assert.match(source, /unresolvedPointConflictRefs/);
  assert.match(source, /assertGate18PhaseBV10Semantics/);
  assert.match(source, /rawNarrativeTextIncludedInConsole:false/);
  assert.match(source, /inferenceExecuted:false/);
  assert.doesNotMatch(source, /requestLoopbackJson/);
  assert.doesNotMatch(source, /\/api\/generate/);
  assert.doesNotMatch(source, /\bfetch\s*\(/);
});
