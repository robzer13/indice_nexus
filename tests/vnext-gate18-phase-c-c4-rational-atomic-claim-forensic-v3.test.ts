import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

test("RATIONAL forensic V2 reveals non-atomic contrastive claim", () => {
  const result = JSON.parse(
    readFileSync(
      "calibration/vnext/OROTITAN_GATE18_PHASE_C_C4_RATIONAL_DIRECTION_ROLE_OVERLAP_FORENSIC_V2_RESULT_001.json",
      "utf8",
    ),
  );

  assert.equal(
    result.status,
    "FORENSIC_COMPLETE_ADDITIONAL_SEMANTIC_DEFECT_FOUND",
  );
  assert.equal(result.diagnostic_observations.removed_conflict_ref_count, 2);
  assert.equal(
    result.diagnostic_observations.removed_support_overlap_ref_count,
    1,
  );
  assert.equal(
    result.diagnostic_observations.downstream_full_validator_pass,
    false,
  );
  assert.equal(
    result.diagnostic_observations.downstream_full_validator_error,
    "VNEXT_GATE18_V10_NON_ATOMIC_CONTRASTIVE_CLAIM",
  );
  assert.deepEqual(result.observed_deterministic_defect_classes, [
    "FINDING_CONFLICT_ID_NOT_GROUNDED_IN_FINDING_EVIDENCE_REFS",
    "DIRECTION_ROLE_OVERLAP",
    "NON_ATOMIC_CONTRASTIVE_CLAIM",
  ]);
  assert.equal(
    result.interpretation_boundary.all_known_deterministic_defects_exhausted,
    false,
  );
});

test("RATIONAL atomic-claim forensic V3 is cumulative, private-safe, and non-inference", () => {
  const prep = JSON.parse(
    readFileSync(
      "calibration/vnext/OROTITAN_GATE18_PHASE_C_C4_RATIONAL_ATOMIC_CLAIM_FORENSIC_V3_PREP_001.json",
      "utf8",
    ),
  );
  const source = readFileSync(
    "scripts/vnext-gate18-phase-c-c4-rational-atomic-claim-forensic-v3.ts",
    "utf8",
  );

  assert.equal(prep.status, "PREPARED_LOCAL_READ_ONLY_NO_INFERENCE");
  assert.equal(prep.diagnostic_normalization.cumulative, true);
  assert.equal(prep.diagnostic_normalization.in_memory_only, true);
  assert.equal(prep.diagnostic_normalization.source_artifact_mutation, false);
  assert.equal(prep.diagnostic_normalization.retroactive_pass_allowed, false);
  assert.equal(prep.authority.inference_authorized, false);
  assert.equal(prep.authority.retry_authorized, false);

  assert.match(source, /NON_ATOMIC_PATTERNS/);
  assert.match(source, /nonAtomicFindings/);
  assert.match(source, /Diagnostic atomic claim\./);
  assert.match(source, /removedConflictRefs/);
  assert.match(source, /removedSupportOverlapRefs/);
  assert.match(source, /assertGate18PhaseBV10Semantics/);
  assert.match(source, /rawClaimTextIncludedInConsole: false/);
  assert.match(source, /rawNarrativeTextIncludedInConsole: false/);
  assert.match(source, /sourceArtifactMutated: false/);
  assert.match(source, /inferenceExecuted: false/);
  assert.doesNotMatch(source, /requestLoopbackJson/);
  assert.doesNotMatch(source, /\/api\/generate/);
  assert.doesNotMatch(source, /\bfetch\s*\(/);
});
