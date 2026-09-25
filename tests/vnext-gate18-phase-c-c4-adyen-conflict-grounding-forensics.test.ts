import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

test("Adyen narrative-saturation forensic V2 reveals ungrounded finding conflict references", () => {
  const result = JSON.parse(
    readFileSync(
      "calibration/vnext/OROTITAN_GATE18_PHASE_C_C4_ADYEN_NARRATIVE_SATURATION_FORENSIC_V2_RESULT_001.json",
      "utf8",
    ),
  );

  assert.equal(
    result.status,
    "FORENSIC_COMPLETE_THIRD_SEMANTIC_DEFECT_FOUND",
  );
  assert.equal(result.diagnostic_observations.saturated_narrative_count, 1);
  assert.equal(
    result.diagnostic_observations.downstream_full_validator_pass,
    false,
  );
  assert.equal(
    result.diagnostic_observations.downstream_full_validator_error,
    "VNEXT_GATE18_V10_CONFLICT_NOT_GROUNDED_IN_FINDING_REFS",
  );
  assert.deepEqual(result.observed_deterministic_defect_classes, [
    "CAUSAL_LINK_INCOMPLETE",
    "NARRATIVE_BOUNDARY_SATURATION",
    "FINDING_CONFLICT_ID_NOT_GROUNDED_IN_FINDING_EVIDENCE_REFS",
  ]);
  assert.equal(
    result.interpretation_boundary.all_known_deterministic_defects_exhausted,
    false,
  );
  assert.equal(result.interpretation_boundary.retry_authorized, false);
});

test("Adyen conflict-grounding forensic V3 is cumulative, private-safe, and non-inference", () => {
  const prep = JSON.parse(
    readFileSync(
      "calibration/vnext/OROTITAN_GATE18_PHASE_C_C4_ADYEN_CONFLICT_GROUNDING_FORENSIC_V3_PREP_001.json",
      "utf8",
    ),
  );
  const source = readFileSync(
    "scripts/vnext-gate18-phase-c-c4-adyen-conflict-grounding-forensic-v3.ts",
    "utf8",
  );

  assert.equal(
    prep.status,
    "PREPARED_LOCAL_READ_ONLY_NO_INFERENCE",
  );
  assert.equal(prep.diagnostic_normalization.cumulative, true);
  assert.equal(prep.diagnostic_normalization.in_memory_only, true);
  assert.equal(
    prep.diagnostic_normalization.source_artifact_mutation,
    false,
  );
  assert.equal(
    prep.diagnostic_normalization.retroactive_pass_allowed,
    false,
  );
  assert.equal(
    prep.diagnostic_normalization.raw_narrative_text_printed,
    false,
  );
  assert.equal(prep.observed_deterministic_defect_classes.length, 3);
  assert.equal(prep.authority.inference_authorized, false);
  assert.equal(prep.authority.retry_authorized, false);
  assert.equal(prep.authority.auto_repair_authorized, false);

  assert.match(source, /const SATURATION_BOUNDARY = 178;/);
  assert.match(source, /punctuationNormalizedFindingIndexes/);
  assert.match(source, /saturatedNarratives/);
  assert.match(source, /const conflictsById = new Map/);
  assert.match(source, /originalFindingConflictGrounding/);
  assert.match(source, /removedConflictRefs/);
  assert.match(source, /overlapWithFindingRefs/);
  assert.match(source, /assertGate18PhaseBV10Semantics/);
  assert.match(source, /rawNarrativeTextIncludedInConsole: false/);
  assert.match(source, /sourceArtifactMutated: false/);
  assert.match(source, /retroactivePassAllowed: false/);
  assert.match(source, /inferenceExecuted: false/);
  assert.doesNotMatch(source, /requestLoopbackJson/);
  assert.doesNotMatch(source, /\/api\/generate/);
  assert.doesNotMatch(source, /\bfetch\s*\(/);
});
