import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

test("RATIONAL atomic forensic V3 is recorded as tooling-inconclusive, not as model evidence", () => {
  const result = JSON.parse(
    readFileSync(
      "calibration/vnext/OROTITAN_GATE18_PHASE_C_C4_RATIONAL_ATOMIC_CLAIM_FORENSIC_V3_RESULT_001.json",
      "utf8",
    ),
  );

  assert.equal(result.status, "FORENSIC_TOOLING_DEFECT_INCONCLUSIVE");
  assert.equal(
    result.tooling_defect.code,
    "FORENSIC_REGEX_WORD_BOUNDARY_DOUBLE_ESCAPED",
  );
  assert.equal(
    result.interpretation_boundary.third_deterministic_defect_still_established_by_v2,
    true,
  );
  assert.equal(
    result.interpretation_boundary.offending_finding_identified,
    false,
  );
  assert.equal(
    result.interpretation_boundary.all_known_deterministic_defects_exhausted,
    false,
  );
  assert.equal(result.interpretation_boundary.inference_executed, false);
});

test("RATIONAL corrected V3R uses validator-parity word-boundary regexes and self-checks them", () => {
  const prep = JSON.parse(
    readFileSync(
      "calibration/vnext/OROTITAN_GATE18_PHASE_C_C4_RATIONAL_ATOMIC_CLAIM_FORENSIC_V3R_PREP_001.json",
      "utf8",
    ),
  );
  const source = readFileSync(
    "scripts/vnext-gate18-phase-c-c4-rational-atomic-claim-forensic-v3r.ts",
    "utf8",
  );

  assert.equal(
    prep.status,
    "PREPARED_LOCAL_READ_ONLY_NO_INFERENCE_TOOLING_CORRECTION",
  );
  assert.equal(prep.diagnostic_normalization.pattern_self_check_required, true);
  assert.equal(prep.diagnostic_normalization.in_memory_only, true);
  assert.equal(prep.diagnostic_normalization.source_artifact_mutation, false);
  assert.equal(prep.authority.inference_authorized, false);
  assert.equal(prep.authority.retry_authorized, false);

  assert.match(source, /\{ label: "but", pattern: \/\\bbut\\b\/i \}/);
  assert.match(
    source,
    /\{ label: "coexist", pattern: \/\\bcoexist\(\?:s\|ed\|ing\)\?\\b\/i \}/,
  );
  assert.doesNotMatch(source, /pattern: \/\\\\bbut\\\\b\/i/);
  assert.match(source, /function assertPatternParitySelfCheck\(\): void/);
  assert.match(source, /assertPatternParitySelfCheck\(\);/);
  assert.match(source, /removedConflictRefs/);
  assert.match(source, /removedSupportOverlapRefs/);
  assert.match(source, /nonAtomicFindings/);
  assert.match(source, /assertGate18PhaseBV10Semantics/);
  assert.match(source, /rawClaimTextIncludedInConsole: false/);
  assert.match(source, /sourceArtifactMutated: false/);
  assert.doesNotMatch(source, /requestLoopbackJson/);
  assert.doesNotMatch(source, /\/api\/generate/);
  assert.doesNotMatch(source, /\bfetch\s*\(/);
});
