import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

test("Adyen forensic V3 exhausts the known deterministic defect set", () => {
  const result = JSON.parse(
    readFileSync(
      "calibration/vnext/OROTITAN_GATE18_PHASE_C_C4_ADYEN_CONFLICT_GROUNDING_FORENSIC_V3_RESULT_001.json",
      "utf8",
    ),
  );

  assert.equal(
    result.status,
    "FORENSIC_COMPLETE_ALL_KNOWN_DETERMINISTIC_DEFECTS_EXHAUSTED",
  );
  assert.equal(
    result.deterministic_defect_set.class_count,
    3,
  );
  assert.deepEqual(result.deterministic_defect_set.classes, [
    "CAUSAL_LINK_INCOMPLETE",
    "NARRATIVE_BOUNDARY_SATURATION",
    "FINDING_CONFLICT_ID_NOT_GROUNDED_IN_FINDING_EVIDENCE_REFS",
  ]);
  assert.equal(
    result.deterministic_defect_set.all_known_deterministic_defects_exhausted,
    true,
  );
  assert.equal(
    result.diagnostic_observations.downstream_full_validator_pass,
    true,
  );
  assert.equal(
    result.disposition_boundary.engineering_failure_concluded,
    true,
  );
  assert.equal(
    result.disposition_boundary.human_adjudication_required,
    true,
  );
  assert.equal(result.disposition_boundary.retry_authorized, false);
  assert.equal(result.disposition_boundary.retroactive_pass_allowed, false);
});

test("Adyen human adjudication export preserves original emitted output and executes no inference", () => {
  const prep = JSON.parse(
    readFileSync(
      "calibration/vnext/OROTITAN_GATE18_PHASE_C_C4_ADYEN_HUMAN_ADJUDICATION_EXPORT_PREP_001.json",
      "utf8",
    ),
  );
  const source = readFileSync(
    "scripts/vnext-gate18-phase-c-c4-adyen-human-adjudication-export.ts",
    "utf8",
  );

  assert.equal(
    prep.status,
    "PREPARED_LOCAL_PRIVATE_NO_INFERENCE_ENGINEERING_FAIL",
  );
  assert.equal(
    prep.engineering_disposition.deterministic_defect_set_exhausted,
    true,
  );
  assert.equal(prep.human_quality_criteria.length, 10);
  assert.equal(
    prep.policy.original_output_must_be_adjudicated_as_emitted,
    true,
  );
  assert.equal(
    prep.policy.diagnostic_normalizations_must_not_be_applied,
    true,
  );
  assert.equal(prep.policy.public_repo_persistence_forbidden, true);
  assert.equal(prep.authority.inference_authorized, false);
  assert.equal(prep.authority.retry_authorized, false);
  assert.equal(prep.authority.auto_repair_authorized, false);

  assert.match(
    source,
    /C4_ADYEN_MOAT_EVIDENCE_AUDIT_QWEN4B_CONTEXT16384_OUTPUT1280_TIMEOUT600_LOOPBACK_GUARDED_001/,
  );
  assert.match(
    source,
    /374163421c4780b70a4cf34ee3443b160d7a5e5e7b7f1701508c9ae5bc0f02e1/,
  );
  assert.match(
    source,
    /89e14b09d58b1305064d76170a15bb31e93a0768737d94f2bfc19d32c39a9b74/,
  );
  assert.match(
    source,
    /ef8f55478017eddfb26d68e652aa4855f9d474dc1a75114e6026a12e17666f0c/,
  );
  assert.match(source, /gate18PhaseBV10OutputSchema\.parse/);
  assert.doesNotMatch(source, /assertGate18PhaseBV10Semantics/);
  assert.match(
    source,
    /diagnosticNormalizationsMustNotBeAppliedToReviewPacket: true/,
  );
  assert.match(source, /generatedContentMustRemainPrivate: true/);
  assert.match(source, /publicRepoPersistenceForbidden: true/);
  assert.doesNotMatch(source, /requestLoopbackJson/);
  assert.doesNotMatch(source, /\/api\/generate/);
  assert.doesNotMatch(source, /\bfetch\s*\(/);
  assert.doesNotMatch(source, /Brookfield|BROOKFIELD/);
});
