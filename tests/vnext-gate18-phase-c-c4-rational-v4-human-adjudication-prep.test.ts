import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

test("RATIONAL forensic V4 exhausts the known deterministic defect set", () => {
  const result = JSON.parse(
    readFileSync(
      "calibration/vnext/OROTITAN_GATE18_PHASE_C_C4_RATIONAL_COUNTEREVIDENCE_LINK_FORENSIC_V4_RESULT_001.json",
      "utf8",
    ),
  );

  assert.equal(
    result.status,
    "FORENSIC_COMPLETE_ALL_KNOWN_DETERMINISTIC_DEFECTS_EXHAUSTED",
  );
  assert.equal(result.deterministic_defect_set.class_count, 4);
  assert.deepEqual(result.deterministic_defect_set.classes, [
    "FINDING_CONFLICT_ID_NOT_GROUNDED_IN_FINDING_EVIDENCE_REFS",
    "DIRECTION_ROLE_OVERLAP",
    "NON_ATOMIC_CONTRASTIVE_CLAIM",
    "COUNTEREVIDENCE_LINK_INCOMPLETE",
  ]);
  assert.equal(
    result.deterministic_defect_set.all_known_deterministic_defects_exhausted,
    true,
  );
  assert.equal(
    result.diagnostic_observations.downstream_full_validator_pass,
    true,
  );
  assert.equal(result.disposition_boundary.engineering_failure_concluded, true);
  assert.equal(result.disposition_boundary.human_adjudication_required, true);
  assert.equal(result.disposition_boundary.retry_authorized, false);
  assert.equal(result.disposition_boundary.retroactive_pass_allowed, false);
  assert.equal(
    result.tooling_note.tooling_defect_not_counted_as_model_semantic_defect,
    true,
  );
});

test("RATIONAL human adjudication export preserves original emitted output and executes no inference", () => {
  const prep = JSON.parse(
    readFileSync(
      "calibration/vnext/OROTITAN_GATE18_PHASE_C_C4_RATIONAL_HUMAN_ADJUDICATION_EXPORT_PREP_001.json",
      "utf8",
    ),
  );
  const source = readFileSync(
    "scripts/vnext-gate18-phase-c-c4-rational-human-adjudication-export.ts",
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
  assert.equal(prep.engineering_disposition.deterministic_defect_classes.length, 4);
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
    /C4_RATIONAL_MOAT_EVIDENCE_AUDIT_QWEN4B_CONTEXT16384_OUTPUT1280_TIMEOUT2700_LOOPBACK_GUARDED_001/,
  );
  assert.match(
    source,
    /0132c86d5372c512b6a7628cd6f06bd8ea79e5348ee1a992b9467cea6c65fe6d/,
  );
  assert.match(
    source,
    /3dc89f69ff39bee857595624cfe65b7772a3591fda538de05af0c757f83ba879/,
  );
  assert.match(
    source,
    /70265015372600e619010150a72e72dad5df32973f61c01a79658994ea02c0a5/,
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
  assert.doesNotMatch(source, /Adyen|ADYEN/);
});
