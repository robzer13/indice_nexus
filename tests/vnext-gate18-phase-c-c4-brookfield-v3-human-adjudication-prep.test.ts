import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

test("Brookfield V3 forensic exhausts the observed deterministic defect set without retroactive pass", () => {
  const result = JSON.parse(
    readFileSync(
      "calibration/vnext/OROTITAN_GATE18_PHASE_C_C4_BROOKFIELD_ATOMIC_CLAIM_FORENSIC_V3_RESULT_001.json",
      "utf8",
    ),
  ) as {
    status: string;
    diagnostic_observations: {
      downstream_full_validator_pass: boolean;
      downstream_full_validator_error: null;
    };
    deterministic_defect_set: {
      class_count: number;
      classes: string[];
      all_known_deterministic_defects_exhausted: boolean;
    };
    disposition_boundary: {
      original_run_status: string;
      original_run_status_changed: boolean;
      retroactive_pass_allowed: boolean;
      engineering_failure_concluded: boolean;
      global_model_capability_failure_concluded: boolean;
      human_adjudication_required: boolean;
      original_output_must_be_adjudicated_as_emitted: boolean;
      auto_repair_forbidden: boolean;
      retry_authorized: boolean;
    };
  };

  assert.equal(
    result.status,
    "FORENSIC_COMPLETE_ALL_KNOWN_DETERMINISTIC_DEFECTS_EXHAUSTED",
  );
  assert.equal(
    result.diagnostic_observations.downstream_full_validator_pass,
    true,
  );
  assert.equal(
    result.diagnostic_observations.downstream_full_validator_error,
    null,
  );
  assert.equal(result.deterministic_defect_set.class_count, 3);
  assert.deepEqual(result.deterministic_defect_set.classes, [
    "COUNTEREVIDENCE_LINK_WITHOUT_IDS",
    "FINDING_CONFLICT_ID_NOT_GROUNDED_IN_FINDING_EVIDENCE_REFS",
    "NON_ATOMIC_CONTRASTIVE_CLAIM",
  ]);
  assert.equal(
    result.deterministic_defect_set.all_known_deterministic_defects_exhausted,
    true,
  );
  assert.equal(
    result.disposition_boundary.original_run_status,
    "FAIL_DETERMINISTIC_SEMANTIC_CONTRACT",
  );
  assert.equal(
    result.disposition_boundary.original_run_status_changed,
    false,
  );
  assert.equal(
    result.disposition_boundary.retroactive_pass_allowed,
    false,
  );
  assert.equal(
    result.disposition_boundary.engineering_failure_concluded,
    true,
  );
  assert.equal(
    result.disposition_boundary.global_model_capability_failure_concluded,
    false,
  );
  assert.equal(
    result.disposition_boundary.human_adjudication_required,
    true,
  );
  assert.equal(
    result.disposition_boundary.original_output_must_be_adjudicated_as_emitted,
    true,
  );
  assert.equal(
    result.disposition_boundary.auto_repair_forbidden,
    true,
  );
  assert.equal(result.disposition_boundary.retry_authorized, false);
});

test("Brookfield human adjudication export preserves original semantic-fail output and performs no inference", () => {
  const prep = JSON.parse(
    readFileSync(
      "calibration/vnext/OROTITAN_GATE18_PHASE_C_C4_BROOKFIELD_HUMAN_ADJUDICATION_EXPORT_PREP_001.json",
      "utf8",
    ),
  ) as {
    status: string;
    source_engineering_status: string;
    deterministic_defect_classes: string[];
    output_policy: {
      public_repo_persistence: boolean;
      generated_content_publication: boolean;
      diagnostic_normalizations_applied: boolean;
      original_output_preserved: boolean;
    };
    authority: {
      inference_authorized: boolean;
      retry_authorized: boolean;
      auto_repair_authorized: boolean;
      final_moat_conclusion_authorized: boolean;
      publication_authority: boolean;
    };
  };

  const source = readFileSync(
    "scripts/vnext-gate18-phase-c-c4-brookfield-human-adjudication-export.ts",
    "utf8",
  );

  assert.equal(
    prep.status,
    "PREPARED_LOCAL_PRIVATE_NO_INFERENCE_ENGINEERING_FAIL",
  );
  assert.equal(
    prep.source_engineering_status,
    "FAIL_DETERMINISTIC_SEMANTIC_CONTRACT",
  );
  assert.equal(prep.deterministic_defect_classes.length, 3);
  assert.equal(prep.output_policy.public_repo_persistence, false);
  assert.equal(
    prep.output_policy.generated_content_publication,
    false,
  );
  assert.equal(
    prep.output_policy.diagnostic_normalizations_applied,
    false,
  );
  assert.equal(prep.output_policy.original_output_preserved, true);
  assert.equal(prep.authority.inference_authorized, false);
  assert.equal(prep.authority.retry_authorized, false);
  assert.equal(prep.authority.auto_repair_authorized, false);
  assert.equal(
    prep.authority.final_moat_conclusion_authorized,
    false,
  );
  assert.equal(prep.authority.publication_authority, false);

  assert.match(source, /Brookfield Corporation/);
  assert.match(source, /OUTPUT1280_TIMEOUT600_LOOPBACK_GUARDED_001/);
  assert.match(
    source,
    /cc51153ec6bdc8041f22800aea2b6aa6418a27837589547b19db88745bc0ff8f/,
  );
  assert.match(
    source,
    /READY_FOR_HUMAN_ADJUDICATION_ENGINEERING_FAIL/,
  );
  assert.match(
    source,
    /PASS_REVIEW_PACKET_READY_ENGINEERING_FAIL/,
  );
  assert.match(source, /originalOutputMustBeAdjudicatedAsEmitted: true/);
  assert.match(
    source,
    /diagnosticNormalizationsMustNotBeAppliedToReviewPacket: true/,
  );
  assert.doesNotMatch(source, /Diagnostic atomic claim/);
  assert.doesNotMatch(source, /counterevidence_link = null/);
  assert.doesNotMatch(source, /requestLoopbackJson/);
  assert.doesNotMatch(source, /\/api\/generate/);
  assert.doesNotMatch(source, /\bfetch\s*\(/);
});
