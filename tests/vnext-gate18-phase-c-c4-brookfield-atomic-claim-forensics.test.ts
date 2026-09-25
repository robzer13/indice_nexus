import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

test("Brookfield conflict-grounding forensic V2 reveals third deterministic defect", () => {
  const result = JSON.parse(
    readFileSync(
      "calibration/vnext/OROTITAN_GATE18_PHASE_C_C4_BROOKFIELD_CONFLICT_GROUNDING_FORENSIC_V2_RESULT_001.json",
      "utf8",
    ),
  ) as {
    status: string;
    observations: {
      removed_conflict_ref_count: number;
    };
    downstream_validation: {
      pass: boolean;
      error: string;
    };
    conclusion: {
      third_deterministic_semantic_defect_found: boolean;
      observed_deterministic_defect_classes: string[];
      all_known_deterministic_defects_exhausted: boolean;
      retry_authorized: boolean;
    };
  };

  assert.equal(
    result.status,
    "FORENSIC_COMPLETE_THIRD_SEMANTIC_DEFECT_FOUND",
  );
  assert.equal(result.observations.removed_conflict_ref_count, 1);
  assert.equal(result.downstream_validation.pass, false);
  assert.equal(
    result.downstream_validation.error,
    "VNEXT_GATE18_V10_NON_ATOMIC_CONTRASTIVE_CLAIM",
  );
  assert.equal(
    result.conclusion.third_deterministic_semantic_defect_found,
    true,
  );
  assert.deepEqual(
    result.conclusion.observed_deterministic_defect_classes,
    [
      "COUNTEREVIDENCE_LINK_WITHOUT_IDS",
      "FINDING_CONFLICT_ID_NOT_GROUNDED_IN_FINDING_EVIDENCE_REFS",
      "NON_ATOMIC_CONTRASTIVE_CLAIM",
    ],
  );
  assert.equal(
    result.conclusion.all_known_deterministic_defects_exhausted,
    false,
  );
  assert.equal(result.conclusion.retry_authorized, false);
});

test("Brookfield atomic-claim forensic V3 is cumulative, private-safe, and non-inference", () => {
  const prep = JSON.parse(
    readFileSync(
      "calibration/vnext/OROTITAN_GATE18_PHASE_C_C4_BROOKFIELD_ATOMIC_CLAIM_FORENSIC_V3_PREP_001.json",
      "utf8",
    ),
  ) as {
    status: string;
    diagnostic_normalization: {
      cumulative: boolean;
      raw_claim_text_printed: boolean;
      in_memory_only: boolean;
      source_artifact_mutation: boolean;
      retroactive_pass_allowed: boolean;
    };
    authority: {
      inference_authorized: boolean;
      retry_authorized: boolean;
      auto_repair_authorized: boolean;
      prompt_change_authorized: boolean;
      schema_change_authorized: boolean;
    };
  };

  const source = readFileSync(
    "scripts/vnext-gate18-phase-c-c4-brookfield-atomic-claim-normalization-forensic-v3.ts",
    "utf8",
  );

  assert.equal(
    prep.status,
    "PREPARED_LOCAL_READ_ONLY_NO_INFERENCE",
  );
  assert.equal(prep.diagnostic_normalization.cumulative, true);
  assert.equal(
    prep.diagnostic_normalization.raw_claim_text_printed,
    false,
  );
  assert.equal(prep.diagnostic_normalization.in_memory_only, true);
  assert.equal(
    prep.diagnostic_normalization.source_artifact_mutation,
    false,
  );
  assert.equal(
    prep.diagnostic_normalization.retroactive_pass_allowed,
    false,
  );

  assert.equal(prep.authority.inference_authorized, false);
  assert.equal(prep.authority.retry_authorized, false);
  assert.equal(prep.authority.auto_repair_authorized, false);
  assert.equal(prep.authority.prompt_change_authorized, false);
  assert.equal(prep.authority.schema_change_authorized, false);

  assert.match(source, /finding\.counterevidence_link = null/);
  assert.match(source, /finding\.conflict_ids = finding\.conflict_ids\.filter/);
  assert.match(source, /NON_ATOMIC_PATTERNS/);
  assert.match(source, /finding\.claim = "Diagnostic atomic claim\."/);
  assert.match(source, /rawClaimTextIncludedInConsole: false/);
  assert.match(source, /assertGate18PhaseBV10Semantics/);
  assert.match(source, /retroactivePassAllowed: false/);
  assert.match(source, /sourceArtifactMutated: false/);
  assert.match(source, /inferenceExecuted: false/);
  assert.doesNotMatch(source, /requestLoopbackJson/);
  assert.doesNotMatch(source, /\/api\/generate/);
  assert.doesNotMatch(source, /\bfetch\s*\(/);
});
