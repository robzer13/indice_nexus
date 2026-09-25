import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

test("Brookfield counterevidence-link forensic finds a second deterministic defect", () => {
  const result = JSON.parse(
    readFileSync(
      "calibration/vnext/OROTITAN_GATE18_PHASE_C_C4_BROOKFIELD_COUNTEREVIDENCE_LINK_FORENSIC_V1_RESULT_001.json",
      "utf8",
    ),
  ) as {
    status: string;
    original_observation: {
      violating_finding_indexes: number[];
      violating_finding_count: number;
    };
    downstream_validation: {
      pass: boolean;
      error: string;
    };
    conclusion: {
      counterevidence_link_defect_not_sole_defect: boolean;
      additional_deterministic_semantic_defect_found: boolean;
      additional_defect_class: string;
      retry_authorized: boolean;
      retroactive_pass_allowed: boolean;
    };
  };

  assert.equal(
    result.status,
    "FORENSIC_COMPLETE_ADDITIONAL_SEMANTIC_DEFECT_FOUND",
  );
  assert.deepEqual(
    result.original_observation.violating_finding_indexes,
    [1, 2, 3],
  );
  assert.equal(
    result.original_observation.violating_finding_count,
    3,
  );
  assert.equal(result.downstream_validation.pass, false);
  assert.equal(
    result.downstream_validation.error,
    "VNEXT_GATE18_V10_CONFLICT_NOT_GROUNDED_IN_FINDING_REFS",
  );
  assert.equal(
    result.conclusion.counterevidence_link_defect_not_sole_defect,
    true,
  );
  assert.equal(
    result.conclusion.additional_deterministic_semantic_defect_found,
    true,
  );
  assert.equal(
    result.conclusion.additional_defect_class,
    "FINDING_CONFLICT_ID_NOT_GROUNDED_IN_FINDING_EVIDENCE_REFS",
  );
  assert.equal(result.conclusion.retry_authorized, false);
  assert.equal(result.conclusion.retroactive_pass_allowed, false);
});

test("Brookfield conflict-grounding forensic V2 is cumulative in-memory diagnostic only", () => {
  const prep = JSON.parse(
    readFileSync(
      "calibration/vnext/OROTITAN_GATE18_PHASE_C_C4_BROOKFIELD_CONFLICT_GROUNDING_FORENSIC_V2_PREP_001.json",
      "utf8",
    ),
  ) as {
    status: string;
    diagnostic_normalization: {
      cumulative: boolean;
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
    "scripts/vnext-gate18-phase-c-c4-brookfield-conflict-grounding-normalization-forensic-v2.ts",
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

  assert.equal(prep.authority.inference_authorized, false);
  assert.equal(prep.authority.retry_authorized, false);
  assert.equal(prep.authority.auto_repair_authorized, false);
  assert.equal(prep.authority.prompt_change_authorized, false);
  assert.equal(prep.authority.schema_change_authorized, false);

  assert.match(
    source,
    /finding\.counterevidence_link = null/,
  );
  assert.match(source, /conflict\.evidence_refs/);
  assert.match(source, /assertGate18PhaseBV10Semantics/);
  assert.match(source, /retroactivePassAllowed: false/);
  assert.match(source, /sourceArtifactMutated: false/);
  assert.match(source, /inferenceExecuted: false/);
  assert.doesNotMatch(source, /requestLoopbackJson/);
  assert.doesNotMatch(source, /\/api\/generate/);
  assert.doesNotMatch(source, /\bfetch\s*\(/);
});
