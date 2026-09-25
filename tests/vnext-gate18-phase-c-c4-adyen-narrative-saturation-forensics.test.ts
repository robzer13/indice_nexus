import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

test("Adyen causal-link forensic V1 reveals narrative-boundary saturation", () => {
  const result = JSON.parse(
    readFileSync(
      "calibration/vnext/OROTITAN_GATE18_PHASE_C_C4_ADYEN_CAUSAL_LINK_FORENSIC_V1_RESULT_001.json",
      "utf8",
    ),
  );

  assert.equal(
    result.status,
    "FORENSIC_COMPLETE_ADDITIONAL_SEMANTIC_DEFECT_FOUND",
  );
  assert.equal(result.original_observation.incomplete_causal_link_count, 1);
  assert.equal(
    result.original_observation.incomplete_causal_links[0].finding_index,
    1,
  );
  assert.equal(
    result.original_observation.incomplete_causal_links[0].length,
    180,
  );
  assert.equal(result.downstream_validation.pass, false);
  assert.equal(
    result.downstream_validation.error,
    "VNEXT_GATE18_V10_NARRATIVE_BOUNDARY_SATURATION",
  );
  assert.equal(result.validator_rule.saturation_boundary, 178);
  assert.equal(result.validator_rule.same_field_sequential_violation, true);
  assert.equal(result.conclusion.retry_authorized, false);
  assert.equal(result.conclusion.retroactive_pass_allowed, false);
});

test("Adyen narrative-saturation forensic V2 is cumulative, private-safe, and non-inference", () => {
  const prep = JSON.parse(
    readFileSync(
      "calibration/vnext/OROTITAN_GATE18_PHASE_C_C4_ADYEN_NARRATIVE_SATURATION_FORENSIC_V2_PREP_001.json",
      "utf8",
    ),
  );
  const source = readFileSync(
    "scripts/vnext-gate18-phase-c-c4-adyen-narrative-saturation-forensic-v2.ts",
    "utf8",
  );

  assert.equal(
    prep.status,
    "PREPARED_LOCAL_READ_ONLY_NO_INFERENCE",
  );
  assert.equal(prep.diagnostic_normalization.cumulative, true);
  assert.equal(
    prep.diagnostic_normalization.saturation_boundary,
    178,
  );
  assert.equal(
    prep.diagnostic_normalization.raw_narrative_text_printed,
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

  assert.match(source, /const SATURATION_BOUNDARY = 178;/);
  assert.match(source, /finding\.causal_link = `\$\{trimmed\}\.`/);
  assert.match(source, /saturatedNarratives/);
  assert.match(source, /Diagnostic causal link\./);
  assert.match(source, /Diagnostic atomic claim\./);
  assert.match(source, /rawNarrativeTextIncludedInConsole: false/);
  assert.match(source, /assertGate18PhaseBV10Semantics/);
  assert.match(source, /retroactivePassAllowed: false/);
  assert.match(source, /sourceArtifactMutated: false/);
  assert.match(source, /inferenceExecuted: false/);
  assert.doesNotMatch(source, /requestLoopbackJson/);
  assert.doesNotMatch(source, /\/api\/generate/);
  assert.doesNotMatch(source, /\bfetch\s*\(/);
});
