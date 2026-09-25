import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

test("Adyen output1280 run is a complete schema-valid deterministic semantic failure", () => {
  const result = JSON.parse(
    readFileSync(
      "calibration/vnext/OROTITAN_GATE18_PHASE_C_C4_ADYEN_OUTPUT1280_TIMEOUT600_RESULT_001.json",
      "utf8",
    ),
  );

  assert.equal(
    result.status,
    "FAIL_DETERMINISTIC_SEMANTIC_CONTRACT",
  );
  assert.equal(result.execution.done_reason, "stop");
  assert.equal(result.execution.eval_count, 1109);
  assert.equal(result.execution.runtime_error, null);
  assert.equal(result.execution.schema_valid, true);
  assert.equal(result.execution.semantic_valid, false);
  assert.equal(
    result.execution.semantic_error,
    "VNEXT_GATE18_V10_CAUSAL_LINK_INCOMPLETE",
  );
  assert.equal(result.output_budget.remaining_token_margin, 171);
  assert.equal(result.output_budget.fully_consumed, false);
  assert.equal(result.output_budget.finish_reason_stop, true);
  assert.equal(result.authority.retry_authorized, false);
  assert.equal(result.authority.auto_repair_authorized, false);
});

test("Adyen causal-link forensic is in-memory diagnostic only and non-inference", () => {
  const prep = JSON.parse(
    readFileSync(
      "calibration/vnext/OROTITAN_GATE18_PHASE_C_C4_ADYEN_CAUSAL_LINK_FORENSIC_V1_PREP_001.json",
      "utf8",
    ),
  );
  const source = readFileSync(
    "scripts/vnext-gate18-phase-c-c4-adyen-causal-link-punctuation-forensic-v1.ts",
    "utf8",
  );

  assert.equal(
    prep.status,
    "PREPARED_LOCAL_READ_ONLY_NO_INFERENCE",
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
  assert.equal(
    prep.diagnostic_normalization.raw_causal_link_text_printed,
    false,
  );

  assert.equal(prep.authority.inference_authorized, false);
  assert.equal(prep.authority.retry_authorized, false);
  assert.equal(prep.authority.auto_repair_authorized, false);
  assert.equal(prep.authority.prompt_change_authorized, false);
  assert.equal(prep.authority.schema_change_authorized, false);

  assert.match(
    source,
    /VNEXT_GATE18_V10_CAUSAL_LINK_INCOMPLETE/,
  );
  assert.match(source, /finding\.causal_link = `\$\{trimmed\}\.`/);
  assert.match(source, /structuredClone/);
  assert.match(source, /assertGate18PhaseBV10Semantics/);
  assert.match(source, /rawCausalLinkTextIncludedInConsole: false/);
  assert.match(source, /retroactivePassAllowed: false/);
  assert.match(source, /sourceArtifactMutated: false/);
  assert.match(source, /inferenceExecuted: false/);
  assert.doesNotMatch(source, /requestLoopbackJson/);
  assert.doesNotMatch(source, /\/api\/generate/);
  assert.doesNotMatch(source, /\bfetch\s*\(/);
});
