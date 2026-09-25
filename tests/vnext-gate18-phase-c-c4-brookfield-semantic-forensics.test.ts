import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

test("Brookfield output1280 run is a complete schema-valid deterministic semantic failure", () => {
  const result = JSON.parse(
    readFileSync(
      "calibration/vnext/OROTITAN_GATE18_PHASE_C_C4_BROOKFIELD_OUTPUT1280_TIMEOUT600_RESULT_001.json",
      "utf8",
    ),
  ) as {
    status: string;
    execution: {
      done_reason: string;
      eval_count: number;
      runtime_error: null;
      schema_valid: boolean;
      semantic_valid: boolean;
      semantic_error: string;
    };
    output_budget: {
      remaining_token_margin: number;
      fully_consumed: boolean;
      finish_reason_stop: boolean;
    };
    diagnosis: {
      deterministic_semantic_failure: boolean;
      prompt_contract_explicit: boolean;
      auto_repair_allowed: boolean;
      sole_defect_concluded: boolean;
    };
    authority: {
      retry_authorized: boolean;
      auto_repair_authorized: boolean;
    };
  };

  assert.equal(
    result.status,
    "FAIL_DETERMINISTIC_SEMANTIC_CONTRACT",
  );
  assert.equal(result.execution.done_reason, "stop");
  assert.equal(result.execution.eval_count, 1096);
  assert.equal(result.execution.runtime_error, null);
  assert.equal(result.execution.schema_valid, true);
  assert.equal(result.execution.semantic_valid, false);
  assert.equal(
    result.execution.semantic_error,
    "VNEXT_GATE18_V10_COUNTEREVIDENCE_LINK_WITHOUT_IDS",
  );
  assert.equal(result.output_budget.remaining_token_margin, 184);
  assert.equal(result.output_budget.fully_consumed, false);
  assert.equal(result.output_budget.finish_reason_stop, true);
  assert.equal(result.diagnosis.deterministic_semantic_failure, true);
  assert.equal(result.diagnosis.prompt_contract_explicit, true);
  assert.equal(result.diagnosis.auto_repair_allowed, false);
  assert.equal(result.diagnosis.sole_defect_concluded, false);
  assert.equal(result.authority.retry_authorized, false);
  assert.equal(result.authority.auto_repair_authorized, false);
});

test("Brookfield counterevidence-link forensic is in-memory diagnostic only and non-inference", () => {
  const prep = JSON.parse(
    readFileSync(
      "calibration/vnext/OROTITAN_GATE18_PHASE_C_C4_BROOKFIELD_COUNTEREVIDENCE_LINK_FORENSIC_V1_PREP_001.json",
      "utf8",
    ),
  ) as {
    status: string;
    diagnostic_normalization: {
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
    "scripts/vnext-gate18-phase-c-c4-brookfield-counterevidence-link-normalization-forensic-v1.ts",
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
  assert.equal(prep.authority.inference_authorized, false);
  assert.equal(prep.authority.retry_authorized, false);
  assert.equal(prep.authority.auto_repair_authorized, false);
  assert.equal(prep.authority.prompt_change_authorized, false);
  assert.equal(prep.authority.schema_change_authorized, false);

  assert.match(
    source,
    /VNEXT_GATE18_V10_COUNTEREVIDENCE_LINK_WITHOUT_IDS/,
  );
  assert.match(
    source,
    /finding\.counterevidence_link = null/,
  );
  assert.match(source, /structuredClone/);
  assert.match(source, /assertGate18PhaseBV10Semantics/);
  assert.match(source, /retroactivePassAllowed: false/);
  assert.match(source, /sourceArtifactMutated: false/);
  assert.match(source, /inferenceExecuted: false/);
  assert.doesNotMatch(source, /requestLoopbackJson/);
  assert.doesNotMatch(source, /\/api\/generate/);
  assert.doesNotMatch(source, /\bfetch\s*\(/);
});
