import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

test("Qwen3.5 first Constellation result preserves raw JSON failure without semantic reclassification", () => {
  const r = JSON.parse(
    readFileSync(
      "calibration/vnext/OROTITAN_GATE18_PHASE_C_C4_CONSTELLATION_QWEN3_5_4B_V1_1_RESULT_001.json",
      "utf8",
    ),
  );

  assert.equal(r.status, "FAIL_RAW_JSON_INCOMPLETE_FORENSICS_REQUIRED");
  assert.equal(r.execution.done_reason, "stop");
  assert.equal(r.execution.eval_count, 842);
  assert.equal(r.execution.output_token_margin, 182);
  assert.equal(r.execution.runtime_error, null);
  assert.equal(r.execution.schema_valid, false);
  assert.equal(r.execution.schema_error, "Unexpected end of JSON input");
  assert.equal(r.interpretation.output_budget_exhaustion_proven, false);
  assert.equal(r.interpretation.deterministic_v1_1_semantic_quality_not_reached, true);
  assert.equal(r.interpretation.model_capability_failure_concluded, false);
  assert.equal(r.interpretation.forensic_required, true);
});

test("Qwen3.5 raw JSON forensic is read-only and non-inferential", () => {
  const a = JSON.parse(
    readFileSync(
      "calibration/vnext/OROTITAN_GATE18_PHASE_C_C4_CONSTELLATION_QWEN3_5_4B_RAW_JSON_FORENSIC_AUTH_001.json",
      "utf8",
    ),
  );

  assert.equal(a.status, "AUTHORIZED_READ_ONLY_NO_INFERENCE");
  assert.equal(a.scope.read_existing_private_artifact, true);
  assert.equal(a.scope.inspect_raw_json_structure, true);
  assert.equal(a.scope.diagnostic_structural_closure_probe_on_copy, true);
  assert.equal(a.scope.mutate_source_artifact, false);
  assert.equal(a.scope.execute_model_inference, false);
  assert.equal(a.scope.call_ollama, false);
  assert.equal(a.scope.external_network_access, false);
  assert.equal(a.scope.retry_inference, false);
});

test("Qwen3.5 raw JSON forensic runner cannot call model or publish raw content", () => {
  const raw = readFileSync(
    "scripts/vnext-gate18-phase-c-c4-constellation-qwen3-5-4b-raw-json-forensic.ts",
    "utf8",
  );

  assert.match(raw, /EXPECTED_DONE_REASON = "stop"/);
  assert.match(raw, /EXPECTED_EVAL_COUNT = 842/);
  assert.match(raw, /EXPECTED_MAX_OUTPUT_TOKENS = 1024/);
  assert.match(raw, /Unexpected end of JSON input/);
  assert.match(raw, /MODEL_STOPPED_WITH_INCOMPLETE_JSON_BEFORE_OUTPUT_BUDGET_EXHAUSTION/);
  assert.match(raw, /structuralClosureProbe/);
  assert.match(raw, /missingRequiredSections/);
  assert.match(raw, /rawOutputPublished: false/);
  assert.match(raw, /modelInferenceExecuted: false/);
  assert.match(raw, /ollamaApiCalled: false/);
  assert.doesNotMatch(raw, /\/api\/generate|\/api\/chat|11434/);
  assert.doesNotMatch(raw, /fetch\(/);
  assert.doesNotMatch(raw, /rawOutput:\s*\{[^}]*rawText/s);
});
