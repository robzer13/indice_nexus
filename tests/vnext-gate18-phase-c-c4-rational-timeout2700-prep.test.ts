import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

test("RATIONAL timeout600 result is classified as runtime timeout only", () => {
  const result = JSON.parse(
    readFileSync(
      "calibration/vnext/OROTITAN_GATE18_PHASE_C_C4_RATIONAL_OUTPUT1280_TIMEOUT600_RESULT_001.json",
      "utf8",
    ),
  );

  assert.equal(result.status, "FAIL_RUNTIME_TIMEOUT");
  assert.equal(result.execution.wall_clock_ms, 600330);
  assert.equal(
    result.execution.runtime_error,
    "LOOPBACK_HTTP_EXPLICIT_TIMEOUT_600000MS",
  );
  assert.equal(result.classification.runtime_completed, false);
  assert.equal(result.classification.structured_output_observed, false);
  assert.equal(result.classification.schema_quality_assessed, false);
  assert.equal(result.classification.semantic_quality_assessed, false);
  assert.equal(result.classification.model_capability_failure_concluded, false);
  assert.equal(result.authority.retry_authorized, false);
});

test("RATIONAL timeout2700 remediation changes only the client timeout", () => {
  const prep = JSON.parse(
    readFileSync(
      "calibration/vnext/OROTITAN_GATE18_PHASE_C_C4_RATIONAL_TIMEOUT2700_SINGLE_CELL_PREP_001.json",
      "utf8",
    ),
  );
  const source = readFileSync(
    "scripts/vnext-gate18-phase-c-c4-rational-qwen4b-context16384-output1280-timeout2700-loopback-guarded.ts",
    "utf8",
  );

  assert.equal(prep.status, "PREPARED_NOT_AUTHORIZED");
  assert.equal(prep.frozen_inference.max_output_tokens, 1280);
  assert.equal(prep.frozen_inference.context_tokens, 16384);
  assert.equal(prep.frozen_inference.temperature, 0);
  assert.equal(prep.frozen_inference.client_timeout_ms, 2700000);
  assert.equal(
    prep.frozen_inference.request_sha256,
    "0132c86d5372c512b6a7628cd6f06bd8ea79e5348ee1a992b9467cea6c65fe6d",
  );
  assert.equal(prep.constraints.authorized_run_count, 0);
  assert.equal(prep.constraints.inference_authorized, false);
  assert.equal(prep.constraints.automatic_retry_authorized, false);

  assert.match(source, /const MAX_OUTPUT_TOKENS = 1280;/);
  assert.match(source, /const CLIENT_TIMEOUT_MS = 2_700_000;/);
  assert.match(source, /requestLoopbackJson/);
  assert.match(source, /acquireWindowsSystemRequiredGuard/);
  assert.match(source, /ollama", \["ps"\]/);
  assert.match(
    source,
    /0132c86d5372c512b6a7628cd6f06bd8ea79e5348ee1a992b9467cea6c65fe6d/,
  );
  assert.doesNotMatch(source, /\bfetch\s*\(/);
});

test("RATIONAL timeout remediation analysis is explicitly heuristic and non-authorizing", () => {
  const analysis = JSON.parse(
    readFileSync(
      "calibration/vnext/OROTITAN_GATE18_PHASE_C_C4_RATIONAL_TIMEOUT_REMEDIATION_ANALYSIS_001.json",
      "utf8",
    ),
  );

  assert.equal(analysis.status, "COMPLETE_NO_INFERENCE");
  assert.equal(analysis.heuristic.selected_timeout_ms, 2700000);
  assert.equal(analysis.boundaries.timeout2700_adequacy_guaranteed, false);
  assert.equal(analysis.boundaries.runtime_completion_guaranteed, false);
  assert.equal(analysis.boundaries.inference_authorized, false);
  assert.equal(analysis.boundaries.retry_authorized, false);
});
