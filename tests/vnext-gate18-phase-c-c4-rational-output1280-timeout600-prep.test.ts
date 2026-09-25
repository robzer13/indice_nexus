import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

test("RATIONAL static preflight freezes exact request identity", () => {
  const result = JSON.parse(
    readFileSync(
      "calibration/vnext/OROTITAN_GATE18_PHASE_C_C4_RATIONAL_OUTPUT1280_TIMEOUT600_STATIC_PREFLIGHT_RESULT_001.json",
      "utf8",
    ),
  );

  assert.equal(result.status, "PASS_STATIC_REQUEST_MEASURED_NO_INFERENCE");
  assert.equal(result.company.display_name, "RATIONAL AG");
  assert.equal(result.company.evidence_count, 56);
  assert.equal(result.company.conflict_count, 9);
  assert.equal(result.runtime.context_tokens, 16384);
  assert.equal(result.runtime.max_output_tokens, 1280);
  assert.equal(result.runtime.client_timeout_ms, 600000);
  assert.equal(
    result.identities.packet_sha256,
    "3dc89f69ff39bee857595624cfe65b7772a3591fda538de05af0c757f83ba879",
  );
  assert.equal(
    result.identities.prompt_sha256,
    "70265015372600e619010150a72e72dad5df32973f61c01a79658994ea02c0a5",
  );
  assert.equal(
    result.identities.request_sha256,
    "0132c86d5372c512b6a7628cd6f06bd8ea79e5348ee1a992b9467cea6c65fe6d",
  );
  assert.equal(result.interpretation_boundary.inference_authorized, false);
});

test("RATIONAL budget decision selects common C4 settings without guarantee", () => {
  const decision = JSON.parse(
    readFileSync(
      "calibration/vnext/OROTITAN_GATE18_PHASE_C_C4_RATIONAL_OUTPUT_BUDGET_DECISION_001.json",
      "utf8",
    ),
  );

  assert.equal(
    decision.status,
    "SELECT_OUTPUT1280_TIMEOUT600_WITH_EMPIRICAL_HEADROOM_NOT_GUARANTEED",
  );
  assert.equal(decision.decision.max_output_tokens, 1280);
  assert.equal(decision.decision.client_timeout_ms, 600000);
  assert.equal(decision.boundaries.adequacy_guaranteed, false);
  assert.equal(decision.boundaries.truncation_impossible_concluded, false);
  assert.equal(decision.boundaries.inference_authorized, false);
});

test("RATIONAL runner is exact-hash-bound and fresh-auth gated", () => {
  const prep = JSON.parse(
    readFileSync(
      "calibration/vnext/OROTITAN_GATE18_PHASE_C_C4_RATIONAL_SINGLE_CELL_PREP_001.json",
      "utf8",
    ),
  );
  const source = readFileSync(
    "scripts/vnext-gate18-phase-c-c4-rational-qwen4b-context16384-output1280-timeout600-loopback-guarded.ts",
    "utf8",
  );

  assert.equal(prep.status, "PREPARED_NOT_AUTHORIZED");
  assert.equal(prep.constraints.authorized_run_count, 0);
  assert.equal(prep.constraints.inference_authorized, false);
  assert.equal(prep.constraints.automatic_retry_authorized, false);
  assert.equal(
    prep.frozen_inference.request_sha256,
    "0132c86d5372c512b6a7628cd6f06bd8ea79e5348ee1a992b9467cea6c65fe6d",
  );

  assert.match(source, /const MAX_OUTPUT_TOKENS = 1280;/);
  assert.match(source, /const CLIENT_TIMEOUT_MS = 600_000;/);
  assert.match(source, /AUTHORIZED_SINGLE_LOCAL_C4_CELL_INFERENCE/);
  assert.match(source, /RATIONAL AG/);
  assert.match(
    source,
    /0132c86d5372c512b6a7628cd6f06bd8ea79e5348ee1a992b9467cea6c65fe6d/,
  );
  assert.match(source, /requestLoopbackJson/);
  assert.match(source, /acquireWindowsSystemRequiredGuard/);
  assert.doesNotMatch(source, /Adyen|ADYEN/);
  assert.doesNotMatch(source, /\bfetch\s*\(/);
});
