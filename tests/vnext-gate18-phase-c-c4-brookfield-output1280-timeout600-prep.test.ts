import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

test("Brookfield output1280 timeout600 preflight result freezes exact request identity", () => {
  const result = JSON.parse(
    readFileSync(
      "calibration/vnext/OROTITAN_GATE18_PHASE_C_C4_BROOKFIELD_OUTPUT1280_TIMEOUT600_STATIC_PREFLIGHT_RESULT_001.json",
      "utf8",
    ),
  ) as {
    status: string;
    runtime: {
      context_tokens: number;
      max_output_tokens: number;
      client_timeout_ms: number;
      transport: string;
    };
    identities: {
      packet_sha256: string;
      prompt_sha256: string;
      prompt_bytes: number;
      request_sha256: string;
      request_bytes: number;
    };
    interpretation_boundary: {
      inference_authorized: boolean;
      retry_authorized: boolean;
    };
  };

  assert.equal(
    result.status,
    "PASS_STATIC_REQUEST_MEASURED_NO_INFERENCE",
  );
  assert.equal(result.runtime.context_tokens, 16384);
  assert.equal(result.runtime.max_output_tokens, 1280);
  assert.equal(result.runtime.client_timeout_ms, 600000);
  assert.equal(
    result.runtime.transport,
    "NODE_HTTP_REQUEST_LOOPBACK",
  );
  assert.equal(
    result.identities.packet_sha256,
    "eb2d779b95fa5f207e91fd3485ece39b2bd101c2589eaaf8c48481f750ef75b3",
  );
  assert.equal(
    result.identities.prompt_sha256,
    "516496bf4dc9bcdce47897c056282eb2bda63b014cd63c1b76e3992b1c0cdad2",
  );
  assert.equal(result.identities.prompt_bytes, 9643);
  assert.equal(
    result.identities.request_sha256,
    "cc51153ec6bdc8041f22800aea2b6aa6418a27837589547b19db88745bc0ff8f",
  );
  assert.equal(result.identities.request_bytes, 18096);
  assert.equal(
    result.interpretation_boundary.inference_authorized,
    false,
  );
  assert.equal(
    result.interpretation_boundary.retry_authorized,
    false,
  );
});

test("Brookfield output1280 timeout600 prep requires fresh one-shot authorization", () => {
  const prep = JSON.parse(
    readFileSync(
      "calibration/vnext/OROTITAN_GATE18_PHASE_C_C4_BROOKFIELD_OUTPUT1280_TIMEOUT600_REMEDIATION_PREP_001.json",
      "utf8",
    ),
  ) as {
    status: string;
    frozen_inference: {
      request_sha256: string;
      max_output_tokens: number;
      client_timeout_ms: number;
    };
    future_authorization_contract: {
      status: string;
      authorized_run_count: number;
      exact_max_output_tokens: number;
      exact_client_timeout_ms: number;
    };
    constraints: {
      authorized_run_count: number;
      inference_authorized: boolean;
      automatic_retry_authorized: boolean;
      max_output_change_execution_authorized: boolean;
      timeout_change_execution_authorized: boolean;
    };
  };

  assert.equal(prep.status, "PREPARED_NOT_AUTHORIZED");
  assert.equal(
    prep.frozen_inference.request_sha256,
    "cc51153ec6bdc8041f22800aea2b6aa6418a27837589547b19db88745bc0ff8f",
  );
  assert.equal(prep.frozen_inference.max_output_tokens, 1280);
  assert.equal(prep.frozen_inference.client_timeout_ms, 600000);
  assert.equal(
    prep.future_authorization_contract.status,
    "AUTHORIZED_SINGLE_LOCAL_C4_OUTPUT_BUDGET_REMEDIATION",
  );
  assert.equal(
    prep.future_authorization_contract.authorized_run_count,
    1,
  );
  assert.equal(
    prep.future_authorization_contract.exact_max_output_tokens,
    1280,
  );
  assert.equal(
    prep.future_authorization_contract.exact_client_timeout_ms,
    600000,
  );
  assert.equal(prep.constraints.authorized_run_count, 0);
  assert.equal(prep.constraints.inference_authorized, false);
  assert.equal(prep.constraints.automatic_retry_authorized, false);
  assert.equal(
    prep.constraints.max_output_change_execution_authorized,
    false,
  );
  assert.equal(
    prep.constraints.timeout_change_execution_authorized,
    false,
  );
});

test("Brookfield output1280 timeout600 runner is hash-bound, loopback-only, and auth-gated", () => {
  const source = readFileSync(
    "scripts/vnext-gate18-phase-c-c4-brookfield-qwen4b-context16384-output1280-timeout600-loopback-guarded.ts",
    "utf8",
  );

  assert.match(source, /const MAX_OUTPUT_TOKENS = 1280;/);
  assert.match(source, /const CLIENT_TIMEOUT_MS = 600_000;/);
  assert.doesNotMatch(source, /OUTPUT1024|TIMEOUT480/);
  assert.match(
    source,
    /AUTHORIZED_SINGLE_LOCAL_C4_OUTPUT_BUDGET_REMEDIATION/,
  );
  assert.match(
    source,
    /cc51153ec6bdc8041f22800aea2b6aa6418a27837589547b19db88745bc0ff8f/,
  );
  assert.match(source, /EXPECTED_PACKET_SHA256/);
  assert.match(source, /EXPECTED_PROMPT_SHA256/);
  assert.match(source, /EXPECTED_REQUEST_SHA256/);
  assert.match(source, /requestLoopbackJson/);
  assert.doesNotMatch(source, /\bfetch\s*\(/);
  assert.match(source, /acquireWindowsSystemRequiredGuard\(\)/);
});
