import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

test("Adyen output1280 timeout600 static preflight result freezes exact request identity", () => {
  const result = JSON.parse(
    readFileSync(
      "calibration/vnext/OROTITAN_GATE18_PHASE_C_C4_ADYEN_OUTPUT1280_TIMEOUT600_STATIC_PREFLIGHT_RESULT_001.json",
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

  assert.equal(result.status, "PASS_STATIC_REQUEST_MEASURED_NO_INFERENCE");
  assert.equal(result.runtime.context_tokens, 16384);
  assert.equal(result.runtime.max_output_tokens, 1280);
  assert.equal(result.runtime.client_timeout_ms, 600000);
  assert.equal(result.runtime.transport, "NODE_HTTP_REQUEST_LOOPBACK");
  assert.equal(
    result.identities.packet_sha256,
    "89e14b09d58b1305064d76170a15bb31e93a0768737d94f2bfc19d32c39a9b74",
  );
  assert.equal(
    result.identities.prompt_sha256,
    "ef8f55478017eddfb26d68e652aa4855f9d474dc1a75114e6026a12e17666f0c",
  );
  assert.equal(result.identities.prompt_bytes, 11671);
  assert.equal(
    result.identities.request_sha256,
    "374163421c4780b70a4cf34ee3443b160d7a5e5e7b7f1701508c9ae5bc0f02e1",
  );
  assert.equal(result.identities.request_bytes, 20584);
  assert.equal(result.interpretation_boundary.inference_authorized, false);
  assert.equal(result.interpretation_boundary.retry_authorized, false);
});

test("Adyen single-cell prep requires fresh one-shot authorization", () => {
  const prep = JSON.parse(
    readFileSync(
      "calibration/vnext/OROTITAN_GATE18_PHASE_C_C4_ADYEN_SINGLE_CELL_PREP_001.json",
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
      exact_company: string;
    };
    constraints: {
      authorized_run_count: number;
      inference_authorized: boolean;
      automatic_retry_authorized: boolean;
    };
  };

  assert.equal(prep.status, "PREPARED_NOT_AUTHORIZED");
  assert.equal(
    prep.frozen_inference.request_sha256,
    "374163421c4780b70a4cf34ee3443b160d7a5e5e7b7f1701508c9ae5bc0f02e1",
  );
  assert.equal(prep.frozen_inference.max_output_tokens, 1280);
  assert.equal(prep.frozen_inference.client_timeout_ms, 600000);
  assert.equal(
    prep.future_authorization_contract.status,
    "AUTHORIZED_SINGLE_LOCAL_C4_CELL_INFERENCE",
  );
  assert.equal(prep.future_authorization_contract.authorized_run_count, 1);
  assert.equal(prep.future_authorization_contract.exact_company, "Adyen");
  assert.equal(prep.constraints.authorized_run_count, 0);
  assert.equal(prep.constraints.inference_authorized, false);
  assert.equal(prep.constraints.automatic_retry_authorized, false);
});

test("Adyen output1280 timeout600 runner is exact-hash-bound, loopback-only, and auth-gated", () => {
  const source = readFileSync(
    "scripts/vnext-gate18-phase-c-c4-adyen-qwen4b-context16384-output1280-timeout600-loopback-guarded.ts",
    "utf8",
  );

  assert.match(source, /const MAX_OUTPUT_TOKENS = 1280;/);
  assert.match(source, /const CLIENT_TIMEOUT_MS = 600_000;/);
  assert.match(source, /AUTHORIZED_SINGLE_LOCAL_C4_CELL_INFERENCE/);
  assert.match(
    source,
    /374163421c4780b70a4cf34ee3443b160d7a5e5e7b7f1701508c9ae5bc0f02e1/,
  );
  assert.match(
    source,
    /89e14b09d58b1305064d76170a15bb31e93a0768737d94f2bfc19d32c39a9b74/,
  );
  assert.match(
    source,
    /ef8f55478017eddfb26d68e652aa4855f9d474dc1a75114e6026a12e17666f0c/,
  );
  assert.match(source, /requestLoopbackJson/);
  assert.match(source, /acquireWindowsSystemRequiredGuard/);
  assert.doesNotMatch(source, /Brookfield|BROOKFIELD/);
  assert.doesNotMatch(source, /\bfetch\s*\(/);
});
