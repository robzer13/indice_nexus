import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

test("STMicro loopback transport remediation prep freezes inference and requires fresh one-shot auth", () => {
  const prep = JSON.parse(
    readFileSync(
      "calibration/vnext/OROTITAN_GATE18_PHASE_C_C4_STMICRO_LOOPBACK_HTTP_TRANSPORT_REMEDIATION_PREP_001.json",
      "utf8",
    ),
  ) as {
    status: string;
    future_authorization_id: string;
    future_attempt_id: string;
    frozen_inference: {
      model_name: string;
      model_digest: string;
      prompt_sha256: string;
      prompt_bytes: number;
      context_tokens: number;
      max_output_tokens: number;
      temperature: number;
      client_timeout_ms: number;
      sleep_guard_required: boolean;
    };
    remediation_delta: Record<string, boolean | string>;
    future_authorization_contract: {
      status: string;
      authorized_run_count: number;
      transport: string;
      exact_attempt_id: string;
      sleep_guard_required: boolean;
      all_frozen_inference_fields_must_match: boolean;
    };
    constraints: {
      authorized_run_count: number;
      inference_authorized: boolean;
      automatic_retry_authorized: boolean;
    };
  };

  assert.equal(prep.status, "PREPARED_NOT_AUTHORIZED");
  assert.equal(
    prep.future_authorization_id,
    "G18-PHASEC-C4-STM-QWEN4B-CONTEXT16384-OUTPUT1024-TIMEOUT420-LOOPBACK-AUTH-001",
  );
  assert.equal(prep.frozen_inference.model_name, "qwen3:4b-instruct");
  assert.equal(
    prep.frozen_inference.model_digest,
    "0edcdef34593eac1aa2be9c7d06c432dcf81945adca5eca2f27662c18f168ba0",
  );
  assert.equal(
    prep.frozen_inference.prompt_sha256,
    "3f06d6e22dbbc88abbbc64fd470811f438a595c71154f7fcb8c004a6d4f5bc91",
  );
  assert.equal(prep.frozen_inference.prompt_bytes, 7558);
  assert.equal(prep.frozen_inference.context_tokens, 16384);
  assert.equal(prep.frozen_inference.max_output_tokens, 1024);
  assert.equal(prep.frozen_inference.temperature, 0);
  assert.equal(prep.frozen_inference.client_timeout_ms, 420000);
  assert.equal(prep.frozen_inference.sleep_guard_required, true);

  assert.equal(
    prep.remediation_delta.transport_from,
    "NODE_GLOBAL_FETCH_UNDICI",
  );
  assert.equal(
    prep.remediation_delta.transport_to,
    "NODE_HTTP_REQUEST_LOOPBACK",
  );
  for (const key of [
    "model_change",
    "model_digest_change",
    "prompt_change",
    "schema_change",
    "packet_change",
    "context_change",
    "max_output_change",
    "temperature_change",
    "explicit_client_timeout_change",
    "sleep_guard_change",
  ]) {
    assert.equal(prep.remediation_delta[key], false);
  }

  assert.equal(
    prep.future_authorization_contract.status,
    "AUTHORIZED_SINGLE_LOCAL_TRANSPORT_REMEDIATION",
  );
  assert.equal(prep.future_authorization_contract.authorized_run_count, 1);
  assert.equal(
    prep.future_authorization_contract.transport,
    "NODE_HTTP_REQUEST_LOOPBACK",
  );
  assert.equal(
    prep.future_authorization_contract.exact_attempt_id,
    prep.future_attempt_id,
  );
  assert.equal(
    prep.future_authorization_contract.all_frozen_inference_fields_must_match,
    true,
  );
  assert.equal(prep.constraints.authorized_run_count, 0);
  assert.equal(prep.constraints.inference_authorized, false);
  assert.equal(prep.constraints.automatic_retry_authorized, false);
});

test("prepared STMicro runner replaces global fetch only and remains authorization-gated", () => {
  const source = readFileSync(
    "scripts/vnext-gate18-phase-c-c4-stmicro-qwen4b-context16384-output1024-timeout420-loopback-guarded.ts",
    "utf8",
  );

  assert.match(
    source,
    /requestLoopbackJson/,
  );
  assert.doesNotMatch(source, /\bfetch\s*\(/);
  assert.match(
    source,
    /artifact\.status !== "AUTHORIZED_SINGLE_LOCAL_TRANSPORT_REMEDIATION"/,
  );
  assert.match(
    source,
    /artifact\.c4_inference\.transport !== "NODE_HTTP_REQUEST_LOOPBACK"/,
  );
  assert.match(
    source,
    /const CONTEXT_TOKENS = 16384;/,
  );
  assert.match(
    source,
    /const MAX_OUTPUT_TOKENS = 1024;/,
  );
  assert.match(
    source,
    /const CLIENT_TIMEOUT_MS = 420_000;/,
  );
  assert.match(
    source,
    /temperature: 0/,
  );
  assert.match(
    source,
    /keep_alive: "0s"/,
  );
  assert.match(
    source,
    /acquireWindowsSystemRequiredGuard\(\)/,
  );
});
