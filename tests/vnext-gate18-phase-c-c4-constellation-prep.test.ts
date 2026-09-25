import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

test("Constellation C4 prep freezes exact static identities and requires fresh one-shot auth", () => {
  const prep = JSON.parse(
    readFileSync(
      "calibration/vnext/OROTITAN_GATE18_PHASE_C_C4_CONSTELLATION_SINGLE_CELL_PREP_001.json",
      "utf8",
    ),
  ) as {
    status: string;
    matrix_cell_id: string;
    future_authorization_id: string;
    future_attempt_id: string;
    frozen_inference: {
      company: string;
      evidence_count: number;
      conflict_count: number;
      packet_sha256: string;
      prompt_sha256: string;
      prompt_bytes: number;
      request_sha256: string;
      request_bytes: number;
      context_tokens: number;
      max_output_tokens: number;
      temperature: number;
      client_timeout_ms: number;
      transport: string;
      sleep_guard_required: boolean;
    };
    future_authorization_contract: {
      status: string;
      authorized_run_count: number;
      exact_company: string;
      exact_attempt_id: string;
      exact_transport: string;
      all_frozen_inference_fields_must_match: boolean;
    };
    constraints: {
      authorized_run_count: number;
      inference_authorized: boolean;
      automatic_retry_authorized: boolean;
      max_output_change_authorized: boolean;
    };
  };

  assert.equal(prep.status, "PREPARED_NOT_AUTHORIZED");
  assert.equal(
    prep.matrix_cell_id,
    "C4_CONSTELLATION_MOAT_EVIDENCE_AUDIT_QWEN4B_CONTEXT16384_001",
  );
  assert.equal(prep.frozen_inference.company, "Constellation Software");
  assert.equal(prep.frozen_inference.evidence_count, 11);
  assert.equal(prep.frozen_inference.conflict_count, 1);
  assert.equal(
    prep.frozen_inference.packet_sha256,
    "9a47bcf15d0c90da55349cbb3fbb2da8e9645b859a535dc8909501e89a20b6d8",
  );
  assert.equal(
    prep.frozen_inference.prompt_sha256,
    "0891fa34d02a47c83e8342d5da5e653a3566d90f1c63c1bfc662f4e6097c10b8",
  );
  assert.equal(prep.frozen_inference.prompt_bytes, 9041);
  assert.equal(
    prep.frozen_inference.request_sha256,
    "145ad3b20eacfe243c0cbbc8089e729b6cb71e0cc2c9a19dcee09e133d347441",
  );
  assert.equal(prep.frozen_inference.request_bytes, 17666);
  assert.equal(prep.frozen_inference.context_tokens, 16384);
  assert.equal(prep.frozen_inference.max_output_tokens, 1024);
  assert.equal(prep.frozen_inference.temperature, 0);
  assert.equal(prep.frozen_inference.client_timeout_ms, 420000);
  assert.equal(
    prep.frozen_inference.transport,
    "NODE_HTTP_REQUEST_LOOPBACK",
  );
  assert.equal(prep.frozen_inference.sleep_guard_required, true);

  assert.equal(
    prep.future_authorization_contract.status,
    "AUTHORIZED_SINGLE_LOCAL_C4_CELL_INFERENCE",
  );
  assert.equal(
    prep.future_authorization_contract.authorized_run_count,
    1,
  );
  assert.equal(
    prep.future_authorization_contract.exact_company,
    "Constellation Software",
  );
  assert.equal(
    prep.future_authorization_contract.exact_attempt_id,
    prep.future_attempt_id,
  );
  assert.equal(
    prep.future_authorization_contract.exact_transport,
    "NODE_HTTP_REQUEST_LOOPBACK",
  );
  assert.equal(
    prep.future_authorization_contract.all_frozen_inference_fields_must_match,
    true,
  );

  assert.equal(prep.constraints.authorized_run_count, 0);
  assert.equal(prep.constraints.inference_authorized, false);
  assert.equal(prep.constraints.automatic_retry_authorized, false);
  assert.equal(prep.constraints.max_output_change_authorized, false);
});

test("Constellation runner is loopback-only, authorization-gated, and identity-bound", () => {
  const source = readFileSync(
    "scripts/vnext-gate18-phase-c-c4-constellation-qwen4b-context16384-output1024-timeout420-loopback-guarded.ts",
    "utf8",
  );

  assert.match(source, /Constellation Software/);
  assert.doesNotMatch(source, /STMicroelectronics/);
  assert.match(
    source,
    /AUTHORIZED_SINGLE_LOCAL_C4_CELL_INFERENCE/,
  );
  assert.match(source, /EXPECTED_PACKET_SHA256/);
  assert.match(source, /EXPECTED_PROMPT_SHA256/);
  assert.match(source, /EXPECTED_REQUEST_SHA256/);
  assert.match(source, /requestLoopbackJson/);
  assert.doesNotMatch(source, /\bfetch\s*\(/);
  assert.match(source, /const CONTEXT_TOKENS = 16384;/);
  assert.match(source, /const MAX_OUTPUT_TOKENS = 1024;/);
  assert.match(source, /const CLIENT_TIMEOUT_MS = 420_000;/);
  assert.match(source, /temperature: 0/);
  assert.match(source, /keep_alive: "0s"/);
  assert.match(source, /acquireWindowsSystemRequiredGuard\(\)/);
});
