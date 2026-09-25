import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

test("Constellation timeout420 forensics confirms healthy slow generation and justifies bounded timeout480 prep", () => {
  const result = JSON.parse(
    readFileSync(
      "calibration/vnext/OROTITAN_GATE18_PHASE_C_C4_CONSTELLATION_TIMEOUT420_LOG_FORENSICS_V1_RESULT_001.json",
      "utf8",
    ),
  ) as {
    status: string;
    observations: {
      latest_observed_n_gen: number;
      configured_max_output_tokens: number;
      observed_average_generation_rate_tps: number;
      generation_progress_continuous_to_terminal_boundary: boolean;
      oom_observed: boolean;
      cuda_failure_observed: boolean;
      runner_exit_observed: boolean;
      server_stall_observed: boolean;
    };
    diagnosis: {
      classification: string;
      timeout_420_inadequate_for_this_cell: boolean;
      estimate_only: {
        projected_additional_generation_seconds: number;
        proposed_timeout_seconds: number;
      };
    };
    authority: {
      inference_authorized: boolean;
      retry_authorized: boolean;
      timeout_change_execution_authorized: boolean;
    };
  };

  assert.equal(
    result.status,
    "PASS_HEALTHY_SLOW_GENERATION_EXPLICIT_TIMEOUT_INADEQUACY_CONFIRMED",
  );
  assert.equal(result.observations.latest_observed_n_gen, 959);
  assert.equal(result.observations.configured_max_output_tokens, 1024);
  assert.equal(
    result.observations.observed_average_generation_rate_tps,
    2.5,
  );
  assert.equal(
    result.observations.generation_progress_continuous_to_terminal_boundary,
    true,
  );
  assert.equal(result.observations.oom_observed, false);
  assert.equal(result.observations.cuda_failure_observed, false);
  assert.equal(result.observations.runner_exit_observed, false);
  assert.equal(result.observations.server_stall_observed, false);
  assert.equal(
    result.diagnosis.timeout_420_inadequate_for_this_cell,
    true,
  );
  assert.equal(
    result.diagnosis.estimate_only.projected_additional_generation_seconds,
    26,
  );
  assert.equal(
    result.diagnosis.estimate_only.proposed_timeout_seconds,
    480,
  );
  assert.equal(result.authority.inference_authorized, false);
  assert.equal(result.authority.retry_authorized, false);
  assert.equal(
    result.authority.timeout_change_execution_authorized,
    false,
  );
});

test("Constellation timeout480 prep changes timeout only and requires fresh one-shot auth", () => {
  const prep = JSON.parse(
    readFileSync(
      "calibration/vnext/OROTITAN_GATE18_PHASE_C_C4_CONSTELLATION_TIMEOUT480_REMEDIATION_PREP_001.json",
      "utf8",
    ),
  ) as {
    status: string;
    frozen_inference: {
      packet_sha256: string;
      prompt_sha256: string;
      request_sha256: string;
      context_tokens: number;
      max_output_tokens: number;
      temperature: number;
      client_timeout_ms: number;
      transport: string;
    };
    remediation_delta: Record<string, boolean | number>;
    future_authorization_contract: {
      status: string;
      authorized_run_count: number;
      exact_client_timeout_ms: number;
    };
    constraints: {
      authorized_run_count: number;
      inference_authorized: boolean;
      automatic_retry_authorized: boolean;
      timeout_change_execution_authorized: boolean;
    };
  };

  assert.equal(prep.status, "PREPARED_NOT_AUTHORIZED");
  assert.equal(
    prep.frozen_inference.packet_sha256,
    "9a47bcf15d0c90da55349cbb3fbb2da8e9645b859a535dc8909501e89a20b6d8",
  );
  assert.equal(
    prep.frozen_inference.prompt_sha256,
    "0891fa34d02a47c83e8342d5da5e653a3566d90f1c63c1bfc662f4e6097c10b8",
  );
  assert.equal(
    prep.frozen_inference.request_sha256,
    "145ad3b20eacfe243c0cbbc8089e729b6cb71e0cc2c9a19dcee09e133d347441",
  );
  assert.equal(prep.frozen_inference.context_tokens, 16384);
  assert.equal(prep.frozen_inference.max_output_tokens, 1024);
  assert.equal(prep.frozen_inference.temperature, 0);
  assert.equal(prep.frozen_inference.client_timeout_ms, 480000);
  assert.equal(
    prep.frozen_inference.transport,
    "NODE_HTTP_REQUEST_LOOPBACK",
  );
  assert.equal(prep.remediation_delta.client_timeout_ms_from, 420000);
  assert.equal(prep.remediation_delta.client_timeout_ms_to, 480000);
  for (const key of [
    "model_change",
    "model_digest_change",
    "prompt_change",
    "schema_change",
    "packet_change",
    "request_body_change",
    "context_change",
    "max_output_change",
    "temperature_change",
    "transport_change",
    "sleep_guard_change",
  ]) {
    assert.equal(prep.remediation_delta[key], false);
  }
  assert.equal(
    prep.future_authorization_contract.status,
    "AUTHORIZED_SINGLE_LOCAL_C4_TIMEOUT_REMEDIATION",
  );
  assert.equal(
    prep.future_authorization_contract.authorized_run_count,
    1,
  );
  assert.equal(
    prep.future_authorization_contract.exact_client_timeout_ms,
    480000,
  );
  assert.equal(prep.constraints.authorized_run_count, 0);
  assert.equal(prep.constraints.inference_authorized, false);
  assert.equal(prep.constraints.automatic_retry_authorized, false);
  assert.equal(
    prep.constraints.timeout_change_execution_authorized,
    false,
  );
});

test("Constellation timeout480 runner remains exact-hash-bound and loopback-only", () => {
  const source = readFileSync(
    "scripts/vnext-gate18-phase-c-c4-constellation-qwen4b-context16384-output1024-timeout480-loopback-guarded.ts",
    "utf8",
  );

  assert.match(source, /const CLIENT_TIMEOUT_MS = 480_000;/);
  assert.doesNotMatch(source, /420_000|TIMEOUT420/);
  assert.match(
    source,
    /AUTHORIZED_SINGLE_LOCAL_C4_TIMEOUT_REMEDIATION/,
  );
  assert.match(source, /EXPECTED_PACKET_SHA256/);
  assert.match(source, /EXPECTED_PROMPT_SHA256/);
  assert.match(source, /EXPECTED_REQUEST_SHA256/);
  assert.match(source, /requestLoopbackJson/);
  assert.doesNotMatch(source, /\bfetch\s*\(/);
  assert.match(source, /const CONTEXT_TOKENS = 16384;/);
  assert.match(source, /const MAX_OUTPUT_TOKENS = 1024;/);
  assert.match(source, /temperature: 0/);
  assert.match(source, /keep_alive: "0s"/);
  assert.match(source, /acquireWindowsSystemRequiredGuard\(\)/);
});
