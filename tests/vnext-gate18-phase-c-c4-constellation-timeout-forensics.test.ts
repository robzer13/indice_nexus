import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

test("Constellation timeout result consumes one-shot authority and leaves no retry authority", () => {
  const result = JSON.parse(
    readFileSync(
      "calibration/vnext/OROTITAN_GATE18_PHASE_C_C4_CONSTELLATION_TIMEOUT420_RESULT_001.json",
      "utf8",
    ),
  ) as {
    status: string;
    execution: {
      wall_clock_ms: number;
      explicit_client_timeout_ms: number;
      runtime_error: string;
      provider_response_observed: boolean;
    };
    interpretation: {
      hidden_undici_headers_timeout_in_active_path: boolean;
      explicit_420s_timeout_reached: boolean;
      timeout_inadequacy_concluded: boolean;
      exact_runtime_mechanism_before_timeout_concluded: boolean;
    };
    authority: {
      retry_authorized: boolean;
      second_retry_authorized: boolean;
      timeout_change_authorized: boolean;
      output_change_authorized: boolean;
    };
  };

  assert.equal(
    result.status,
    "FAIL_EXPLICIT_CLIENT_TIMEOUT_REACHED_NO_PROVIDER_RESPONSE",
  );
  assert.equal(result.execution.wall_clock_ms, 420485);
  assert.equal(
    result.execution.explicit_client_timeout_ms,
    420000,
  );
  assert.equal(
    result.execution.runtime_error,
    "LOOPBACK_HTTP_EXPLICIT_TIMEOUT_420000MS",
  );
  assert.equal(
    result.execution.provider_response_observed,
    false,
  );
  assert.equal(
    result.interpretation.hidden_undici_headers_timeout_in_active_path,
    false,
  );
  assert.equal(
    result.interpretation.explicit_420s_timeout_reached,
    true,
  );
  assert.equal(
    result.interpretation.timeout_inadequacy_concluded,
    false,
  );
  assert.equal(
    result.interpretation.exact_runtime_mechanism_before_timeout_concluded,
    false,
  );
  assert.equal(result.authority.retry_authorized, false);
  assert.equal(result.authority.second_retry_authorized, false);
  assert.equal(result.authority.timeout_change_authorized, false);
  assert.equal(result.authority.output_change_authorized, false);
});

test("Constellation timeout log forensics V1 is read-only and non-inference", () => {
  const prep = JSON.parse(
    readFileSync(
      "calibration/vnext/OROTITAN_GATE18_PHASE_C_C4_CONSTELLATION_TIMEOUT420_LOG_FORENSICS_V1_PREP_001.json",
      "utf8",
    ),
  ) as {
    status: string;
    collector: {
      direct_node_file_read: boolean;
      ollama_api_call: boolean;
      inference: boolean;
      model_load: boolean;
      extracts_generation_progress: boolean;
    };
    authority: {
      inference_authorized: boolean;
      retry_authorized: boolean;
      timeout_change_authorized: boolean;
    };
  };

  const source = readFileSync(
    "scripts/vnext-gate18-phase-c-c4-constellation-timeout420-log-forensics-v1.ts",
    "utf8",
  );

  assert.equal(
    prep.status,
    "PREPARED_READ_ONLY_NO_INFERENCE",
  );
  assert.equal(prep.collector.direct_node_file_read, true);
  assert.equal(prep.collector.ollama_api_call, false);
  assert.equal(prep.collector.inference, false);
  assert.equal(prep.collector.model_load, false);
  assert.equal(
    prep.collector.extracts_generation_progress,
    true,
  );
  assert.equal(prep.authority.inference_authorized, false);
  assert.equal(prep.authority.retry_authorized, false);
  assert.equal(prep.authority.timeout_change_authorized, false);

  assert.match(source, /server\.log/);
  assert.match(source, /n_gen/);
  assert.match(source, /10:35:49\.839\+02:00/);
  assert.match(source, /LOOPBACK_HTTP_EXPLICIT_TIMEOUT_420000MS/);
  assert.doesNotMatch(source, /requestLoopbackJson/);
  assert.doesNotMatch(source, /\/api\/tags/);
  assert.doesNotMatch(source, /\bfetch\s*\(/);
  assert.doesNotMatch(source, /ollama", \["run"/);
});
