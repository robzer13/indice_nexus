import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

test("STMicro output-budget remediation is prepared, invariant-preserving, and unauthorized", () => {
  const result = JSON.parse(
    readFileSync(
      "calibration/vnext/OROTITAN_GATE18_PHASE_C_C4_STMICRO_GUARDED_RETRY_RESULT_001.json",
      "utf8",
    ),
  ) as {
    classification: string;
    execution: {
      done_reason: string;
      eval_count: number;
      runtime_error: string | null;
    };
    model: { max_output_tokens: number };
    diagnosis: {
      max_output_768_inadequate_for_this_packet_concluded: boolean;
      semantic_failure_concluded: boolean;
    };
  };
  const diagnostic = JSON.parse(
    readFileSync(
      "calibration/vnext/OROTITAN_GATE18_PHASE_C_C4_STMICRO_OUTPUT_BUDGET_STATIC_DIAGNOSTIC_001.json",
      "utf8",
    ),
  ) as {
    status: string;
    remediation: {
      proposed_max_output_tokens: number;
      proposed_client_timeout_ms: number;
      model_change: boolean;
      prompt_change: boolean;
      schema_change: boolean;
      packet_change: boolean;
      context_change: boolean;
      temperature_change: boolean;
    };
  };
  const prep = JSON.parse(
    readFileSync(
      "calibration/vnext/OROTITAN_GATE18_PHASE_C_C4_STMICRO_OUTPUT_BUDGET_REMEDIATION_PREP_001.json",
      "utf8",
    ),
  ) as {
    status: string;
    future_execution: {
      inference_authorized: boolean;
      retry_authorized: boolean;
    };
    authority: {
      inference_authorized: boolean;
      automatic_retry_authorized: boolean;
    };
    remediation_delta: {
      prior_max_output_tokens: number;
      proposed_max_output_tokens: number;
      prior_client_timeout_ms: number;
      proposed_client_timeout_ms: number;
    };
  };
  const runner = readFileSync(
    "scripts/vnext-gate18-phase-c-c4-stmicro-qwen4b-context16384-output1024-timeout420-guarded.ts",
    "utf8",
  );

  assert.equal(
    result.classification,
    "OUTPUT_BUDGET_EXHAUSTED_TRUNCATED_JSON_NO_SEMANTIC_RESULT",
  );
  assert.equal(result.execution.done_reason, "length");
  assert.equal(result.execution.eval_count, 768);
  assert.equal(result.model.max_output_tokens, 768);
  assert.equal(result.execution.runtime_error, null);
  assert.equal(
    result.diagnosis.max_output_768_inadequate_for_this_packet_concluded,
    true,
  );
  assert.equal(result.diagnosis.semantic_failure_concluded, false);

  assert.equal(
    diagnostic.status,
    "PASS_OUTPUT_BUDGET_REMEDIATION_JUSTIFIED",
  );
  assert.equal(
    diagnostic.remediation.proposed_max_output_tokens,
    1024,
  );
  assert.equal(
    diagnostic.remediation.proposed_client_timeout_ms,
    420000,
  );
  assert.equal(diagnostic.remediation.model_change, false);
  assert.equal(diagnostic.remediation.prompt_change, false);
  assert.equal(diagnostic.remediation.schema_change, false);
  assert.equal(diagnostic.remediation.packet_change, false);
  assert.equal(diagnostic.remediation.context_change, false);
  assert.equal(diagnostic.remediation.temperature_change, false);

  assert.equal(prep.status, "PREPARED_NOT_AUTHORIZED");
  assert.equal(prep.future_execution.inference_authorized, false);
  assert.equal(prep.future_execution.retry_authorized, false);
  assert.equal(prep.authority.inference_authorized, false);
  assert.equal(prep.authority.automatic_retry_authorized, false);
  assert.equal(prep.remediation_delta.prior_max_output_tokens, 768);
  assert.equal(prep.remediation_delta.proposed_max_output_tokens, 1024);
  assert.equal(prep.remediation_delta.prior_client_timeout_ms, 300000);
  assert.equal(prep.remediation_delta.proposed_client_timeout_ms, 420000);

  assert.match(runner, /const MAX_OUTPUT_TOKENS = 1024;/);
  assert.match(runner, /const CLIENT_TIMEOUT_MS = 420_000;/);
  assert.match(
    runner,
    /AUTHORIZED_SINGLE_LOCAL_OUTPUT_BUDGET_REMEDIATION/,
  );
  assert.match(
    runner,
    /G18-PHASEC-C4-STM-QWEN4B-CONTEXT16384-OUTPUT1024-TIMEOUT420-AUTH-001/,
  );
  assert.match(runner, /acquireWindowsSystemRequiredGuard/);
  assert.match(runner, /await sleepGuard\.release\(\)/);
});
