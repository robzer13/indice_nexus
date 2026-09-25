import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

test("STMicro human adjudication result closes one C4 cell with carry and no new authority", () => {
  const result = JSON.parse(
    readFileSync(
      "calibration/vnext/OROTITAN_GATE18_PHASE_C_C4_STMICRO_HUMAN_ADJUDICATION_RESULT_001.json",
      "utf8",
    ),
  ) as {
    status: string;
    human_quality: Record<string, string>;
    disposition: {
      human_adjudication_completed: boolean;
      critical_human_quality_failure: boolean;
      clean_pass: boolean;
      pass_with_carry: boolean;
      matrix_cell_completed: boolean;
    };
    authority: {
      retry_authorized: boolean;
      auto_repair_authorized: boolean;
      parameter_change_authorized: boolean;
      model_ranking_authority: boolean;
      routing_authority: boolean;
      production_candidate_decision_authority: boolean;
    };
  };

  assert.equal(
    result.status,
    "COMPLETED_WITH_HUMAN_QUALITY_CARRY",
  );
  assert.equal(
    Object.keys(result.human_quality).length,
    10,
  );
  assert.equal(
    result.disposition.human_adjudication_completed,
    true,
  );
  assert.equal(
    result.disposition.critical_human_quality_failure,
    false,
  );
  assert.equal(result.disposition.clean_pass, false);
  assert.equal(result.disposition.pass_with_carry, true);
  assert.equal(result.disposition.matrix_cell_completed, true);
  assert.equal(result.authority.retry_authorized, false);
  assert.equal(result.authority.auto_repair_authorized, false);
  assert.equal(result.authority.parameter_change_authorized, false);
  assert.equal(result.authority.model_ranking_authority, false);
  assert.equal(result.authority.routing_authority, false);
  assert.equal(
    result.authority.production_candidate_decision_authority,
    false,
  );
});

test("remaining matrix static preflight V2 uses validated candidate runtime and no inference", () => {
  const prep = JSON.parse(
    readFileSync(
      "calibration/vnext/OROTITAN_GATE18_PHASE_C_C4_REMAINING_MATRIX_STATIC_PREFLIGHT_V2_PREP_001.json",
      "utf8",
    ),
  ) as {
    status: string;
    proposed_common_runtime: {
      context_tokens: number;
      max_output_tokens: number;
      temperature: number;
      client_timeout_ms: number;
      transport: string;
      sleep_guard_required: boolean;
    };
    remaining_matrix: Array<{ company: string }>;
    authority: {
      inference_authorized: boolean;
      automatic_retry_authorized: boolean;
      parameter_change_execution_authorized: boolean;
    };
  };

  const source = readFileSync(
    "scripts/vnext-gate18-phase-c-c4-remaining-matrix-static-preflight-v2.ts",
    "utf8",
  );

  assert.equal(prep.status, "PREPARED_NO_INFERENCE");
  assert.equal(prep.proposed_common_runtime.context_tokens, 16384);
  assert.equal(prep.proposed_common_runtime.max_output_tokens, 1024);
  assert.equal(prep.proposed_common_runtime.temperature, 0);
  assert.equal(
    prep.proposed_common_runtime.client_timeout_ms,
    420000,
  );
  assert.equal(
    prep.proposed_common_runtime.transport,
    "NODE_HTTP_REQUEST_LOOPBACK",
  );
  assert.equal(
    prep.proposed_common_runtime.sleep_guard_required,
    true,
  );
  assert.deepEqual(
    prep.remaining_matrix.map((row) => row.company),
    [
      "RATIONAL AG",
      "Constellation Software",
      "Brookfield Corporation",
      "Adyen",
    ],
  );
  assert.equal(prep.authority.inference_authorized, false);
  assert.equal(prep.authority.automatic_retry_authorized, false);
  assert.equal(
    prep.authority.parameter_change_execution_authorized,
    false,
  );

  assert.match(source, /const CONTEXT_TOKENS = 16384;/);
  assert.match(source, /const MAX_OUTPUT_TOKENS = 1024;/);
  assert.match(source, /const CLIENT_TIMEOUT_MS = 420_000;/);
  assert.match(source, /NODE_HTTP_REQUEST_LOOPBACK/);
  assert.match(source, /modelInferenceExecuted: false/);
  assert.match(source, /ollamaApiCalled: false/);
  assert.doesNotMatch(source, /\/api\/generate/);
  assert.doesNotMatch(source, /\bfetch\s*\(/);
});
