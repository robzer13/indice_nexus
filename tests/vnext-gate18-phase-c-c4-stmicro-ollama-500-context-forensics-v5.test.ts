import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

test("STMicro Ollama HTTP 500 context forensics V5 is targeted and non-inference", () => {
  const prep = JSON.parse(
    readFileSync(
      "calibration/vnext/OROTITAN_GATE18_PHASE_C_C4_STMICRO_OLLAMA_500_CONTEXT_FORENSICS_V5_PREP_001.json",
      "utf8",
    ),
  ) as {
    status: string;
    target: {
      local_timestamp: string;
      http_status: number;
      route: string;
      expected_server_duration: string;
    };
    collection_scope: {
      powershell_used: boolean;
      raw_context_before_lines: number;
      raw_context_after_lines: number;
    };
    authority: {
      inference_authorized: boolean;
      retry_authorized: boolean;
      model_load_authorized: boolean;
      parameter_change_authorized: boolean;
    };
  };

  const script = readFileSync(
    "scripts/vnext-gate18-phase-c-c4-stmicro-ollama-500-context-forensics-v5.ts",
    "utf8",
  );

  assert.equal(prep.status, "PREPARED_READ_ONLY_NO_INFERENCE");
  assert.equal(prep.target.local_timestamp, "2026/09/24 - 22:08:47");
  assert.equal(prep.target.http_status, 500);
  assert.equal(prep.target.route, "POST /api/generate");
  assert.equal(prep.target.expected_server_duration, "5m4s");
  assert.equal(prep.collection_scope.powershell_used, false);
  assert.equal(prep.collection_scope.raw_context_before_lines, 180);
  assert.equal(prep.collection_scope.raw_context_after_lines, 80);
  assert.equal(prep.authority.inference_authorized, false);
  assert.equal(prep.authority.retry_authorized, false);
  assert.equal(prep.authority.model_load_authorized, false);
  assert.equal(prep.authority.parameter_change_authorized, false);

  assert.match(script, /TARGET_LOCAL_STAMP = "2026\/09\/24 - 22:08:47"/);
  assert.match(script, /TARGET_ROUTE = 'POST     "\/api\/generate"'/);
  assert.match(script, /readFileSync\(SERVER_LOG, "utf8"\)/);
  assert.match(script, /diagnosticContextLines/);
  assert.match(script, /ollamaApiCalled: false/);
  assert.doesNotMatch(script, /fetch\(/);
  assert.doesNotMatch(script, /ollama", \["run"/);
});
