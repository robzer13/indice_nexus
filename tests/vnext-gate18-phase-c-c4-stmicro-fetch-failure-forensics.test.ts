import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

test("STMicro fetch-failure forensics stays read-only and non-inference", () => {
  const prep = JSON.parse(
    readFileSync(
      "calibration/vnext/OROTITAN_GATE18_PHASE_C_C4_STMICRO_FETCH_FAILURE_FORENSICS_PREP_001.json",
      "utf8",
    ),
  ) as {
    status: string;
    authority: {
      inference_authorized: boolean;
      retry_authorized: boolean;
      model_load_authorized: boolean;
      parameter_change_authorized: boolean;
    };
  };
  const script = readFileSync(
    "scripts/vnext-gate18-phase-c-c4-stmicro-fetch-failure-forensics.ts",
    "utf8",
  );

  assert.equal(prep.status, "PREPARED_READ_ONLY_NO_INFERENCE");
  assert.equal(prep.authority.inference_authorized, false);
  assert.equal(prep.authority.retry_authorized, false);
  assert.equal(prep.authority.model_load_authorized, false);
  assert.equal(prep.authority.parameter_change_authorized, false);

  assert.match(script, /RUN_WALL_CLOCK_MS = 304_446/);
  assert.match(script, /configuredClientTimeoutMs: 420000/);
  assert.match(script, /runtimeError: "fetch failed"/);
  assert.match(script, /ollamaGenerateApiCalled: false/);
  assert.doesNotMatch(script, /\/api\/generate/);
  assert.doesNotMatch(script, /ollama", \["run"/);
});
