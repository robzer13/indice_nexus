import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

test("Qwen3.5 think-false retry authorization freezes a single same-cell compatibility retry", () => {
  const a = JSON.parse(
    readFileSync(
      "calibration/vnext/OROTITAN_GATE18_PHASE_C_C4_CONSTELLATION_QWEN3_5_4B_V1_1_THINK_FALSE_RETRY_AUTH_001.json",
      "utf8",
    ),
  );

  assert.equal(a.status, "AUTHORIZED_SINGLE_LOCAL_INFERENCE");
  assert.equal(a.c4_inference.retry_kind, "SAME_CELL_RUNTIME_COMPATIBILITY_RETRY");
  assert.equal(a.c4_inference.company, "Constellation Software");
  assert.equal(a.c4_inference.model_name, "qwen3.5:4b-q4_K_M");
  assert.equal(a.c4_inference.context_tokens, 16384);
  assert.equal(a.c4_inference.max_output_tokens, 1024);
  assert.equal(a.c4_inference.temperature, 0);
  assert.equal(a.c4_inference.client_timeout_ms, 600000);
  assert.equal(a.c4_inference.think, false);
  assert.equal(a.constraints.authorized_run_count, 1);
  assert.equal(a.constraints.automatic_second_retry_authorized, false);
  assert.equal(a.invariants.only_runtime_change, "think:false");
});

test("Qwen3.5 think-false retry runner explicitly disables thinking and keeps all other C4 invariants", () => {
  const raw = readFileSync(
    "scripts/vnext-gate18-phase-c-c4-constellation-qwen3-5-4b-v1-1-context16384-output1024-timeout600-thinkfalse-loopback-guarded.ts",
    "utf8",
  );

  assert.match(raw, /qwen3\.5:4b-q4_K_M/);
  assert.match(raw, /CONTEXT_TOKENS = 16384/);
  assert.match(raw, /MAX_OUTPUT_TOKENS = 1024/);
  assert.match(raw, /CLIENT_TIMEOUT_MS = 600_000/);
  assert.match(raw, /think:\s*false/);
  assert.match(raw, /temperature:\s*0/);
  assert.match(raw, /num_ctx:\s*CONTEXT_TOKENS/);
  assert.match(raw, /num_predict:\s*MAX_OUTPUT_TOKENS/);
  assert.match(raw, /keep_alive:\s*"0s"/);
  assert.match(raw, /evaluateGate18V11Validation/);
  assert.match(raw, /GATE18_PHASE_B_V10_SYSTEM_PROMPT/);
  assert.match(raw, /GATE18_PHASE_B_V10_GENERATION_SCHEMA_SPEC/);
  assert.match(raw, /0891fa34d02a47c83e8342d5da5e653a3566d90f1c63c1bfc662f4e6097c10b8/);
  assert.match(raw, /rawOutputPreserved:\s*true/);
  assert.match(raw, /humanAdjudicationRequired:\s*true/);
});
