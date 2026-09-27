import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

test("Qwen3.5 Constellation C4 authorization is consumed while preserving the frozen cell", () => {
  const a = JSON.parse(
    readFileSync(
      "calibration/vnext/OROTITAN_GATE18_PHASE_C_C4_CONSTELLATION_QWEN3_5_4B_V1_1_CONTEXT16384_OUTPUT1024_TIMEOUT600_LOOPBACK_AUTH_001.json",
      "utf8",
    ),
  );

  assert.equal(a.status, "CONSUMED_SINGLE_LOCAL_INFERENCE_FAIL_RAW_JSON_INCOMPLETE");
  assert.equal(a.c4_inference.authorized, false);
  assert.equal(a.c4_inference.company, "Constellation Software");
  assert.equal(a.c4_inference.archetype, "SERIAL_ACQUIRER");
  assert.equal(a.c4_inference.model_name, "qwen3.5:4b-q4_K_M");
  assert.equal(
    a.c4_inference.model_digest,
    "2a654d98e6fba55d452b7043684e9b57a947e393bbffa62485a7aac05ee4eefd",
  );
  assert.equal(a.c4_inference.packet_sha256, "9a47bcf15d0c90da55349cbb3fbb2da8e9645b859a535dc8909501e89a20b6d8");
  assert.equal(a.c4_inference.prompt_sha256, "0891fa34d02a47c83e8342d5da5e653a3566d90f1c63c1bfc662f4e6097c10b8");
  assert.equal(a.c4_inference.prompt_bytes, 9041);
  assert.equal(a.c4_inference.context_tokens, 16384);
  assert.equal(a.c4_inference.max_output_tokens, 1024);
  assert.equal(a.c4_inference.temperature, 0);
  assert.equal(a.c4_inference.client_timeout_ms, 600000);
  assert.equal(a.constraints.authorized_run_count, 0);
  assert.equal(a.constraints.automatic_retry_authorized, false);
  assert.equal(a.constraints.prompt_change_authorized, false);
  assert.equal(a.constraints.schema_change_authorized, false);
  assert.equal(a.constraints.packet_change_authorized, false);
  assert.equal(a.constraints.external_model_api_cost_usd, 0);
});

test("Qwen3.5 Constellation runner keeps generation v1.0 and validates with v1.1", () => {
  const raw = readFileSync(
    "scripts/vnext-gate18-phase-c-c4-constellation-qwen3-5-4b-v1-1-context16384-output1024-timeout600-loopback-guarded.ts",
    "utf8",
  );

  assert.match(raw, /qwen3\.5:4b-q4_K_M/);
  assert.match(raw, /CONTEXT_TOKENS = 16384/);
  assert.match(raw, /MAX_OUTPUT_TOKENS = 1024/);
  assert.match(raw, /CLIENT_TIMEOUT_MS = 600_000/);
  assert.match(raw, /0891fa34d02a47c83e8342d5da5e653a3566d90f1c63c1bfc662f4e6097c10b8/);
  assert.match(raw, /evaluateGate18V11Validation/);
  assert.match(raw, /GATE18_PHASE_B_V10_SYSTEM_PROMPT/);
  assert.match(raw, /GATE18_PHASE_B_V10_GENERATION_SCHEMA_SPEC/);
  assert.match(raw, /temperature:\s*0/);
  assert.match(raw, /num_ctx:\s*CONTEXT_TOKENS/);
  assert.match(raw, /num_predict:\s*MAX_OUTPUT_TOKENS/);
  assert.match(raw, /keep_alive:\s*"0s"/);
  assert.match(raw, /humanAdjudicationRequired:\s*true/);
  assert.match(raw, /rawOutputPreserved:\s*true/);
  assert.doesNotMatch(raw, /phi4-mini:3\.8b-q4_K_M/);
});
