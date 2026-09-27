import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

test("Gemma 3 pinned download result is exact and non-inferential", () => {
  const result = JSON.parse(
    readFileSync(
      "calibration/vnext/OROTITAN_GATE18_PHASE_C_GEMMA3_4B_DOWNLOAD_RESULT_001.json",
      "utf8",
    ),
  );

  assert.equal(result.status, "PASS_PINNED_DOWNLOAD_ONLY");
  assert.equal(
    result.model.digest,
    "a2af6cc3eb7fa8be8504abaf9b04e88f17a119ec3f04a3addf55f92841195f5a",
  );
  assert.equal(result.model.size_bytes, 3338801804);
  assert.equal(result.model.quantization, "Q4_K_M");
  assert.equal(result.model.ollama_reported_parameter_size, "4.3B");
  assert.equal(result.safety.load_smoke_executed, false);
  assert.equal(result.safety.model_inference_executed, false);
});

test("Gemma 3 context4096 load-only authorization is consumed after the measured pass", () => {
  const auth = JSON.parse(
    readFileSync(
      "calibration/vnext/OROTITAN_GATE18_PHASE_C_GEMMA3_4B_LOAD_SMOKE_AUTH_001.json",
      "utf8",
    ),
  );

  assert.equal(auth.status, "CONSUMED_SINGLE_LOAD_ONLY_COMPLETE");
  assert.equal(auth.authority.load_smoke_authorized, false);
  assert.equal(auth.authority.inference_authorized, false);
  assert.equal(auth.authority.authorized_run_count, 0);
  assert.equal(auth.planned_execution.context_tokens, 4096);
  assert.equal(auth.execution_result, "G18-PHASEC-GEMMA3-4B-LOAD-SMOKE-RESULT-001");
});

test("Gemma 3 load-only runner requires exact digest and requests no prompt", () => {
  const raw = readFileSync(
    "scripts/vnext-gate18-phase-c-gemma3-4b-load-smoke.ts",
    "utf8",
  );

  assert.match(raw, /gemma3:4b-it-q4_K_M/);
  assert.match(
    raw,
    /a2af6cc3eb7fa8be8504abaf9b04e88f17a119ec3f04a3addf55f92841195f5a/,
  );
  assert.match(raw, /AUTHORIZED_SINGLE_LOAD_ONLY_UNCONSUMED/);
  assert.match(raw, /CONTEXT_TOKENS = 4096/);
  assert.match(raw, /num_ctx:\s*CONTEXT_TOKENS/);
  assert.match(raw, /keep_alive:\s*"2m"/);
  assert.match(raw, /keep_alive:\s*0/);
  assert.match(raw, /semanticInferenceExecuted:\s*false/);
  assert.doesNotMatch(raw, /\/api\/chat/);
  assert.doesNotMatch(raw, /prompt\s*:/);
});
