import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

test("Granite 4 3B pinned download result is exact and non-inferential", () => {
  const result = JSON.parse(
    readFileSync(
      "calibration/vnext/OROTITAN_GATE18_PHASE_C_GRANITE4_3B_DOWNLOAD_RESULT_001.json",
      "utf8",
    ),
  );

  assert.equal(result.status, "PASS_PINNED_DOWNLOAD_ONLY");
  assert.equal(
    result.model.digest,
    "89962fcc75239ac434cdebceb6b7e0669397f92eaef9c487774b718bc36a3e5f",
  );
  assert.equal(result.model.size_bytes, 2099521385);
  assert.equal(result.model.quantization, "Q4_K_M");
  assert.equal(result.model.ollama_reported_parameter_size, "3.4B");
  assert.equal(result.safety.load_smoke_executed, false);
  assert.equal(result.safety.model_inference_executed, false);
});

test("Granite 4 3B context4096 load-only authorization allows one guarded run", () => {
  const auth = JSON.parse(
    readFileSync(
      "calibration/vnext/OROTITAN_GATE18_PHASE_C_GRANITE4_3B_LOAD_SMOKE_AUTH_001.json",
      "utf8",
    ),
  );

  assert.equal(auth.status, "AUTHORIZED_SINGLE_LOAD_ONLY_UNCONSUMED");
  assert.equal(auth.planned_execution.context_tokens, 4096);
  assert.equal(auth.authority.load_smoke_authorized, true);
  assert.equal(auth.authority.inference_authorized, false);
  assert.equal(auth.authority.authorized_run_count, 1);
});

test("Granite 4 load-only runner requires exact digest and requests no prompt", () => {
  const raw = readFileSync(
    "scripts/vnext-gate18-phase-c-granite4-3b-load-smoke.ts",
    "utf8",
  );

  assert.match(raw, /granite4:3b/);
  assert.match(
    raw,
    /89962fcc75239ac434cdebceb6b7e0669397f92eaef9c487774b718bc36a3e5f/,
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

test("Qwen3 8B is explicitly queued immediately after Granite", () => {
  const queue = JSON.parse(
    readFileSync(
      "calibration/vnext/OROTITAN_GATE18_POST_GRANITE_QWEN3_8B_QUEUE_001.json",
      "utf8",
    ),
  );
  const registry = JSON.parse(
    readFileSync(
      "calibration/vnext/OROTITAN_GATE18_PHASE_C_LOCAL_CANDIDATE_REGISTRY_REFRESH_2026_001.json",
      "utf8",
    ),
  );

  assert.equal(queue.status, "QWEN3_8B_QUEUED_AFTER_GRANITE");
  assert.equal(queue.sequence.current_candidate, "GRANITE4_3B_OLLAMA_Q4_K_M");
  assert.equal(queue.sequence.next_candidate_after_granite, "QWEN3_8B_LOCAL");
  assert.equal(queue.qwen3_8b_plan.direct_inference_authorized, false);
  assert.equal(registry.next_candidate_after_selected, "QWEN3_8B_LOCAL");
  assert.deepEqual(registry.user_directed_sequence.sequence, [
    "GRANITE4_3B_OLLAMA_Q4_K_M",
    "QWEN3_8B_LOCAL",
  ]);
});
