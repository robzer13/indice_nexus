import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

test("Qwen3.5 context4096 load-only result is persisted and non-inferential", () => {
  const r = JSON.parse(
    readFileSync(
      "calibration/vnext/OROTITAN_GATE18_PHASE_C_QWEN3_5_4B_LOAD_SMOKE_RESULT_001.json",
      "utf8",
    ),
  );

  assert.equal(r.status, "PASS_LOAD_ONLY_MEASURED");
  assert.equal(r.target.context_tokens, 4096);
  assert.equal(r.loaded.vram_used_mib, 2601);
  assert.equal(r.loaded.vram_free_mib, 1362);
  assert.equal(r.loaded.processor_split, "50%/50% CPU/GPU");
  assert.equal(r.derived_measurements.vram_delta_vs_qwen3_4b_context4096_mib, 290);
  assert.equal(r.safety.semantic_inference_executed, false);
  assert.equal(r.after.model_unloaded, true);
});

test("Qwen3.5 context8192 load-only step is standing-authorized and bounded", () => {
  const a = JSON.parse(
    readFileSync(
      "calibration/vnext/OROTITAN_GATE18_PHASE_C_QWEN3_5_4B_CONTEXT8192_LOAD_SMOKE_AUTH_001.json",
      "utf8",
    ),
  );

  assert.equal(a.status, "AUTHORIZED_SINGLE_LOAD_ONLY_UNCONSUMED");
  assert.equal(a.authorization_source.authorization_id, "OROTITAN-STANDING-TECHNICAL-AUTH-002");
  assert.equal(a.authorization_source.separate_user_reprompt_required, false);
  assert.equal(a.authorization_source.cost_usd, 0);
  assert.equal(a.planned_execution.context_tokens, 8192);
  assert.equal(a.planned_execution.prompt_provided, false);
  assert.equal(a.planned_execution.semantic_inference_requested, false);
  assert.equal(a.authority.load_smoke_authorized, true);
  assert.equal(a.authority.inference_authorized, false);
  assert.equal(a.authority.authorized_run_count, 1);
  assert.equal(a.constraints.context_change_beyond_8192_authorized, false);
});

test("Qwen3.5 context8192 runner preserves load-only guardrails", () => {
  const raw = readFileSync(
    "scripts/vnext-gate18-phase-c-qwen3-5-4b-context8192-load-smoke.ts",
    "utf8",
  );

  assert.match(raw, /CONTEXT_TOKENS = 8192/);
  assert.match(raw, /CONTEXT8192_LOAD_SMOKE_AUTH_001\.json/);
  assert.match(raw, /AUTHORIZED_SINGLE_LOAD_ONLY_UNCONSUMED/);
  assert.match(raw, /keep_alive:"2m"|keep_alive:\s*"2m"/);
  assert.match(raw, /keep_alive:0|keep_alive:\s*0/);
  assert.doesNotMatch(raw, /\/api\/chat/);
  assert.doesNotMatch(raw, /prompt\s*:/);
});
