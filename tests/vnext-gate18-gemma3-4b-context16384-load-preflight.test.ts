import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

test("Gemma 3 context8192 result is persisted and non-inferential", () => {
  const r = JSON.parse(
    readFileSync(
      "calibration/vnext/OROTITAN_GATE18_PHASE_C_GEMMA3_4B_CONTEXT8192_LOAD_SMOKE_RESULT_001.json",
      "utf8",
    ),
  );

  assert.equal(r.status, "PASS_LOAD_ONLY_MEASURED");
  assert.equal(r.target.context_tokens, 8192);
  assert.equal(r.loaded.vram_used_mib, 2423);
  assert.equal(r.loaded.vram_free_mib, 1540);
  assert.equal(r.loaded.processor_split, "56%/44% CPU/GPU");
  assert.equal(r.loaded.free_ram_gib, 0.77);
  assert.equal(r.safety.semantic_inference_executed, false);
  assert.equal(r.after.model_unloaded, true);
});

test("Gemma 3 context8192 authorization is consumed", () => {
  const a = JSON.parse(
    readFileSync(
      "calibration/vnext/OROTITAN_GATE18_PHASE_C_GEMMA3_4B_CONTEXT8192_LOAD_SMOKE_AUTH_001.json",
      "utf8",
    ),
  );

  assert.equal(a.status, "CONSUMED_SINGLE_LOAD_ONLY_COMPLETE");
  assert.equal(a.planned_execution.context_tokens, 8192);
  assert.equal(a.authority.load_smoke_authorized, false);
  assert.equal(a.authority.inference_authorized, false);
  assert.equal(a.authority.authorized_run_count, 0);
  assert.equal(
    a.execution_result,
    "G18-PHASEC-GEMMA3-4B-CONTEXT8192-LOAD-SMOKE-RESULT-001",
  );
});

test("Gemma 3 context16384 result is persisted and authorization is consumed", () => {
  const r = JSON.parse(
    readFileSync(
      "calibration/vnext/OROTITAN_GATE18_PHASE_C_GEMMA3_4B_CONTEXT16384_LOAD_SMOKE_RESULT_001.json",
      "utf8",
    ),
  );

  assert.equal(r.status, "PASS_LOAD_ONLY_MEASURED");
  assert.equal(r.target.context_tokens, 16384);
  assert.equal(r.loaded.vram_used_mib, 2449);
  assert.equal(r.loaded.vram_free_mib, 1514);
  assert.equal(r.loaded.free_ram_gib, 0.33);
  assert.equal(r.loaded.processor_split, "57%/43% CPU/GPU");
  assert.equal(r.interpretation.hardware_qualification_for_c4_target_context, "PASS_WITH_HIGH_RAM_PRESSURE");
  assert.equal(r.safety.semantic_inference_executed, false);
  assert.equal(r.after.model_unloaded, true);


  const a = JSON.parse(
    readFileSync(
      "calibration/vnext/OROTITAN_GATE18_PHASE_C_GEMMA3_4B_CONTEXT16384_LOAD_SMOKE_AUTH_001.json",
      "utf8",
    ),
  );

  assert.equal(a.status, "CONSUMED_SINGLE_LOAD_ONLY_COMPLETE");
  assert.equal(a.planned_execution.context_tokens, 16384);
  assert.equal(a.authority.load_smoke_authorized, false);
  assert.equal(a.authority.inference_authorized, false);
  assert.equal(a.authority.authorized_run_count, 0);
  assert.equal(a.constraints.context_change_beyond_16384_authorized, false);
  assert.equal(a.execution_result, "G18-PHASEC-GEMMA3-4B-CONTEXT16384-LOAD-SMOKE-RESULT-001");
});

test("Gemma 3 context16384 runner requires exact digest and no prompt", () => {
  const raw = readFileSync(
    "scripts/vnext-gate18-phase-c-gemma3-4b-context16384-load-smoke.ts",
    "utf8",
  );

  assert.match(raw, /CONTEXT_TOKENS = 16384/);
  assert.match(
    raw,
    /a2af6cc3eb7fa8be8504abaf9b04e88f17a119ec3f04a3addf55f92841195f5a/,
  );
  assert.match(raw, /AUTHORIZED_SINGLE_LOAD_ONLY_UNCONSUMED/);
  assert.match(raw, /keep_alive:\s*"2m"/);
  assert.match(raw, /keep_alive:\s*0/);
  assert.doesNotMatch(raw, /\/api\/chat/);
  assert.doesNotMatch(raw, /prompt\s*:/);
});
