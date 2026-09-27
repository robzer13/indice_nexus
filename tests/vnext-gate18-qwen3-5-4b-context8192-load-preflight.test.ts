import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

test("Qwen3.5 context16384 load-only result is persisted and non-inferential", () => {
  const r = JSON.parse(
    readFileSync(
      "calibration/vnext/OROTITAN_GATE18_PHASE_C_QWEN3_5_4B_CONTEXT16384_LOAD_SMOKE_RESULT_001.json",
      "utf8",
    ),
  );

  assert.equal(r.status, "PASS_LOAD_ONLY_MEASURED");
  assert.equal(r.target.context_tokens, 16384);
  assert.equal(r.loaded.vram_used_mib, 2589);
  assert.equal(r.loaded.vram_free_mib, 1374);
  assert.equal(r.loaded.processor_split, "56%/44% CPU/GPU");
  assert.equal(r.loaded.ollama_reported_size, "4.2 GB");
  assert.equal(r.safety.semantic_inference_executed, false);
  assert.equal(r.after.model_unloaded, true);
});

test("Qwen3.5 context16384 load-only authority is consumed after the measured pass", () => {
  const a = JSON.parse(
    readFileSync(
      "calibration/vnext/OROTITAN_GATE18_PHASE_C_QWEN3_5_4B_CONTEXT16384_LOAD_SMOKE_AUTH_001.json",
      "utf8",
    ),
  );

  assert.equal(a.status, "CONSUMED_SINGLE_LOAD_ONLY_COMPLETE");
  assert.equal(a.planned_execution.context_tokens, 16384);
  assert.equal(a.authority.load_smoke_authorized, false);
  assert.equal(a.authority.inference_authorized, false);
  assert.equal(a.authority.authorized_run_count, 0);
  assert.equal(
    a.execution_result,
    "G18-PHASEC-QWEN3_5-4B-CONTEXT16384-LOAD-SMOKE-RESULT-001",
  );
});
