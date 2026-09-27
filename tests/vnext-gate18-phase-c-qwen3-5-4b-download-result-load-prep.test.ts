import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

test("Qwen3.5 4B pinned download result is exact and non-inferential", () => {
  const r=JSON.parse(readFileSync("calibration/vnext/OROTITAN_GATE18_PHASE_C_QWEN3_5_4B_DOWNLOAD_RESULT_001.json","utf8"));
  assert.equal(r.status,"PASS_PINNED_DOWNLOAD_ONLY");
  assert.equal(r.model.digest,"2a654d98e6fba55d452b7043684e9b57a947e393bbffa62485a7aac05ee4eefd");
  assert.equal(r.model.size_bytes,3389983735);
  assert.equal(r.model.quantization,"Q4_K_M");
  assert.equal(r.model.ollama_reported_parameter_size,"4.7B");
  assert.equal(r.safety.load_smoke_executed,false);
  assert.equal(r.safety.model_inference_executed,false);
});

test("Qwen3.5 load-only preflight is authorized under standing zero-cost authority",()=>{
  const a=JSON.parse(readFileSync("calibration/vnext/OROTITAN_GATE18_PHASE_C_QWEN3_5_4B_LOAD_SMOKE_AUTH_001.json","utf8"));
  assert.equal(a.status,"AUTHORIZED_SINGLE_LOAD_ONLY_UNCONSUMED");
  assert.equal(a.authority.load_smoke_authorized,true);
  assert.equal(a.authority.inference_authorized,false);
  assert.equal(a.authority.authorized_run_count,1);
});

test("Qwen3.5 load-only runner requires explicit authorization and requests no prompt",()=>{
  const raw=readFileSync("scripts/vnext-gate18-phase-c-qwen3-5-4b-load-smoke.ts","utf8");
  assert.match(raw,/AUTHORIZED_SINGLE_LOAD_ONLY_UNCONSUMED/);
  assert.match(raw,/CONTEXT_TOKENS = 4096/);
  assert.match(raw,/num_ctx:\s*CONTEXT_TOKENS/);
  assert.match(raw,/keep_alive:\s*"2m"/);
  assert.match(raw,/keep_alive:\s*0/);
  assert.match(raw,/semanticInferenceExecuted:false/);
  assert.doesNotMatch(raw,/\/api\/chat/);
  assert.doesNotMatch(raw,/prompt\s*:/);
});
