import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

const source = readFileSync(
  "scripts/vnext-gate18-phase-c-c3-adyen-timeout-static-diagnostic.ts",
  "utf8",
);

test("Adyen timeout diagnostic performs no Ollama or network request", () => {
  assert.match(source, /NO_INFERENCE_NO_OLLAMA_REQUEST/);
  assert.match(source, /ollamaApiCalled: false/);
  assert.match(source, /modelInferenceExecuted: false/);
  assert.doesNotMatch(source, /fetch\(/);
  assert.doesNotMatch(source, /\/api\/generate|\/api\/chat|127\.0\.0\.1:11434/);
});

test("Adyen timeout diagnostic rebuilds both exact compact probes", () => {
  assert.match(source, /findCompany\("Adyen"\)/);
  assert.match(source, /findCompany\("Brookfield Corporation"\)/);
  for (const id of [
    "E-036",
    "E-037",
    "E-040",
    "E-041",
    "E-042",
    "E-043",
    "E-055",
    "E-056",
    "C-005",
    "C-010",
  ]) {
    assert.match(source, new RegExp(id));
  }
  assert.match(source, /E-039/);
});

test("Adyen timeout diagnostic preserves the pinned 4B generation envelope", () => {
  assert.match(source, /qwen3:4b-instruct/);
  assert.match(source, /CONTEXT_TOKENS = 4096/);
  assert.match(source, /MAX_OUTPUT_TOKENS = 768/);
  assert.match(source, /temperature: 0/);
});

test("Adyen timeout diagnostic reports exact byte sizes and hashes", () => {
  assert.match(source, /Buffer\.byteLength/);
  assert.match(source, /promptSha256/);
  assert.match(source, /requestSha256/);
  assert.match(source, /promptBytesAdyenVsBrookfield/);
  assert.match(source, /requestBytesAdyenVsBrookfield/);
  assert.match(source, /observedWallClockMs: 101077/);
  assert.match(source, /observedEvalCount: 405/);
});
