import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

const source = readFileSync(
  "scripts/vnext-gate18-phase-c-qwen4b-memory-preflight.ts",
  "utf8",
);

test("Qwen3 4B memory preflight is read-only and local", () => {
  assert.match(source, /LOCAL_RESOURCE_MEASUREMENT_ONLY/);
  assert.match(source, /networkAccessRequested: false/);
  assert.match(source, /modelDownloadExecuted: false/);
  assert.match(source, /modelInferenceExecuted: false/);
  assert.doesNotMatch(source, /ollama", \["pull"/);
  assert.doesNotMatch(source, /\/api\/generate|\/api\/chat/);
  assert.doesNotMatch(source, /Invoke-WebRequest|curl\.exe|wget|Start-BitsTransfer/i);
});

test("Qwen3 4B memory preflight pins the escalation candidate", () => {
  assert.match(source, /QWEN3_4B_INSTRUCT_OLLAMA_Q4_K_M/);
  assert.match(source, /qwen3:4b-instruct/);
  assert.match(source, /Q4_K_M/);
  assert.match(source, /TARGET_ARTIFACT_CLASS_GIB = 2\.5/);
});

test("Qwen3 4B memory preflight measures current RAM and VRAM headroom", () => {
  assert.match(source, /os\.freemem\(\)/);
  assert.match(source, /memory\.total,memory\.free,memory\.used/);
  assert.match(source, /freeRamGiBAtProbe/);
  assert.match(source, /memoryFreeMiB/);
  assert.match(source, /memoryUsedMiB/);
});

test("Qwen3 4B memory preflight observes loaded Ollama models without inference", () => {
  assert.match(source, /runText\("ollama", \["ps"\]\)/);
  assert.match(source, /loadedModelCount/);
  assert.match(source, /qwen3_4bResourceFitConcluded: false/);
  assert.match(source, /qwen3_4bDownloadAuthorized: false/);
  assert.match(source, /qwen3_4bInferenceAuthorized: false/);
});
