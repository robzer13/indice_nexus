import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

const source = readFileSync(
  "scripts/vnext-gate18-phase-c-c3-brookfield-runtime-diagnostic.ts",
  "utf8",
);

test("Brookfield runtime diagnostic is no-inference metadata only", () => {
  assert.match(source, /LOCAL_METADATA_ONLY_NO_INFERENCE/);
  assert.match(source, /modelInferenceExecuted: false/);
  assert.doesNotMatch(source, /\/api\/generate/);
  assert.doesNotMatch(source, /\/api\/chat/);
  assert.doesNotMatch(source, /\/api\/pull/);
});

test("Brookfield runtime diagnostic pins the exact Qwen model", () => {
  assert.match(source, /MODEL_NAME = "qwen3:1\.7b"/);
  assert.match(
    source,
    /8f68893c685c3ddff2aa3fffce2aa60a30bb2da65ca488b61fff134a4d1730e7/,
  );
  assert.match(source, /MODEL_DIGEST_MISMATCH/);
});

test("Brookfield runtime diagnostic inspects only local Ollama metadata and memory", () => {
  assert.match(source, /"\/api\/version"/);
  assert.match(source, /"\/api\/tags"/);
  assert.match(source, /"\/api\/show"/);
  assert.match(source, /"\/api\/ps"/);
  assert.match(source, /nvidia-smi/);
  assert.match(source, /freeRamGiB/);
  assert.match(source, /freeVramMiB/);
});

test("Brookfield runtime diagnostic cannot authorize retry", () => {
  assert.match(source, /authorizesRetry: false/);
  assert.match(source, /authorizesModelSwitch: false/);
  assert.match(source, /authorizesDecodingChange: false/);
});
