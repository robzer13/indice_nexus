import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

const source = readFileSync(
  "scripts/vnext-gate18-phase-c-c3-qwen17-preflight.ts",
  "utf8",
);

test("C3 Qwen preflight is loopback-only and inference-free", () => {
  assert.match(source, /http:\/\/127\.0\.0\.1:11434/);
  assert.doesNotMatch(source, /https:\/\//);
  assert.match(source, /modelInferenceExecuted: false/);
  assert.match(source, /externalModelApiCall: false/);
  assert.doesNotMatch(source, /\/api\/generate/);
  assert.doesNotMatch(source, /\/api\/chat/);
});

test("C3 Qwen preflight pins exact model identity", () => {
  assert.match(source, /MODEL_NAME = "qwen3:1\.7b"/);
  assert.match(
    source,
    /8f68893c685c3ddff2aa3fffce2aa60a30bb2da65ca488b61fff134a4d1730e7/,
  );
  assert.match(source, /PINNED_MODEL_DIGEST_MISMATCH/);
});

test("C3 Qwen preflight inspects metadata without pulling", () => {
  assert.match(source, /"\/api\/tags"/);
  assert.match(source, /"\/api\/show"/);
  assert.match(source, /"\/api\/ps"/);
  assert.doesNotMatch(source, /\/api\/pull/);
  assert.match(source, /modelDownloadExecuted: false/);
});

test("C3 Qwen preflight keeps C3 inference unauthorized", () => {
  assert.match(source, /c3InferenceAuthorized: false/);
  assert.match(source, /fullPacketContextNotYetAuthorized: true/);
  assert.match(source, /TARGETED_REGRESSION_COMPACT_FIRST/);
});
