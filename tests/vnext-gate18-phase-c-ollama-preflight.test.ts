import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

const source = readFileSync(
  "scripts/vnext-gate18-phase-c-ollama-preflight.ts",
  "utf8",
);

test("Phase C Ollama preflight is loopback-only", () => {
  assert.match(source, /http:\/\/127\.0\.0\.1:11434/);
  assert.doesNotMatch(source, /https:\/\//);
});

test("Phase C Ollama preflight cannot infer or download", () => {
  assert.doesNotMatch(source, /\/api\/chat/);
  assert.doesNotMatch(source, /\/api\/generate/);
  assert.doesNotMatch(source, /\/api\/pull/);
  assert.doesNotMatch(source, /ollama", \["run"/);
  assert.doesNotMatch(source, /ollama", \["pull"/);
});

test("Phase C Ollama preflight only inspects version, tags and ps", () => {
  assert.match(source, /"\/api\/version"/);
  assert.match(source, /"\/api\/tags"/);
  assert.match(source, /"\/api\/ps"/);
  assert.match(source, /modelInferenceExecuted: false/);
  assert.match(source, /modelDownloadExecuted: false/);
  assert.match(source, /externalModelApiCall: false/);
});
