import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

const source = readFileSync(
  "scripts/vnext-gate18-phase-c-ollama-model-registry-recovery.ts",
  "utf8",
);

test("registry recovery check cannot infer or mutate models", () => {
  assert.match(source, /LOCAL_METADATA_ONLY_NO_INFERENCE/);
  assert.match(source, /modelInferenceExecuted: false/);
  assert.match(source, /modelDownloadExecuted: false/);
  assert.match(source, /modelDeleted: false/);
  assert.doesNotMatch(source, /\/api\/generate/);
  assert.doesNotMatch(source, /\/api\/chat/);
  assert.doesNotMatch(source, /\/api\/pull/);
  assert.doesNotMatch(source, /ollama", \["pull"/);
  assert.doesNotMatch(source, /ollama", \["rm"/);
});

test("registry recovery check pins Qwen3 exact digest", () => {
  assert.match(source, /MODEL_NAME = "qwen3:1\.7b"/);
  assert.match(
    source,
    /8f68893c685c3ddff2aa3fffce2aa60a30bb2da65ca488b61fff134a4d1730e7/,
  );
  assert.match(source, /exactDigestValid/);
});

test("registry recovery check uses longer tags timeout and independent corroboration", () => {
  assert.match(source, /TAGS_TIMEOUT_MS = 60_000/);
  assert.match(source, /ollama", \["list"\]/);
  assert.match(source, /"\/api\/tags"/);
  assert.match(source, /"\/api\/show"/);
  assert.match(source, /localPresenceCorroborated/);
});

test("registry recovery check remains loopback only", () => {
  assert.match(source, /http:\/\/127\.0\.0\.1:11434/);
  assert.doesNotMatch(source, /https:\/\//);
});

test("registry recovery check cannot authorize inference or driver upgrade", () => {
  assert.match(source, /c3InferenceAuthorized: false/);
  assert.match(source, /driverUpgradeAuthorized: false/);
});
