import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

const source = readFileSync(
  "scripts/vnext-gate18-phase-c-ollama-upgrade-preflight.ts",
  "utf8",
);

test("Ollama upgrade preflight cannot install or infer", () => {
  assert.match(source, /NO_INSTALL_NO_INFERENCE_PREFLIGHT/);
  assert.match(source, /installerExecuted: false/);
  assert.match(source, /modelInferenceExecuted: false/);
  assert.doesNotMatch(source, /install\.ps1/);
  assert.doesNotMatch(source, /OllamaSetup\.exe/);
  assert.doesNotMatch(source, /\/api\/generate/);
  assert.doesNotMatch(source, /\/api\/chat/);
  assert.doesNotMatch(source, /\/api\/pull/);
});

test("Ollama upgrade preflight pins stable target and driver minimum", () => {
  assert.match(source, /TARGET_OLLAMA_VERSION = "0\.34\.3"/);
  assert.match(source, /MIN_NVIDIA_DRIVER_VERSION = "551\.61"/);
  assert.match(source, /driver_version/);
});

test("Ollama upgrade preflight preserves exact Qwen digest", () => {
  assert.match(source, /MODEL_NAME = "qwen3:1\.7b"/);
  assert.match(
    source,
    /8f68893c685c3ddff2aa3fffce2aa60a30bb2da65ca488b61fff134a4d1730e7/,
  );
  assert.match(source, /modelDigestValid/);
});

test("Ollama upgrade preflight is loopback-only", () => {
  assert.match(source, /http:\/\/127\.0\.0\.1:11434/);
  assert.doesNotMatch(source, /https:\/\//);
});
