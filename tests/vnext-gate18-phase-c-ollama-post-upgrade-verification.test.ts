import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

const source = readFileSync(
  "scripts/vnext-gate18-phase-c-ollama-post-upgrade-verification.ts",
  "utf8",
);

test("post-upgrade verification cannot install, infer, or pull models", () => {
  assert.match(source, /NO_INSTALL_NO_INFERENCE_POSTCHECK/);
  assert.match(source, /installerExecutedByThisScript: false/);
  assert.match(source, /modelInferenceExecuted: false/);
  assert.match(source, /modelDownloadExecuted: false/);
  assert.doesNotMatch(source, /install\.ps1/);
  assert.doesNotMatch(source, /OllamaSetup\.exe/);
  assert.doesNotMatch(source, /\/api\/generate/);
  assert.doesNotMatch(source, /\/api\/chat/);
  assert.doesNotMatch(source, /\/api\/pull/);
});

test("post-upgrade verification pins Ollama 0.34.3 and driver minimum", () => {
  assert.match(source, /EXPECTED_OLLAMA_VERSION = "0\.34\.3"/);
  assert.match(source, /MIN_NVIDIA_DRIVER_VERSION = "551\.61"/);
  assert.match(source, /BLOCKED_DRIVER/);
});

test("post-upgrade verification preserves exact Qwen3 digest", () => {
  assert.match(source, /MODEL_NAME = "qwen3:1\.7b"/);
  assert.match(
    source,
    /8f68893c685c3ddff2aa3fffce2aa60a30bb2da65ca488b61fff134a4d1730e7/,
  );
  assert.match(source, /exactDigestValid/);
});

test("post-upgrade verification checks CLI/API coherence and local endpoints", () => {
  assert.match(source, /where\.exe/);
  assert.match(source, /"\/api\/version"/);
  assert.match(source, /"\/api\/tags"/);
  assert.match(source, /"\/api\/ps"/);
  assert.match(source, /binaryApiCoherent/);
});

test("post-upgrade verification cannot authorize inference or driver upgrade", () => {
  assert.match(source, /c3InferenceAuthorized: false/);
  assert.match(source, /structuredOutputSmokeAuthorized: false/);
  assert.match(source, /driverUpgradeAuthorized: false/);
});
