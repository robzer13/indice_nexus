import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

const source = readFileSync(
  "scripts/vnext-gate18-phase-c-qwen4b-load-smoke.ts",
  "utf8",
);

test("Qwen3 4B load smoke pins exact model identity", () => {
  assert.match(source, /qwen3:4b-instruct/);
  assert.match(
    source,
    /0edcdef34593eac1aa2be9c7d06c432dcf81945adca5eca2f27662c18f168ba0/,
  );
  assert.match(source, /QWEN3_4B_IDENTITY_MISMATCH/);
});

test("Qwen3 4B load smoke performs loopback load-only requests", () => {
  assert.match(source, /http:\/\/127\.0\.0\.1:11434/);
  assert.match(source, /promptProvided: false/);
  assert.match(source, /inferenceRequested: false/);
  assert.match(source, /keep_alive: "2m"/);
  assert.match(source, /keep_alive: 0/);
  assert.doesNotMatch(source, /prompt:/);
  assert.doesNotMatch(source, /messages:/);
});

test("Qwen3 4B load smoke measures resource state before, during and after load", () => {
  assert.match(source, /const before =/);
  assert.match(source, /const loaded =/);
  assert.match(source, /const after =/);
  assert.match(source, /memory\.total,memory\.free,memory\.used/);
  assert.match(source, /ollama", \["ps"\]/);
  assert.match(source, /os\.freemem\(\)/);
});

test("Qwen3 4B load smoke remains non-inference and fail closed", () => {
  assert.match(source, /LOCAL_MODEL_LOAD_WITHOUT_INFERENCE/);
  assert.match(source, /modelInferenceExecuted: false/);
  assert.match(source, /c3InferenceAuthorized: false/);
  assert.match(source, /PASS_LOAD_ONLY_MEASURED/);
  assert.match(source, /FAIL_LOAD_ONLY_GUARD/);
  assert.match(source, /process\.exitCode = 1/);
});
