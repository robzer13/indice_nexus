import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

const source = readFileSync(
  "scripts/vnext-gate18-phase-c-c4-qwen4b-static-request-preflight.ts",
  "utf8",
);

test("C4 preflight performs no inference or Ollama request", () => {
  assert.match(source, /NO_INFERENCE_NO_OLLAMA_REQUEST/);
  assert.doesNotMatch(source, /fetch\(/);
  assert.doesNotMatch(source, /127\.0\.0\.1:11434|\/api\/generate|\/api\/chat/);
});

test("C4 preflight spans all five pinned companies through pilotJson", () => {
  assert.match(source, /pilotJson\.companies\.map/);
  assert.match(source, /MOAT_EVIDENCE_AUDIT_ASSISTED_V0_3/);
  assert.match(source, /GATE18_MOAT_EVIDENCE_AUDIT_V0_8/);
});

test("C4 preflight pins Qwen3 4B generation envelope", () => {
  assert.match(source, /MODEL_NAME = "qwen3:4b-instruct"/);
  assert.match(source, /CONTEXT_TOKENS = 4096/);
  assert.match(source, /MAX_OUTPUT_TOKENS = 768/);
  assert.match(source, /temperature: 0/);
});

test("C4 preflight reports exact sizes and hashes", () => {
  assert.match(source, /promptSha256/);
  assert.match(source, /requestSha256/);
  assert.match(source, /promptBytes/);
  assert.match(source, /requestBytes/);
  assert.match(source, /evidenceCount/);
  assert.match(source, /conflictCount/);
});
