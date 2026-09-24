import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

const source = readFileSync(
  "scripts/vnext-gate18-phase-c-c4-qwen4b-context16384-load-smoke.ts",
  "utf8",
);

test("C4 context16384 smoke pins exact Qwen3 4B identity and context", () => {
  assert.match(source, /qwen3:4b-instruct/);
  assert.match(
    source,
    /0edcdef34593eac1aa2be9c7d06c432dcf81945adca5eca2f27662c18f168ba0/,
  );
  assert.match(source, /CONTEXT_TOKENS = 16384/);
  assert.match(source, /num_ctx: CONTEXT_TOKENS/);
});

test("C4 context16384 smoke is load-only without prompt or messages", () => {
  assert.doesNotMatch(source, /prompt\s*:/);
  assert.doesNotMatch(source, /messages\s*:/);
  assert.match(source, /loadOnlyConfirmed/);
  assert.match(source, /modelInferenceExecuted: false/);
});

test("C4 context16384 smoke measures and unloads resources", () => {
  assert.match(source, /nvidia-smi/);
  assert.match(source, /os\.freemem/);
  assert.match(source, /ollamaPs/);
  assert.match(source, /keep_alive: 0/);
});

test("C4 context16384 smoke does not authorize inference", () => {
  assert.match(source, /c4InferenceAuthorized: false/);
  assert.match(source, /productionMutation: false/);
  assert.match(source, /publicationAuthority: false/);
});
