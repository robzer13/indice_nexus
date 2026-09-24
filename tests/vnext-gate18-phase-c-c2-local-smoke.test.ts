import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

const source = readFileSync(
  "scripts/vnext-gate18-phase-c-c2-local-smoke.ts",
  "utf8",
);

test("Phase C C2 smoke is loopback-only and local-inference-only", () => {
  assert.match(source, /http:\/\/127\.0\.0\.1:11434/);
  assert.doesNotMatch(source, /https:\/\//);
  assert.match(source, /externalModelApiCall: false/);
  assert.match(source, /modelDownloadExecuted: false/);
});

test("Phase C C2 smoke pins the existing phi model by exact digest", () => {
  assert.match(source, /MODEL_NAME = "phi:latest"/);
  assert.match(
    source,
    /e2fd6321a5fe6bb3ac8a4e6f1cf04477fd2dea2924cf53237a995387e152ee9c/,
  );
  assert.match(source, /PINNED_MODEL_DIGEST_MISMATCH/);
});

test("Phase C C2 smoke cannot download or call paid model infrastructure", () => {
  assert.doesNotMatch(source, /\/api\/pull/);
  assert.doesNotMatch(source, /ai-gateway/i);
  assert.doesNotMatch(source, /vercel/i);
  assert.doesNotMatch(source, /openai\/gpt/i);
});

test("Phase C C2 smoke enforces deterministic bounded generation", () => {
  assert.match(source, /temperature: 0/);
  assert.match(source, /num_ctx: 2048/);
  assert.match(source, /num_predict: 128/);
  assert.match(source, /keep_alive: "0s"/);
  assert.match(source, /format: "json"/);
});

test("Phase C C2 smoke validates exact synthetic semantics", () => {
  assert.match(source, /"E-001"/);
  assert.match(source, /synthetic-local-smoke/);
  assert.match(source, /OUTPUT_KEYSET_MISMATCH/);
  assert.match(source, /COUNTEREVIDENCE_MISMATCH/);
  assert.match(source, /exactModelIdentityValid: true/);
});
