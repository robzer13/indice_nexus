import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

const source = readFileSync(
  "scripts/vnext-gate18-phase-c-c3-brookfield-qwen4b-local-run.ts",
  "utf8",
);

test("Qwen3 4B Brookfield runner pins exact model identity", () => {
  assert.match(source, /MODEL_NAME = "qwen3:4b-instruct"/);
  assert.match(
    source,
    /0edcdef34593eac1aa2be9c7d06c432dcf81945adca5eca2f27662c18f168ba0/,
  );
  assert.match(source, /MODEL_DIGEST_MISMATCH/);
});

test("Qwen3 4B Brookfield runner pins bounded 4096 context", () => {
  assert.match(source, /CONTEXT_TOKENS = 4096/);
  assert.match(source, /MAX_OUTPUT_TOKENS = 768/);
  assert.match(source, /temperature: 0/);
  assert.match(source, /keep_alive: "0s"/);
});

test("Qwen3 4B Brookfield runner requires exact future authorization", () => {
  assert.match(
    source,
    /G18-PHASEC-C3-BROOKFIELD-QWEN4B-INFERENCE-AUTH-001/,
  );
  assert.match(
    source,
    /OROTITAN_GATE18_PHASE_C_C3_BROOKFIELD_QWEN3_4B_INFERENCE_AUTH_001\.json/,
  );
  assert.match(source, /AUTHORIZED_SINGLE_LOCAL_INFERENCE/);
  assert.match(source, /context_tokens !== CONTEXT_TOKENS/);
  assert.match(source, /max_output_tokens !== MAX_OUTPUT_TOKENS/);
  assert.match(source, /temperature !== 0/);
});

test("Qwen3 4B Brookfield runner preserves exact targeted semantics", () => {
  for (const id of ["E-036", "E-037", "E-039", "E-042"]) {
    assert.match(source, new RegExp(id));
  }
  assert.match(
    source,
    /BROOKFIELD_PEER_ROLE_CORE_001_LOCAL_COMPACT_QWEN4B_4096/,
  );
  assert.match(source, /assertGate18V10BrookfieldTargetedProbeSemantics/);
});

test("Qwen3 4B Brookfield runner requires no preloaded Ollama model", () => {
  assert.match(source, /ollama", \["ps"\]/);
  assert.match(source, /QWEN4B_OTHER_MODEL_ALREADY_LOADED/);
  assert.match(source, /QWEN4B_OLLAMA_PS_CHECK_FAILED/);
});

test("Qwen3 4B Brookfield runner remains local and private", () => {
  assert.match(source, /http:\/\/127\.0\.0\.1:11434/);
  assert.match(source, /externalModelApiCostUsd: 0/);
  assert.match(source, /calibration\/vnext\/private-runs/);
  assert.match(source, /comparisonAdmissible: false/);
  assert.match(source, /modelRankingAuthority: false/);
});
