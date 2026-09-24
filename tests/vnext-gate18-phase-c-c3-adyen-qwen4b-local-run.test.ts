import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

const source = readFileSync(
  "scripts/vnext-gate18-phase-c-c3-adyen-qwen4b-local-run.ts",
  "utf8",
);

test("Adyen Qwen3 4B runner pins exact local model and bounded decoding", () => {
  assert.match(source, /MODEL_NAME = "qwen3:4b-instruct"/);
  assert.match(
    source,
    /0edcdef34593eac1aa2be9c7d06c432dcf81945adca5eca2f27662c18f168ba0/,
  );
  assert.match(source, /CONTEXT_TOKENS = 4096/);
  assert.match(source, /MAX_OUTPUT_TOKENS = 768/);
  assert.match(source, /temperature: 0/);
  assert.match(source, /keep_alive: "0s"/);
});

test("Adyen Qwen3 4B runner replays the exact v1.0 targeted regression", () => {
  assert.match(source, /model-calibration-targeted-regression-v10/);
  assert.match(source, /GATE18_V10_TARGETED_PROBE_ID/);
  assert.match(source, /assertGate18V10TargetedProbeSemantics/);
  assert.match(source, /ADYEN_CLAIM_TARGET_CORE_001_LOCAL_COMPACT_QWEN4B_4096/);
  for (const id of [
    "E-036",
    "E-037",
    "E-040",
    "E-041",
    "E-042",
    "E-043",
    "E-055",
    "E-056",
    "C-005",
    "C-010",
  ]) {
    assert.match(source, new RegExp(id));
  }
});

test("Adyen Qwen3 4B runner requires a separate future authorization", () => {
  assert.match(
    source,
    /G18-PHASEC-C3-ADYEN-QWEN4B-INFERENCE-AUTH-001/,
  );
  assert.match(
    source,
    /OROTITAN_GATE18_PHASE_C_C3_ADYEN_QWEN3_4B_INFERENCE_AUTH_001\.json/,
  );
  assert.match(source, /AUTHORIZED_SINGLE_LOCAL_INFERENCE/);
  assert.match(source, /context_tokens !== CONTEXT_TOKENS/);
  assert.match(source, /max_output_tokens !== MAX_OUTPUT_TOKENS/);
  assert.match(source, /temperature !== 0/);
});

test("Adyen Qwen3 4B runner is local, private and fail closed", () => {
  assert.match(source, /http:\/\/127\.0\.0\.1:11434/);
  assert.match(source, /externalModelApiCostUsd: 0/);
  assert.match(source, /calibration\/vnext\/private-runs/);
  assert.match(source, /QWEN4B_OTHER_MODEL_ALREADY_LOADED/);
  assert.match(source, /comparisonAdmissible: false/);
  assert.match(source, /modelRankingAuthority: false/);
});
