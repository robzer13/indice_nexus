import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

const source = readFileSync(
  "scripts/vnext-gate18-phase-c-c4-stmicro-qwen4b-context16384-local-run.ts",
  "utf8",
);

test("C4 STMicro runner pins exact Qwen3 4B context16384 envelope", () => {
  assert.match(source, /MODEL_NAME = "qwen3:4b-instruct"/);
  assert.match(
    source,
    /0edcdef34593eac1aa2be9c7d06c432dcf81945adca5eca2f27662c18f168ba0/,
  );
  assert.match(source, /CONTEXT_TOKENS = 16384/);
  assert.match(source, /MAX_OUTPUT_TOKENS = 768/);
  assert.match(source, /CLIENT_TIMEOUT_MS = 300_000/);
  assert.match(source, /temperature: 0/);
});

test("C4 STMicro runner uses the full pinned V10 packet and prompt", () => {
  assert.match(source, /findSTMicroelectronics/);
  assert.match(source, /buildVerifiedGate18V10MoatPacket/);
  assert.match(source, /buildGate18PhaseBV10ModelInput/);
  assert.match(source, /GATE18_PHASE_B_V10_SYSTEM_PROMPT/);
  assert.match(source, /assertGate18PhaseBV10Semantics/);
  assert.doesNotMatch(source, /buildCompactPacket|compactPacket/);
});

test("C4 STMicro runner pins static-preflight prompt identity", () => {
  assert.match(
    source,
    /3f06d6e22dbbc88abbbc64fd470811f438a595c71154f7fcb8c004a6d4f5bc91/,
  );
  assert.match(source, /EXPECTED_PROMPT_BYTES = 7558/);
  assert.match(source, /EXPECTED_EVIDENCE_COUNT = 9/);
  assert.match(source, /EXPECTED_CONFLICT_COUNT = 2/);
  assert.match(source, /PROMPT_IDENTITY_MISMATCH/);
});

test("C4 STMicro runner is one-shot authorization gated and fail closed", () => {
  assert.match(
    source,
    /G18-PHASEC-C4-STM-QWEN4B-CONTEXT16384-INFERENCE-AUTH-001/,
  );
  assert.match(source, /AUTHORIZED_SINGLE_LOCAL_INFERENCE/);
  assert.match(source, /--authorization-id/);
  assert.match(source, /QWEN4B_OTHER_MODEL_ALREADY_LOADED/);
  assert.match(source, /comparisonAdmissible: false/);
  assert.match(source, /modelRankingAuthority: false/);
});

test("C4 STMicro runner keeps output private and performs no cloud fallback", () => {
  assert.match(source, /calibration\/vnext\/private-runs/);
  assert.match(source, /http:\/\/127\.0\.0\.1:11434/);
  assert.match(source, /externalModelApiCostUsd: 0/);
  assert.doesNotMatch(source, /ai-gateway|VERCEL_OIDC_TOKEN|generateText\(/);
});
