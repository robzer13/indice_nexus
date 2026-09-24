import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

const source = readFileSync(
  "scripts/vnext-gate18-phase-c-c3-brookfield-local-run.ts",
  "utf8",
);

test("Brookfield local C3 runner is loopback-only and zero external API cost", () => {
  assert.match(source, /http:\/\/127\.0\.0\.1:11434/);
  assert.doesNotMatch(source, /https:\/\//);
  assert.match(source, /externalModelApiCostUsd: 0/);
});

test("Brookfield local C3 runner pins exact Qwen3 model identity", () => {
  assert.match(source, /MODEL_NAME = "qwen3:1\.7b"/);
  assert.match(
    source,
    /8f68893c685c3ddff2aa3fffce2aa60a30bb2da65ca488b61fff134a4d1730e7/,
  );
  assert.match(source, /MODEL_DIGEST_MISMATCH/);
});

test("Brookfield local C3 runner requires explicit authorization artifact and execute flag", () => {
  assert.match(source, /AUTHORIZED_SINGLE_LOCAL_INFERENCE/);
  assert.match(source, /--execute/);
  assert.match(source, /--authorization-id/);
  assert.match(
    source,
    /OROTITAN_GATE18_PHASE_C_C3_BROOKFIELD_QWEN3_1_7B_INFERENCE_AUTH_001\.json/,
  );
  assert.match(source, /AUTHORIZATION_ARTIFACT_MISSING/);
});

test("Brookfield local C3 runner preserves the exact targeted regression", () => {
  for (const id of ["E-036", "E-037", "E-039", "E-042"]) {
    assert.match(source, new RegExp(id));
  }
  assert.match(source, /BROOKFIELD_PEER_ROLE_CORE_001_LOCAL_COMPACT/);
  assert.match(source, /assertGate18V10BrookfieldTargetedProbeSemantics/);
});

test("Brookfield local C3 runner uses structured schema and bounded 8k execution", () => {
  assert.match(source, /format: GATE18_PHASE_B_V10_GENERATION_SCHEMA_SPEC/);
  assert.match(source, /CONTEXT_TOKENS = 8192/);
  assert.match(source, /MAX_OUTPUT_TOKENS = 768/);
  assert.match(source, /temperature: 0/);
  assert.match(source, /keep_alive: "0s"/);
});

test("Brookfield local C3 runner writes generated content only to gitignored private-runs", () => {
  assert.match(source, /calibration\/vnext\/private-runs/);
  assert.match(source, /privateArtifact: true/);
  assert.match(source, /humanAdjudicationRequired: true/);
  assert.match(source, /comparisonAdmissible: false/);
  assert.match(source, /modelRankingAuthority: false/);
});
