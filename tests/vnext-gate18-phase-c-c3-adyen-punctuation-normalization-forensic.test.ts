import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

const source = readFileSync(
  "scripts/vnext-gate18-phase-c-c3-adyen-punctuation-normalization-forensic.ts",
  "utf8",
);

test("Adyen punctuation forensic performs no inference or Ollama request", () => {
  assert.match(
    source,
    /IN_MEMORY_DIAGNOSTIC_ONLY_NO_ARTIFACT_MUTATION_NO_INFERENCE/,
  );
  assert.doesNotMatch(source, /fetch\(/);
  assert.doesNotMatch(source, /127\.0\.0\.1:11434|\/api\/generate|\/api\/chat/);
});

test("Adyen punctuation forensic only normalizes missing terminal punctuation", () => {
  assert.match(source, /terminalPunctuation/);
  assert.match(source, /normalizedConflictIds\.push/);
  assert.match(
    source,
    /Append one period only when terminal punctuation is absent/,
  );
  assert.match(source, /sourceArtifactMutated: false/);
});

test("Adyen punctuation forensic re-runs the exact full targeted validator", () => {
  assert.match(source, /assertGate18V10TargetedProbeSemantics/);
  assert.match(source, /buildVerifiedGate18V10MoatPacket/);
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

test("Adyen punctuation forensic cannot retroactively change run status", () => {
  assert.match(source, /retroactivePassAllowed: false/);
  assert.match(source, /originalRunStatusChanged: false/);
  assert.match(source, /modelCapabilityConclusionMade: false/);
});
