import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

const source = readFileSync(
  "scripts/vnext-gate18-phase-c-c4-stmicro-post-abort-runtime-forensics.ts",
  "utf8",
);

test("C4 STMicro runtime forensic performs no inference or Ollama API request", () => {
  assert.match(source, /LOCAL_READ_ONLY_NO_INFERENCE/);
  assert.doesNotMatch(source, /fetch\(/);
  assert.doesNotMatch(source, /\/api\/generate|\/api\/chat|127\.0\.0\.1:11434/);
  assert.match(source, /modelInferenceExecuted: false/);
});

test("C4 STMicro runtime forensic inspects current Ollama and GPU state", () => {
  assert.match(source, /ollama/);
  assert.match(source, /nvidia-smi/);
  assert.match(source, /os\.freemem/);
  assert.match(source, /loadedModelCount/);
});

test("C4 STMicro runtime forensic inspects Windows sleep and resume events", () => {
  assert.match(source, /Microsoft-Windows-Kernel-Power/);
  assert.match(source, /Microsoft-Windows-Power-Troubleshooter/);
  assert.match(source, /Id=42,107,506,507/);
  assert.match(source, /Id=1/);
  assert.match(source, /sleepOrResumeEventObserved/);
});

test("C4 STMicro runtime forensic does not infer causality", () => {
  assert.match(source, /sleepOrResumeEventCausalityConcluded: false/);
  assert.match(source, /timeoutGuardDelayRootCauseConcluded: false/);
  assert.match(source, /modelCapabilityFailureConcluded: false/);
  assert.match(source, /semanticFailureConcluded: false/);
});
