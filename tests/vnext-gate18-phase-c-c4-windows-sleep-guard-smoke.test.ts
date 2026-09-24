import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

const source = readFileSync(
  "scripts/vnext-gate18-phase-c-c4-windows-sleep-guard-smoke.ts",
  "utf8",
);

test("sleep guard smoke is non-inferential and makes no Ollama calls", () => {
  assert.match(source, /NO_INFERENCE/);
  assert.doesNotMatch(source, /fetch\(/);
  assert.doesNotMatch(source, /\/api\/generate|\/api\/chat/);
  assert.match(source, /modelInferenceExecuted: false/);
});

test("sleep guard smoke uses process-scoped SetThreadExecutionState", () => {
  assert.match(source, /SetThreadExecutionState/);
  assert.match(source, /ES_CONTINUOUS/);
  assert.match(source, /ES_SYSTEM_REQUIRED/);
  assert.doesNotMatch(source, /ES_DISPLAY_REQUIRED|ES_AWAYMODE_REQUIRED/);
});

test("sleep guard smoke does not persistently mutate the power plan", () => {
  assert.match(source, /persistentPowerPlanMutation: false/);
  assert.doesNotMatch(source, /\/change|\/setactive|\/setacvalueindex|\/setdcvalueindex/i);
});

test("sleep guard smoke observes power requests before during and after", () => {
  assert.match(source, /powercfg\.exe/);
  assert.match(source, /"\/requests"/);
  assert.match(source, /before/);
  assert.match(source, /during/);
  assert.match(source, /after/);
});

test("sleep guard smoke does not claim protection against lid or manual sleep", () => {
  assert.match(source, /lidOrManualSleepPreventionConcluded: false/);
  assert.match(source, /actualSleepPreventionProven: false/);
});
