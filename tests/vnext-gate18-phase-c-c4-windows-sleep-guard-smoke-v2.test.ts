import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

const source = readFileSync(
  "scripts/vnext-gate18-phase-c-c4-windows-sleep-guard-smoke-v2.ts",
  "utf8",
);

test("V2 sleep guard smoke remains non-inferential", () => {
  assert.match(source, /NO_INFERENCE/);
  assert.doesNotMatch(source, /fetch\(/);
  assert.doesNotMatch(source, /\/api\/generate|\/api\/chat/);
  assert.match(source, /modelInferenceExecuted: false/);
});

test("V2 constructs ES_CONTINUOUS without signed literal cast failure", () => {
  assert.match(source, /Convert\]::ToUInt32\('80000000', 16\)/);
  assert.doesNotMatch(source, /\[uint32\]0x80000000/);
  assert.match(source, /\$flags = \[uint32\]\(\$ES_CONTINUOUS -bor \$ES_SYSTEM_REQUIRED\)/);
});

test("V2 proves direct API acquisition and explicit release", () => {
  assert.match(source, /SetThreadExecutionState\(\$flags\)/);
  assert.match(source, /SetThreadExecutionState\(\$ES_CONTINUOUS\)/);
  assert.match(source, /OROTITAN_SLEEP_GUARD_READY/);
  assert.match(source, /OROTITAN_SLEEP_GUARD_RELEASED/);
  assert.match(source, /PASS_API_GUARD_ACQUIRED_AND_RELEASED/);
});

test("V2 treats powercfg telemetry as optional", () => {
  assert.match(source, /verificationRequiredForPass: false/);
  assert.match(source, /powercfgTelemetryRequiredForPass: false/);
});

test("V2 does not persistently modify Windows power policy", () => {
  assert.match(source, /persistentPowerPlanMutation: false/);
  assert.doesNotMatch(source, /\/change|\/setactive|\/setacvalueindex|\/setdcvalueindex/i);
  assert.doesNotMatch(source, /ES_DISPLAY_REQUIRED|ES_AWAYMODE_REQUIRED/);
});
