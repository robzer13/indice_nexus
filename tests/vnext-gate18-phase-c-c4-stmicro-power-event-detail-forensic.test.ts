import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

const source = readFileSync(
  "scripts/vnext-gate18-phase-c-c4-stmicro-power-event-detail-forensic.ts",
  "utf8",
);

test("power-event forensic is local read-only with no model calls", () => {
  assert.match(source, /LOCAL_READ_ONLY_NO_INFERENCE/);
  assert.doesNotMatch(source, /fetch\(/);
  assert.doesNotMatch(source, /\/api\/generate|\/api\/chat/);
  assert.match(source, /modelInferenceExecuted: false/);
});

test("power-event forensic captures full event messages and XML", () => {
  assert.match(source, /Message=\$_.Message/);
  assert.match(source, /Xml=\$_.ToXml\(\)/);
  assert.match(source, /TimeCreatedUtc/);
});

test("power-event forensic includes relevant providers and IDs", () => {
  assert.match(source, /Microsoft-Windows-Kernel-Power/);
  assert.match(source, /Microsoft-Windows-Power-Troubleshooter/);
  assert.match(source, /Id=42,107,506,507/);
  assert.match(source, /Id=1/);
});

test("power-event forensic makes no automatic causal conclusion", () => {
  assert.match(source, /sleepWakeTimingConcluded: false/);
  assert.match(source, /timeoutDelayRootCauseConcluded: false/);
  assert.match(source, /modelCapabilityFailureConcluded: false/);
});
