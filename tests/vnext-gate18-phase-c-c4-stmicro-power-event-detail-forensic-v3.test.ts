import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

const source = readFileSync(
  "scripts/vnext-gate18-phase-c-c4-stmicro-power-event-detail-forensic-v3.ts",
  "utf8",
);

test("V3 remains local read-only and non-inferential", () => {
  assert.match(source, /LOCAL_READ_ONLY_NO_INFERENCE/);
  assert.doesNotMatch(source, /fetch\(/);
  assert.doesNotMatch(source, /\/api\/generate|\/api\/chat/);
  assert.match(source, /modelInferenceExecuted: false/);
});

test("V3 preserves multiline PowerShell source with newlines", () => {
  assert.match(source, /\.join\("\\n"\)/);
  assert.doesNotMatch(source, /\.join\(";"\)/);
});

test("V3 keeps independent provider and event queries", () => {
  for (const id of ["42", "107", "506", "507"]) {
    assert.match(source, new RegExp(`Id=${id}`));
  }
  assert.match(source, /Microsoft-Windows-Power-Troubleshooter/);
  assert.match(source, /Id=1/);
  assert.match(source, /foreach\(\$spec in \$specs\)/);
});

test("V3 captures message XML EventData and per-query errors", () => {
  assert.match(source, /Message=\$evt.Message/);
  assert.match(source, /Xml=\$xmlText/);
  assert.match(source, /EventData.Data/);
  assert.match(source, /QuerySucceeded=\$false/);
  assert.match(source, /Error=\$_.Exception.Message/);
});

test("V3 makes no automatic causal conclusion", () => {
  assert.match(source, /sleepWakeTimingConcluded: false/);
  assert.match(source, /timeoutDelayRootCauseConcluded: false/);
  assert.match(source, /modelCapabilityFailureConcluded: false/);
});
