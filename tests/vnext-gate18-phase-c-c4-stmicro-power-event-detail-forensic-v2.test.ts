import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

const source = readFileSync(
  "scripts/vnext-gate18-phase-c-c4-stmicro-power-event-detail-forensic-v2.ts",
  "utf8",
);

test("V2 power-event forensic remains local read-only and non-inferential", () => {
  assert.match(source, /LOCAL_READ_ONLY_NO_INFERENCE/);
  assert.doesNotMatch(source, /fetch\(/);
  assert.doesNotMatch(source, /\/api\/generate|\/api\/chat/);
  assert.match(source, /modelInferenceExecuted: false/);
});

test("V2 queries each provider and event id independently", () => {
  for (const id of ["42", "107", "506", "507"]) {
    assert.match(source, new RegExp(`Id=${id}`));
  }
  assert.match(source, /Microsoft-Windows-Power-Troubleshooter/);
  assert.match(source, /Id=1/);
  assert.match(source, /foreach\(\$spec in \$specs\)/);
});

test("V2 captures per-query failures instead of failing the whole forensic", () => {
  assert.match(source, /QuerySucceeded=\$false/);
  assert.match(source, /Error=\$_.Exception.Message/);
  assert.match(source, /failedQueryCount/);
});

test("V2 captures message, XML and named EventData fields", () => {
  assert.match(source, /Message=\$evt.Message/);
  assert.match(source, /Xml=\$xmlText/);
  assert.match(source, /EventData.Data/);
  assert.match(source, /Data=\$data/);
});

test("V2 makes no automatic causal conclusion", () => {
  assert.match(source, /sleepWakeTimingConcluded: false/);
  assert.match(source, /timeoutDelayRootCauseConcluded: false/);
  assert.match(source, /modelCapabilityFailureConcluded: false/);
});
