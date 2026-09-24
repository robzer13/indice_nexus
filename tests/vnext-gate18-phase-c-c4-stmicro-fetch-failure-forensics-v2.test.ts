import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

test("STMicro fetch-failure forensics V2 isolates probes and remains non-inference", () => {
  const prep = JSON.parse(
    readFileSync(
      "calibration/vnext/OROTITAN_GATE18_PHASE_C_C4_STMICRO_FETCH_FAILURE_FORENSICS_V2_PREP_001.json",
      "utf8",
    ),
  ) as {
    status: string;
    design_delta: {
      monolithic_powershell_collector: boolean;
      independent_powershell_probes: boolean;
      encoded_command_transport: boolean;
      per_probe_stderr_capture: boolean;
      per_probe_failure_isolation: boolean;
    };
    authority: {
      inference_authorized: boolean;
      retry_authorized: boolean;
      model_load_authorized: boolean;
      parameter_change_authorized: boolean;
    };
  };

  const script = readFileSync(
    "scripts/vnext-gate18-phase-c-c4-stmicro-fetch-failure-forensics-v2.ts",
    "utf8",
  );

  assert.equal(prep.status, "PREPARED_READ_ONLY_NO_INFERENCE");
  assert.equal(prep.design_delta.monolithic_powershell_collector, false);
  assert.equal(prep.design_delta.independent_powershell_probes, true);
  assert.equal(prep.design_delta.encoded_command_transport, true);
  assert.equal(prep.design_delta.per_probe_stderr_capture, true);
  assert.equal(prep.design_delta.per_probe_failure_isolation, true);
  assert.equal(prep.authority.inference_authorized, false);
  assert.equal(prep.authority.retry_authorized, false);
  assert.equal(prep.authority.model_load_authorized, false);
  assert.equal(prep.authority.parameter_change_authorized, false);

  assert.match(script, /-EncodedCommand/);
  assert.match(script, /successfulProbeCount/);
  assert.match(script, /resourceExhaustion/);
  assert.match(script, /applicationError/);
  assert.match(script, /windowsErrorReporting/);
  assert.match(script, /ollamaLogs/);
  assert.match(script, /ollamaGenerateApiCalled: false/);
  assert.doesNotMatch(script, /\/api\/generate/);
  assert.doesNotMatch(script, /ollama", \["run"/);
});
