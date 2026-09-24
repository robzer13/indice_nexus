import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

test("STMicro Ollama log forensics V3 fixes $Matches collision and remains non-inference", () => {
  const prep = JSON.parse(
    readFileSync(
      "calibration/vnext/OROTITAN_GATE18_PHASE_C_C4_STMICRO_OLLAMA_LOG_FORENSICS_V3_PREP_001.json",
      "utf8",
    ),
  ) as {
    status: string;
    implementation_fix: {
      forbidden_variable_name: string;
      replacement_variable_name: string;
    };
    authority: {
      inference_authorized: boolean;
      retry_authorized: boolean;
      model_load_authorized: boolean;
      parameter_change_authorized: boolean;
    };
  };

  const script = readFileSync(
    "scripts/vnext-gate18-phase-c-c4-stmicro-ollama-log-forensics-v3.ts",
    "utf8",
  );

  assert.equal(prep.status, "PREPARED_READ_ONLY_NO_INFERENCE");
  assert.equal(prep.implementation_fix.forbidden_variable_name, "matches");
  assert.equal(prep.implementation_fix.replacement_variable_name, "matchedLines");
  assert.equal(prep.authority.inference_authorized, false);
  assert.equal(prep.authority.retry_authorized, false);
  assert.equal(prep.authority.model_load_authorized, false);
  assert.equal(prep.authority.parameter_change_authorized, false);

  assert.match(script, /matchedLines/);
  assert.doesNotMatch(script, /\$matches\s*=/i);
  assert.match(script, /time=\(\?<ts>/);
  assert.match(script, /status=5\[0-9\]\[0-9\]/);
  assert.match(script, /ollamaApiCalled: false/);
  assert.doesNotMatch(script, /\/api\/generate/);
  assert.doesNotMatch(script, /ollama", \["run"/);
});
