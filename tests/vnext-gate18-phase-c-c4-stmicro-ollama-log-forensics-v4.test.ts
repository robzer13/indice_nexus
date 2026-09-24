import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

test("STMicro Ollama log forensics V4 uses base64 JSON transport and remains non-inference", () => {
  const prep = JSON.parse(
    readFileSync(
      "calibration/vnext/OROTITAN_GATE18_PHASE_C_C4_STMICRO_OLLAMA_LOG_FORENSICS_V4_PREP_001.json",
      "utf8",
    ),
  ) as {
    status: string;
    transport_fix: {
      powershell_json_encoding: string;
      powershell_stdout_encoding: string;
      node_transport_decode: string;
      node_json_parse_after_decode: boolean;
      raw_stdout_success_no_longer_equates_to_parse_success: boolean;
    };
    authority: {
      inference_authorized: boolean;
      retry_authorized: boolean;
      model_load_authorized: boolean;
      parameter_change_authorized: boolean;
    };
  };

  const script = readFileSync(
    "scripts/vnext-gate18-phase-c-c4-stmicro-ollama-log-forensics-v4.ts",
    "utf8",
  );

  assert.equal(prep.status, "PREPARED_READ_ONLY_NO_INFERENCE");
  assert.equal(prep.transport_fix.powershell_json_encoding, "UTF8");
  assert.equal(prep.transport_fix.powershell_stdout_encoding, "BASE64_ASCII");
  assert.equal(prep.transport_fix.node_transport_decode, "BASE64_TO_UTF8");
  assert.equal(prep.transport_fix.node_json_parse_after_decode, true);
  assert.equal(
    prep.transport_fix.raw_stdout_success_no_longer_equates_to_parse_success,
    true,
  );
  assert.equal(prep.authority.inference_authorized, false);
  assert.equal(prep.authority.retry_authorized, false);
  assert.equal(prep.authority.model_load_authorized, false);
  assert.equal(prep.authority.parameter_change_authorized, false);

  assert.match(script, /ToBase64String/);
  assert.match(script, /Buffer\.from\(candidate, "base64"\)/);
  assert.match(script, /JSON\.parse\(decodedText\)/);
  assert.match(script, /base64DecodeAndJsonParseSucceeded/);
  assert.match(script, /ollamaApiCalled: false/);
  assert.doesNotMatch(script, /\/api\/generate/);
  assert.doesNotMatch(script, /ollama", \["run"/);
});
