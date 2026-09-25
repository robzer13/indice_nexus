import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

test("Adyen adjudication closes the matrix cell", () => {
  const result = JSON.parse(readFileSync("calibration/vnext/OROTITAN_GATE18_PHASE_C_C4_ADYEN_HUMAN_ADJUDICATION_RESULT_001.json","utf8"));
  assert.equal(result.disposition,"COMPLETED_WITH_ENGINEERING_FAIL_AND_HUMAN_QUALITY_CRITICAL_FAILURE");
  assert.equal(result.engineering.automated_engineering_pass,false);
  assert.equal(result.human_quality.critical_failure,true);
  assert.equal(result.matrix_cell.completed,true);
  assert.equal(result.authority.retry_authorized,false);
});

test("RATIONAL static preflight remains no-inference", () => {
  const prep = JSON.parse(readFileSync("calibration/vnext/OROTITAN_GATE18_PHASE_C_C4_RATIONAL_OUTPUT1280_TIMEOUT600_STATIC_PREFLIGHT_PREP_001.json","utf8"));
  const source = readFileSync("scripts/vnext-gate18-phase-c-c4-rational-output1280-timeout600-static-preflight.ts","utf8");
  assert.equal(prep.status,"PREPARED_NO_INFERENCE");
  assert.equal(prep.company.display_name,"RATIONAL AG");
  assert.equal(prep.company.evidence_count,56);
  assert.equal(prep.company.conflict_count,9);
  assert.equal(prep.interpretation_boundary.output1280_adequacy_concluded,false);
  assert.equal(prep.authority.inference_authorized,false);
  assert.match(source,/const COMPANY = "RATIONAL AG";/);
  assert.match(source,/const EXPECTED_PROMPT_BYTES = 35756;/);
  assert.match(source,/const EXPECTED_EVIDENCE_COUNT = 56;/);
  assert.match(source,/const EXPECTED_CONFLICT_COUNT = 9;/);
  assert.doesNotMatch(source,/requestLoopbackJson/);
  assert.doesNotMatch(source,/\/api\/generate/);
});
