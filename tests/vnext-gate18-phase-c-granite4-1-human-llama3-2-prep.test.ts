import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

test("Granite 4.1 human adjudication preserves engineering pass but stops expansion", () => {
  const r = JSON.parse(
    readFileSync(
      "calibration/vnext/OROTITAN_GATE18_PHASE_C_C4_CONSTELLATION_GRANITE4_1_3B_V1_1_HUMAN_ADJUDICATION_RESULT_001.json",
      "utf8",
    ),
  );

  assert.equal(
    r.status,
    "COMPLETED_WITH_ENGINEERING_PASS_AND_HUMAN_QUALITY_CRITICAL_FAILURE",
  );
  assert.equal(r.engineering_disposition.runtime_pass, true);
  assert.equal(r.engineering_disposition.schema_valid, true);
  assert.equal(r.engineering_disposition.v1_1_raw_presentation_compliant, true);
  assert.equal(r.engineering_disposition.v1_1_safe_normalization_count, 0);
  assert.equal(r.engineering_disposition.output_token_margin, 589);
  assert.equal(r.human_quality.unresolved_point_usefulness, "FAIL");
  assert.equal(r.human_quality.priority_selection_usefulness, "FAIL");
  assert.equal(r.critical_failures.duplicate_priority_finding_critical_failure, true);
  assert.equal(r.critical_failures.unresolved_point_usefulness_critical_failure, true);
  assert.equal(r.critical_failures.priority_selection_critical_failure, true);
  assert.equal(r.authority.retry_authorized, false);
  assert.equal(r.authority.new_granite4_1_inference_authorized, false);
});

test("Granite 4.1 post-Constellation disposition activates Llama 3.2", () => {
  const d = JSON.parse(
    readFileSync(
      "calibration/vnext/OROTITAN_GATE18_PHASE_C_GRANITE4_1_POST_CONSTELLATION_DISPOSITION_001.json",
      "utf8",
    ),
  );

  assert.equal(
    d.status,
    "STOP_GRANITE4_1_C4_EXPANSION_RETAIN_CALIBRATION_EVIDENCE",
  );
  assert.equal(d.interpretation.granite4_1_candidate_admitted, false);
  assert.equal(d.interpretation.retry_authorized, false);
  assert.equal(d.next_candidate.candidate_id, "LLAMA3_2_3B_OLLAMA_Q4_K_M");
  assert.equal(d.next_candidate.model_id, "llama3.2:3b-instruct-q4_K_M");
});

test("Llama 3.2 download authorization is pinned and download-only", () => {
  const a = JSON.parse(
    readFileSync(
      "calibration/vnext/OROTITAN_GATE18_PHASE_C_LLAMA3_2_3B_DOWNLOAD_AUTH_001.json",
      "utf8",
    ),
  );

  assert.equal(a.status, "AUTHORIZED_SINGLE_DOWNLOAD_ONLY");
  assert.equal(a.model.ollama_model_name, "llama3.2:3b-instruct-q4_K_M");
  assert.equal(a.model.expected_digest_prefix, "a80c4f17acd5");
  assert.equal(a.model.expected_quantization, "Q4_K_M");
  assert.equal(a.execution.action_count_authorized, 1);
  assert.equal(a.authority.llama3_2_download_authorized, true);
  assert.equal(a.authority.llama3_2_load_smoke_authorized, false);
  assert.equal(a.authority.llama3_2_inference_authorized, false);
  assert.equal(a.constraints.automatic_retry, false);
  assert.equal(a.constraints.automatic_model_switch, false);
});

test("Llama 3.2 download verifier contains no load or inference path", () => {
  const raw = readFileSync(
    "scripts/vnext-gate18-phase-c-llama3-2-3b-download-verify.ts",
    "utf8",
  );

  assert.match(raw, /const MODEL = "llama3\.2:3b-instruct-q4_K_M";/);
  assert.match(raw, /const EXPECTED_DIGEST_PREFIX = "a80c4f17acd5";/);
  assert.match(raw, /ollama", \["pull", MODEL\]/);
  assert.doesNotMatch(raw, /\/api\/generate|\/api\/chat/);
  assert.match(raw, /modelInferenceExecuted: false/);
  assert.match(raw, /loadSmokeExecuted: false/);
});
