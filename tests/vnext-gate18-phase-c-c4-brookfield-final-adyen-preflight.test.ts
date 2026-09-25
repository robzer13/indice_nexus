import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

test("Brookfield human adjudication closes the C4 cell with engineering and human-quality failure", () => {
  const result = JSON.parse(
    readFileSync(
      "calibration/vnext/OROTITAN_GATE18_PHASE_C_C4_BROOKFIELD_HUMAN_ADJUDICATION_RESULT_001.json",
      "utf8",
    ),
  );

  assert.equal(
    result.status,
    "COMPLETED_WITH_ENGINEERING_FAIL_AND_HUMAN_QUALITY_CRITICAL_FAILURE",
  );
  assert.equal(result.engineering_disposition.semantic_validator_pass, false);
  assert.equal(
    result.engineering_disposition.deterministic_defect_class_count,
    3,
  );
  assert.equal(result.human_quality.claim_atomicity, "FAIL");
  assert.equal(result.human_quality.claim_target_alignment, "FAIL");
  assert.equal(result.human_quality.conflict_handling, "FAIL");
  assert.equal(result.human_quality.judgment_boundary_compliance, "PASS");
  assert.equal(result.disposition.automated_engineering_pass, false);
  assert.equal(result.disposition.human_adjudication_completed, true);
  assert.equal(result.disposition.critical_human_quality_failure, true);
  assert.equal(result.disposition.matrix_cell_completed, true);
  assert.equal(result.authority.retry_authorized, false);
  assert.equal(result.authority.auto_repair_authorized, false);
});

test("Adyen output1280 timeout600 static preflight is non-inference and exact-packet-bound", () => {
  const prep = JSON.parse(
    readFileSync(
      "calibration/vnext/OROTITAN_GATE18_PHASE_C_C4_ADYEN_OUTPUT1280_TIMEOUT600_STATIC_PREFLIGHT_PREP_001.json",
      "utf8",
    ),
  );
  const source = readFileSync(
    "scripts/vnext-gate18-phase-c-c4-adyen-output1280-timeout600-static-preflight.ts",
    "utf8",
  );

  assert.equal(prep.status, "PREPARED_NO_INFERENCE");
  assert.equal(prep.selection_basis.company, "Adyen");
  assert.equal(
    prep.selection_basis.archetype,
    "DIFFICULT_MOAT_CONFLICTING_EVIDENCE",
  );
  assert.equal(prep.runtime_basis.selected_context_tokens, 16384);
  assert.equal(prep.runtime_basis.selected_max_output_tokens, 1280);
  assert.equal(prep.runtime_basis.selected_client_timeout_ms, 600000);
  assert.equal(
    prep.expected_static_identity.packet_sha256,
    "89e14b09d58b1305064d76170a15bb31e93a0768737d94f2bfc19d32c39a9b74",
  );
  assert.equal(
    prep.expected_static_identity.prompt_sha256,
    "ef8f55478017eddfb26d68e652aa4855f9d474dc1a75114e6026a12e17666f0c",
  );
  assert.equal(prep.expected_static_identity.prompt_bytes, 11671);
  assert.equal(prep.expected_static_identity.evidence_count, 16);
  assert.equal(prep.expected_static_identity.conflict_count, 3);
  assert.equal(prep.authority.inference_authorized, false);
  assert.equal(prep.authority.retry_authorized, false);

  assert.match(source, /const COMPANY = "Adyen";/);
  assert.match(source, /const MAX_OUTPUT_TOKENS = 1280;/);
  assert.match(source, /const CLIENT_TIMEOUT_MS = 600_000;/);
  assert.match(source, /buildVerifiedGate18V10MoatPacket/);
  assert.match(source, /modelInferenceExecuted: false/);
  assert.match(source, /ollamaApiCalled: false/);
  assert.doesNotMatch(source, /requestLoopbackJson/);
  assert.doesNotMatch(source, /\/api\/generate/);
  assert.doesNotMatch(source, /\bfetch\s*\(/);
});
