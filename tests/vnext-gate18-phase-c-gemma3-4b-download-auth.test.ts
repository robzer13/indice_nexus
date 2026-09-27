import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

test("Gemma 3 4B download authorization is consumed after one pinned download", () => {
  const auth = JSON.parse(
    readFileSync(
      "calibration/vnext/OROTITAN_GATE18_PHASE_C_GEMMA3_4B_DOWNLOAD_AUTH_001.json",
      "utf8",
    ),
  );

  assert.equal(auth.status, "CONSUMED_SINGLE_DOWNLOAD_ONLY");
  assert.equal(auth.user_authorization.explicit, true);
  assert.equal(auth.user_authorization.gemma_terms_accepted, true);
  assert.equal(auth.model.ollama_model_name, "gemma3:4b-it-q4_K_M");
  assert.equal(auth.model.expected_digest_prefix, "a2af6cc3eb7f");
  assert.equal(auth.model.expected_quantization, "Q4_K_M");
  assert.equal(auth.execution.action_count_authorized, 1);
  assert.equal(auth.constraints.inference_after_download, false);
  assert.equal(auth.constraints.load_smoke_after_download, false);
  assert.equal(auth.constraints.automatic_retry, false);
  assert.equal(auth.consumption.consumed, true);
  assert.equal(auth.consumption.result_status, "PASS_PINNED_DOWNLOAD_ONLY");
  assert.equal(auth.authority.gemma3_download_authorized, false);
  assert.equal(auth.authority.gemma3_load_smoke_authorized, false);
  assert.equal(auth.authority.gemma3_inference_authorized, false);
});

test("Gemma 3 4B download runner pulls exact tag and contains no generation or load-smoke execution", () => {
  const raw = readFileSync(
    "scripts/vnext-gate18-phase-c-gemma3-4b-download-verify.ts",
    "utf8",
  );

  assert.match(raw, /gemma3:4b-it-q4_K_M/);
  assert.match(raw, /ollama", \["pull", MODEL\]/);
  assert.match(raw, /a2af6cc3eb7f/);
  assert.match(raw, /\/api\/tags/);
  assert.match(raw, /\/api\/show/);
  assert.doesNotMatch(raw, /\/api\/generate/);
  assert.doesNotMatch(raw, /\/api\/chat/);
  assert.doesNotMatch(raw, /keep_alive/);
  assert.doesNotMatch(raw, /prompt\s*:/);
  assert.match(raw, /modelInferenceExecuted: false/);
  assert.match(raw, /loadSmokeExecuted: false/);
});

test("Phase C advances from Gemma 3 first forensic to unknown-conflict-reference forensic", () => {
  const entry = JSON.parse(
    readFileSync(
      "calibration/vnext/OROTITAN_GATE18_PHASE_C_ENTRY_V0.1.json",
      "utf8",
    ),
  );

  assert.equal(
    entry.next_action,
    "RUN_LOCAL_READ_ONLY_GEMMA3_CONSTELLATION_UNKNOWN_CONFLICT_REF_FORENSIC",
  );
  assert.equal(entry.gemma3_terms_user_accepted, true);
  assert.equal(entry.gemma3_download_authorized, false);
  assert.equal(entry.gemma3_download_executed, true);
  assert.equal(entry.gemma3_context4096_load_authorized, false);
  assert.equal(entry.gemma3_context4096_load_authorized_run_count, 0);
  assert.equal(entry.gemma3_context4096_load_status, "PASS_LOAD_ONLY_MEASURED");
  assert.equal(entry.gemma3_context8192_load_authorized, false);
  assert.equal(entry.gemma3_context8192_load_authorized_run_count, 0);
  assert.equal(entry.gemma3_context8192_load_status, "PASS_LOAD_ONLY_MEASURED");
  assert.equal(entry.gemma3_context16384_load_authorized, false);
  assert.equal(entry.gemma3_context16384_load_authorized_run_count, 0);
  assert.equal(entry.gemma3_context16384_load_status, "PASS_LOAD_ONLY_MEASURED");
  assert.equal(entry.gemma3_hardware_qualification, "PASS_WITH_HIGH_RAM_PRESSURE");
  assert.equal(entry.gemma3_first_c4_company, "Constellation Software");
  assert.equal(entry.gemma3_first_c4_inference_authorized, false);
  assert.equal(entry.gemma3_first_c4_authorized_run_count, 0);
  assert.equal(entry.gemma3_first_c4_result_status, "FAIL_DETERMINISTIC_SEMANTIC_CONTRACT");
  assert.equal(entry.gemma3_first_c4_semantic_error, "VNEXT_GATE18_V10_COUNTEREVIDENCE_LINK_WITHOUT_IDS");
  assert.equal(entry.gemma3_first_c4_forensic_required, true);
  assert.equal(
    entry.gemma3_first_c4_first_forensic_status,
    "FORENSIC_COMPLETE_ADDITIONAL_SEMANTIC_DEFECT_FOUND",
  );
  assert.deepEqual(
    entry.gemma3_first_c4_counterevidence_link_violating_finding_indexes,
    [1, 2, 3],
  );
  assert.equal(
    entry.gemma3_first_c4_downstream_semantic_error_after_first_forensic,
    "VNEXT_GATE18_V10_UNKNOWN_CONFLICT_REF",
  );
  assert.equal(entry.gemma3_first_c4_second_forensic_required, true);
});
