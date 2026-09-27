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

test("Phase C advances from Ministral 3 3B pinned download to context4096 load-only", () => {
  const entry = JSON.parse(
    readFileSync(
      "calibration/vnext/OROTITAN_GATE18_PHASE_C_ENTRY_V0.1.json",
      "utf8",
    ),
  );

  assert.equal(
    entry.next_action,
    "EXECUTE_MINISTRAL3_3B_CONTEXT4096_LOAD_ONLY_MEMORY_PREFLIGHT",
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
  assert.equal(
    entry.gemma3_first_c4_second_forensic_status,
    "FORENSIC_COMPLETE_ALL_KNOWN_DETERMINISTIC_DEFECTS_EXHAUSTED",
  );
  assert.equal(entry.gemma3_first_c4_unknown_conflict_id, "C-006");
  assert.equal(entry.gemma3_first_c4_unknown_conflict_ref_count, 3);
  assert.equal(
    entry.gemma3_first_c4_downstream_semantic_pass_after_cumulative_diagnostic_normalization,
    true,
  );
  assert.equal(entry.gemma3_c4_expansion_status, "STOPPED_RETAIN_CALIBRATION_EVIDENCE");
  assert.equal(entry.gemma3_family_global_failure_concluded, false);
  assert.equal(entry.next_candidate, "MINISTRAL3_3B_INSTRUCT_2512_Q4_K_M");
  assert.equal(entry.granite4_3b_model_name, "granite4:3b");
  assert.equal(entry.granite4_3b_download_authorized, false);
  assert.equal(entry.granite4_3b_download_executed, true);
  assert.equal(entry.granite4_3b_download_status, "PASS_PINNED_DOWNLOAD_ONLY");
  assert.equal(
    entry.granite4_3b_full_digest,
    "89962fcc75239ac434cdebceb6b7e0669397f92eaef9c487774b718bc36a3e5f",
  );
  assert.equal(entry.granite4_3b_context4096_load_authorized, false);
  assert.equal(entry.granite4_3b_context4096_load_authorized_run_count, 0);
  assert.equal(entry.granite4_3b_context4096_load_status, "PASS_LOAD_ONLY_MEASURED");
  assert.equal(entry.granite4_3b_context4096_vram_used_mib, 2313);
  assert.equal(entry.granite4_3b_context4096_vram_free_mib, 1650);
  assert.equal(entry.granite4_3b_context4096_loaded_free_ram_gib, 1.06);
  assert.equal(entry.granite4_3b_context4096_processor_split, "15%/85% CPU/GPU");
  assert.equal(entry.granite4_3b_context8192_load_authorized, false);
  assert.equal(entry.granite4_3b_context8192_load_authorized_run_count, 0);
  assert.equal(entry.granite4_3b_context8192_load_status, "PASS_LOAD_ONLY_MEASURED");
  assert.equal(entry.granite4_3b_context8192_vram_used_mib, 2355);
  assert.equal(entry.granite4_3b_context8192_vram_free_mib, 1608);
  assert.equal(entry.granite4_3b_context8192_loaded_free_ram_gib, 0.94);
  assert.equal(entry.granite4_3b_context8192_processor_split, "22%/78% CPU/GPU");
  assert.equal(entry.granite4_3b_context16384_load_authorized, false);
  assert.equal(entry.granite4_3b_context16384_load_authorized_run_count, 0);
  assert.equal(entry.granite4_3b_context16384_load_status, "PASS_LOAD_ONLY_MEASURED");
  assert.equal(entry.granite4_3b_context16384_vram_used_mib, 2305);
  assert.equal(entry.granite4_3b_context16384_vram_free_mib, 1658);
  assert.equal(entry.granite4_3b_context16384_loaded_free_ram_gib, 0.41);
  assert.equal(entry.granite4_3b_context16384_processor_split, "38%/62% CPU/GPU");
  assert.equal(entry.granite4_3b_hardware_qualification, "PASS_WITH_HIGH_RAM_PRESSURE");
  assert.equal(entry.granite4_first_c4_company, "Constellation Software");
  assert.equal(entry.granite4_first_c4_inference_authorized, false);
  assert.equal(entry.granite4_first_c4_authorized_run_count, 0);
  assert.equal(entry.granite4_first_c4_result_status, "ENGINEERING_PASS_HUMAN_QUALITY_CRITICAL_FAILURE");
  assert.equal(entry.granite4_3b_inference_authorized, false);
  assert.equal(entry.qwen3_8b_download_authorized, false);
  assert.equal(entry.qwen3_8b_download_executed, true);
  assert.equal(entry.qwen3_8b_download_status, "PASS_PINNED_DOWNLOAD_ONLY");
  assert.equal(entry.qwen3_8b_model_name, "qwen3:8b-q4_K_M");
  assert.equal(entry.qwen3_8b_expected_digest_prefix, "500a1f067a9f");
  assert.equal(
    entry.qwen3_8b_full_digest,
    "500a1f067a9f782620b40bee6f7b0c89e17ae61f686b92c24933e4ca4b2b8b41",
  );
  assert.equal(entry.qwen3_8b_size_bytes, 5225388164);
  assert.equal(entry.qwen3_8b_context4096_load_authorized, false);
  assert.equal(entry.qwen3_8b_context4096_load_authorized_run_count, 0);
  assert.equal(entry.qwen3_8b_context4096_load_status, "PASS_LOAD_ONLY_MEASURED");
  assert.equal(entry.qwen3_8b_context4096_loaded_free_ram_gib, 0.21);
  assert.equal(entry.qwen3_8b_context4096_vram_used_mib, 2279);
  assert.equal(entry.qwen3_8b_context4096_vram_free_mib, 1684);
  assert.equal(entry.qwen3_8b_context4096_processor_split, "61%/39% CPU/GPU");
  assert.equal(entry.qwen3_8b_context4096_hardware_fit, "PASS_WITH_CRITICAL_RAM_PRESSURE");
  assert.equal(entry.qwen3_8b_context8192_load_authorized, false);
  assert.equal(entry.qwen3_8b_context8192_load_authorized_run_count, 0);
  assert.equal(entry.qwen3_8b_context8192_load_status, "PASS_LOAD_ONLY_MEASURED");
  assert.equal(entry.qwen3_8b_context8192_loaded_free_ram_gib, 0.22);
  assert.equal(entry.qwen3_8b_context8192_vram_used_mib, 2323);
  assert.equal(entry.qwen3_8b_context8192_vram_free_mib, 1640);
  assert.equal(entry.qwen3_8b_context8192_processor_split, "64%/36% CPU/GPU");
  assert.equal(entry.qwen3_8b_context8192_hardware_fit, "PASS_WITH_CRITICAL_RAM_PRESSURE");
  assert.equal(entry.qwen3_8b_context16384_load_authorized, false);
  assert.equal(entry.qwen3_8b_context16384_load_authorized_run_count, 0);
  assert.equal(entry.qwen3_8b_context16384_load_status, "PASS_LOAD_ONLY_MEASURED");
  assert.equal(entry.qwen3_8b_context16384_loaded_free_ram_gib, 0.14);
  assert.equal(entry.qwen3_8b_context16384_vram_used_mib, 2345);
  assert.equal(entry.qwen3_8b_context16384_vram_free_mib, 1618);
  assert.equal(entry.qwen3_8b_context16384_processor_split, "70%/30% CPU/GPU");
  assert.equal(entry.qwen3_8b_context16384_hardware_fit, "PASS_WITH_EXTREME_RAM_PRESSURE");
  assert.equal(entry.qwen3_8b_hardware_qualification, "EXPERIMENTAL_INFERENCE_ONLY_EXTREME_RAM_PRESSURE");
  assert.equal(entry.qwen3_8b_first_c4_company, "Constellation Software");
  assert.equal(entry.qwen3_8b_first_c4_context_tokens, 16384);
  assert.equal(entry.qwen3_8b_first_c4_max_output_tokens, 1024);
  assert.equal(entry.qwen3_8b_first_c4_think, false);
  assert.equal(entry.qwen3_8b_first_c4_minimum_free_ram_gib, 1.0);
  assert.equal(entry.qwen3_8b_first_c4_inference_authorized, false);
  assert.equal(entry.qwen3_8b_first_c4_authorized_run_count, 0);
  assert.equal(entry.qwen3_8b_first_c4_result_status, "ENGINEERING_PASS_HUMAN_QUALITY_CRITICAL_FAILURE");
  assert.equal(entry.qwen3_8b_first_c4_engineering_pass, true);
  assert.equal(entry.qwen3_8b_first_c4_raw_presentation_compliant, true);
  assert.equal(entry.qwen3_8b_first_c4_normalized_path_count, 0);
  assert.equal(entry.qwen3_8b_first_c4_semantic_valid, true);
  assert.equal(entry.qwen3_8b_first_c4_eval_count, 860);
  assert.equal(entry.qwen3_8b_first_c4_output_token_margin, 164);
  assert.equal(entry.qwen3_8b_first_c4_human_adjudication_completed, true);
  assert.equal(entry.qwen3_8b_first_c4_human_quality, "CRITICAL_FAILURE");
  assert.equal(entry.qwen3_8b_c4_expansion_status, "STOPPED_RETAIN_CALIBRATION_EVIDENCE");
  assert.equal(entry.qwen3_8b_family_global_failure_concluded, false);
  assert.equal(entry.qwen3_8b_inference_authorized, false);
  assert.equal(entry.next_candidate_after_granite, "QWEN3_8B_LOCAL");
  assert.equal(entry.qwen3_8b_user_directed_test_after_granite, true);
  assert.equal(entry.qwen3_8b_direct_inference_authorized, false);
  assert.equal(entry.ministral3_3b_model_name, "ministral-3:3b-instruct-2512-q4_K_M");
  assert.equal(entry.ministral3_3b_expected_digest_prefix, "f04aa1c738f6");
  assert.equal(entry.ministral3_3b_download_authorized, false);
  assert.equal(entry.ministral3_3b_download_executed, true);
  assert.equal(entry.ministral3_3b_download_status, "PASS_PINNED_DOWNLOAD_ONLY");
  assert.equal(entry.ministral3_3b_load_smoke_authorized, true);
  assert.equal(entry.ministral3_3b_context4096_load_authorized_run_count, 1);
  assert.equal(entry.ministral3_3b_inference_authorized, false);
});
