import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

test("Qwen3 8B post-hardware decision authorizes only one bounded think-false C4 experiment", () => {
  const d = JSON.parse(
    readFileSync(
      "calibration/vnext/OROTITAN_GATE18_QWEN3_8B_POST_HARDWARE_DECISION_001.json",
      "utf8",
    ),
  );

  assert.equal(d.status, "ONE_BOUNDED_CONSTELLATION_C4_THINK_FALSE_EXPERIMENT_SELECTED");
  assert.equal(d.hardware_basis.context16384, "PASS_WITH_EXTREME_RAM_PRESSURE");
  assert.equal(d.hardware_basis.context16384_loaded_free_ram_gib, 0.14);
  assert.equal(d.selected_experiment.company, "Constellation Software");
  assert.equal(d.selected_experiment.context_tokens, 16384);
  assert.equal(d.selected_experiment.max_output_tokens, 1024);
  assert.equal(d.selected_experiment.temperature, 0);
  assert.equal(d.selected_experiment.client_timeout_ms, 600000);
  assert.equal(d.selected_experiment.think, false);
  assert.equal(d.safeguards.pre_inference_minimum_free_ram_gib, 1.0);
  assert.equal(d.safeguards.automatic_retry, false);
  assert.equal(d.interpretation_boundary.production_fit, false);
  assert.equal(d.interpretation_boundary.model_ranking_authority, false);
});

test("Qwen3 8B Constellation C4 authorization freezes one experimental local inference", () => {
  const a = JSON.parse(
    readFileSync(
      "calibration/vnext/OROTITAN_GATE18_PHASE_C_C4_CONSTELLATION_QWEN3_8B_V1_1_AUTH_001.json",
      "utf8",
    ),
  );

  assert.equal(a.status, "AUTHORIZED_SINGLE_LOCAL_INFERENCE");
  assert.equal(a.c4_inference.authorized, true);
  assert.equal(a.c4_inference.experiment_kind, "USER_DIRECTED_EXPERIMENTAL_SAME_PACKET_C4");
  assert.equal(a.c4_inference.company, "Constellation Software");
  assert.equal(a.c4_inference.model_name, "qwen3:8b-q4_K_M");
  assert.equal(
    a.c4_inference.model_digest,
    "500a1f067a9f782620b40bee6f7b0c89e17ae61f686b92c24933e4ca4b2b8b41",
  );
  assert.equal(a.c4_inference.context_tokens, 16384);
  assert.equal(a.c4_inference.max_output_tokens, 1024);
  assert.equal(a.c4_inference.temperature, 0);
  assert.equal(a.c4_inference.client_timeout_ms, 600000);
  assert.equal(a.c4_inference.think, false);
  assert.equal(a.c4_inference.pre_inference_minimum_free_ram_gib, 1.0);
  assert.equal(a.constraints.authorized_run_count, 1);
  assert.equal(a.constraints.automatic_retry_authorized, false);
  assert.equal(a.constraints.thinking_mode_change_authorized, false);
  assert.equal(a.interpretation_boundary.production_candidate_decision_authority, false);
});

test("Qwen3 8B C4 runner preserves packet contract, disables thinking, and guards baseline RAM", () => {
  const raw = readFileSync(
    "scripts/vnext-gate18-phase-c-c4-constellation-qwen3-8b-v1-1-context16384-output1024-timeout600-thinkfalse-loopback-guarded.ts",
    "utf8",
  );

  assert.match(raw, /qwen3:8b-q4_K_M/);
  assert.match(raw, /500a1f067a9f782620b40bee6f7b0c89e17ae61f686b92c24933e4ca4b2b8b41/);
  assert.match(raw, /CONTEXT_TOKENS = 16384/);
  assert.match(raw, /MAX_OUTPUT_TOKENS = 1024/);
  assert.match(raw, /CLIENT_TIMEOUT_MS = 600_000/);
  assert.match(raw, /think:\s*false/);
  assert.match(raw, /temperature:\s*0/);
  assert.match(raw, /num_ctx:\s*CONTEXT_TOKENS/);
  assert.match(raw, /num_predict:\s*MAX_OUTPUT_TOKENS/);
  assert.match(raw, /keep_alive:\s*"0s"/);
  assert.match(raw, /pre_inference_minimum_free_ram_gib/);
  assert.match(raw, /baselineFreeRamGiB/);
  assert.match(raw, /INSUFFICIENT_BASELINE_FREE_RAM/);
  assert.match(raw, /evaluateGate18V11Validation/);
  assert.match(raw, /0891fa34d02a47c83e8342d5da5e653a3566d90f1c63c1bfc662f4e6097c10b8/);
  assert.match(raw, /9a47bcf15d0c90da55349cbb3fbb2da8e9645b859a535dc8909501e89a20b6d8/);
  assert.match(raw, /humanAdjudicationRequired:\s*true/);
  assert.match(raw, /comparisonAdmissible:\s*false/);
  assert.match(raw, /modelRankingAuthority:\s*false/);
  assert.doesNotMatch(raw, /qwen3\.5/i);
});

test("Qwen3 8B C4 prep keeps generation and validation contracts pinned", () => {
  const p = JSON.parse(
    readFileSync(
      "calibration/vnext/OROTITAN_GATE18_PHASE_C_C4_CONSTELLATION_QWEN3_8B_V1_1_PREP_001.json",
      "utf8",
    ),
  );

  assert.equal(p.status, "AUTHORIZED_READY_TO_EXECUTE");
  assert.equal(p.parameters.context_tokens, 16384);
  assert.equal(p.parameters.max_output_tokens, 1024);
  assert.equal(p.parameters.temperature, 0);
  assert.equal(p.parameters.client_timeout_ms, 600000);
  assert.equal(p.parameters.think, false);
  assert.equal(p.contract.generation_prompt_contract, "V1_0_UNCHANGED");
  assert.equal(p.contract.generation_schema_contract, "V1_0_UNCHANGED");
  assert.equal(p.contract.validation_contract, "GATE18_MOAT_EVIDENCE_AUDIT_VALIDATION_V1_1");
  assert.equal(p.hardware_carry.context16384_fit, "PASS_WITH_EXTREME_RAM_PRESSURE");
  assert.equal(p.hardware_carry.inference_fit, "EXPERIMENTAL_ONLY_NOT_PROVEN");
});
