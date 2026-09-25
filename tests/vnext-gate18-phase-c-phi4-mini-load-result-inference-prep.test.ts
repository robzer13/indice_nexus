import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

test("Phi-4 mini 4096 load result records measured headroom and full unload", () => {
  const result = JSON.parse(
    readFileSync(
      "calibration/vnext/OROTITAN_GATE18_PHASE_C_PHI4_MINI_LOAD_SMOKE_RESULT_001.json",
      "utf8",
    ),
  );

  assert.equal(result.status, "PASS_LOAD_ONLY_MEASURED");
  assert.equal(result.target.model, "phi4-mini:3.8b-q4_K_M");
  assert.equal(
    result.target.digest,
    "78fad5d182a7c33065e153a5f8ba210754207ba9d91973f57dffa7f487363753",
  );
  assert.equal(result.target.context_tokens_observed, 4096);
  assert.equal(result.loaded.vram_used_mib, 2295);
  assert.equal(result.loaded.vram_free_mib, 1668);
  assert.equal(result.loaded.free_ram_gib, 0.81);
  assert.equal(result.loaded.processor_split, "36%/64% CPU/GPU");
  assert.equal(result.derived_measurements.full_vram_release_after_unload, true);
  assert.equal(result.derived_measurements.ollama_model_residency_cleared_after_unload, true);
  assert.equal(result.after.vram_used_mib, 0);
  assert.equal(result.after.ollama_loaded_model_count, 0);
  assert.equal(result.interpretation.load_fit_at_4096_context, "PASS");
  assert.equal(result.interpretation.inference_fit_at_4096_context, "NOT_YET_EXECUTION_PROVEN");
  assert.equal(result.interpretation.system_ram_pressure, "HIGH");
  assert.equal(result.interpretation.bounded_4096_inference_preparation_allowed, true);
  assert.equal(result.interpretation.context_growth_allowed, false);
  assert.equal(result.interpretation.inference_authorized, false);
});

test("Phi-4 mini first semantic probe prep is compact, 4096, deterministic, and not authorized", () => {
  const prep = JSON.parse(
    readFileSync(
      "calibration/vnext/OROTITAN_GATE18_PHASE_C_PHI4_MINI_BROOKFIELD_COMPACT4096_INFERENCE_PREP_001.json",
      "utf8",
    ),
  );

  assert.equal(prep.status, "PREPARED_NOT_AUTHORIZED");
  assert.equal(prep.probe.semantic_probe_id, "BROOKFIELD_PEER_ROLE_CORE_001");
  assert.equal(
    prep.probe.execution_id,
    "BROOKFIELD_PEER_ROLE_CORE_001_LOCAL_COMPACT_PHI4MINI_4096",
  );
  assert.equal(prep.execution_plan.context_tokens, 4096);
  assert.equal(prep.execution_plan.max_output_tokens, 768);
  assert.equal(prep.execution_plan.temperature, 0);
  assert.equal(prep.execution_plan.client_timeout_ms, 180000);
  assert.equal(prep.resource_boundary.load_only_fit_proven, true);
  assert.equal(prep.resource_boundary.semantic_inference_fit_proven, false);
  assert.equal(prep.resource_boundary.system_ram_pressure, "HIGH");
  assert.equal(prep.resource_boundary.automatic_retry_authorized, false);
  assert.equal(prep.guard.authorization_artifact_present, false);
  assert.equal(prep.guard.inference_currently_authorized, false);
  assert.equal(prep.authority.phi4_mini_inference_authorized, false);
  assert.equal(prep.authority.context_growth_authorized, false);
  assert.equal(prep.authority.model_switch_authorized, false);
});

test("Phi-4 mini Brookfield runner is authorization-gated, digest-pinned, private, and bounded", () => {
  const raw = readFileSync(
    "scripts/vnext-gate18-phase-c-phi4-mini-brookfield-compact4096-local-run.ts",
    "utf8",
  );

  assert.match(raw, /phi4-mini:3\.8b-q4_K_M/);
  assert.match(raw, /78fad5d182a7c33065e153a5f8ba210754207ba9d91973f57dffa7f487363753/);
  assert.match(raw, /CONTEXT_TOKENS = 4096/);
  assert.match(raw, /MAX_OUTPUT_TOKENS = 768/);
  assert.match(raw, /G18-PHASEC-PHI4-MINI-BROOKFIELD-COMPACT4096-INFERENCE-AUTH-001/);
  assert.match(raw, /AUTHORIZED_SINGLE_LOCAL_INFERENCE/);
  assert.match(raw, /phi4_mini_inference\?\.authorized !== true/);
  assert.match(raw, /options\.execute/);
  assert.match(raw, /authorizationId !== AUTHORIZATION_ID/);
  assert.match(raw, /assertNoLoadedModels/);
  assert.match(raw, /temperature: 0/);
  assert.match(raw, /num_ctx: CONTEXT_TOKENS/);
  assert.match(raw, /num_predict: MAX_OUTPUT_TOKENS/);
  assert.match(raw, /keep_alive: "0s"/);
  assert.match(raw, /calibration\/vnext\/private-runs/);
  assert.match(raw, /externalModelApiCostUsd: 0/);
  assert.match(raw, /productionMutation: false/);
  assert.match(raw, /publicationAuthority: false/);
});

test("Phase C stops before Phi-4 mini semantic inference authorization", () => {
  const entry = JSON.parse(
    readFileSync(
      "calibration/vnext/OROTITAN_GATE18_PHASE_C_ENTRY_V0.1.json",
      "utf8",
    ),
  );

  assert.equal(entry.phi4_mini_load_fit_at_4096, "PASS");
  assert.equal(entry.phi4_mini_loaded_vram_free_mib, 1668);
  assert.equal(entry.phi4_mini_loaded_free_ram_gib, 0.81);
  assert.equal(entry.phi4_mini_bounded_4096_inference_preparation_allowed, true);
  assert.equal(entry.phi4_mini_context_growth_authorized, false);
  assert.equal(entry.phi4_mini_inference_authorized, false);
  assert.equal(entry.model_switch_authorized, false);
  assert.equal(
    entry.next_action,
    "REQUEST_EXPLICIT_USER_AUTHORIZATION_FOR_ONE_PHI4_MINI_BROOKFIELD_COMPACT4096_INFERENCE",
  );
});
