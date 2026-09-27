import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

test("Granite 4.1 context16384 result is persisted and load auth consumed", () => {
  const r = JSON.parse(
    readFileSync(
      "calibration/vnext/OROTITAN_GATE18_PHASE_C_GRANITE4_1_3B_CONTEXT16384_LOAD_SMOKE_RESULT_001.json",
      "utf8",
    ),
  );
  const a = JSON.parse(
    readFileSync(
      "calibration/vnext/OROTITAN_GATE18_PHASE_C_GRANITE4_1_3B_CONTEXT16384_LOAD_SMOKE_AUTH_001.json",
      "utf8",
    ),
  );

  assert.equal(r.status, "PASS_LOAD_ONLY_MEASURED");
  assert.equal(r.context_tokens, 16384);
  assert.equal(r.measured.loaded_free_ram_gib, 0.84);
  assert.equal(r.measured.loaded_vram_used_mib, 2305);
  assert.equal(r.measured.loaded_vram_free_mib, 1658);
  assert.equal(r.measured.processor_split, "38%/62% CPU/GPU");
  assert.equal(r.measured.explicit_unload_complete, true);
  assert.equal(r.safety.semantic_inference_executed, false);
  assert.equal(
    r.interpretation.hardware_fit_at_16384,
    "PASS_WITH_USEFUL_HEADROOM",
  );

  assert.equal(a.status, "CONSUMED_SINGLE_LOAD_ONLY_COMPLETE");
  assert.equal(a.authority.load_smoke_authorized, false);
  assert.equal(a.authority.inference_authorized, false);
  assert.equal(a.authority.authorized_run_count, 0);
  assert.equal(
    a.execution_result,
    "G18-PHASEC-GRANITE4_1-3B-CONTEXT16384-LOAD-SMOKE-RESULT-001",
  );
});

test("Granite 4.1 Constellation C4 prep preserves the frozen comparison cell", () => {
  const p = JSON.parse(
    readFileSync(
      "calibration/vnext/OROTITAN_GATE18_PHASE_C_C4_CONSTELLATION_GRANITE4_1_3B_V1_1_PREP_001.json",
      "utf8",
    ),
  );

  assert.equal(p.status, "AUTHORIZED_READY_TO_EXECUTE");
  assert.equal(p.candidate.model, "granite4.1:3b-q4_K_M");
  assert.equal(
    p.candidate.digest,
    "6fd349357287c7ffc9e38189a93b48ea175d24fc566b38f09cfc564fb7f303eb",
  );
  assert.equal(p.cell.company, "Constellation Software");
  assert.equal(p.cell.evidence_count, 11);
  assert.equal(p.cell.conflict_count, 1);
  assert.equal(
    p.cell.packet_sha256,
    "9a47bcf15d0c90da55349cbb3fbb2da8e9645b859a535dc8909501e89a20b6d8",
  );
  assert.equal(
    p.cell.prompt_sha256,
    "0891fa34d02a47c83e8342d5da5e653a3566d90f1c63c1bfc662f4e6097c10b8",
  );
  assert.equal(p.cell.prompt_bytes, 9041);
  assert.equal(p.parameters.context_tokens, 16384);
  assert.equal(p.parameters.max_output_tokens, 1024);
  assert.equal(p.parameters.temperature, 0);
  assert.equal(p.parameters.client_timeout_ms, 600000);
  assert.equal(p.parameters.pre_inference_minimum_free_ram_gib, 1);
  assert.equal(p.contract.human_adjudication_required, true);
});

test("Granite 4.1 C4 authorization is one local run only", () => {
  const a = JSON.parse(
    readFileSync(
      "calibration/vnext/OROTITAN_GATE18_PHASE_C_C4_CONSTELLATION_GRANITE4_1_3B_V1_1_AUTH_001.json",
      "utf8",
    ),
  );

  assert.equal(a.status, "AUTHORIZED_SINGLE_LOCAL_INFERENCE");
  assert.equal(a.c4_inference.authorized, true);
  assert.equal(a.c4_inference.model_name, "granite4.1:3b-q4_K_M");
  assert.equal(a.c4_inference.context_tokens, 16384);
  assert.equal(a.c4_inference.max_output_tokens, 1024);
  assert.equal(a.c4_inference.temperature, 0);
  assert.equal(a.c4_inference.client_timeout_ms, 600000);
  assert.equal(a.c4_inference.pre_inference_minimum_free_ram_gib, 1);
  assert.equal(a.constraints.authorized_run_count, 1);
  assert.equal(a.constraints.automatic_retry_authorized, false);
  assert.equal(a.constraints.prompt_change_authorized, false);
  assert.equal(a.constraints.packet_change_authorized, false);
  assert.equal(a.constraints.context_change_authorized, false);
  assert.equal(a.constraints.model_switch_authorized, false);
  assert.equal(a.output_policy.public_repo_generated_content_forbidden, true);
  assert.equal(a.output_policy.human_adjudication_required_if_engineering_pass, true);
});

test("Granite 4.1 C4 runner is pinned, guarded, and private-output only", () => {
  const raw = readFileSync(
    "scripts/vnext-gate18-phase-c-c4-constellation-granite4-1-3b-v1-1-context16384-output1024-timeout600-loopback-guarded.ts",
    "utf8",
  );

  assert.match(raw, /const MODEL_NAME = "granite4\.1:3b-q4_K_M";/);
  assert.match(
    raw,
    /6fd349357287c7ffc9e38189a93b48ea175d24fc566b38f09cfc564fb7f303eb/,
  );
  assert.match(raw, /const CONTEXT_TOKENS = 16384;/);
  assert.match(raw, /const MAX_OUTPUT_TOKENS = 1024;/);
  assert.match(raw, /const CLIENT_TIMEOUT_MS = 600_000;/);
  assert.match(raw, /pre_inference_minimum_free_ram_gib/);
  assert.match(raw, /baselineFreeRamGiB/);
  assert.match(raw, /calibration\/vnext\/private-runs/);
  assert.match(raw, /humanAdjudicationRequired: true/);
  assert.doesNotMatch(raw, /MINISTRAL3|ministral-3|Ministral/);
});
