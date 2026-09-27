import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

test("Gemma 3 first C4 discriminator is same-packet Constellation", () => {
  const d = JSON.parse(
    readFileSync(
      "calibration/vnext/OROTITAN_GATE18_GEMMA3_FIRST_C4_DISCRIMINATOR_DECISION_001.json",
      "utf8",
    ),
  );

  assert.equal(d.status, "CONSTELLATION_SAME_PACKET_SELECTED");
  assert.equal(d.selected_cell.company, "Constellation Software");
  assert.equal(
    d.selected_cell.packet_sha256,
    "9a47bcf15d0c90da55349cbb3fbb2da8e9645b859a535dc8909501e89a20b6d8",
  );
  assert.equal(
    d.selected_cell.prompt_sha256,
    "0891fa34d02a47c83e8342d5da5e653a3566d90f1c63c1bfc662f4e6097c10b8",
  );
  assert.equal(d.boundaries.model_ranking_authority, false);
});

test("Gemma 3 Constellation authorization is consumed after one local C4 inference", () => {
  const a = JSON.parse(
    readFileSync(
      "calibration/vnext/OROTITAN_GATE18_PHASE_C_C4_CONSTELLATION_GEMMA3_4B_V1_1_AUTH_001.json",
      "utf8",
    ),
  );

  assert.equal(a.status, "CONSUMED_SINGLE_LOCAL_INFERENCE");
  assert.equal(a.c4_inference.authorized, false);
  assert.equal(a.c4_inference.company, "Constellation Software");
  assert.equal(a.c4_inference.model_name, "gemma3:4b-it-q4_K_M");
  assert.equal(
    a.c4_inference.model_digest,
    "a2af6cc3eb7fa8be8504abaf9b04e88f17a119ec3f04a3addf55f92841195f5a",
  );
  assert.equal(a.c4_inference.context_tokens, 16384);
  assert.equal(a.c4_inference.max_output_tokens, 1024);
  assert.equal(a.c4_inference.temperature, 0);
  assert.equal(a.c4_inference.client_timeout_ms, 600000);
  assert.equal(a.constraints.authorized_run_count, 0);
  assert.equal(a.execution_result, "G18-PHASEC-C4-CONSTELLATION-GEMMA3-4B-V1_1-RESULT-001");
  assert.equal(a.constraints.automatic_retry_authorized, false);
  assert.equal(a.gemma_terms_boundary.assist_only, true);
  assert.equal(a.gemma_terms_boundary.autonomous_financial_decision_authority, false);
});

test("Gemma 3 Constellation runner preserves packet/prompt and private-output boundaries", () => {
  const raw = readFileSync(
    "scripts/vnext-gate18-phase-c-c4-constellation-gemma3-4b-v1-1-context16384-output1024-timeout600-loopback-guarded.ts",
    "utf8",
  );

  assert.match(raw, /gemma3:4b-it-q4_K_M/);
  assert.match(
    raw,
    /a2af6cc3eb7fa8be8504abaf9b04e88f17a119ec3f04a3addf55f92841195f5a/,
  );
  assert.match(
    raw,
    /9a47bcf15d0c90da55349cbb3fbb2da8e9645b859a535dc8909501e89a20b6d8/,
  );
  assert.match(
    raw,
    /0891fa34d02a47c83e8342d5da5e653a3566d90f1c63c1bfc662f4e6097c10b8/,
  );
  assert.match(raw, /CONTEXT_TOKENS = 16384/);
  assert.match(raw, /MAX_OUTPUT_TOKENS = 1024/);
  assert.match(raw, /CLIENT_TIMEOUT_MS = 600_000/);
  assert.match(raw, /evaluateGate18V11Validation/);
  assert.match(raw, /calibration\/vnext\/private-runs/);
  assert.match(raw, /humanAdjudicationRequired: true/);
  assert.match(raw, /comparisonAdmissible: false/);
  assert.match(raw, /modelRankingAuthority: false/);
  assert.doesNotMatch(raw, /think:\s*false/);
  assert.doesNotMatch(raw, /qwen3\.5/i);
});

test("Gemma 3 Constellation prep keeps generation and validation contracts unchanged", () => {
  const p = JSON.parse(
    readFileSync(
      "calibration/vnext/OROTITAN_GATE18_PHASE_C_C4_CONSTELLATION_GEMMA3_4B_V1_1_PREP_001.json",
      "utf8",
    ),
  );

  assert.equal(p.status, "AUTHORIZED_READY_TO_EXECUTE");
  assert.equal(p.contract.generation_prompt_contract, "V1_0_UNCHANGED");
  assert.equal(p.contract.generation_schema_contract, "V1_0_UNCHANGED");
  assert.equal(p.contract.validation_contract, "GATE18_MOAT_EVIDENCE_AUDIT_VALIDATION_V1_1");
  assert.equal(p.contract.raw_output_preserved, true);
  assert.equal(p.contract.human_adjudication_required, true);
});


test("Gemma 3 Constellation result is a schema-valid deterministic semantic failure", () => {
  const r = JSON.parse(
    readFileSync(
      "calibration/vnext/OROTITAN_GATE18_PHASE_C_C4_CONSTELLATION_GEMMA3_4B_V1_1_RESULT_001.json",
      "utf8",
    ),
  );

  assert.equal(r.status, "FAIL_DETERMINISTIC_SEMANTIC_CONTRACT");
  assert.equal(r.execution.done_reason, "stop");
  assert.equal(r.execution.eval_count, 683);
  assert.equal(r.execution.schema_valid, true);
  assert.equal(r.execution.semantic_valid, false);
  assert.equal(
    r.execution.semantic_error,
    "VNEXT_GATE18_V10_COUNTEREVIDENCE_LINK_WITHOUT_IDS",
  );
  assert.equal(r.validation_v1_1.normalized_path_count, 13);
  assert.equal(r.validation_v1_1.substantive_status, "FAIL");
  assert.equal(r.interpretation.retry_authorized, false);
  assert.equal(r.authority.additional_inference_authorized, false);
});
