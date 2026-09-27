import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

test("Granite 4 first C4 discriminator is same-packet Constellation", () => {
  const d = JSON.parse(
    readFileSync(
      "calibration/vnext/OROTITAN_GATE18_GRANITE4_FIRST_C4_DISCRIMINATOR_DECISION_001.json",
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
  assert.equal(d.hardware_basis.context16384_load_fit, "PASS_WITH_HIGH_RAM_PRESSURE");
  assert.equal(d.boundaries.model_ranking_authority, false);
});

test("Granite 4 Constellation authorization permits exactly one local C4 inference", () => {
  const a = JSON.parse(
    readFileSync(
      "calibration/vnext/OROTITAN_GATE18_PHASE_C_C4_CONSTELLATION_GRANITE4_3B_V1_1_AUTH_001.json",
      "utf8",
    ),
  );

  assert.equal(a.status, "AUTHORIZED_SINGLE_LOCAL_INFERENCE");
  assert.equal(a.c4_inference.authorized, true);
  assert.equal(a.c4_inference.company, "Constellation Software");
  assert.equal(a.c4_inference.model_name, "granite4:3b");
  assert.equal(
    a.c4_inference.model_digest,
    "89962fcc75239ac434cdebceb6b7e0669397f92eaef9c487774b718bc36a3e5f",
  );
  assert.equal(a.c4_inference.context_tokens, 16384);
  assert.equal(a.c4_inference.max_output_tokens, 1024);
  assert.equal(a.c4_inference.temperature, 0);
  assert.equal(a.c4_inference.client_timeout_ms, 600000);
  assert.equal(a.constraints.authorized_run_count, 1);
  assert.equal(a.constraints.automatic_retry_authorized, false);
  assert.equal(a.license_boundary.license, "Apache-2.0");
  assert.equal(a.license_boundary.assist_only, true);
});

test("Granite 4 Constellation runner preserves packet, prompt, and private-output boundaries", () => {
  const raw = readFileSync(
    "scripts/vnext-gate18-phase-c-c4-constellation-granite4-3b-v1-1-context16384-output1024-timeout600-loopback-guarded.ts",
    "utf8",
  );

  assert.match(raw, /granite4:3b/);
  assert.match(
    raw,
    /89962fcc75239ac434cdebceb6b7e0669397f92eaef9c487774b718bc36a3e5f/,
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
  assert.doesNotMatch(raw, /gemma3:/i);
  assert.doesNotMatch(raw, /qwen3\.5/i);
});

test("Granite 4 Constellation prep keeps generation and validation contracts unchanged", () => {
  const p = JSON.parse(
    readFileSync(
      "calibration/vnext/OROTITAN_GATE18_PHASE_C_C4_CONSTELLATION_GRANITE4_3B_V1_1_PREP_001.json",
      "utf8",
    ),
  );

  assert.equal(p.status, "AUTHORIZED_READY_TO_EXECUTE");
  assert.equal(p.contract.generation_prompt_contract, "V1_0_UNCHANGED");
  assert.equal(p.contract.generation_schema_contract, "V1_0_UNCHANGED");
  assert.equal(p.contract.validation_contract, "GATE18_MOAT_EVIDENCE_AUDIT_VALIDATION_V1_1");
  assert.equal(p.contract.raw_output_preserved, true);
  assert.equal(p.contract.human_adjudication_required, true);
  assert.equal(p.runtime_basis.context16384_load_fit, "PASS_WITH_HIGH_RAM_PRESSURE");
});
