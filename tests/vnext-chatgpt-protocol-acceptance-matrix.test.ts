import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

const matrix = JSON.parse(
  readFileSync(
    "calibration/vnext/OROTITAN_CHATGPT_PROTOCOL_ACCEPTANCE_MATRIX_001.json",
    "utf8",
  ),
);

test("ChatGPT protocol acceptance matrix covers all 25 frozen scenarios", () => {
  assert.equal(matrix.format, "OROTITAN_CHATGPT_PROTOCOL_ACCEPTANCE_MATRIX_V1");
  assert.equal(matrix.protocol, "OROTITAN_CHATGPT_OPERATING_PROTOCOL_V1_FREEZE_V1.0");
  assert.equal(matrix.production_mutation, false);
  assert.equal(matrix.live_validation_mode, "READ_ONLY_ONLY");
  assert.equal(matrix.scenarios.length, 25);
  assert.equal(new Set(matrix.scenarios.map((x: { id: string }) => x.id)).size, 25);
});

test("acceptance matrix preserves the critical reliability scenarios", () => {
  const byName = new Map(
    matrix.scenarios.map((x: { name: string }) => [x.name, x]),
  );

  for (const name of [
    "RESUME_ACTIVE_RUN_NEW_CHAT",
    "POST_CUTOFF_SOURCE",
    "PRIOR_CANONICAL_CONCLUSION_OVERTURNED",
    "STALE_STATE_VERSION_ON_SAVE",
    "CONNECTOR_FAILURE_DURING_LOAD",
    "PERSISTENCE_FAILURE_DURING_SAVE",
    "INVALID_EVIDENCE_ID",
    "RED_TEAM_REOPENS_EARLIER_BLOCK",
    "PRICE_ONLY_DELTA",
    "SAVE_NEVER_PUBLISHES",
    "GO_PUBLISH_INVALID_INTEGRATION",
  ]) {
    assert.ok(byName.has(name), `missing acceptance scenario ${name}`);
  }
});

test("all pre-vertical-slice gaps have a concrete implementation destination", () => {
  const required = matrix.scenarios.filter(
    (x: { required_before_vertical_slice: boolean }) =>
      x.required_before_vertical_slice,
  );

  assert.ok(required.length > 0);
  for (const scenario of required) {
    assert.ok(
      [
        "DATA_CONTRACTS_V2",
        "PROCESS_ENGINE_V2",
        "CHATGPT_SUPABASE_BRIDGE",
      ].includes(scenario.next_phase),
      `${scenario.id} has invalid implementation destination ${scenario.next_phase}`,
    );
  }
});

test("live preflight records RLS, bounded RPCs and publication separation", () => {
  const live = matrix.live_readonly_findings;
  assert.equal(live.mutation_firewall_verified, true);
  assert.equal(live.run_stage_version_guards_verified, true);
  assert.equal(live.artifact_hash_authority_resolution_verified, true);
  assert.equal(live.publication_separation_verified, true);
  assert.ok(live.rls_tables_verified.includes("orotitan_runs"));
  assert.ok(live.rls_tables_verified.includes("orotitan_artifacts"));
  assert.ok(
    live.guarded_rpcs_service_role_only.includes("checkpoint_orotitan_stage"),
  );
  assert.ok(
    live.guarded_rpcs_service_role_only.includes(
      "record_orotitan_publish_authorization",
    ),
  );
});

test("acceptance battery keeps Data Contracts V2 as exact next action", () => {
  assert.equal(
    matrix.conclusion,
    "FOUNDATION_IS_STRONG_BUT_NOT_VERTICAL_SLICE_READY",
  );
  assert.equal(
    matrix.exact_next_action,
    "DESIGN_ANALYTICAL_ENGINE_V2_DATA_CONTRACTS",
  );
});
