import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

const replay = JSON.parse(
  readFileSync(
    "calibration/vnext/OROTITAN_CHATGPT_PROTOCOL_BATTERY_RUN_002.json",
    "utf8",
  ),
);

test("protocol battery replay 002 is non-mutating and targets frozen protocol", () => {
  assert.equal(replay.run_id, "OROTITAN-CHATGPT-PROTOCOL-BATTERY-002");
  assert.equal(
    replay.protocol,
    "OROTITAN_CHATGPT_OPERATING_PROTOCOL_V1_FREEZE_V1.0",
  );
  assert.equal(replay.production_mutation, false);
  assert.equal(replay.test_mode, "CI_PLUS_LIVE_READ_ONLY_PREFLIGHT");
});

test("live replay keeps critical registry surfaces behind RLS", () => {
  for (const [table, enabled] of Object.entries(
    replay.live_supabase.registry_tables_rls,
  )) {
    assert.equal(enabled, true, `RLS must remain enabled on ${table}`);
  }
});

test("live replay keeps guarded RPCs inaccessible to anon/authenticated roles", () => {
  const p = replay.live_supabase.guarded_rpc_permissions;
  assert.equal(p.checked, 9);
  assert.equal(p.anon_execute_count, 0);
  assert.equal(p.authenticated_execute_count, 0);
  assert.equal(p.service_role_execute_count, 9);
  assert.equal(p.all_security_definer, true);
});

test("live replay preserves concurrency, publication and artifact authority guards", () => {
  const g = replay.live_supabase.guard_checks;
  for (const key of [
    "checkpoint_checks_run_state_version",
    "checkpoint_has_idempotency_logic",
    "finalize_checks_run_state_version",
    "finalize_has_idempotency_logic",
    "reopen_checks_run_state_version",
    "reopen_has_idempotency_logic",
    "publish_authorization_checks_run_state_version",
    "publish_authorization_checks_ready_to_publish",
    "publish_authorization_has_idempotency_logic",
    "publish_result_checks_run_state_version",
    "publish_result_checks_ready_to_publish",
    "artifact_resolver_checks_expected_sha256",
    "artifact_resolver_checks_authority_class",
  ]) {
    assert.equal(g[key], true, `missing guard: ${key}`);
  }
});

test("advisor drift remains explicit and non-blocking for protocol foundation", () => {
  const s = replay.live_supabase.security_advisors;
  assert.equal(s.rls_enabled_no_policy.count, 14);
  assert.equal(s.function_search_path_mutable.count, 3);
  assert.equal(
    s.function_search_path_mutable.critical_orotitan_registry_rpc_affected,
    false,
  );
  assert.equal(
    replay.live_supabase.performance_advisors.unindexed_foreign_keys.count,
    10,
  );
  assert.equal(replay.replay_result.advisor_drift, "NO_MATERIAL_CHANGE");
});

test("replay does not falsely declare vertical-slice readiness", () => {
  assert.equal(replay.expected_summary.total, 25);
  assert.equal(replay.expected_summary.required_before_vertical_slice, 15);
  assert.equal(replay.readiness.vertical_slice_ready, false);
  assert.equal(
    replay.conclusion,
    "FOUNDATION_REPLAY_PASS_IMPLEMENTATION_GAPS_UNCHANGED",
  );
  assert.equal(
    replay.exact_next_action,
    "DESIGN_ANALYTICAL_ENGINE_V2_DATA_CONTRACTS",
  );
});
