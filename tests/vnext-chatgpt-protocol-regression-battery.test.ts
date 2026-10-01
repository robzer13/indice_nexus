import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

const battery = JSON.parse(
  readFileSync(
    "calibration/vnext/OROTITAN_CHATGPT_PROTOCOL_BATTERY_RUN_001.json",
    "utf8",
  ),
);

const matrix = JSON.parse(
  readFileSync(
    "calibration/vnext/OROTITAN_CHATGPT_PROTOCOL_ACCEPTANCE_MATRIX_001.json",
    "utf8",
  ),
);

test("protocol battery is non-mutating and based on green merged baseline", () => {
  assert.equal(battery.production_mutation, false);
  assert.equal(battery.test_mode, "CI_PLUS_LIVE_READ_ONLY_PREFLIGHT");
  assert.equal(battery.ci_baseline.vnext_ci, "PASS");
  assert.equal(battery.ci_baseline.screener_ci, "PASS");
  assert.equal(
    battery.ci_baseline.merge_commit,
    "e4dcec16e9790e5dff88a032db64a742bcaca3cf",
  );
});

test("live preflight verifies RLS on the canonical registry surfaces", () => {
  for (const [table, enabled] of Object.entries(
    battery.live_supabase.registry_tables_rls,
  )) {
    assert.equal(enabled, true, `RLS must remain enabled on ${table}`);
  }
});

test("critical analytical registry RPCs remain service-role only", () => {
  const permissions = battery.live_supabase.guarded_rpc_permissions;

  assert.equal(permissions.checked, 9);
  assert.equal(permissions.anon_execute_count, 0);
  assert.equal(permissions.authenticated_execute_count, 0);
  assert.equal(permissions.service_role_execute_count, 9);
  assert.equal(permissions.all_security_definer, true);
});

test("checkpoint/finalize/reopen preserve optimistic concurrency and idempotency guards", () => {
  const guards = battery.live_supabase.guard_checks;

  for (const key of [
    "checkpoint_checks_run_state_version",
    "checkpoint_checks_stage_state_version",
    "checkpoint_has_idempotency_logic",
    "finalize_checks_run_state_version",
    "finalize_checks_stage_state_version",
    "finalize_has_idempotency_logic",
    "reopen_checks_run_state_version",
    "reopen_checks_stage_state_version",
  ]) {
    assert.equal(guards[key], true, `missing guard: ${key}`);
  }
});

test("publication and artifact authority remain separate guarded operations", () => {
  const guards = battery.live_supabase.guard_checks;

  assert.equal(guards.publish_authorization_checks_run_state_version, true);
  assert.equal(guards.publish_authorization_checks_ready_to_publish, true);
  assert.equal(guards.publish_authorization_has_idempotency_logic, true);
  assert.equal(guards.publish_result_checks_run_state_version, true);
  assert.equal(guards.publish_result_checks_ready_to_publish, true);
  assert.equal(guards.artifact_resolver_checks_expected_sha256, true);
  assert.equal(guards.artifact_resolver_checks_authority_class, true);
});

test("security-advisor findings stay explicit rather than silently greenwashed", () => {
  const advisors = battery.live_supabase.security_advisors;

  assert.equal(advisors.rls_enabled_no_policy.level, "INFO");
  assert.equal(
    advisors.rls_enabled_no_policy.interpretation,
    "REVIEW_AND_DOCUMENT_SERVICE_ONLY_POSTURE",
  );

  assert.equal(advisors.function_search_path_mutable.level, "WARN");
  assert.equal(advisors.function_search_path_mutable.count, 3);
  assert.equal(
    advisors.function_search_path_mutable.critical_orotitan_registry_rpc_affected,
    false,
  );
  assert.equal(
    advisors.function_search_path_mutable.disposition,
    "TECHNICAL_DEBT_TO_HARDEN",
  );
});

test("battery and frozen 25-scenario matrix reconcile exactly", () => {
  assert.equal(battery.acceptance_matrix.total, matrix.summary.total);
  assert.equal(
    battery.acceptance_matrix.pass_live,
    matrix.summary.pass_live,
  );
  assert.equal(
    battery.acceptance_matrix.pass_infrastructure,
    matrix.summary.pass_infrastructure,
  );
  assert.equal(
    battery.acceptance_matrix.pass_protocol,
    matrix.summary.pass_protocol,
  );
  assert.equal(
    battery.acceptance_matrix.partial_infrastructure,
    matrix.summary.partial_infrastructure,
  );
  assert.equal(
    battery.acceptance_matrix.partial_code,
    matrix.summary.partial_code,
  );
  assert.equal(
    battery.acceptance_matrix.implementation_gaps,
    matrix.summary.implementation_gaps,
  );
  assert.equal(
    battery.acceptance_matrix.required_before_vertical_slice,
    matrix.summary.required_before_vertical_slice,
  );
});

test("battery blocks the vertical slice until the 15 required gaps are closed", () => {
  assert.equal(battery.readiness.vertical_slice_ready, false);
  assert.equal(battery.blockers_before_vertical_slice.length, 15);
  assert.equal(
    battery.conclusion,
    "FOUNDATION_PASS_WITH_IMPLEMENTATION_GAPS",
  );
  assert.equal(
    battery.exact_next_action,
    "DESIGN_ANALYTICAL_ENGINE_V2_DATA_CONTRACTS",
  );
});

test("French-first remains a product-surface invariant", () => {
  const direction = JSON.parse(
    readFileSync(
      "calibration/vnext/OROTITAN_POST_C7_PRODUCT_DIRECTION_ANALYTICAL_ENGINE_V2_001.json",
      "utf8",
    ),
  );

  assert.equal(direction.product_requirements.french_first_ui, true);
});
