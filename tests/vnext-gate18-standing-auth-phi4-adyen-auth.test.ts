import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

test("OroTitan standing technical authorization is active with explicit exclusions", () => {
  const standing = JSON.parse(
    readFileSync(
      "calibration/vnext/OROTITAN_STANDING_TECHNICAL_EXECUTION_AUTHORIZATION_001.json",
      "utf8",
    ),
  );

  assert.equal(standing.status, "ACTIVE");
  assert.equal(standing.user_authorization.explicit, true);
  assert.equal(
    standing.execution_policy.per_execution_authorization_artifacts_may_be_derived_from_this_standing_authorization,
    true,
  );
  assert.equal(standing.execution_policy.user_reprompt_required_for_in_scope_actions, false);
  assert.equal(standing.execution_policy.user_reprompt_required_for_excluded_actions, true);
  assert.equal(
    standing.exclusions_requiring_new_explicit_authorization.includes(
      "SIGNIFICANT_EXTERNAL_PAID_EXECUTION_OR_PURCHASE",
    ),
    true,
  );
  assert.equal(
    standing.exclusions_requiring_new_explicit_authorization.includes(
      "PRODUCTION_MUTATION_OR_PRODUCTION_ROUTING_FREEZE",
    ),
    true,
  );
  assert.equal(standing.protocol_guards.private_generated_content_stays_private, true);
  assert.equal(standing.protocol_guards.no_retroactive_pass, true);
});

test("Phi-4 mini Adyen authorization is a one-run derivative of standing authority", () => {
  const auth = JSON.parse(
    readFileSync(
      "calibration/vnext/OROTITAN_GATE18_PHASE_C_PHI4_MINI_ADYEN_COMPACT4096_INFERENCE_AUTH_001.json",
      "utf8",
    ),
  );

  assert.equal(auth.status, "AUTHORIZED_SINGLE_LOCAL_INFERENCE");
  assert.equal(auth.authorization_source.type, "STANDING_USER_AUTHORIZATION");
  assert.equal(
    auth.authorization_source.authorization_id,
    "OROTITAN-STANDING-TECHNICAL-AUTH-001",
  );
  assert.equal(auth.authorization_source.separate_user_reprompt_required, false);
  assert.equal(auth.phi4_mini_inference.authorized, true);
  assert.equal(auth.phi4_mini_inference.model_name, "phi4-mini:3.8b-q4_K_M");
  assert.equal(
    auth.phi4_mini_inference.model_digest,
    "78fad5d182a7c33065e153a5f8ba210754207ba9d91973f57dffa7f487363753",
  );
  assert.equal(auth.phi4_mini_inference.semantic_probe_id, "ADYEN_CLAIM_TARGET_CORE_001");
  assert.equal(
    auth.phi4_mini_inference.execution_id,
    "ADYEN_CLAIM_TARGET_CORE_001_LOCAL_COMPACT_PHI4MINI_4096",
  );
  assert.equal(auth.phi4_mini_inference.context_tokens, 4096);
  assert.equal(auth.phi4_mini_inference.max_output_tokens, 768);
  assert.equal(auth.phi4_mini_inference.temperature, 0);
  assert.equal(auth.constraints.authorized_run_count, 1);
  assert.equal(auth.constraints.automatic_retry_authorized, false);
  assert.equal(auth.constraints.context_growth_authorized, false);
  assert.equal(auth.constraints.automatic_model_switch_authorized, false);
  assert.equal(auth.constraints.external_model_api_cost_usd, 0);
  assert.equal(auth.constraints.production_mutation, false);
  assert.equal(auth.constraints.publication_authority, false);
});

test("Phase C current state exposes only the bounded Adyen run", () => {
  const entry = JSON.parse(
    readFileSync(
      "calibration/vnext/OROTITAN_GATE18_PHASE_C_ENTRY_V0.1.json",
      "utf8",
    ),
  );

  assert.equal(entry.standing_technical_authorization_status, "ACTIVE");
  assert.equal(entry.standing_in_scope_reprompt_required, false);
  assert.equal(entry.phi4_mini_inference_authorized, true);
  assert.equal(entry.phi4_mini_adyen_compact4096_inference_authorized, true);
  assert.equal(
    entry.phi4_mini_adyen_compact4096_inference_authorization_status,
    "AUTHORIZED_SINGLE_LOCAL_INFERENCE",
  );
  assert.equal(entry.phi4_mini_adyen_compact4096_authorized_run_count, 1);
  assert.equal(entry.phi4_mini_adyen_compact4096_context_tokens, 4096);
  assert.equal(entry.phi4_mini_adyen_compact4096_max_output_tokens, 768);
  assert.equal(entry.phi4_mini_adyen_compact4096_temperature, 0);
  assert.equal(entry.phi4_mini_context_growth_authorized, false);
  assert.equal(entry.phi4_mini_automatic_retry_authorized, false);
  assert.equal(entry.model_switch_authorized, false);
  assert.equal(entry.next_action, "RUN_ONE_LOCAL_PHI4_MINI_ADYEN_COMPACT4096_INFERENCE");
});
