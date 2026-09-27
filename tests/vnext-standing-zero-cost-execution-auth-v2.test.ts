import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

test("standing zero-cost execution authorization v2 removes free-action reprompts", () => {
  const auth = JSON.parse(
    readFileSync(
      "calibration/vnext/OROTITAN_STANDING_TECHNICAL_EXECUTION_AUTHORIZATION_002.json",
      "utf8",
    ),
  );

  assert.equal(auth.status, "ACTIVE");
  assert.equal(auth.supersedes, "OROTITAN-STANDING-TECHNICAL-AUTH-001");
  assert.equal(auth.user_authorization.explicit, true);
  assert.equal(auth.execution_policy.user_reprompt_required_for_zero_cost_actions, false);
  assert.equal(auth.execution_policy.user_reprompt_required_for_paid_actions, true);
  assert.deepEqual(
    auth.only_category_requiring_new_user_authorization,
    ["ANY_ACTION_WITH_NONZERO_EXTERNAL_COST_OR_PURCHASE"],
  );
  assert.equal(auth.paid_action_definition.free_download_requires_reprompt, false);
  assert.equal(auth.paid_action_definition.free_inference_requires_reprompt, false);
  assert.equal(auth.paid_action_definition.free_retry_requires_reprompt, false);
  assert.equal(auth.paid_action_definition.free_model_switch_requires_reprompt, false);
  assert.equal(auth.protocol_guards.no_retroactive_pass, true);
  assert.equal(auth.protocol_guards.private_generated_content_stays_private, true);
});

test("current Qwen3.5 think-false retry derives authority from standing v2", () => {
  const auth = JSON.parse(
    readFileSync(
      "calibration/vnext/OROTITAN_GATE18_PHASE_C_C4_CONSTELLATION_QWEN3_5_4B_V1_1_THINK_FALSE_RETRY_AUTH_001.json",
      "utf8",
    ),
  );
  assert.equal(auth.status, "AUTHORIZED_SINGLE_LOCAL_INFERENCE");
  assert.equal(auth.authorization_source.authorization_id, "OROTITAN-STANDING-TECHNICAL-AUTH-002");
  assert.equal(auth.authorization_source.separate_user_reprompt_required, false);
  assert.equal(auth.authorization_source.cost_usd, 0);
  assert.equal(auth.c4_inference.think, false);
  assert.equal(auth.constraints.authorized_run_count, 1);
  assert.equal(auth.constraints.external_model_api_cost_usd, 0);
});
