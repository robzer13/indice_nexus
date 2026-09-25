import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

test("Phi-4 mini Brookfield compact inference authorization is one-run and exact", () => {
  const auth = JSON.parse(
    readFileSync(
      "calibration/vnext/OROTITAN_GATE18_PHASE_C_PHI4_MINI_BROOKFIELD_COMPACT4096_INFERENCE_AUTH_001.json",
      "utf8",
    ),
  );

  assert.equal(auth.status, "AUTHORIZED_SINGLE_LOCAL_INFERENCE");
  assert.equal(auth.user_authorization.explicit, true);
  assert.equal(
    auth.user_authorization.authorized_action,
    "ONE_PHI4_MINI_BROOKFIELD_COMPACT4096_INFERENCE",
  );
  assert.equal(auth.phi4_mini_inference.authorized, true);
  assert.equal(auth.phi4_mini_inference.model_name, "phi4-mini:3.8b-q4_K_M");
  assert.equal(
    auth.phi4_mini_inference.model_digest,
    "78fad5d182a7c33065e153a5f8ba210754207ba9d91973f57dffa7f487363753",
  );
  assert.equal(auth.phi4_mini_inference.semantic_probe_id, "BROOKFIELD_PEER_ROLE_CORE_001");
  assert.equal(
    auth.phi4_mini_inference.execution_id,
    "BROOKFIELD_PEER_ROLE_CORE_001_LOCAL_COMPACT_PHI4MINI_4096",
  );
  assert.equal(auth.phi4_mini_inference.context_tokens, 4096);
  assert.equal(auth.phi4_mini_inference.max_output_tokens, 768);
  assert.equal(auth.phi4_mini_inference.temperature, 0);
  assert.equal(auth.constraints.authorized_run_count, 1);
  assert.equal(auth.constraints.automatic_retry_authorized, false);
  assert.equal(auth.constraints.context_growth_authorized, false);
  assert.equal(auth.constraints.automatic_model_switch_authorized, false);
  assert.equal(auth.constraints.external_model_api_call, false);
  assert.equal(auth.constraints.external_model_api_cost_usd, 0);
  assert.equal(auth.authority.comparison_admissible, false);
  assert.equal(auth.authority.model_ranking_authority, false);
  assert.equal(auth.authority.production_candidate_decision_authority, false);
});

test("Phi-4 mini Brookfield generated content remains private and carries no production authority", () => {
  const auth = JSON.parse(
    readFileSync(
      "calibration/vnext/OROTITAN_GATE18_PHASE_C_PHI4_MINI_BROOKFIELD_COMPACT4096_INFERENCE_AUTH_001.json",
      "utf8",
    ),
  );

  assert.equal(auth.output_policy.generated_content_destination, "calibration/vnext/private-runs/");
  assert.equal(auth.output_policy.public_repo_generated_content_forbidden, true);
  assert.equal(auth.output_policy.human_adjudication_required, true);
  assert.equal(auth.constraints.production_mutation, false);
  assert.equal(auth.constraints.publication_authority, false);
});

test("Phase C exposes only the one authorized Phi-4 mini compact inference", () => {
  const entry = JSON.parse(
    readFileSync(
      "calibration/vnext/OROTITAN_GATE18_PHASE_C_ENTRY_V0.1.json",
      "utf8",
    ),
  );

  assert.equal(entry.phi4_mini_inference_authorized, true);
  assert.equal(
    entry.phi4_mini_brookfield_compact4096_inference_authorization_status,
    "AUTHORIZED_SINGLE_LOCAL_INFERENCE",
  );
  assert.equal(entry.phi4_mini_brookfield_compact4096_authorized_run_count, 1);
  assert.equal(entry.phi4_mini_brookfield_compact4096_context_tokens, 4096);
  assert.equal(entry.phi4_mini_brookfield_compact4096_max_output_tokens, 768);
  assert.equal(entry.phi4_mini_brookfield_compact4096_temperature, 0);
  assert.equal(entry.phi4_mini_automatic_retry_authorized, false);
  assert.equal(entry.phi4_mini_context_growth_authorized, false);
  assert.equal(entry.model_switch_authorized, false);
});
