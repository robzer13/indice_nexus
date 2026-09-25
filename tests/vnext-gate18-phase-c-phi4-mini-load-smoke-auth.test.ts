import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

test("Phi-4 mini load-smoke authorization is consumed after one 4096 load-only execution", () => {
  const auth = JSON.parse(
    readFileSync(
      "calibration/vnext/OROTITAN_GATE18_PHASE_C_PHI4_MINI_LOAD_SMOKE_AUTH_001.json",
      "utf8",
    ),
  );

  assert.equal(auth.status, "CONSUMED_SINGLE_LOAD_ONLY");
  assert.equal(auth.user_authorization.explicit, true);
  assert.equal(auth.user_authorization.authorized_model, "phi4-mini:3.8b-q4_K_M");
  assert.equal(auth.user_authorization.authorized_context_tokens, 4096);
  assert.equal(auth.execution.action_count_authorized, 1);
  assert.equal(auth.execution.mode, "LOCAL_MODEL_LOAD_WITHOUT_SEMANTIC_INFERENCE");
  assert.equal(auth.execution.loopback_only, true);
  assert.equal(auth.execution.prompt_provided, false);
  assert.equal(auth.execution.messages_provided, false);
  assert.equal(auth.execution.semantic_inference_authorized, false);
  assert.equal(auth.execution.context_tokens, 4096);
  assert.equal(auth.execution.explicit_unload, true);
  assert.equal(auth.constraints.model_download_authorized, false);
  assert.equal(auth.constraints.automatic_retry_authorized, false);
  assert.equal(auth.constraints.automatic_model_switch_authorized, false);
  assert.equal(auth.constraints.context_growth_authorized, false);
  assert.equal(auth.authority.phi4_mini_load_smoke_authorized, false);
  assert.equal(auth.authority.phi4_mini_inference_authorized, false);
  assert.equal(auth.authority.model_switch_authorized, false);
});

test("Phi-4 mini load-smoke authorization pins the exact downloaded artifact", () => {
  const auth = JSON.parse(
    readFileSync(
      "calibration/vnext/OROTITAN_GATE18_PHASE_C_PHI4_MINI_LOAD_SMOKE_AUTH_001.json",
      "utf8",
    ),
  );

  assert.equal(
    auth.basis.digest,
    "78fad5d182a7c33065e153a5f8ba210754207ba9d91973f57dffa7f487363753",
  );
  assert.equal(auth.basis.quantization, "Q4_K_M");
  assert.equal(auth.basis.parameter_size, "3.8B");
  assert.equal(auth.basis.identity_pin_status, "PASS");
});

test("Phase C preserves consumed load-only result while inference remains forbidden", () => {
  const entry = JSON.parse(
    readFileSync(
      "calibration/vnext/OROTITAN_GATE18_PHASE_C_ENTRY_V0.1.json",
      "utf8",
    ),
  );

  assert.equal(entry.phi4_mini_load_smoke_authorized, false);
  assert.equal(
    entry.phi4_mini_load_smoke_authorization_status,
    "CONSUMED_SINGLE_LOAD_ONLY",
  );
  assert.equal(
    entry.phi4_mini_load_smoke_authorization_id,
    "G18-PHASEC-PHI4-MINI-LOAD-SMOKE-AUTH-001",
  );
  assert.equal(entry.phi4_mini_load_smoke_context_tokens, 4096);
  assert.equal(entry.phi4_mini_inference_authorized, false);
  assert.equal(entry.model_switch_authorized, false);
  assert.equal(entry.phi4_mini_load_fit_at_4096, "PASS");
  assert.equal(entry.phi4_mini_inference_authorized, false);
});
