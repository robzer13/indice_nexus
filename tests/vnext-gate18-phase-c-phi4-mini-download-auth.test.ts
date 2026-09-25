import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

test("Phi-4 mini download authorization is single-use scoped and forbids load/inference", () => {
  const auth = JSON.parse(
    readFileSync(
      "calibration/vnext/OROTITAN_GATE18_PHASE_C_PHI4_MINI_DOWNLOAD_AUTH_001.json",
      "utf8",
    ),
  );

  assert.equal(auth.status, "AUTHORIZED_SINGLE_DOWNLOAD_UNCONSUMED");
  assert.equal(auth.user_authorization.explicit, true);
  assert.equal(auth.model.ollama_model_name, "phi4-mini:3.8b-q4_K_M");
  assert.equal(auth.model.expected_digest_prefix, "78fad5d182a7");
  assert.equal(auth.execution.action_count_authorized, 1);
  assert.equal(auth.constraints.inference_after_download, false);
  assert.equal(auth.constraints.load_smoke_after_download, false);
  assert.equal(auth.constraints.automatic_retry, false);
  assert.equal(auth.constraints.automatic_model_switch, false);
  assert.equal(auth.authority.phi4_mini_download_authorized, true);
  assert.equal(auth.authority.phi4_mini_load_smoke_authorized, false);
  assert.equal(auth.authority.phi4_mini_inference_authorized, false);
});

test("Phi-4 mini download runner pulls exact model and contains no inference or load-smoke execution", () => {
  const raw = readFileSync(
    "scripts/vnext-gate18-phase-c-phi4-mini-download-verify.ts",
    "utf8",
  );

  assert.match(raw, /phi4-mini:3\.8b-q4_K_M/);
  assert.match(raw, /ollama", \["pull", MODEL\]/);
  assert.match(raw, /78fad5d182a7/);
  assert.match(raw, /\/api\/tags/);
  assert.match(raw, /\/api\/show/);
  assert.doesNotMatch(raw, /\/api\/generate/);
  assert.doesNotMatch(raw, /\/api\/chat/);
  assert.doesNotMatch(raw, /keep_alive/);
  assert.doesNotMatch(raw, /prompt\s*:/);
  assert.match(raw, /modelInferenceExecuted: false/);
  assert.match(raw, /loadSmokeExecuted: false/);
});

test("Phase C entry stops at one authorized Phi-4 mini download with inference still forbidden", () => {
  const entry = JSON.parse(
    readFileSync(
      "calibration/vnext/OROTITAN_GATE18_PHASE_C_ENTRY_V0.1.json",
      "utf8",
    ),
  );

  assert.equal(entry.phi4_mini_download_authorized, true);
  assert.equal(
    entry.phi4_mini_download_authorization_status,
    "AUTHORIZED_SINGLE_DOWNLOAD_UNCONSUMED",
  );
  assert.equal(entry.phi4_mini_load_smoke_authorized, false);
  assert.equal(entry.phi4_mini_inference_authorized, false);
  assert.equal(entry.model_switch_authorized, false);
  assert.equal(entry.next_action, "RUN_LOCAL_PHI4_MINI_DOWNLOAD_VERIFY_ONCE");
});
