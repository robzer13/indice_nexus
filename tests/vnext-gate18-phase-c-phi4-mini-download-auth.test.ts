import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

test("Phi-4 mini download authorization is consumed after one pinned download and still forbids load/inference", () => {
  const auth = JSON.parse(
    readFileSync(
      "calibration/vnext/OROTITAN_GATE18_PHASE_C_PHI4_MINI_DOWNLOAD_AUTH_001.json",
      "utf8",
    ),
  );

  assert.equal(auth.status, "CONSUMED_SINGLE_DOWNLOAD_ONLY");
  assert.equal(auth.user_authorization.explicit, true);
  assert.equal(auth.model.ollama_model_name, "phi4-mini:3.8b-q4_K_M");
  assert.equal(auth.model.expected_digest_prefix, "78fad5d182a7");
  assert.equal(auth.execution.action_count_authorized, 1);
  assert.equal(auth.constraints.inference_after_download, false);
  assert.equal(auth.constraints.load_smoke_after_download, false);
  assert.equal(auth.constraints.automatic_retry, false);
  assert.equal(auth.constraints.automatic_model_switch, false);
  assert.equal(auth.authority.phi4_mini_download_authorized, false);
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

test("Phase C entry preserves the pinned Phi-4 mini download while stopping before load/inference", () => {
  const entry = JSON.parse(
    readFileSync(
      "calibration/vnext/OROTITAN_GATE18_PHASE_C_ENTRY_V0.1.json",
      "utf8",
    ),
  );

  assert.equal(entry.phi4_mini_download_authorized, false);
  assert.equal(
    entry.phi4_mini_download_authorization_status,
    "CONSUMED_SINGLE_DOWNLOAD_ONLY",
  );
  assert.equal(entry.phi4_mini_download_executed, true);
  assert.equal(entry.phi4_mini_identity_pin_status, "PASS");
  assert.equal(
    entry.phi4_mini_digest,
    "78fad5d182a7c33065e153a5f8ba210754207ba9d91973f57dffa7f487363753",
  );
  assert.equal(entry.phi4_mini_load_smoke_authorized, false);
  assert.equal(entry.phi4_mini_inference_authorized, false);
  assert.equal(entry.model_switch_authorized, false);
  assert.equal(entry.phi4_mini_load_smoke_context_tokens, 4096);
});
