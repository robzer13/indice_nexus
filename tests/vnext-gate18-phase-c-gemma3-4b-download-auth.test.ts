import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

test("Gemma 3 4B download authorization permits one pinned download and forbids load/inference", () => {
  const auth = JSON.parse(
    readFileSync(
      "calibration/vnext/OROTITAN_GATE18_PHASE_C_GEMMA3_4B_DOWNLOAD_AUTH_001.json",
      "utf8",
    ),
  );

  assert.equal(auth.status, "AUTHORIZED_SINGLE_DOWNLOAD_ONLY");
  assert.equal(auth.user_authorization.explicit, true);
  assert.equal(auth.user_authorization.gemma_terms_accepted, true);
  assert.equal(auth.model.ollama_model_name, "gemma3:4b-it-q4_K_M");
  assert.equal(auth.model.expected_digest_prefix, "a2af6cc3eb7f");
  assert.equal(auth.model.expected_quantization, "Q4_K_M");
  assert.equal(auth.execution.action_count_authorized, 1);
  assert.equal(auth.constraints.inference_after_download, false);
  assert.equal(auth.constraints.load_smoke_after_download, false);
  assert.equal(auth.constraints.automatic_retry, false);
});

test("Gemma 3 4B download runner pulls exact tag and contains no generation or load-smoke execution", () => {
  const raw = readFileSync(
    "scripts/vnext-gate18-phase-c-gemma3-4b-download-verify.ts",
    "utf8",
  );

  assert.match(raw, /gemma3:4b-it-q4_K_M/);
  assert.match(raw, /ollama", \["pull", MODEL\]/);
  assert.match(raw, /a2af6cc3eb7f/);
  assert.match(raw, /\/api\/tags/);
  assert.match(raw, /\/api\/show/);
  assert.doesNotMatch(raw, /\/api\/generate/);
  assert.doesNotMatch(raw, /\/api\/chat/);
  assert.doesNotMatch(raw, /keep_alive/);
  assert.doesNotMatch(raw, /prompt\s*:/);
  assert.match(raw, /modelInferenceExecuted: false/);
  assert.match(raw, /loadSmokeExecuted: false/);
});

test("Phase C advances explicitly to the Gemma 3 pinned download checkpoint", () => {
  const entry = JSON.parse(
    readFileSync(
      "calibration/vnext/OROTITAN_GATE18_PHASE_C_ENTRY_V0.1.json",
      "utf8",
    ),
  );

  assert.equal(entry.next_action, "EXECUTE_GEMMA3_4B_PINNED_DOWNLOAD_VERIFY");
  assert.equal(entry.gemma3_terms_user_accepted, true);
  assert.equal(entry.gemma3_download_authorized, true);
  assert.equal(entry.gemma3_download_executed, false);
  assert.equal(entry.gemma3_load_smoke_authorized, false);
  assert.equal(entry.gemma3_inference_authorized, false);
});

