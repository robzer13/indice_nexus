import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

test("Granite 4 3B pinned download authorization is zero-cost and download-only", () => {
  const auth = JSON.parse(
    readFileSync(
      "calibration/vnext/OROTITAN_GATE18_PHASE_C_GRANITE4_3B_DOWNLOAD_AUTH_001.json",
      "utf8",
    ),
  );

  assert.equal(auth.status, "AUTHORIZED_SINGLE_DOWNLOAD_ONLY_UNCONSUMED");
  assert.equal(auth.authorization_source.authorization_id, "OROTITAN-STANDING-TECHNICAL-AUTH-002");
  assert.equal(auth.authorization_source.cost_usd, 0);
  assert.equal(auth.model.ollama_model_name, "granite4:3b");
  assert.equal(auth.model.expected_digest_prefix, "89962fcc7523");
  assert.equal(auth.model.expected_quantization, "Q4_K_M");
  assert.equal(auth.model.license, "Apache-2.0");
  assert.equal(auth.constraints.inference_after_download, false);
  assert.equal(auth.constraints.load_smoke_after_download, false);
  assert.equal(auth.constraints.automatic_retry, false);
  assert.equal(auth.authority.granite4_download_authorized, true);
  assert.equal(auth.authority.granite4_load_smoke_authorized, false);
  assert.equal(auth.authority.granite4_inference_authorized, false);
});

test("Granite 4 3B download verifier pulls only the exact tag and performs no inference", () => {
  const raw = readFileSync(
    "scripts/vnext-gate18-phase-c-granite4-3b-download-verify.ts",
    "utf8",
  );

  assert.match(raw, /const MODEL = "granite4:3b"/);
  assert.match(raw, /const EXPECTED_DIGEST_PREFIX = "89962fcc7523"/);
  assert.match(raw, /const EXPECTED_QUANTIZATION = "Q4_K_M"/);
  assert.match(raw, /ollama", \["pull", MODEL\]/);
  assert.match(raw, /\/api\/tags/);
  assert.match(raw, /\/api\/show/);
  assert.doesNotMatch(raw, /\/api\/generate/);
  assert.doesNotMatch(raw, /\/api\/chat/);
  assert.doesNotMatch(raw, /prompt\s*:/);
  assert.doesNotMatch(raw, /keep_alive/);
  assert.match(raw, /modelInferenceExecuted: false/);
  assert.match(raw, /loadSmokeExecuted: false/);
});
