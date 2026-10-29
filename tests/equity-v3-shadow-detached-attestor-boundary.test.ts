import test from "node:test";
import assert from "node:assert/strict";
import { randomUUID } from "node:crypto";
import {
  parseDetachedHandoff, parseDetachedAttestorConfig, parseDetachedVerifierConfig,
  SHADOW_REF, PRODUCTION_REF, SYNTHETIC_SHA, SYNTHETIC_URL,
} from "../lib/orotitan-equity/v3/shadow-detached-attestor-boundary";

function validHandoff() {
  const runId = randomUUID();
  const artifactId = randomUUID();
  const path = "preadmission/" + runId + "/" + artifactId + "/v0001/" +
    SYNTHETIC_SHA + ".pdf";
  return {
    schema: "OROTITAN_SHADOW_DETACHED_HANDOFF_V1",
    challengeNonce: randomUUID(),
    locator: {
      runId, artifactId, version: 1, bucket: "orotitan-source-files-v1",
      path, uri: "supabase://orotitan-source-files-v1/" + path,
      backend: "SUPABASE_STORAGE", mediaType: "application/pdf",
      sizeBytes: 715, sha256: SYNTHETIC_SHA, sourceUrl: SYNTHETIC_URL,
      publishedOn: "2026-01-01", dataCutoff: "2026-09-21",
      dateAuthority: "DECLARED_NOT_INDEPENDENTLY_VERIFIED",
    },
  };
}
function validAttestorEnv(): NodeJS.ProcessEnv {
  return {
    NODE_ENV: "test",
    PREFLIGHT_NONPROD_APPROVED: "YES",
    SUPABASE_TEST_REF: SHADOW_REF,
    SUPABASE_PRODUCTION_REF: PRODUCTION_REF,
    SUPABASE_TEST_URL: "https://" + SHADOW_REF + ".supabase.co",
    SUPABASE_TEST_PUBLISHABLE_KEY: "sb_publishable_placeholder-not-real",
    SHADOW_DETACHED_ATTESTOR_JWT: "test-jwt-placeholder",
    SHADOW_DETACHED_PRIVATE_KEY_FILE: "/private/key.pem",
    SHADOW_DETACHED_HANDOFF_FILE: "/tmp/handoff.json",
    SHADOW_DETACHED_RECEIPT_FILE: "/tmp/receipt.json",
    SHADOW_DETACHED_ATTESTOR_ID: "separate-attestor",
    SHADOW_DETACHED_KEY_ID: "test-key-1",
  };
}

test("detached handoff: exact synthetic locator only", () => {
  const handoff = validHandoff();
  assert.deepEqual(parseDetachedHandoff(handoff), handoff);
});
test("detached handoff: altered source bytes or dates denied", () => {
  const a = validHandoff();
  assert.throws(() => parseDetachedHandoff({
    ...a, locator: { ...a.locator, sha256: "a".repeat(64) },
  }), /DETACHED_NON_SYNTHETIC_INPUT_DENIED/);
  assert.throws(() => parseDetachedHandoff({
    ...a, locator: { ...a.locator, publishedOn: "2026-10-01" },
  }), /DETACHED_NON_SYNTHETIC_INPUT_DENIED/);
});
test("detached handoff: forged path and unexpected authority claims denied", () => {
  const a = validHandoff();
  assert.throws(() => parseDetachedHandoff({
    ...a, locator: { ...a.locator, path: "preadmission/unrelated.pdf" },
  }), /DETACHED_LOCATOR_NOT_CONTENT_ADDRESSED/);
  assert.throws(() => parseDetachedHandoff({ ...a, registryAdmitted: true }),
    /DETACHED_HANDOFF_SHAPE_INVALID/);
  assert.throws(() => parseDetachedHandoff({ ...a, locator: {
    ...a.locator, authority: "AUTHORITATIVE",
  } }), /DETACHED_LOCATOR_SHAPE_INVALID/);
});
test("detached attestor: proper shadow config with no Writer secrets passes", () => {
  assert.equal(parseDetachedAttestorConfig(validAttestorEnv()).attestorId,
    "separate-attestor");
});
test("detached attestor: production or typo project fails closed", () => {
  assert.throws(() => parseDetachedAttestorConfig({
    ...validAttestorEnv(), SUPABASE_TEST_REF: PRODUCTION_REF,
  }), /DETACHED_PRODUCTION_OR_UNPINNED_PROJECT/);
  assert.throws(() => parseDetachedAttestorConfig({
    ...validAttestorEnv(), SUPABASE_TEST_URL: "https://evil.example",
  }), /DETACHED_PRODUCTION_OR_UNPINNED_PROJECT/);
});
test("detached attestor: no opting in, service key, or Writer credentials", () => {
  assert.throws(() => parseDetachedAttestorConfig({
    ...validAttestorEnv(), PREFLIGHT_NONPROD_APPROVED: "NO",
  }), /DETACHED_SHADOW_OPT_IN_REQUIRED/);
  assert.throws(() => parseDetachedAttestorConfig({
    ...validAttestorEnv(), SUPABASE_SERVICE_ROLE_KEY: "service-key",
  }), /DETACHED_FORBIDDEN_SHARED_CREDENTIAL/);
  assert.throws(() => parseDetachedAttestorConfig({
    ...validAttestorEnv(), SUPABASE_TEST_WRITER_JWT: "writer-secret",
  }), /DETACHED_FORBIDDEN_SHARED_CREDENTIAL/);
  assert.throws(() => parseDetachedAttestorConfig({
    ...validAttestorEnv(), SUPABASE_TEST_PUBLISHABLE_KEY: "sb_secret_bad",
  }), /DETACHED_PUBLISHABLE_KEY_REQUIRED/);
});
test("detached verifier: independent pinned nonce and key required", () => {
  const cfg: NodeJS.ProcessEnv = {
    NODE_ENV: "test",
    PREFLIGHT_NONPROD_APPROVED: "YES",
    SUPABASE_TEST_REF: SHADOW_REF, SUPABASE_PRODUCTION_REF: PRODUCTION_REF,
    SHADOW_GATE_TRUSTED_ATTESTOR_ID: "separate-attestor",
    SHADOW_GATE_TRUSTED_KEY_ID: "test-key-1",
    SHADOW_GATE_PINNED_CHALLENGE_NONCE: randomUUID(),
    SHADOW_GATE_TRUSTED_PUBLIC_PEM_FILE: "/trusted/pub.pem",
    SHADOW_GATE_TRUSTED_PUBLIC_PEM_SHA256: "a".repeat(64),
    SHADOW_GATE_HANDOFF_FILE: "/tmp/handoff.json",
    SHADOW_GATE_RECEIPT_FILE: "/tmp/receipt.json",
  };
  assert.equal(parseDetachedVerifierConfig(cfg).keyId, "test-key-1");
  assert.throws(() => parseDetachedVerifierConfig({
    ...cfg, SHADOW_GATE_PINNED_CHALLENGE_NONCE: "1234",
  }), /DETACHED_VERIFIER_NONCE_INVALID/);
  assert.throws(() => parseDetachedVerifierConfig({
    ...cfg, SHADOW_GATE_TRUSTED_PUBLIC_PEM_FILE: "",
  }), /DETACHED_ENV_REQUIRED/);
  assert.throws(() => parseDetachedVerifierConfig({
    ...cfg, SHADOW_GATE_TRUSTED_PUBLIC_PEM_SHA256: "not-a-digest",
  }), /DETACHED_PUBLIC_KEY_PIN_INVALID/);
});
