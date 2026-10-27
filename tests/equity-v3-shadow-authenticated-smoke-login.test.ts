import test from "node:test";
import assert from "node:assert/strict";
import { parseShadowLocalConfig } from "../scripts/orotitan-equity/shadow-authenticated-smoke-login";

const valid = (): NodeJS.ProcessEnv => ({
  PREFLIGHT_NONPROD_APPROVED: "YES",
  SUPABASE_TEST_URL: "https://awgsurdyvsyolcgpnygh.supabase.co",
  SUPABASE_TEST_REF: "awgsurdyvsyolcgpnygh",
  SUPABASE_PRODUCTION_REF: "cugpgtzygqqlxetyeven",
  SUPABASE_TEST_PUBLISHABLE_KEY: "sb_publishable_synthetic_key_without_authority",
  SHADOW_WRITER_EMAIL: "uploader@example.test",
  SHADOW_WRITER_PASSWORD: "local-test-credential-not-used",
  SHADOW_ATTESTOR_EMAIL: "reviewer@example.test",
  SHADOW_ATTESTOR_PASSWORD: "local-test-credential-not-used",
});
function rejects(env: NodeJS.ProcessEnv, code: string) {
  assert.throws(() => parseShadowLocalConfig(env), err =>
    err instanceof Error && err.message === code);
}
test("opt-in shadow-only config accepts distinct accounts without exposing credentials", () => {
  const cfg = parseShadowLocalConfig(valid());
  assert.equal(cfg.testRef, "awgsurdyvsyolcgpnygh");
  assert.notEqual(cfg.writerEmail, cfg.attestorEmail);
  assert.equal(cfg.writerPassword, "local-test-credential-not-used");
});
test("explicit manual opt-in is mandatory", () => {
  rejects({...valid(), PREFLIGHT_NONPROD_APPROVED:"NO"}, "NONPRODUCTION_APPROVAL_REQUIRED");
});
test("production reference cannot be used as shadow ref", () => {
  rejects({...valid(), SUPABASE_PRODUCTION_REF:"awgsurdyvsyolcgpnygh"}, "PRODUCTION_OR_AMBIGUOUS_PROJECT_DENIED");
});
test("shadow URL must match exact verified project reference", () => {
  rejects({...valid(), SUPABASE_TEST_URL:"https://cugpgtzygqqlxetyeven.supabase.co"}, "PRODUCTION_OR_AMBIGUOUS_PROJECT_DENIED");
  rejects({...valid(), SUPABASE_TEST_URL:"http://awgsurdyvsyolcgpnygh.supabase.co"}, "PRODUCTION_OR_AMBIGUOUS_PROJECT_DENIED");
});
test("service-role and secret keys are not accepted as publishable keys", () => {
  rejects({...valid(), SUPABASE_TEST_PUBLISHABLE_KEY:"sb_secret_cannot_use"}, "PUBLISHABLE_KEY_REQUIRED_NO_SECRET_KEY");
  rejects({...valid(), SUPABASE_TEST_PUBLISHABLE_KEY:"eyJhbGciOiJIUzI1NiJ9.secret"}, "PUBLISHABLE_KEY_REQUIRED_NO_SECRET_KEY");
});
test("distinct Auth user accounts mandatory before any login", () => {
  rejects({...valid(), SHADOW_ATTESTOR_EMAIL:"UPLOADER@example.test"}, "DISTINCT_ACCOUNT_EMAILS_REQUIRED");
});
test("passwords must be locally supplied and are never invented", () => {
  rejects({...valid(), SHADOW_WRITER_PASSWORD:undefined}, "SHADOW_CONFIG_REQUIRED_SHADOW_WRITER_PASSWORD");
  rejects({...valid(), SHADOW_ATTESTOR_PASSWORD:undefined}, "SHADOW_CONFIG_REQUIRED_SHADOW_ATTESTOR_PASSWORD");
});
