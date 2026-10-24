import assert from 'node:assert/strict';
import { test } from 'node:test';
import {
  isHostedShadowResearchEnabled,
  SHADOW_RESEARCH_ORIGIN,
  SHADOW_RESEARCH_PROJECT_REF,
} from '../lib/orotitan-equity/product/hosted-shadow-research-gate';

const SYNTHETIC_KEY = 'synthetic-only-not-a-key-aaaaaaaaaaaaaaaaaaaaaaaaaaaa';

function validShadowPreview(): Record<string, string> {
  return {
    NODE_ENV: 'production',
    VERCEL_ENV: 'preview',
    OROTITAN_HOSTED_SHADOW_RESEARCH_ENABLED: '1',
    OROTITAN_SHARED_INBOX_TEST_ENABLED: '1',
    OROTITAN_SHARED_INBOX_TEST_PROJECT_REF: SHADOW_RESEARCH_PROJECT_REF,
    OROTITAN_EXPECTED_SUPABASE_PROJECT_REF: SHADOW_RESEARCH_PROJECT_REF,
    OROTITAN_SHARED_INBOX_TEST_URL: SHADOW_RESEARCH_ORIGIN,
    NEXT_PUBLIC_SUPABASE_URL: SHADOW_RESEARCH_ORIGIN,
    OROTITAN_SHARED_INBOX_TEST_SERVICE_KEY: SYNTHETIC_KEY,
    SUPABASE_SERVICE_ROLE_KEY: SYNTHETIC_KEY,
    ADMIN_PASSWORD: 'synthetic-admin-password-32characters',
    ADMIN_SESSION_SECRET: 'synthetic-session-secret-for-shadow-only-32characters',
  };
}

test('hosted Research gate admits exact, explicitly enabled SHADOW Preview', () => {
  assert.equal(isHostedShadowResearchEnabled(validShadowPreview()), true);
  const withTrailingSlash = {
    ...validShadowPreview(),
    NEXT_PUBLIC_SUPABASE_URL: SHADOW_RESEARCH_ORIGIN + '/',
    OROTITAN_SHARED_INBOX_TEST_URL: SHADOW_RESEARCH_ORIGIN + '/',
  };
  assert.equal(isHostedShadowResearchEnabled(withTrailingSlash), true);
});

const denied: Array<{ name: string; key: string; value: string | undefined }> = [
  { name: 'Vercel production', key: 'VERCEL_ENV', value: 'production' },
  { name: 'Vercel development', key: 'VERCEL_ENV', value: 'development' },
  { name: 'non-Vercel environment', key: 'VERCEL_ENV', value: undefined },
  { name: 'Next development', key: 'NODE_ENV', value: 'development' },
  { name: 'hosted enable flag absent', key: 'OROTITAN_HOSTED_SHADOW_RESEARCH_ENABLED', value: undefined },
  { name: 'hosted enable flag off', key: 'OROTITAN_HOSTED_SHADOW_RESEARCH_ENABLED', value: '0' },
  { name: 'shared backend flag missing', key: 'OROTITAN_SHARED_INBOX_TEST_ENABLED', value: undefined },
  { name: 'production ref selected', key: 'OROTITAN_SHARED_INBOX_TEST_PROJECT_REF', value: 'cugpgtzygqqlxetyeven' },
  { name: 'production ref expected', key: 'OROTITAN_EXPECTED_SUPABASE_PROJECT_REF', value: 'cugpgtzygqqlxetyeven' },
  { name: 'production backend URL', key: 'OROTITAN_SHARED_INBOX_TEST_URL', value: 'https://cugpgtzygqqlxetyeven.supabase.co' },
  { name: 'production app URL', key: 'NEXT_PUBLIC_SUPABASE_URL', value: 'https://cugpgtzygqqlxetyeven.supabase.co' },
  { name: 'untrusted external host', key: 'OROTITAN_SHARED_INBOX_TEST_URL', value: 'https://example.org' },
  { name: 'HTTP shared URL', key: 'OROTITAN_SHARED_INBOX_TEST_URL', value: 'http://awgsurdyvsyolcgpnygh.supabase.co' },
  { name: 'secret missing', key: 'OROTITAN_SHARED_INBOX_TEST_SERVICE_KEY', value: undefined },
  { name: 'secret too short', key: 'OROTITAN_SHARED_INBOX_TEST_SERVICE_KEY', value: '123' },
  { name: 'app/service key mismatch', key: 'SUPABASE_SERVICE_ROLE_KEY', value: 'another-test-not-a-key-aaaaaaaaaaaaaaaaaaaaaaa' },
  { name: 'admin password missing', key: 'ADMIN_PASSWORD', value: undefined },
  { name: 'weak admin password', key: 'ADMIN_PASSWORD', value: 'weak' },
  { name: 'session secret missing', key: 'ADMIN_SESSION_SECRET', value: undefined },
  { name: 'short session secret', key: 'ADMIN_SESSION_SECRET', value: 'short' },
];

for (const { name, key, value } of denied) {
  test('hosted Research gate denies ' + name, () => {
    assert.equal(isHostedShadowResearchEnabled({ ...validShadowPreview(), [key]: value }), false);
  });
}

test('hosted gate fails closed on an empty environment', () => {
  assert.equal(isHostedShadowResearchEnabled({}), false);
});
