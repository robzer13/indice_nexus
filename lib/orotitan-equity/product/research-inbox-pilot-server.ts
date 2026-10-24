import 'server-only';
import { isHostedShadowResearchEnabled } from './hosted-shadow-research-gate';
import { join } from 'node:path';
import { createClient, type SupabaseClient } from '@supabase/supabase-js';
import { getAllCompanies } from '@/lib/data/company-admin';
import {
  enqueueLocalResearchRequest, listLocalResearchRequests,
  type ResearchInboxCompany,
} from './local-research-inbox-v1';
import {
  enqueueSharedResearchCandidate, listSharedResearchInboxCandidate,
} from './shared-research-inbox-candidate-v1';

/**
 * One web UI, two strictly gated pilot backends.
 * "shared" is available in isolated development or the exact, explicitly
 * enabled Vercel Preview targeting SHADOW, never OroTitan production.
 */
export type ResearchInboxPilotMode = 'disabled' | 'local' | 'shared-test';

const PROD_REF = 'cugpgtzygqqlxetyeven';

export function researchInboxPilotMode(): ResearchInboxPilotMode {
  // On Vercel, both Preview and Production MUST obey the single strict
  // SHADOW-only hosted gate, even if someone overrides NODE_ENV.
  if (process.env.VERCEL_ENV) {
    return isHostedShadowResearchEnabled(process.env) ? 'shared-test' : 'disabled';
  }
  if (process.env.NODE_ENV !== 'development') return 'disabled';
  if (process.env.OROTITAN_SHARED_INBOX_TEST_ENABLED === '1') return 'shared-test';
  if (process.env.OROTITAN_LOCAL_REQUESTS_ENABLED === '1') return 'local';
  return 'disabled';
}

function localDirectory() {
  return join(process.cwd(), '.orotitan-local');
}

function sharedTestClient(): SupabaseClient {
  if (researchInboxPilotMode() !== 'shared-test') {
    throw new Error('PRODUCT_SHARED_INBOX_DISABLED');
  }
  const ref = process.env.OROTITAN_SHARED_INBOX_TEST_PROJECT_REF;
  const rawUrl = process.env.OROTITAN_SHARED_INBOX_TEST_URL;
  const secret = process.env.OROTITAN_SHARED_INBOX_TEST_SERVICE_KEY;
  // CI-only loopback fixture emulates the minimum PostgREST API over a
  // disposable PG17 database. It is intentionally unreachable in production.
  // No credentials or hostname supplied by the browser are trusted here.
  if (process.env.NODE_ENV === 'development' &&
    process.env.CI === 'true' &&
    process.env.OROTITAN_PRODUCT_BROWSER_FIXTURE === '1') {
    if (ref !== 'local-browser-fixture' ||
      rawUrl !== 'http://127.0.0.1:48761' ||
      !secret || secret.length < 40 ||
      process.env.NEXT_PUBLIC_SUPABASE_URL !== rawUrl) {
      throw new Error('PRODUCT_BROWSER_FIXTURE_CONFIG_DENIED');
    }
    return createClient(rawUrl, secret, {
      auth: { persistSession: false, autoRefreshToken: false, detectSessionInUrl: false },
    });
  }
  if (!ref || !/^[a-z]{20}$/.test(ref) || ref === PROD_REF ||
    !rawUrl || !secret || secret.length < 40) {
    throw new Error('PRODUCT_SHARED_TEST_CONFIG_DENIED');
  }
  let url: URL;
  try { url = new URL(rawUrl); }
  catch { throw new Error('PRODUCT_SHARED_TEST_URL_INVALID'); }
  if (url.protocol !== 'https:' || url.hostname !== ref + '.supabase.co' ||
    url.username || url.password || url.pathname !== '/' ||
    url.search || url.hash ||
    (url.origin === process.env.NEXT_PUBLIC_SUPABASE_URL &&
      !isHostedShadowResearchEnabled(process.env))) {
    throw new Error('PRODUCT_SHARED_TEST_URL_DENIED');
  }
  return createClient(url.toString(), secret, {
    auth: { persistSession: false, autoRefreshToken: false, detectSessionInUrl: false },
  });
}

export async function getPilotResearchCompanies(): Promise<ResearchInboxCompany[]> {
  const mode = researchInboxPilotMode();
  if (mode === 'disabled') throw new Error('PRODUCT_PILOT_DISABLED');
  if (mode === 'local') return getAllCompanies();
  const { data, error } = await sharedTestClient().from('companies')
    .select('id,name,ticker,exchange,active')
    .order('name', { ascending: true });
  if (error || !data) throw new Error('PRODUCT_TEST_COMPANY_READ_FAILED');
  return data as ResearchInboxCompany[];
}

export async function loadPilotResearchInbox() {
  const mode = researchInboxPilotMode();
  if (mode === 'local') return listLocalResearchRequests(localDirectory());
  if (mode === 'shared-test') {
    return listSharedResearchInboxCandidate(sharedTestClient());
  }
  throw new Error('PRODUCT_PILOT_DISABLED');
}

export async function requestPilotResearch(companyId: string) {
  const mode = researchInboxPilotMode();
  if (mode === 'local') {
    const companies = await getAllCompanies();
    return enqueueLocalResearchRequest({
      directory: localDirectory(), companyId, companies,
    });
  }
  if (mode === 'shared-test') {
    // SQL re-reads active company and creates the only authoritative identity
    // snapshot in the test database transaction.
    return enqueueSharedResearchCandidate({ client: sharedTestClient(), companyId });
  }
  throw new Error('PRODUCT_PILOT_DISABLED');
}
