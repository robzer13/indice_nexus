import 'server-only';
import { isHostedShadowResearchEnabled } from './hosted-shadow-research-gate';
import {isRecoveryCiBrowserFixture} from './research-recovery-ci-fixture';
import type { ResearchIdentitySnapshot } from './research-intake-preflight-v1';
import type { ResearchRegistryRoutingSnapshot } from './research-pre-run-card-v1';
import { join } from 'node:path';
import { createClient, type SupabaseClient } from '@supabase/supabase-js';
import { getAllCompanies } from '@/lib/data/company-admin';
import {
  enqueueLocalResearchRequest, listLocalResearchRequests,
  type ResearchInboxCompany, type ResearchInboxRequest,
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
const UUID_RE = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;

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

/**
 * Read-only Registry candidate lookup. This does not confirm issuer identity,
 * pick a canonical mode, attach an execution GO, or write any run or request.
 * CI/local fixture does not expose Registry data and must fail closed.
 */
export async function getPilotResearchIdentitySnapshot(
  requests: readonly ResearchInboxRequest[],
): Promise<ResearchIdentitySnapshot> {
  if (!isHostedShadowResearchEnabled(process.env)) {
    return { verified: false, securities: [] };
  }
  if (requests.length === 0) return { verified: true, securities: [] };
  const tickers = [...new Set(requests.map(r => r.ticker.trim().toUpperCase()))];
  if (tickers.length === 0 || tickers.length > 500) {
    return { verified: false, securities: [] };
  }
  try {
    const { data, error } = await sharedTestClient()
      .from('securities')
      .select('security_id,issuer_id,ticker,exchange')
      .in('ticker', tickers)
      .limit(1000);
    // A capped/incomplete dataset must NEVER be mistaken for a unique match.
    if (error || !data || data.length >= 1000) {
      return { verified: false, securities: [] };
    }
    const candidateRows = data as Array<Record<string, unknown>>;
    if (candidateRows.some(row =>
      typeof row.security_id !== 'string' || !UUID_RE.test(row.security_id) ||
      typeof row.issuer_id !== 'string' || !UUID_RE.test(row.issuer_id) ||
      typeof row.ticker !== 'string' || !row.ticker.trim() ||
      typeof row.exchange !== 'string' || !row.exchange.trim(),
    )) {
      return { verified: false, securities: [] };
    }
    return {
      verified: true,
      securities: candidateRows.map(row => ({
        securityId: row.security_id as string,
        issuerId: row.issuer_id as string,
        ticker: row.ticker as string,
        exchange: row.exchange as string,
      })),
    };
  } catch {
    // Explicit unknown, rather than erroneously claiming no issuer exists.
    return { verified: false, securities: [] };
  }
}

/**
 * Read-only SHADOW Registry routing snapshot. Never upgrades a Research
 * request to a run, never writes dossiers, and never confirms legal identity.
 * Only the exact hosted SHADOW Preview may query canonical Registry tables.
 */
export async function getPilotResearchRegistryRoutingSnapshot(
  identity: ResearchIdentitySnapshot,
): Promise<ResearchRegistryRoutingSnapshot> {
  const unavailable: ResearchRegistryRoutingSnapshot = {
    verified: false, issuers: [], dossiers: [], snapshots: [], runs: [],
  };
  if (!isHostedShadowResearchEnabled(process.env) || !identity.verified) {
    return unavailable;
  }
  const issuerIds = [...new Set(identity.securities.map(s => s.issuerId))];
  if (!issuerIds.every(id => UUID_RE.test(id)) || issuerIds.length > 100) {
    return unavailable;
  }
  if (issuerIds.length === 0) {
    return { verified: true, issuers: [], dossiers: [], snapshots: [], runs: [] };
  }
  try {
    const client = sharedTestClient();
    const [issuerResult, dossierResult, runResult] = await Promise.all([
      client.from('issuers')
        .select('issuer_id,display_name,legal_name,country,reporting_currency')
        .in('issuer_id', issuerIds).limit(500),
      client.from('research_dossiers')
        .select('dossier_id,issuer_id,current_snapshot_id,active')
        .in('issuer_id', issuerIds).limit(500),
      client.from('orotitan_runs')
        .select('run_id,issuer_id,run_status')
        .in('issuer_id', issuerIds).limit(500),
    ]);
    if (issuerResult.error || dossierResult.error || runResult.error ||
      !issuerResult.data || !dossierResult.data || !runResult.data ||
      issuerResult.data.length >= 500 || dossierResult.data.length >= 500 ||
      runResult.data.length >= 500) return unavailable;

    const issuers = issuerResult.data as Record<string, unknown>[];
    const dossiers = dossierResult.data as Record<string, unknown>[];
    const runs = runResult.data as Record<string, unknown>[];
    if (issuers.some(i => !UUID_RE.test(String(i.issuer_id ?? '')) ||
      typeof i.display_name !== 'string' ||
      (i.legal_name !== null && typeof i.legal_name !== 'string') ||
      (i.country !== null && typeof i.country !== 'string') ||
      (i.reporting_currency !== null && typeof i.reporting_currency !== 'string'))) {
      return unavailable;
    }
    if (dossiers.some(d => !UUID_RE.test(String(d.dossier_id ?? '')) ||
      !UUID_RE.test(String(d.issuer_id ?? '')) ||
      typeof d.active !== 'boolean' ||
      (d.current_snapshot_id !== null && !UUID_RE.test(String(d.current_snapshot_id))))) {
      return unavailable;
    }
    if (runs.some(run => !UUID_RE.test(String(run.run_id ?? '')) ||
      !UUID_RE.test(String(run.issuer_id ?? '')) ||
      typeof run.run_status !== 'string')) return unavailable;

    const snapshotIds = [...new Set(dossiers
      .filter(d => d.active && d.current_snapshot_id)
      .map(d => String(d.current_snapshot_id)))];
    let snapshots: Record<string, unknown>[] = [];
    if (snapshotIds.length > 0) {
      const result = await client.from('research_snapshots')
        .select('snapshot_id,issuer_id,dossier_id')
        .in('snapshot_id', snapshotIds).limit(500);
      if (result.error || !result.data || result.data.length >= 500) return unavailable;
      snapshots = result.data as Record<string, unknown>[];
      if (snapshots.some(snapshot =>
        !UUID_RE.test(String(snapshot.snapshot_id ?? '')) ||
        !UUID_RE.test(String(snapshot.issuer_id ?? '')) ||
        !UUID_RE.test(String(snapshot.dossier_id ?? '')))) return unavailable;
    }

    return {
      verified: true,
      issuers: issuers.map(i => ({
        issuerId: i.issuer_id as string,
        displayName: i.display_name as string,
        legalName: i.legal_name as string | null,
        country: i.country as string | null,
        reportingCurrency: i.reporting_currency as string | null,
      })),
      dossiers: dossiers.map(d => ({
        dossierId: d.dossier_id as string,
        issuerId: d.issuer_id as string,
        active: d.active as boolean,
        currentSnapshotId: d.current_snapshot_id as string | null,
      })),
      snapshots: snapshots.map(snapshot => ({
        snapshotId: snapshot.snapshot_id as string,
        issuerId: snapshot.issuer_id as string,
        dossierId: snapshot.dossier_id as string,
      })),
      runs: runs.map(run => ({
        runId: run.run_id as string,
        issuerId: run.issuer_id as string,
        runStatus: run.run_status as string,
      })),
    };
  } catch {
    return unavailable;
  }
}

/** Service-role server-only SHADOW client for non-authoritative recovery journal. */
export function getHostedShadowRecoveryClient(): SupabaseClient {
  // CI-only loopback disposable PG17 fixture is NOT a hosted project.
  // The normal hosted production safety gate remains unchanged.
  if (!(isHostedShadowResearchEnabled(process.env) ||
        isRecoveryCiBrowserFixture(process.env)) ||
      researchInboxPilotMode() !== 'shared-test') {
    throw new Error('PRODUCT_RECOVERY_NOT_HOSTED_SHADOW');
  }
  return sharedTestClient();
}
