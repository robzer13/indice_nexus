import 'server-only';
import { createServerSupabaseClient } from '@/lib/supabase/server';
import type { MarketPriceRow, MarketSyncRun } from '@/lib/domain/types';

export interface MarketDataCompany {
  id: string;
  slug: string;
  name: string;
  market_data_symbol: string;
  market_data_multiplier: number;
  latest_price: number | null;
}

function slugifyMarketCompany(value: string): string {
  return value
    .normalize('NFD')
    .replace(/[\u0300-\u036f]/g, '')
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-+|-+$/g, '');
}

function deriveMarketDataSymbol(ticker: string, exchange: string, explicit: string | null): string | null {
  if (explicit) return explicit;
  const normalized = exchange.toUpperCase().replace(/[^A-Z0-9]/g, '');
  if (normalized.includes('NASDAQ')) return `${ticker}:NASDAQ`;
  if (normalized === 'NYSE' || normalized.includes('NEWYORKSTOCKEXCHANGE')) return `${ticker}:NYSE`;
  if (normalized.includes('NYSEARCA')) return `${ticker}:NYSEARCA`;
  return null;
}

async function ensureCanonicalMarketCompanyBridges(): Promise<void> {
  const supabase = createServerSupabaseClient();
  const { data: dossiers, error: dossierError } = await supabase
    .from('research_dossiers')
    .select('issuer_id,current_snapshot_id')
    .eq('active', true)
    .not('current_snapshot_id', 'is', null);
  if (dossierError) throw new Error(`Unable to load published dossiers for market bridge: ${dossierError.message}`);
  if (!dossiers || dossiers.length === 0) return;

  const issuerIds = dossiers.map((row) => row.issuer_id as string);
  const snapshotIds = dossiers.map((row) => row.current_snapshot_id as string);
  const [snapshotResult, issuerResult, existingResult, slugResult] = await Promise.all([
    supabase.from('research_snapshots').select('snapshot_id,issuer_id,security_id').in('snapshot_id', snapshotIds),
    supabase.from('issuers').select('issuer_id,display_name,legal_name,country,reporting_currency').in('issuer_id', issuerIds),
    supabase.from('companies').select('id,market_data_symbol').in('id', issuerIds),
    supabase.from('companies').select('id,slug'),
  ]);
  if (snapshotResult.error) throw new Error(`Unable to load snapshots for market bridge: ${snapshotResult.error.message}`);
  if (issuerResult.error) throw new Error(`Unable to load issuers for market bridge: ${issuerResult.error.message}`);
  if (existingResult.error) throw new Error(`Unable to load existing market bridges: ${existingResult.error.message}`);
  if (slugResult.error) throw new Error(`Unable to load company slugs for market bridge: ${slugResult.error.message}`);

  const snapshots = snapshotResult.data ?? [];
  const securityIds = snapshots.map((row) => row.security_id as string);
  const { data: securities, error: securityError } = await supabase
    .from('securities')
    .select('security_id,issuer_id,ticker,exchange,trading_currency,quote_unit,price_decimals,market_data_symbol,market_data_multiplier,country')
    .in('security_id', securityIds);
  if (securityError) throw new Error(`Unable to load securities for market bridge: ${securityError.message}`);

  const issuerById = new Map((issuerResult.data ?? []).map((row) => [row.issuer_id as string, row]));
  const securityById = new Map((securities ?? []).map((row) => [row.security_id as string, row]));
  const existingById = new Map((existingResult.data ?? []).map((row) => [row.id as string, row]));
  const occupiedSlugs = new Map((slugResult.data ?? []).map((row) => [row.slug as string, row.id as string]));

  for (const snapshot of snapshots) {
    const issuerId = snapshot.issuer_id as string;
    const issuer = issuerById.get(issuerId);
    const security = securityById.get(snapshot.security_id as string);
    if (!issuer || !security) continue;

    const ticker = String(security.ticker ?? '').trim();
    const exchange = String(security.exchange ?? '').trim();
    const explicitSymbol = typeof security.market_data_symbol === 'string' && security.market_data_symbol.trim()
      ? security.market_data_symbol.trim()
      : null;
    const marketDataSymbol = deriveMarketDataSymbol(ticker, exchange, explicitSymbol);
    if (!ticker || !exchange || !marketDataSymbol) continue;

    const existing = existingById.get(issuerId);
    if (existing) {
      if (!existing.market_data_symbol) {
        const { error } = await supabase
          .from('companies')
          .update({ market_data_symbol: marketDataSymbol, market_data_multiplier: Number(security.market_data_multiplier ?? 1), updated_at: new Date().toISOString() })
          .eq('id', issuerId);
        if (error) throw new Error(`Unable to update canonical market bridge for ${ticker}: ${error.message}`);
      }
      continue;
    }

    const name = String(issuer.display_name ?? issuer.legal_name ?? ticker);
    const baseSlug = slugifyMarketCompany(name) || ticker.toLowerCase();
    const occupiedBy = occupiedSlugs.get(baseSlug);
    const slug = occupiedBy && occupiedBy !== issuerId ? `${baseSlug}-${ticker.toLowerCase()}` : baseSlug;
    const { error } = await supabase.from('companies').insert({
      id: issuerId,
      slug,
      ticker,
      name,
      exchange,
      currency: String(security.trading_currency ?? issuer.reporting_currency ?? 'USD'),
      quote_unit: security.quote_unit === 'MINOR' ? 'MINOR' : 'MAJOR',
      price_decimals: Number(security.price_decimals ?? 2),
      market_data_symbol: marketDataSymbol,
      market_data_multiplier: Number(security.market_data_multiplier ?? 1),
      country: security.country ?? issuer.country ?? null,
      sector: null,
      active: true,
    });
    if (error) throw new Error(`Unable to create canonical market bridge for ${ticker}: ${error.message}`);
    occupiedSlugs.set(slug, issuerId);
  }
}

export async function getMarketDataCompanies(): Promise<MarketDataCompany[]> {
  const supabase = createServerSupabaseClient();
  await ensureCanonicalMarketCompanyBridges();
  const { data, error } = await supabase
    .from('latest_company_state')
    .select('id,slug,name,market_data_symbol,market_data_multiplier,price')
    .not('market_data_symbol', 'is', null)
    .order('name');
  if (error) throw new Error(`Unable to load market-data companies: ${error.message}`);
  return (data ?? [])
    .filter((row) => row.market_data_symbol)
    .map((row) => ({
      id: row.id,
      slug: row.slug,
      name: row.name,
      market_data_symbol: row.market_data_symbol,
      market_data_multiplier: Number(row.market_data_multiplier),
      latest_price: row.price === null ? null : Number(row.price),
    })) as MarketDataCompany[];
}

export async function insertMarketPrice(input: {
  companyId: string;
  price: number;
  asOf: string;
  source: string;
  raw: unknown;
}): Promise<void> {
  const supabase = createServerSupabaseClient();
  const { error } = await supabase.from('market_prices').insert({
    company_id: input.companyId,
    price: input.price,
    as_of: input.asOf,
    source: input.source,
    raw: input.raw,
  });
  if (error) throw new Error(`Unable to insert market price: ${error.message}`);
}

export async function getMarketPriceHistory(companyId: string, limit = 180): Promise<MarketPriceRow[]> {
  const supabase = createServerSupabaseClient();
  const { data, error } = await supabase
    .from('market_prices')
    .select('*')
    .eq('company_id', companyId)
    .order('as_of', { ascending: false })
    .limit(limit);
  if (error) throw new Error(`Unable to load market price history: ${error.message}`);
  return ((data ?? []) as MarketPriceRow[]).reverse();
}

export async function recordMarketSyncRun(input: {
  startedAt: string;
  finishedAt: string;
  triggerSource: 'CRON' | 'ADMIN';
  companies: number;
  inserted: number;
  failed: number;
  results: unknown;
}): Promise<void> {
  const supabase = createServerSupabaseClient();
  const { error } = await supabase.from('market_sync_runs').insert({
    started_at: input.startedAt,
    finished_at: input.finishedAt,
    trigger_source: input.triggerSource,
    companies: input.companies,
    inserted: input.inserted,
    failed: input.failed,
    results: input.results,
  });
  if (error) throw new Error(`Unable to record market sync run: ${error.message}`);
}

function isMissingMarketSyncRelation(error: { code?: string; message?: string } | null): boolean {
  if (!error) return false;
  return (
    error.code === '42P01' ||
    error.code === 'PGRST205' ||
    Boolean(error.message?.includes('market_sync_runs') && error.message?.includes('schema cache'))
  );
}

export async function getRecentMarketSyncRuns(limit = 20): Promise<MarketSyncRun[]> {
  const supabase = createServerSupabaseClient();
  const { data, error } = await supabase
    .from('market_sync_runs')
    .select('*')
    .order('created_at', { ascending: false })
    .limit(limit);
  if (isMissingMarketSyncRelation(error)) return [];
  if (error) throw new Error(`Unable to load market sync runs: ${error.message}`);
  return (data ?? []) as MarketSyncRun[];
}
