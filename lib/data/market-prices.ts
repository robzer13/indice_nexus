import 'server-only';
import { createServerSupabaseClient } from '@/lib/supabase/server';
import type { MarketPriceRow, MarketSyncRun } from '@/lib/domain/types';

type UnknownRecord = Record<string, unknown>;

export interface MarketDataCompany {
  securityId: string;
  issuerId: string;
  slug: string;
  name: string;
  market_data_symbol: string;
  market_data_multiplier: number;
  latest_price: number | null;
}

function asString(value: unknown): string | null {
  return typeof value === 'string' && value.length > 0 ? value : null;
}

function asNumber(value: unknown): number | null {
  if (typeof value === 'number' && Number.isFinite(value)) return value;
  if (typeof value === 'string' && value.trim() !== '') {
    const parsed = Number(value);
    return Number.isFinite(parsed) ? parsed : null;
  }
  return null;
}

function slugify(value: string): string {
  return value
    .normalize('NFD')
    .replace(/[\u0300-\u036f]/g, '')
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-+|-+$/g, '');
}

function isMissingRelation(error: { code?: string; message?: string } | null | undefined, relation: string): boolean {
  if (!error) return false;
  return error.code === '42P01' || error.code === 'PGRST205' || Boolean(error.message?.includes(relation));
}

export async function getMarketDataCompanies(): Promise<MarketDataCompany[]> {
  const supabase = createServerSupabaseClient();
  const { data: dossiers, error: dossierError } = await supabase
    .from('research_dossiers')
    .select('issuer_id,current_snapshot_id')
    .eq('active', true)
    .not('current_snapshot_id', 'is', null);
  if (dossierError) throw new Error(`Unable to load canonical dossiers for market sync: ${dossierError.message}`);
  if (!dossiers || dossiers.length === 0) return [];

  const snapshotIds = dossiers
    .map((row) => row.current_snapshot_id)
    .filter((value): value is string => typeof value === 'string');
  const { data: snapshots, error: snapshotError } = await supabase
    .from('research_snapshots')
    .select('snapshot_id,issuer_id,security_id')
    .in('snapshot_id', snapshotIds);
  if (snapshotError) throw new Error(`Unable to load canonical snapshots for market sync: ${snapshotError.message}`);

  const securityIds = (snapshots ?? []).map((row) => row.security_id as string);
  const issuerIds = (snapshots ?? []).map((row) => row.issuer_id as string);
  if (securityIds.length === 0) return [];

  const [securitiesResult, issuersResult, companiesResult, pricesResult] = await Promise.all([
    supabase.from('securities').select('security_id,issuer_id,ticker,market_data_symbol,market_data_multiplier').in('security_id', securityIds),
    supabase.from('issuers').select('issuer_id,display_name,legal_name').in('issuer_id', issuerIds),
    supabase.from('companies').select('id,slug').in('id', issuerIds),
    supabase.from('security_market_prices').select('security_id,price,as_of').in('security_id', securityIds).order('as_of', { ascending: false }),
  ]);

  if (securitiesResult.error) throw new Error(`Unable to load securities for market sync: ${securitiesResult.error.message}`);
  if (issuersResult.error) throw new Error(`Unable to load issuers for market sync: ${issuersResult.error.message}`);
  if (companiesResult.error) throw new Error(`Unable to load display metadata for market sync: ${companiesResult.error.message}`);
  if (pricesResult.error && !isMissingRelation(pricesResult.error, 'security_market_prices')) {
    throw new Error(`Unable to load canonical market prices: ${pricesResult.error.message}`);
  }

  const issuerById = new Map((issuersResult.data ?? []).map((row) => [row.issuer_id as string, row as UnknownRecord]));
  const slugByIssuer = new Map((companiesResult.data ?? []).map((row) => [row.id as string, row.slug as string]));
  const latestPriceBySecurity = new Map<string, number>();
  for (const row of pricesResult.data ?? []) {
    const securityId = row.security_id as string;
    if (!latestPriceBySecurity.has(securityId)) {
      const price = asNumber(row.price);
      if (price !== null) latestPriceBySecurity.set(securityId, price);
    }
  }

  return (securitiesResult.data ?? []).flatMap((security) => {
    const symbol = asString(security.market_data_symbol);
    if (!symbol) return [];

    const securityId = security.security_id as string;
    const issuerId = security.issuer_id as string;
    const issuer = issuerById.get(issuerId);
    const name = asString(issuer?.display_name) ?? asString(issuer?.legal_name) ?? asString(security.ticker) ?? issuerId;
    const multiplier = asNumber(security.market_data_multiplier) ?? 1;
    return [{
      securityId,
      issuerId,
      slug: slugByIssuer.get(issuerId) ?? slugify(name),
      name,
      market_data_symbol: symbol,
      market_data_multiplier: multiplier,
      latest_price: latestPriceBySecurity.get(securityId) ?? null,
    }];
  }).sort((a, b) => a.name.localeCompare(b.name));
}

export async function insertSecurityMarketPrice(input: {
  securityId: string;
  price: number;
  asOf: string;
  source: string;
  raw: unknown;
}): Promise<void> {
  const supabase = createServerSupabaseClient();
  const { error } = await supabase.from('security_market_prices').insert({
    security_id: input.securityId,
    price: input.price,
    as_of: input.asOf,
    source: input.source,
    raw: input.raw,
  });
  if (error) throw new Error(`Unable to insert canonical security market price: ${error.message}`);
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
  if (error) throw new Error(`Unable to insert legacy market price: ${error.message}`);
}

export async function getMarketPriceHistory(
  companyId: string,
  securityId?: string | null,
  limit = 180,
): Promise<MarketPriceRow[]> {
  const supabase = createServerSupabaseClient();
  const [legacyResult, canonicalResult] = await Promise.all([
    supabase
      .from('market_prices')
      .select('*')
      .eq('company_id', companyId)
      .order('as_of', { ascending: false })
      .limit(limit),
    securityId
      ? supabase
          .from('security_market_prices')
          .select('*')
          .eq('security_id', securityId)
          .order('as_of', { ascending: false })
          .limit(limit)
      : Promise.resolve({ data: [], error: null }),
  ]);

  if (legacyResult.error) throw new Error(`Unable to load legacy market price history: ${legacyResult.error.message}`);
  if (canonicalResult.error && !isMissingRelation(canonicalResult.error, 'security_market_prices')) {
    throw new Error(`Unable to load canonical security price history: ${canonicalResult.error.message}`);
  }

  const legacy = (legacyResult.data ?? []) as MarketPriceRow[];
  const canonical = ((canonicalResult.data ?? []) as Array<Record<string, unknown>>).map((row) => ({
    id: Number(row.id),
    company_id: companyId,
    security_id: typeof row.security_id === 'string' ? row.security_id : securityId ?? null,
    price: Number(row.price),
    as_of: String(row.as_of),
    source: String(row.source),
    raw: (row.raw ?? null) as MarketPriceRow['raw'],
    created_at: String(row.created_at),
  } satisfies MarketPriceRow));

  const merged = [...legacy, ...canonical]
    .filter((row) => Number.isFinite(Number(row.price)) && Number(row.price) > 0)
    .sort((a, b) => Date.parse(a.as_of) - Date.parse(b.as_of));

  const deduped: MarketPriceRow[] = [];
  const seen = new Set<string>();
  for (const row of merged) {
    const key = `${row.as_of}|${row.price}|${row.source}`;
    if (seen.has(key)) continue;
    seen.add(key);
    deduped.push(row);
  }
  return deduped.slice(-limit);
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
  if (error) throw new Error(`Unable to record sync run: ${error.message}`);
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
