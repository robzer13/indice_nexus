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

export async function getMarketDataCompanies(): Promise<MarketDataCompany[]> {
  const supabase = createServerSupabaseClient();
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
