import 'server-only';
import {
  getMarketDataCompanies,
  insertMarketPrice,
  recordMarketSyncRun,
} from '@/lib/data/market-prices';
import { fetchTwelveDataPrice } from '@/lib/market/twelve-data';

export interface PriceRefreshItem {
  slug: string;
  symbol: string;
  status: 'inserted' | 'failed';
  price?: number;
  error?: string;
}

export interface PriceRefreshResult {
  ok: boolean;
  triggerSource: 'CRON' | 'ADMIN';
  startedAt: string;
  finishedAt: string;
  companies: number;
  inserted: number;
  failed: number;
  results: PriceRefreshItem[];
  logRecorded: boolean;
  logError?: string;
}

export async function refreshMarketPrices(triggerSource: 'CRON' | 'ADMIN'): Promise<PriceRefreshResult> {
  const startedAt = new Date().toISOString();
  const companies = await getMarketDataCompanies();
  const results: PriceRefreshItem[] = [];

  for (const company of companies) {
    try {
      if (!company.market_data_symbol) {
        results.push({ slug: company.slug, symbol: '', status: 'failed', error: 'Missing market_data_symbol' });
        continue;
      }
      if (!Number.isFinite(company.market_data_multiplier) || company.market_data_multiplier <= 0) {
        results.push({ slug: company.slug, symbol: company.market_data_symbol, status: 'failed', error: 'Invalid market_data_multiplier' });
        continue;
      }

      const provider = await fetchTwelveDataPrice(company.market_data_symbol);
      const normalizedPrice = provider.providerPrice * company.market_data_multiplier;
      if (!Number.isFinite(normalizedPrice) || normalizedPrice <= 0) {
        throw new Error('Normalized price is invalid');
      }

      await insertMarketPrice({
        companyId: company.id,
        price: normalizedPrice,
        asOf: provider.fetchedAt,
        source: 'TWELVE_DATA',
        raw: provider.raw,
      });
      results.push({ slug: company.slug, symbol: company.market_data_symbol, status: 'inserted', price: normalizedPrice });
    } catch (error) {
      results.push({
        slug: company.slug,
        symbol: company.market_data_symbol,
        status: 'failed',
        error: error instanceof Error ? error.message : 'Unknown error',
      });
    }
  }

  const finishedAt = new Date().toISOString();
  const inserted = results.filter((result) => result.status === 'inserted').length;
  const failed = results.length - inserted;
  let logRecorded = false;
  let logError: string | undefined;

  try {
    await recordMarketSyncRun({
      startedAt,
      finishedAt,
      triggerSource,
      companies: results.length,
      inserted,
      failed,
      results,
    });
    logRecorded = true;
  } catch (error) {
    logError = error instanceof Error ? error.message : 'Unable to record sync run';
  }

  return {
    ok: failed === 0,
    triggerSource,
    startedAt,
    finishedAt,
    companies: results.length,
    inserted,
    failed,
    results,
    logRecorded,
    ...(logError ? { logError } : {}),
  };
}
