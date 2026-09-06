import 'server-only';
import { fetchTwelveDataPrice } from '@/lib/market/twelve-data';
import { fetchYahooFinancePrice } from '@/lib/market/yahoo-finance';

export type MarketPriceProvider = 'TWELVE_DATA' | 'YAHOO_FINANCE';

export interface ProviderPriceResult {
  provider: MarketPriceProvider;
  providerPrice: number;
  fetchedAt: string;
  raw: unknown;
}

export async function fetchProviderPrice(reference: string): Promise<ProviderPriceResult> {
  if (reference.includes(':')) {
    const result = await fetchYahooFinancePrice(reference);
    return {
      provider: 'YAHOO_FINANCE',
      providerPrice: result.providerPrice,
      fetchedAt: result.fetchedAt,
      raw: result.raw,
    };
  }

  const result = await fetchTwelveDataPrice(reference);
  return {
    provider: 'TWELVE_DATA',
    providerPrice: result.providerPrice,
    fetchedAt: result.fetchedAt,
    raw: result.raw,
  };
}
