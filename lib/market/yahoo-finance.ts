import 'server-only';
import YahooFinance from 'yahoo-finance2';
import { toYahooFinanceSymbol } from '@/lib/market/yahoo-symbol';

const yahooFinance = new YahooFinance();

export interface YahooFinancePriceResult {
  providerPrice: number;
  fetchedAt: string;
  providerSymbol: string;
  providerCurrency: string | null;
  raw: unknown;
}

function toIso(value: unknown): string {
  if (value instanceof Date && !Number.isNaN(value.getTime())) return value.toISOString();
  if (typeof value === 'number' && Number.isFinite(value)) {
    const milliseconds = value > 10_000_000_000 ? value : value * 1000;
    const parsed = new Date(milliseconds);
    if (!Number.isNaN(parsed.getTime())) return parsed.toISOString();
  }
  if (typeof value === 'string') {
    const parsed = new Date(value);
    if (!Number.isNaN(parsed.getTime())) return parsed.toISOString();
  }
  return new Date().toISOString();
}

export async function fetchYahooFinancePrice(reference: string): Promise<YahooFinancePriceResult> {
  const providerSymbol = toYahooFinanceSymbol(reference);
  const quote = await yahooFinance.quote(providerSymbol) as {
    symbol?: string;
    regularMarketPrice?: number;
    currency?: string;
    regularMarketTime?: unknown;
    quoteSourceName?: string;
    marketState?: string;
  } | undefined;

  const providerPrice = Number(quote?.regularMarketPrice);
  if (!Number.isFinite(providerPrice) || providerPrice <= 0) {
    throw new Error(`Yahoo Finance returned no valid price for ${providerSymbol}`);
  }

  const fetchedAt = toIso(quote?.regularMarketTime);
  const providerCurrency = typeof quote?.currency === 'string' ? quote.currency : null;

  return {
    providerPrice,
    fetchedAt,
    providerSymbol,
    providerCurrency,
    raw: {
      symbol: quote?.symbol ?? providerSymbol,
      regularMarketPrice: providerPrice,
      currency: providerCurrency,
      regularMarketTime: fetchedAt,
      quoteSourceName: quote?.quoteSourceName ?? null,
      marketState: quote?.marketState ?? null,
    },
  };
}
