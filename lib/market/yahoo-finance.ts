import 'server-only';
import YahooFinance from 'yahoo-finance2';
import { toYahooFinanceSymbol } from '@/lib/market/yahoo-symbol';

const yahooFinance = new YahooFinance();

export interface YahooFinancePriceResult {
  providerPrice: number;
  fetchedAt: string;
  providerSymbol: string;
  providerCurrency: string | null;
  providerExchange: string | null;
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
  throw new Error('YAHOO_MARKET_TIMESTAMP_MISSING');
}

export async function fetchYahooFinancePrice(reference: string): Promise<YahooFinancePriceResult> {
  const providerSymbol = toYahooFinanceSymbol(reference);
  const quote = await yahooFinance.quote(providerSymbol) as {
    symbol?: string;
    regularMarketPrice?: number;
    currency?: string;
    regularMarketTime?: unknown;
    quoteSourceName?: string;
    exchange?: string;
    marketState?: string;
  } | undefined;

  const providerPrice = Number(quote?.regularMarketPrice);
  if (!Number.isFinite(providerPrice) || providerPrice <= 0) {
    throw new Error(`Yahoo Finance returned no valid price for ${providerSymbol}`);
  }

  const fetchedAt = toIso(quote?.regularMarketTime);
  const providerCurrency = typeof quote?.currency === 'string' ? quote.currency : null;
  const providerExchange = typeof quote?.exchange === 'string' ? quote.exchange : null;
  if (!quote?.symbol || quote.symbol.toUpperCase() !== providerSymbol.toUpperCase())
    throw new Error('YAHOO_QUOTE_SYMBOL_MISMATCH');

  return {
    providerPrice,
    fetchedAt,
    providerSymbol,
    providerCurrency,
    providerExchange,
    raw: {
      symbol: quote.symbol,
      regularMarketPrice: providerPrice,
      currency: providerCurrency,
      exchange: providerExchange,
      regularMarketTime: fetchedAt,
      quoteSourceName: quote?.quoteSourceName ?? null,
      marketState: quote?.marketState ?? null,
    },
  };
}
