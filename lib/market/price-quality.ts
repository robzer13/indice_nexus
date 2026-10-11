/** Pure, deterministic admission checks. No provider may bypass these gates. */
export type MarketQuoteUnit = 'MAJOR' | 'MINOR';
export type MarketFeedName = 'YAHOO_FINANCE' | 'TWELVE_DATA';
export interface ExpectedMarketInstrument {
  reference: string;
  ticker: string;
  exchange: string;
  currency: string;
  quoteUnit: MarketQuoteUnit;
  multiplier: number;
  previousPrice: number | null;
}
export interface CandidateMarketQuote {
  provider: MarketFeedName;
  providerSymbol: string;
  providerCurrency: string | null;
  providerExchange: string | null;
  providerPrice: number;
  observedAt: string;
  raw: unknown;
}
const MAX_PRICE_AGE_MS = 96 * 60 * 60 * 1000;
const FUTURE_GRACE_MS = 5 * 60 * 1000;
const US_EXCHANGES = new Set(['NASDAQ','NYSE','NYSE ARCA','NYSE AMERICAN']);
const YAHOO_NASDAQ = new Set(['NMS','NGM','NCM','NAS','NMQ']);
const YAHOO_NYSE = new Set(['NYQ','NYE','ASE','PCX']);

export function isUnambiguousUSExchange(exchange: string): boolean {
  return US_EXCHANGES.has(exchange.trim().toUpperCase());
}

export function requireUsListingMatch(expectedExchange: string, providerExchange: string | null): void {
  const venue = expectedExchange.trim().toUpperCase();
  const actual = providerExchange?.trim().toUpperCase() ?? '';
  if (venue === 'NASDAQ' && !YAHOO_NASDAQ.has(actual))
    throw new Error('QUOTE_EXCHANGE_MISMATCH');
  if (venue === 'NYSE' && !YAHOO_NYSE.has(actual))
    throw new Error('QUOTE_EXCHANGE_MISMATCH');
  if ((venue === 'NYSE ARCA' || venue === 'NYSE AMERICAN') && !YAHOO_NYSE.has(actual))
    throw new Error('QUOTE_EXCHANGE_MISMATCH');
}

export function admitMarketQuote(
  expected: ExpectedMarketInstrument,
  candidate: CandidateMarketQuote,
  expectedProviderSymbol: string,
  now: Date = new Date(),
): number {
  if (!expected.reference.trim() || !expected.ticker.trim()
    || !/^[A-Z]{3}$/.test(expected.currency)
    || !['MAJOR','MINOR'].includes(expected.quoteUnit)
    || !Number.isFinite(expected.multiplier) || expected.multiplier <= 0) {
    throw new Error('INVALID_CANONICAL_INSTRUMENT');
  }
  const match = (s: string) => s.trim().toUpperCase();
  if (!candidate.providerSymbol || match(candidate.providerSymbol) !== match(expectedProviderSymbol))
    throw new Error('QUOTE_SYMBOL_MISMATCH');
  const quotedCurrency = candidate.providerCurrency;
  if (!quotedCurrency) throw new Error('QUOTE_CURRENCY_MISSING');
  if (expected.quoteUnit === 'MINOR') {
    if (expected.currency !== 'GBP' || !['GBp','GBX'].includes(quotedCurrency))
      throw new Error('QUOTE_CURRENCY_UNIT_MISMATCH');
  } else if (quotedCurrency !== expected.currency) {
    throw new Error('QUOTE_CURRENCY_UNIT_MISMATCH');
  }
  const observedMs = Date.parse(candidate.observedAt);
  const nowMs = now.getTime();
  if (!Number.isFinite(observedMs) || !Number.isFinite(nowMs)
    || observedMs > nowMs + FUTURE_GRACE_MS || nowMs - observedMs > MAX_PRICE_AGE_MS)
    throw new Error('QUOTE_TIMESTAMP_OUT_OF_WINDOW');
  const normalizedPrice = candidate.providerPrice * expected.multiplier;
  if (!Number.isFinite(candidate.providerPrice) || candidate.providerPrice <= 0
    || !Number.isFinite(normalizedPrice) || normalizedPrice <= 0) {
    throw new Error('QUOTE_NON_POSITIVE_OR_INVALID');
  }
  // Preserve the existing extreme-move guard, including provider fallbacks.
  if (expected.previousPrice !== null
    && Number.isFinite(expected.previousPrice) && expected.previousPrice > 0) {
    const ratio = normalizedPrice / expected.previousPrice;
    if (ratio > 5 || ratio < 0.2) throw new Error('QUOTE_EXTREME_PRICE_CHANGE');
  }
  return normalizedPrice;
}
