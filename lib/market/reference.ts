export function deriveMarketDataReference(
  ticker: string,
  exchange: string,
  explicit: string | null,
): string | null {
  if (explicit?.trim()) return explicit.trim();

  const symbol = ticker.trim();
  const normalized = exchange.toUpperCase().replace(/[^A-Z0-9]/g, '');
  if (!symbol) return null;

  if (normalized.includes('NASDAQ')) return symbol + ':NASDAQ';
  if (normalized === 'NYSE' || normalized.includes('NEWYORKSTOCKEXCHANGE')) return symbol + ':NYSE';
  if (normalized.includes('NYSEARCA')) return symbol + ':NYSEARCA';
  if (normalized.includes('AMEX')) return symbol + ':AMEX';
  if (normalized.includes('LONDON') || normalized === 'LSE') return symbol + ':LSE';
  if (normalized.includes('EURONEXTPARIS') || normalized === 'EPA' || normalized === 'EURONEXT') return symbol + ':EPA';
  if (normalized.includes('OSLO') || normalized === 'OSE') return symbol + ':OSE';
  if (normalized.includes('XETRA') || normalized === 'XETR') return symbol + ':XETR';

  return null;
}
