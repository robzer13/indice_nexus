const EXCHANGE_ALIASES: Record<string, string> = {
  EPA: 'EURONEXT',
};

export interface TwelveDataInstrumentRef {
  symbol: string;
  exchange: string | null;
}

export function parseTwelveDataInstrumentRef(value: string): TwelveDataInstrumentRef {
  const trimmed = value.trim();
  const separator = trimmed.lastIndexOf(':');
  if (separator <= 0 || separator === trimmed.length - 1) {
    return { symbol: trimmed, exchange: null };
  }

  const symbol = trimmed.slice(0, separator).trim();
  const rawExchange = trimmed.slice(separator + 1).trim().toUpperCase();
  return {
    symbol,
    exchange: EXCHANGE_ALIASES[rawExchange] ?? rawExchange,
  };
}
