import { parseTwelveDataInstrumentRef } from '@/lib/market/twelve-data-symbol';

const EXCHANGE_SUFFIXES: Record<string, string> = {
  LSE: '.L',
  EURONEXT: '.PA',
  OSE: '.OL',
  XETR: '.DE',
  NASDAQ: '',
  NYSE: '',
};

// Only recognized exchange suffixes may pass through unchanged. Never infer a venue.
const RECOGNIZED_DOTTED_SUFFIXES = new Set(['.AS','.SW','.TO','.SI','.CO','.V','.TW','.PA','.L','.OL','.DE']);

export function toYahooFinanceSymbol(reference: string): string {
  const dotted = reference.trim().toUpperCase().match(/^([A-Z0-9-]+)(\.[A-Z]+)$/);
  if (dotted) {
    if (!RECOGNIZED_DOTTED_SUFFIXES.has(dotted[2]))
      throw new Error('Unsupported Yahoo Finance exchange suffix: '+dotted[2]);
    return dotted[0];
  }
  const instrument = parseTwelveDataInstrumentRef(reference);
  if (!instrument.exchange) return instrument.symbol;

  const suffix = EXCHANGE_SUFFIXES[instrument.exchange];
  if (!suffix) {
    throw new Error(`Unsupported Yahoo Finance exchange mapping: ${instrument.exchange}`);
  }
  return `${instrument.symbol}${suffix}`;
}
