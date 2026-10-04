import { parseTwelveDataInstrumentRef } from '@/lib/market/twelve-data-symbol';

const EXCHANGE_SUFFIXES: Record<string, string> = {
  LSE: '.L',
  EURONEXT: '.PA',
  OSE: '.OL',
  XETR: '.DE',
  AMS: '.AS',
  SWX: '.SW',
  TSX: '.TO',
  TSXV: '.V',
  CPH: '.CO',
  TWSE: '.TW',
  SGX: '.SI',
};

export function toYahooFinanceSymbol(reference: string): string {
  const instrument = parseTwelveDataInstrumentRef(reference);
  if (!instrument.exchange) return instrument.symbol;

  const suffix = EXCHANGE_SUFFIXES[instrument.exchange];
  if (!suffix) {
    throw new Error(`Unsupported Yahoo Finance exchange mapping: ${instrument.exchange}`);
  }
  return `${instrument.symbol}${suffix}`;
}
