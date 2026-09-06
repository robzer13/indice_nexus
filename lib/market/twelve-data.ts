import 'server-only';
import { parseTwelveDataInstrumentRef } from '@/lib/market/twelve-data-symbol';

const REQUEST_TIMEOUT_MS = 8_000;

export interface TwelveDataPriceResult {
  providerPrice: number;
  fetchedAt: string;
  raw: unknown;
}

function providerMessage(raw: unknown): string | null {
  if (!raw || typeof raw !== 'object') return null;
  if ('message' in raw && typeof raw.message === 'string') return raw.message;
  if ('status' in raw && typeof raw.status === 'string') return raw.status;
  return null;
}

export async function fetchTwelveDataPrice(reference: string): Promise<TwelveDataPriceResult> {
  const apiKey = process.env.TWELVE_DATA_API_KEY;
  if (!apiKey) throw new Error('Missing required environment variable: TWELVE_DATA_API_KEY');

  const instrument = parseTwelveDataInstrumentRef(reference);
  const url = new URL('https://api.twelvedata.com/price');
  url.searchParams.set('symbol', instrument.symbol);
  if (instrument.exchange) url.searchParams.set('exchange', instrument.exchange);
  url.searchParams.set('apikey', apiKey);

  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);
  try {
    const response = await fetch(url, { signal: controller.signal, cache: 'no-store' });
    const raw: unknown = await response.json();
    const message = providerMessage(raw);

    if (!response.ok) {
      throw new Error(
        `Twelve Data HTTP ${response.status}${message ? `: ${message}` : ''}`,
      );
    }

    if (!raw || typeof raw !== 'object' || !('price' in raw)) {
      throw new Error(`Twelve Data provider error: ${message ?? 'invalid payload'}`);
    }

    const providerPrice = Number(raw.price);
    if (!Number.isFinite(providerPrice) || providerPrice <= 0) {
      throw new Error('Twelve Data returned an invalid price');
    }

    return { providerPrice, fetchedAt: new Date().toISOString(), raw };
  } catch (error) {
    if (error instanceof Error && error.name === 'AbortError') {
      throw new Error('Twelve Data request timed out');
    }
    throw error;
  } finally {
    clearTimeout(timeout);
  }
}
