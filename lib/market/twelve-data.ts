import 'server-only';

const REQUEST_TIMEOUT_MS = 8_000;

export interface TwelveDataPriceResult {
  providerPrice: number;
  fetchedAt: string;
  raw: unknown;
}

export async function fetchTwelveDataPrice(symbol: string): Promise<TwelveDataPriceResult> {
  const apiKey = process.env.TWELVE_DATA_API_KEY;
  if (!apiKey) throw new Error('Missing required environment variable: TWELVE_DATA_API_KEY');

  const url = new URL('https://api.twelvedata.com/price');
  url.searchParams.set('symbol', symbol);
  url.searchParams.set('apikey', apiKey);

  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);
  try {
    const response = await fetch(url, { signal: controller.signal, cache: 'no-store' });
    const raw: unknown = await response.json();
    if (!response.ok) throw new Error(`Twelve Data HTTP ${response.status}`);
    if (!raw || typeof raw !== 'object' || !('price' in raw)) {
      const message = raw && typeof raw === 'object' && 'message' in raw ? String(raw.message) : 'invalid payload';
      throw new Error(`Twelve Data provider error: ${message}`);
    }
    const providerPrice = Number(raw.price);
    if (!Number.isFinite(providerPrice) || providerPrice <= 0) {
      throw new Error('Twelve Data returned an invalid price');
    }
    return { providerPrice, fetchedAt: new Date().toISOString(), raw };
  } catch (error) {
    if (error instanceof Error && error.name === 'AbortError') throw new Error('Twelve Data request timed out');
    throw error;
  } finally {
    clearTimeout(timeout);
  }
}
