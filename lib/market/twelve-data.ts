import 'server-only';
import { parseTwelveDataInstrumentRef } from '@/lib/market/twelve-data-symbol';

const REQUEST_TIMEOUT_MS = 6_000;
export interface TwelveDataPriceResult {
  providerPrice: number;
  fetchedAt: string;
  providerSymbol: string;
  providerCurrency: string | null;
  providerExchange: string | null;
  raw: unknown;
}
function objectPayload(raw: unknown): Record<string, unknown> {
  if (raw === null || typeof raw !== 'object' || Array.isArray(raw))
    throw new Error('TWELVE_DATA_INVALID_QUOTE_PAYLOAD');
  return raw as Record<string, unknown>;
}
export async function fetchTwelveDataPrice(reference: string): Promise<TwelveDataPriceResult> {
  const key = process.env.TWELVE_DATA_API_KEY;
  if (!key) throw new Error('TWELVE_DATA_NOT_CONFIGURED');
  const instrument = parseTwelveDataInstrumentRef(reference);
  const url = new URL('https://api.twelvedata.com/quote');
  url.searchParams.set('symbol',instrument.symbol);
  if (instrument.exchange) url.searchParams.set('exchange',instrument.exchange);
  url.searchParams.set('timezone','UTC');
  url.searchParams.set('apikey',key);
  const controller=new AbortController();
  const timeout=setTimeout(()=>controller.abort(),REQUEST_TIMEOUT_MS);
  try {
    const response=await fetch(url,{signal:controller.signal,cache:'no-store'});
    const raw: unknown=await response.json();
    const quote=objectPayload(raw);
    if (!response.ok || quote.status==='error' || quote.code!==undefined)
      throw new Error('TWELVE_DATA_QUOTE_REJECTED_'+String(quote.code??response.status));
    const price=Number(quote.close);
    if (!Number.isFinite(price) || price<=0)throw new Error('TWELVE_DATA_INVALID_CLOSE');
    const timestamp=typeof quote.timestamp==='number'?quote.timestamp:
      typeof quote.timestamp==='string' && /^\d+$/.test(quote.timestamp)?Number(quote.timestamp):NaN;
    if (!Number.isFinite(timestamp)||timestamp<=0)throw new Error('TWELVE_DATA_TIMESTAMP_MISSING');
    const observedAt=new Date(timestamp*1000).toISOString();
    return {
      providerPrice:price,fetchedAt:observedAt,
      providerSymbol:typeof quote.symbol==='string'?quote.symbol:'',
      providerCurrency:typeof quote.currency==='string'?quote.currency:null,
      providerExchange:typeof quote.exchange==='string'?quote.exchange:null,
      raw,
    };
  }catch(e) {
    if (e instanceof Error&&e.name==='AbortError')throw new Error('TWELVE_DATA_TIMEOUT');
    throw e;
  }finally{clearTimeout(timeout);}
}
