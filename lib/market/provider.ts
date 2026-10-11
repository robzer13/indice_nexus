import 'server-only';
import { fetchTwelveDataPrice } from '@/lib/market/twelve-data';
import { fetchYahooFinancePrice } from '@/lib/market/yahoo-finance';
import { toYahooFinanceSymbol } from '@/lib/market/yahoo-symbol';
import { parseTwelveDataInstrumentRef } from '@/lib/market/twelve-data-symbol';
import {
  admitMarketQuote,isUnambiguousUSExchange,requireUsListingMatch,
  type CandidateMarketQuote,type ExpectedMarketInstrument,type MarketFeedName,
} from '@/lib/market/price-quality';
export type MarketPriceProvider = MarketFeedName;
export interface ProviderPriceResult {
  provider: MarketPriceProvider;
  providerPrice: number;
  normalizedPrice: number;
  fetchedAt: string;
  raw: unknown;
}

/**
 * Free-tier chain: Yahoo first for ALL canonical symbols, including those
 * without ':'. Twelve Data /quote is a separately verified fallback only
 * when a user-provisioned FREE API key exists. No timestamp synthesis.
 */
export async function fetchProviderPrice(instrument: ExpectedMarketInstrument): Promise<ProviderPriceResult> {
  const failures: string[]=[];
  const ref=instrument.reference;
  try {
    const yahooSymbol=toYahooFinanceSymbol(ref);
    // A bare, suffix-less ticker must refer to a US venue; otherwise it could
    // silently resolve to an unrelated US company with the same ticker.
    if (!ref.includes(':') && !ref.includes('.') && !isUnambiguousUSExchange(instrument.exchange))
      throw new Error('BARE_SYMBOL_NON_US_LISTING');
    const fetched=await fetchYahooFinancePrice(ref);
    const quote: CandidateMarketQuote={
      provider:'YAHOO_FINANCE',providerSymbol:fetched.providerSymbol,
      providerCurrency:fetched.providerCurrency,providerExchange:fetched.providerExchange,
      providerPrice:fetched.providerPrice,observedAt:fetched.fetchedAt,raw:fetched.raw,
    };
    if (!ref.includes(':') && !ref.includes('.'))requireUsListingMatch(instrument.exchange,quote.providerExchange);
    if (ref.toUpperCase().endsWith(':NASDAQ')||ref.toUpperCase().endsWith(':NYSE'))
      requireUsListingMatch(ref.split(':').at(-1)!,quote.providerExchange);
    const normalizedPrice=admitMarketQuote(instrument,quote,yahooSymbol);
    return {provider:'YAHOO_FINANCE',providerPrice:quote.providerPrice,
      normalizedPrice,fetchedAt:quote.observedAt,raw:quote.raw};
  } catch(e){failures.push('YAHOO_FINANCE: '+(e instanceof Error?e.message:'UNKNOWN_ERROR'));}
  if (process.env.TWELVE_DATA_API_KEY) {
    try{
      const requested=parseTwelveDataInstrumentRef(ref);
      // Dotted Yahoo ticker aliases are not verified Twelve Data identities.
      // Do not strip the suffix or guess an exchange.
      if(ref.includes('.') && !ref.includes(':'))throw new Error('DOTTED_SYMBOL_UNVERIFIED_ON_TWELVE_DATA');
      const fetched=await fetchTwelveDataPrice(ref);
      const quote: CandidateMarketQuote={
        provider:'TWELVE_DATA',providerSymbol:fetched.providerSymbol,
        providerCurrency:fetched.providerCurrency,providerExchange:fetched.providerExchange,
        providerPrice:fetched.providerPrice,observedAt:fetched.fetchedAt,raw:fetched.raw,
      };
      // The payload exchange must be present for cross-listed instruments.
      if(requested.exchange && !quote.providerExchange)
        throw new Error('TWELVE_DATA_EXCHANGE_MISSING');
      const normalizedPrice=admitMarketQuote(instrument,quote,requested.symbol);
      return {provider:'TWELVE_DATA',providerPrice:quote.providerPrice,
        normalizedPrice,fetchedAt:quote.observedAt,raw:quote.raw};
    }catch(e){failures.push('TWELVE_DATA: '+(e instanceof Error?e.message:'UNKNOWN_ERROR'));}
  }else failures.push('TWELVE_DATA: OPTIONAL_FREE_KEY_NOT_CONFIGURED');
  throw new Error('NO_ADMISSIBLE_MARKET_QUOTE | '+failures.join(' | '));
}
