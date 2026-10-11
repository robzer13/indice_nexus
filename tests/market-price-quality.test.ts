import assert from 'node:assert/strict';
import test from 'node:test';
import { admitMarketQuote, requireUsListingMatch, isUnambiguousUSExchange, type CandidateMarketQuote, type ExpectedMarketInstrument } from '../lib/market/price-quality';
const now=new Date('2026-10-11T18:00:00Z');
const eur:ExpectedMarketInstrument={
  reference:'ADYEN.AS',ticker:'ADYEN',exchange:'Euronext Amsterdam',
  currency:'EUR',quoteUnit:'MAJOR',multiplier:1,previousPrice:1500,
};
const base:CandidateMarketQuote={
  provider:'YAHOO_FINANCE',providerSymbol:'ADYEN.AS',
  providerCurrency:'EUR',providerExchange:'AMS',
  providerPrice:1475,observedAt:'2026-10-09T16:30:00Z',raw:{},
};
test('admits exchange-identified European EUR quote with real trade timestamp',()=>{
  assert.equal(admitMarketQuote(eur,base,'ADYEN.AS',now),1475);
});
test('rejects missing currency and wrong listing even when price matches',()=>{
  assert.throws(()=>admitMarketQuote(eur,{...base,providerCurrency:null},'ADYEN.AS',now),/QUOTE_CURRENCY_MISSING/);
  assert.throws(()=>admitMarketQuote(eur,{...base,providerCurrency:'USD'},'ADYEN.AS',now),/QUOTE_CURRENCY_UNIT_MISMATCH/);
  assert.throws(()=>admitMarketQuote(eur,{...base,providerSymbol:'ADYEN.PA'},'ADYEN.AS',now),/QUOTE_SYMBOL_MISMATCH/);
});
test('accepts genuine GBP minor quotes only for a MINOR listing',()=>{
  const gbp={...eur,currency:'GBP',quoteUnit:'MINOR' as const};
  assert.equal(admitMarketQuote(gbp,{...base,providerCurrency:'GBp'},'ADYEN.AS',now),1475);
  assert.throws(()=>admitMarketQuote(gbp,{...base,providerCurrency:'GBP'},'ADYEN.AS',now),/QUOTE_CURRENCY_UNIT_MISMATCH/);
  assert.throws(()=>admitMarketQuote({...gbp,quoteUnit:'MAJOR'}, {...base,providerCurrency:'GBp'},'ADYEN.AS',now),/QUOTE_CURRENCY_UNIT_MISMATCH/);
});
test('invalid timestamps, future timestamps, and prices fail closed',()=>{
  assert.throws(()=>admitMarketQuote(eur,{...base,observedAt:'2026-10-20T12:00:00Z'},'ADYEN.AS',now),/QUOTE_TIMESTAMP_OUT_OF_WINDOW/);
  assert.throws(()=>admitMarketQuote(eur,{...base,observedAt:'2026-10-01T12:00:00Z'},'ADYEN.AS',now),/QUOTE_TIMESTAMP_OUT_OF_WINDOW/);
  assert.throws(()=>admitMarketQuote(eur,{...base,observedAt:'invalid'},'ADYEN.AS',now),/QUOTE_TIMESTAMP_OUT_OF_WINDOW/);
  assert.throws(()=>admitMarketQuote(eur,{...base,providerPrice:0},'ADYEN.AS',now),/QUOTE_NON_POSITIVE/);
});
test('conservative extreme-price guard enforces existing 5x / 0.2x bounds',()=>{
  assert.throws(()=>admitMarketQuote(eur,{...base,providerPrice:9000},'ADYEN.AS',now),/QUOTE_EXTREME_PRICE_CHANGE/);
  assert.throws(()=>admitMarketQuote(eur,{...base,providerPrice:1},'ADYEN.AS',now),/QUOTE_EXTREME_PRICE_CHANGE/);
});
test('bare tickers may not silently switch listing exchange',()=>{
  assert.equal(isUnambiguousUSExchange('Nasdaq'),true);
  assert.equal(isUnambiguousUSExchange('Euronext Amsterdam'),false);
  assert.doesNotThrow(()=>requireUsListingMatch('Nasdaq','NMS'));
  assert.throws(()=>requireUsListingMatch('Nasdaq','NYQ'),/QUOTE_EXCHANGE_MISMATCH/);
  assert.doesNotThrow(()=>requireUsListingMatch('NYSE','NYQ'));
  assert.throws(()=>requireUsListingMatch('NYSE',null),/QUOTE_EXCHANGE_MISMATCH/);
});
