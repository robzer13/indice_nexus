import assert from 'node:assert/strict';
import test from 'node:test';
import { admitMarketQuote, requireUsListingMatch, isUnambiguousUSExchange, assertCanonicalMarketReference, assertTwelveDataVenue, isStoredQuoteCurrencyCompatible, selectLatestCompatibleStoredQuote, fetchAllStoredQuotePages, type CandidateMarketQuote, type ExpectedMarketInstrument } from '../lib/market/price-quality';
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

test('reference must match canonical ticker AND exchange before valuation',()=>{
  const valid=[
    ['ADYEN.AS','ADYEN','Euronext Amsterdam'],
    ['AI:EPA','AI','Euronext Paris'],['BEAN.SW','BEAN','SIX Swiss Exchange'],
    ['BN.TO','BN','Toronto Stock Exchange'],['D05.SI','D05','SGX Mainboard'],
    ['NOVO-B.CO','NOVO-B','Nasdaq Copenhagen'],['TOI.V','TOI','TSX Venture Exchange'],
    ['2330.TW','2330','Taiwan Stock Exchange'],['AUTO:LSE','AUTO','London Stock Exchange'],
    ['RAA:XETR','RAA','Xetra'],['MEDI:OSE','MEDI','Oslo Børs'],
    ['ISRG:NASDAQ','ISRG','Nasdaq'],['WSE:NASDAQ','WSE','Nasdaq'],
    ['V','V','NYSE'],['META','META','Nasdaq'],
  ];
  for(const [reference,ticker,exchange] of valid)
    assert.doesNotThrow(()=>assertCanonicalMarketReference({...eur,reference,ticker,exchange}));
  assert.throws(()=>assertCanonicalMarketReference({...eur,reference:'MSFT',ticker:'NVDA',exchange:'Nasdaq'}),/QUOTE_REFERENCE_SECURITY_MISMATCH/);
  assert.throws(()=>assertCanonicalMarketReference({...eur,reference:'AI:EPA',ticker:'AI',exchange:'Euronext Amsterdam'}),/QUOTE_REFERENCE_SECURITY_MISMATCH/);
  assert.throws(()=>assertCanonicalMarketReference({...eur,exchange:'Unverified Venue'}),/CANONICAL_EXCHANGE_UNMAPPED/);
});
test('secondary source must corroborate requested exchange',()=>{
  assert.doesNotThrow(()=>assertTwelveDataVenue('NASDAQ','NASDAQ'));
  assert.doesNotThrow(()=>assertTwelveDataVenue('EURONEXT','Euronext Paris'));
  assert.doesNotThrow(()=>assertTwelveDataVenue('XETR','Xetra'));
  assert.throws(()=>assertTwelveDataVenue('NASDAQ','NYSE'),/TWELVE_DATA_EXCHANGE_MISMATCH/);
  assert.throws(()=>assertTwelveDataVenue('LSE',null),/TWELVE_DATA_EXCHANGE_MISMATCH/);
});

test('NYSE, NYSE Arca and NYSE American Yahoo exchange codes must not cross-match',()=>{
  assert.doesNotThrow(()=>requireUsListingMatch('NYSE','NYQ'));
  assert.doesNotThrow(()=>requireUsListingMatch('NYSE','NYE'));
  assert.throws(()=>requireUsListingMatch('NYSE','ASE'),/QUOTE_EXCHANGE_MISMATCH/);
  assert.throws(()=>requireUsListingMatch('NYSE','PCX'),/QUOTE_EXCHANGE_MISMATCH/);
  assert.doesNotThrow(()=>requireUsListingMatch('NYSE Arca','PCX'));
  assert.throws(()=>requireUsListingMatch('NYSE Arca','NYQ'),/QUOTE_EXCHANGE_MISMATCH/);
  assert.doesNotThrow(()=>requireUsListingMatch('NYSE American','ASE'));
  assert.throws(()=>requireUsListingMatch('NYSE American','NYQ'),/QUOTE_EXCHANGE_MISMATCH/);
});
test('secondary quote on bare US ticker must match canonical venue exactly',()=>{
  assert.doesNotThrow(()=>assertTwelveDataVenue('Nasdaq','NASDAQ'));
  assert.throws(()=>assertTwelveDataVenue('Nasdaq','NYSE'),/TWELVE_DATA_EXCHANGE_MISMATCH/);
  assert.doesNotThrow(()=>assertTwelveDataVenue('NYSE','NYSE'));
  assert.throws(()=>assertTwelveDataVenue('NYSE','NASDAQ'),/TWELVE_DATA_EXCHANGE_MISMATCH/);
  assert.throws(()=>assertTwelveDataVenue('NYSE Arca','NYSE'),/TWELVE_DATA_EXCHANGE_MISMATCH/);
});

test('BCG quote currency is EUR-major since 2026-09-02; historical guard uses canonical currency',()=>{
  // Official issuer trading-currency notice: https://balticclassifieds.com/change-in-trading-currency
  // Nominal share denomination GBP0.01 is distinct from exchange quote currency.
  assert.equal(isStoredQuoteCurrencyCompatible({symbol:'BCG.L',currency:'EUR'},'EUR','MAJOR'),true);
  assert.equal(isStoredQuoteCurrencyCompatible({symbol:'BCG.L',currency:'GBP'},'EUR','MAJOR'),false);
  assert.equal(isStoredQuoteCurrencyCompatible({symbol:'BCG.L',currency:'GBp'},'EUR','MAJOR'),false);
  // Legacy companies.currency=GBP is stale; this test does NOT certify EUR prices as GBP.
  assert.equal(isStoredQuoteCurrencyCompatible({symbol:'BCG.L',currency:'EUR'},'GBP','MAJOR'),false);
  assert.equal(isStoredQuoteCurrencyCompatible({currency:'GBp'},'GBP','MAJOR'),false);
  assert.equal(isStoredQuoteCurrencyCompatible({currency:'GBp'},'GBP','MINOR'),true);
  assert.equal(isStoredQuoteCurrencyCompatible({currency:'GBX'},'GBP','MINOR'),true);
  assert.equal(isStoredQuoteCurrencyCompatible({currency:123},'EUR','MAJOR'),false);
  // Unattested pre-provenance rows remain observable, not certified as currency-consistent.
  assert.equal(isStoredQuoteCurrencyCompatible(null,'EUR','MAJOR'),true);
  assert.equal(isStoredQuoteCurrencyCompatible({},'EUR','MAJOR'),true);
});

test('latest compatible quote skips an incompatible newest row instead of falling back to a stale reference',()=>{
  const rows = [
    {price:3.20,raw:{currency:'GBP'}},
    {price:3.10,raw:{currency:'EUR'}},
    {price:3.00,raw:{currency:'EUR'}},
  ]; // Newest first, as ordered by market_prices.as_of DESC.
  assert.equal(selectLatestCompatibleStoredQuote(rows,'EUR','MAJOR'),rows[1]);
  assert.equal(selectLatestCompatibleStoredQuote(rows,'GBP','MAJOR'),rows[0]);
  assert.equal(selectLatestCompatibleStoredQuote(rows,'USD','MAJOR'),null);
  assert.equal(selectLatestCompatibleStoredQuote([], 'EUR','MAJOR'),null);
});

test('legacy rows without attested raw currency remain eligible after contradictory newer quotes',()=>{
  const rows = [
    {price:3.20,raw:{currency:'GBP'}},
    {price:2.70,raw:null},
  ];
  assert.equal(selectLatestCompatibleStoredQuote(rows,'EUR','MAJOR'),rows[1]);
  assert.equal(selectLatestCompatibleStoredQuote([{price:3.20,raw:{currency:'GBP'}}],'EUR','MAJOR'),null);
});

test('quote pagination passes response caps and finds a compatible older quote',async()=>{
  const rows = [
    {price:9,raw:{currency:'GBP'}},
    {price:8,raw:{currency:'GBP'}},
    {price:7,raw:{currency:'GBP'}},
    {price:6,raw:{currency:'GBP'}},
    {price:5,raw:{currency:'GBP'}},
    {price:4,raw:{currency:'GBP'}},
    {price:3,raw:{currency:'EUR'}},
  ]; // Server ordering is newest first.
  const requested: number[][] = [];
  const found = await fetchAllStoredQuotePages(async(start,end)=>{
    requested.push([start,end]);
    // Simulate PostgREST silently capping to TWO rows when THREE were requested.
    return rows.slice(start,Math.min(end+1,start+2));
  },3);
  assert.deepEqual(found,rows);
  assert.deepEqual(requested,[[0,2],[2,4],[4,6],[6,8],[7,9]]);
  assert.equal(selectLatestCompatibleStoredQuote(found,'EUR','MAJOR'),rows[6]);
});

test('quote pagination handles empty results and fails closed on page errors',async()=>{
  assert.deepEqual(await fetchAllStoredQuotePages(async()=>[],3),[]);
  let requests = 0;
  await assert.rejects(fetchAllStoredQuotePages(async()=>{
    requests++;
    if(requests===1) return [{price:1,raw:{currency:'GBP'}}];
    throw new Error('MARKET_PAGE_FAILURE');
  },2),/MARKET_PAGE_FAILURE/);
  assert.equal(requests,2);
  await assert.rejects(fetchAllStoredQuotePages(async()=>[],0),/STORED_QUOTE_PAGE_SIZE_INVALID/);
});
