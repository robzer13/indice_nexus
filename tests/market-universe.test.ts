import assert from 'node:assert/strict';
import test from 'node:test';
import fs from 'node:fs';
import {assertCanonicalMarketReference} from '../lib/market/price-quality';
import {toYahooFinanceSymbol} from '../lib/market/yahoo-symbol';
type Instrument={ticker:string;exchange:string;currency:string;quote_unit:'MAJOR'|'MINOR';market_data_symbol:string};
const rows=JSON.parse(fs.readFileSync('tests/fixtures/market-universe-20261011.json','utf8')) as Instrument[];
test('all 39 current canonical listings map to unambiguous Yahoo references',()=>{
 assert.equal(rows.length,39);
 const unique=new Set<string>();
 for(const r of rows){
  assertCanonicalMarketReference({reference:r.market_data_symbol,ticker:r.ticker,exchange:r.exchange,currency:r.currency,quoteUnit:r.quote_unit,multiplier:1,previousPrice:null});
  const symbol=toYahooFinanceSymbol(r.market_data_symbol);
  assert.ok(/^[A-Z0-9.^-]+$/i.test(symbol),'Invalid symbol: '+symbol);
  assert.ok(!unique.has(symbol),'Duplicate Yahoo instrument: '+symbol);
  unique.add(symbol);
 }
});
