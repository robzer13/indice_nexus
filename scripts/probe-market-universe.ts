/* Read-only nonproduction Yahoo coverage probe, no credentials or database writes. */
import fs from 'node:fs';
import YahooFinance from 'yahoo-finance2';
import {toYahooFinanceSymbol} from '../lib/market/yahoo-symbol';
import {admitMarketQuote,requireUsListingMatch,
  type CandidateMarketQuote,type ExpectedMarketInstrument} from '../lib/market/price-quality';

type Instrument={ticker:string;exchange:string;currency:string;quote_unit:'MAJOR'|'MINOR';market_data_symbol:string};
const instruments=JSON.parse(fs.readFileSync('tests/fixtures/market-universe-20261011.json','utf8')) as Instrument[];
const finance=new YahooFinance();
function iso(v:unknown):string{
 if(v instanceof Date&&!Number.isNaN(v.getTime()))return v.toISOString();
 if(typeof v==='number'&&Number.isFinite(v)){const t=new Date(v>1e10?v:v*1000);if(!Number.isNaN(t.getTime()))return t.toISOString();}
 if(typeof v==='string'&&!Number.isNaN(Date.parse(v)))return new Date(v).toISOString();
 return 'UNVERIFIED_TIMESTAMP';
}
async function main(){
 let next=0;
 const results:Array<{symbol:string;status:string;detail:string}>=new Array(instruments.length);
 const worker=async()=>{
  while(true){
   const i=next++;if(i>=instruments.length)return;
   const c=instruments[i];const symbol=toYahooFinanceSymbol(c.market_data_symbol);
   try{
    const q=await finance.quote(symbol) as {symbol?:string;currency?:string;exchange?:string;regularMarketPrice?:number;regularMarketTime?:unknown};
    const candidate:CandidateMarketQuote={provider:'YAHOO_FINANCE',providerSymbol:q?.symbol??'',providerCurrency:q?.currency??null,providerExchange:q?.exchange??null,providerPrice:Number(q?.regularMarketPrice),observedAt:iso(q?.regularMarketTime),raw:q};
    if(!c.market_data_symbol.includes('.')&&!c.market_data_symbol.includes(':'))requireUsListingMatch(c.exchange,candidate.providerExchange);
    if(c.market_data_symbol.toUpperCase().endsWith(':NASDAQ')||c.market_data_symbol.toUpperCase().endsWith(':NYSE'))requireUsListingMatch(c.market_data_symbol.split(':').at(-1)!,candidate.providerExchange);
    const expected:ExpectedMarketInstrument={reference:c.market_data_symbol,ticker:c.ticker,exchange:c.exchange,currency:c.currency,quoteUnit:c.quote_unit,multiplier:1,previousPrice:null};
    admitMarketQuote(expected,candidate,symbol);
    results[i]={symbol,status:'ACCEPTED',detail:candidate.observedAt};
   }catch(e){results[i]={symbol,status:'REJECTED',detail:e instanceof Error?e.message.slice(0,140):'UNKNOWN'};}
  }
 };
 await Promise.all([worker(),worker()]);
 const accepted=results.filter(r=>r.status==='ACCEPTED').length;
 const rejected=results.length-accepted;
 console.log('MARKET_PROBE_RESULTS='+JSON.stringify({checked:results.length,accepted,rejected,results}));
 if(process.env.GITHUB_STEP_SUMMARY){
  const lines=['# Nonproduction 39-source coverage','Admitted: '+accepted+' / '+results.length,'No credentials / DB writes. GitHub egress may differ from Vercel.','| Yahoo instrument | Status | Reason/time |','|---|---|---|'];
  for(const r of results)lines.push('| '+r.symbol+' | '+r.status+' | '+r.detail.replace(/[|]/g,'/')+' |');
  fs.appendFileSync(process.env.GITHUB_STEP_SUMMARY,lines.join('\n')+'\n');
 }
}
void main().catch(e=>{console.error(e);process.exitCode=1;});
