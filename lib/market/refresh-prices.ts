import 'server-only';
import { getMarketDataCompanies,insertMarketPrice,recordMarketSyncRun } from '@/lib/data/market-prices';
import { fetchProviderPrice,type MarketPriceProvider } from '@/lib/market/provider';

export interface PriceRefreshItem {
  slug:string;symbol:string;status:'inserted'|'failed';
  provider?:MarketPriceProvider;price?:number;error?:string;
}
export interface PriceRefreshResult {
  ok:boolean;triggerSource:'CRON'|'ADMIN';startedAt:string;finishedAt:string;
  companies:number;inserted:number;failed:number;results:PriceRefreshItem[];
  logRecorded:boolean;logError?:string;
}
const MAX_PARALLEL_REQUESTS=4;
export async function refreshMarketPrices(triggerSource:'CRON'|'ADMIN'):Promise<PriceRefreshResult> {
  const startedAt=new Date().toISOString();
  const companies=await getMarketDataCompanies();
  const results:PriceRefreshItem[]=new Array(companies.length);
  let next=0;
  // Bounded concurrency preserves stable, per-issuer audit results and reduces
  // cron wall time without firing 39 simultaneous external requests.
  const worker=async()=>{
    while(true){
      const idx=next++;
      if(idx>=companies.length)return;
      const c=companies[idx];
      try{
        if(!c.market_data_symbol)throw new Error('MISSING_MARKET_DATA_SYMBOL');
        const price=await fetchProviderPrice({
          reference:c.market_data_symbol,ticker:c.ticker,exchange:c.exchange,
          currency:c.currency,quoteUnit:c.quote_unit,
          multiplier:c.market_data_multiplier,previousPrice:c.latest_price,
        });
        await insertMarketPrice({
          companyId:c.id,price:price.normalizedPrice,
          asOf:price.fetchedAt,source:price.provider,raw:price.raw,
        });
        results[idx]={slug:c.slug,symbol:c.market_data_symbol,
          status:'inserted',provider:price.provider,price:price.normalizedPrice};
      }catch(e){
        results[idx]={slug:c.slug,symbol:c.market_data_symbol,
          status:'failed',error:e instanceof Error?e.message:'UNKNOWN_ERROR'};
      }
    }
  };
  await Promise.all(Array.from({length:Math.min(companies.length,MAX_PARALLEL_REQUESTS)},()=>worker()));
  const finishedAt=new Date().toISOString();
  const inserted=results.filter(r=>r.status==='inserted').length;
  const failed=results.length-inserted;
  let logRecorded=false,logError:string|undefined;
  try{await recordMarketSyncRun({
    startedAt,finishedAt,triggerSource,companies:results.length,inserted,failed,results,
  });logRecorded=true;}catch(e){logError=e instanceof Error?e.message:'SYNC_AUDIT_FAILED';}
  return {ok:failed===0,triggerSource,startedAt,finishedAt,companies:results.length,
    inserted,failed,results,logRecorded,...(logError?{logError}:{})};
}
