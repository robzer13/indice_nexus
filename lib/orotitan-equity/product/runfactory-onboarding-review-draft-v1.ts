/**
 * Run Factory — generic READ-ONLY review draft transport.
 *
 * Permits a research agent to supply an issuer's candidate fields in a URL
 * without creating code or a database fixture for each issuer.
 * This transport grants NO authority: requires fresh P0-C registry scan,
 * reviewed official sources, admin session, checkbox and exact typed GO.
 */
import {assertIsin} from './runfactory-identity-preflight-v1';

export type P0DReviewDraft=Readonly<{
 isin:string;ticker:string;exchange:string;displayName:string;legalName:string;
 issuerCountry:string;reportingCurrency:string;tradingCurrency:string;
 primaryListing:boolean;quoteUnit:'MAJOR'|'MINOR';priceDecimals:number;
 issuerSourceUrl:string;listingSourceUrl:string;
}>;
type Param=string|string[]|undefined;
export type DraftQuery=Readonly<Record<string,Param>>;

const scalar=(value:Param):string|null=>
 typeof value==='string'&&value.length<=2048?value:null;
function source(v:Param):string{
 const raw=scalar(v);
 if(!raw)return '';
 try{
  const u=new URL(raw);
  if(u.protocol!=='https:'||u.username||u.password||u.port||
   !/^[a-z0-9-]+(?:\.[a-z0-9-]+)+$/i.test(u.hostname)||
   u.hostname.endsWith('.local')||
   /^\\d+(?:\\.\\d+){3}$/.test(u.hostname)||u.hash||u.pathname==='/')
   return '';
  return u.toString();
 }catch{return '';}
}
function textParam(v:Param,max:number):string{
 const value=scalar(v)?.trim()??'';
 return value.length<=max&&!/[\u0000-\u001f\u007f]/.test(value)?
  value:'';
}
export function parseUntrustedOnboardingDraft(query:DraftQuery):
 P0DReviewDraft|null{
 const rawIsin=scalar(query.isin);
 if(!rawIsin)return null;
 let isin:string;
 try{isin=assertIsin(rawIsin);}catch{return null;}
 const ticker=textParam(query.ticker,30).toUpperCase();
 const exchange=textParam(query.exchange,30).toUpperCase();
 if(!/^[A-Z0-9._-]{1,30}$/.test(ticker)||
    !/^[A-Z0-9._-]{2,30}$/.test(exchange))return null;
 const issuerCountry=textParam(query.issuerCountry,2).toUpperCase();
 const reportingCurrency=textParam(query.reportingCurrency,3).toUpperCase();
 const tradingCurrency=textParam(query.tradingCurrency,3).toUpperCase();
 const quoteUnit=query.quoteUnit==='MINOR'?'MINOR':'MAJOR';
 const priceDecimals=scalar(query.priceDecimals);
 const decimals=priceDecimals!==null&&/^[0-6]$/.test(priceDecimals)?
  Number(priceDecimals):2;
 return {
  isin,ticker,exchange,
  displayName:textParam(query.displayName,180),
  legalName:textParam(query.legalName,180),
  issuerCountry:/^[A-Z]{2}$/.test(issuerCountry)?issuerCountry:'',
  reportingCurrency:/^[A-Z]{3}$/.test(reportingCurrency)?reportingCurrency:'',
  tradingCurrency:/^[A-Z]{3}$/.test(tradingCurrency)?tradingCurrency:'',
  primaryListing:query.primaryListing!=='false',
  quoteUnit,priceDecimals:decimals,
  issuerSourceUrl:source(query.issuerSourceUrl),
  listingSourceUrl:source(query.listingSourceUrl),
 };
}
export function buildUntrustedReviewDraftQuery(draft:P0DReviewDraft):string{
 const q=new URLSearchParams();
 for(const [key,value] of Object.entries(draft)){
  q.set(key,String(value));
 }
 return q.toString();
}
