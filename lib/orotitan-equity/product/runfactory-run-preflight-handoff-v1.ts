/**
 * Run Factory P0-B: URL handoff from P0-D's persisted identity receipt.
 * URL fields are untrusted convenience inputs only. Never authorize a run,
 * assign methodology, invent IDs or imply original contract bytes verified.
 */
import {assertIsin} from './runfactory-identity-preflight-v1';

type QueryField=string|string[]|undefined;
export type RunPreflightQuery=Readonly<Record<string,QueryField>>;
export type RunPreflightSeed=Readonly<{
 isin:string;exchange:string;dataCutoff:string;
 profileId:'';
}>;
export function parseUntrustedRunPreflightQuery(query:RunPreflightQuery):RunPreflightSeed{
 let isin='';
 if(typeof query.isin==='string'){
  try{isin=assertIsin(query.isin);}catch{isin='';}
 }
 const rawExchange=typeof query.exchange==='string'?
  query.exchange.trim().toUpperCase():'';
 const exchange=/^[A-Z0-9_.-]{2,30}$/.test(rawExchange)?rawExchange:'';
 const rawCutoff=typeof query.dataCutoff==='string'?query.dataCutoff:'';
 let dataCutoff='';
 if(/^\d{4}-\d{2}-\d{2}$/.test(rawCutoff)){
  const d=new Date(rawCutoff+'T00:00:00.000Z');
  if(Number.isFinite(d.getTime())&&d.toISOString().slice(0,10)===rawCutoff)
   dataCutoff=rawCutoff;
 }
 return {isin,exchange,dataCutoff,profileId:''};
}
export function buildUntrustedRunPreflightHandoff(input:{
 isin:string;exchange:string;dataCutoff?:string|null;
}):string{
 const seed=parseUntrustedRunPreflightQuery({
  isin:input.isin,exchange:input.exchange,dataCutoff:input.dataCutoff??'',
 });
 const params=new URLSearchParams();
 if(seed.isin)params.set('isin',seed.isin);
 if(seed.exchange)params.set('exchange',seed.exchange);
 if(seed.dataCutoff)params.set('dataCutoff',seed.dataCutoff);
 return '/admin/research-factory/run-preflight?'+params.toString();
}
