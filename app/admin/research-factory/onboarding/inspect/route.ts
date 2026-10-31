import {NextResponse,type NextRequest} from 'next/server';
import {requireAdminSession} from '@/lib/auth/admin-session';
import {isHostedShadowResearchEnabled} from
 '@/lib/orotitan-equity/product/hosted-shadow-research-gate';
import {getHostedShadowRecoveryClient} from
 '@/lib/orotitan-equity/product/research-inbox-pilot-server';
import {assertIsin} from
 '@/lib/orotitan-equity/product/runfactory-identity-preflight-v1';
import {
 planCanonicalIdentityOnboarding,type P0CInventory,type P0CIssuer,type P0CSecurity,
} from '@/lib/orotitan-equity/product/runfactory-canonical-onboarding-candidate-v1';

export const runtime='nodejs';
export const dynamic='force-dynamic';
export const maxDuration=30;
const headers={'Cache-Control':'private, no-store'};
const MAX_ROWS=100;
const UUID=/^[0-9a-f]{8}-(?:[0-9a-f]{4}-){3}[0-9a-f]{12}$/i;
const fail=():never=>{throw Error('IDENTITY_REGISTRY_READ_FAILED');};
function asString(v:unknown):string{
 if(typeof v!=='string')throw Error('IDENTITY_REGISTRY_READ_FAILED');
 return v;
}
function asUuid(v:unknown):string{
 const s=asString(v);if(!UUID.test(s))fail();return s;
}
function rows(result:{data:unknown,error:unknown}):Record<string,unknown>[]{
 if(result.error||!Array.isArray(result.data)||result.data.length>=MAX_ROWS)
  throw Error('IDENTITY_REGISTRY_READ_FAILED');
 if(result.data.some((v:unknown)=>v===null||typeof v!=='object'||Array.isArray(v)))
  throw Error('IDENTITY_REGISTRY_READ_FAILED');
 return result.data as Record<string,unknown>[];
}
function uniqueBy<T>(values:T[],key:(v:T)=>string):T[]{
 const map=new Map<string,T>();
 for(const value of values){
  const k=key(value);
  if(map.has(k)){
   if(JSON.stringify(map.get(k))!==JSON.stringify(value))
    throw Error('IDENTITY_REGISTRY_READ_FAILED');
  }else map.set(k,value);
 }
 return [...map.values()];
}
export async function POST(req:NextRequest){
 if(!isHostedShadowResearchEnabled(process.env))
  return NextResponse.json({ok:false,error:'SHADOW_ONLY'},{status:404,headers});
 try{await requireAdminSession();}
 catch{return NextResponse.json({ok:false,error:'UNAUTHORIZED'},{status:401,headers});}
 if(req.headers.get('origin')!==new URL(req.url).origin)
  return NextResponse.json({ok:false,error:'ORIGIN_MISMATCH'},{status:403,headers});
 const length=Number(req.headers.get('content-length')??'NaN');
 if(req.headers.get('content-type')?.split(';')[0]?.trim()!=='application/json'||
    !Number.isSafeInteger(length)||length<5||length>1500)
  return NextResponse.json({ok:false,error:'REQUEST_INVALID'},{status:400,headers});
 try{
  const payload:unknown=await req.json();
  if(!payload||typeof payload!=='object'||Array.isArray(payload))fail();
  const input=payload as Record<string,unknown>;
  if(typeof input.ticker!=='string'||typeof input.exchange!=='string'||
     typeof input.isin!=='string')throw Error('IDENTITY_INPUT_INVALID');
  const isin=assertIsin(input.isin);
  const ticker=input.ticker.trim().toUpperCase();
  const exchange=input.exchange.trim().toUpperCase();
  if(!/^[A-Z0-9._-]{1,30}$/.test(ticker)||
     !/^[A-Z0-9._-]{2,30}$/.test(exchange))
    throw Error('IDENTITY_LISTING_INPUT_INVALID');
  if(input.issuerDisplayName!==null&&input.issuerDisplayName!==undefined&&
     (typeof input.issuerDisplayName!=='string'||
      input.issuerDisplayName.length>180||
      !/^[\p{L}\p{N} .,&'’()+\/-]+$/u.test(input.issuerDisplayName)))
    throw Error('IDENTITY_NAME_INPUT_INVALID');
  const displayName=(input.issuerDisplayName as string|null|undefined)?.trim()||null;
  if(input.evidenceUrl!==null&&input.evidenceUrl!==undefined&&
     typeof input.evidenceUrl!=='string')throw Error('INVALID_EVIDENCE_URL');
  const started=Date.now();
  const client=getHostedShadowRecoveryClient();
  // Two distinct predicates discover existing identities even when ISIN
  // conflicts with exchange/ticker. Legacy is ONLY a candidate source.
  const [isinResult,listingResult,legacyResult]=await Promise.all([
   client.from('securities')
    .select('security_id,issuer_id,isin,ticker,exchange,trading_currency,primary_listing')
    .eq('isin',isin).limit(MAX_ROWS),
   client.from('securities')
    .select('security_id,issuer_id,isin,ticker,exchange,trading_currency,primary_listing')
    .ilike('ticker',ticker).ilike('exchange',exchange).limit(MAX_ROWS),
   client.from('companies')
    .select('id,name,ticker,exchange,currency,active,country')
    .ilike('ticker',ticker).ilike('exchange',exchange).limit(MAX_ROWS),
  ]);
  const securities:P0CSecurity[]=uniqueBy(
   [...rows(isinResult),...rows(listingResult)].map(r=>({
    securityId:asUuid(r.security_id),issuerId:asUuid(r.issuer_id),
    isin:asString(r.isin),ticker:asString(r.ticker),
    exchange:asString(r.exchange),tradingCurrency:asString(r.trading_currency),
    primaryListing:r.primary_listing===null?null:r.primary_listing===true,
   })),r=>r.securityId);
  const companies=rows(legacyResult).map(r=>({
   id:asUuid(r.id),name:asString(r.name),ticker:asString(r.ticker),
   exchange:asString(r.exchange),currency:asString(r.currency),
   active:r.active===true,
   country:r.country===null?null:asString(r.country),
  }));
  const names=new Set([displayName,...companies.map(c=>c.name)].filter(
   (x):x is string=>typeof x==='string'&&x.length>0&&x.length<=180&&
   /^[\p{L}\p{N} .,&'’()+\/-]+$/u.test(x)));
  const baseIssuerIds=[...new Set(securities.map(s=>s.issuerId))];
  const issuerQueries=[
   ...(baseIssuerIds.length?[
    client.from('issuers').select('issuer_id,display_name,legal_name')
     .in('issuer_id',baseIssuerIds).limit(MAX_ROWS),
   ]:[]),
   ...[...names].flatMap(name=>[
    client.from('issuers').select('issuer_id,display_name,legal_name')
     .ilike('display_name',name).limit(MAX_ROWS),
    client.from('issuers').select('issuer_id,display_name,legal_name')
     .ilike('legal_name',name).limit(MAX_ROWS),
   ]),
  ];
  const issuerResults=await Promise.all(issuerQueries);
  const issuers:P0CIssuer[]=uniqueBy(
   issuerResults.flatMap(rows).map(r=>({
    issuerId:asUuid(r.issuer_id),
    displayName:asString(r.display_name),
    legalName:r.legal_name===null?null:asString(r.legal_name),
   })),r=>r.issuerId);
  const issuerIds=[...new Set(issuers.map(i=>i.issuerId))];
  const companyIds=[...new Set(companies.map(c=>c.id))];
  const [dossierResult,mapResult]=await Promise.all([
   issuerIds.length?
    client.from('research_dossiers').select('dossier_id,issuer_id,active')
     .in('issuer_id',issuerIds).limit(MAX_ROWS):Promise.resolve({data:[],error:null}),
   companyIds.length?
    client.from('legacy_company_identity_map')
     .select('legacy_company_id,issuer_id,security_id,dossier_id')
     .in('legacy_company_id',companyIds).limit(MAX_ROWS):
    Promise.resolve({data:[],error:null}),
  ]);
  const inventory:P0CInventory={
   companies,issuers,securities,
   dossiers:rows(dossierResult).map(r=>({
    dossierId:asUuid(r.dossier_id),issuerId:asUuid(r.issuer_id),
    active:r.active===true,
   })),
   maps:rows(mapResult).map(r=>({
    legacyCompanyId:asUuid(r.legacy_company_id),
    issuerId:asUuid(r.issuer_id),
    securityId:asUuid(r.security_id),
    dossierId:asUuid(r.dossier_id),
   })),
  };
  const result=planCanonicalIdentityOnboarding({
   isin,ticker,exchange,issuerDisplayName:displayName,
   evidenceUrl:input.evidenceUrl as string|null|undefined,
  },inventory);
  return NextResponse.json({
   ok:true,mode:'PRE_ADMISSION_CANDIDATE_ONLY',
   checkedAt:new Date().toISOString(),queryDurationMs:Date.now()-started,
   result,sourceVerified:false,legacyCrosswalkAuthorized:false,
   rpcExecuted:false,registryMutated:false,
   productionAccessed:false,
  },{headers});
 }catch(e){
  const code=e instanceof Error?e.message:'';
  const safe=new Set(['INVALID_ISIN','INVALID_ISIN_CHECK_DIGIT','INVALID_EVIDENCE_URL',
   'IDENTITY_INPUT_INVALID','IDENTITY_LISTING_INPUT_INVALID',
   'IDENTITY_NAME_INPUT_INVALID']);
  return NextResponse.json({ok:false,
   error:safe.has(code)?code:'IDENTITY_REGISTRY_READ_FAILED',
   registryMutated:false,rpcExecuted:false},
   {status:422,headers});
 }
}
