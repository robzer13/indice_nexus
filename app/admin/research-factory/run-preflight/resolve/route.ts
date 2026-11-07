import {NextResponse,type NextRequest} from 'next/server';
import {requireAdminSession} from '@/lib/auth/admin-session';
import {isHostedJ1RunFactoryEnabled} from
 '@/lib/orotitan-equity/product/hosted-shadow-research-gate';
import {getJ1RunFactoryShadowClient} from
 '@/lib/orotitan-equity/product/runfactory-j1-shadow-server';
import {resolveJ1CompanyIdentity,type J1CompanyIdentity} from
 '@/lib/orotitan-equity/product/runfactory-j1-request-resolution-v1';

export const runtime='nodejs';
export const dynamic='force-dynamic';
const headers={'Cache-Control':'private, no-store'};
const LIMIT=40;
const MAX_INPUT_BYTES=300;
const UUID=/^[a-f0-9]{8}-(?:[a-f0-9]{4}-){3}[a-f0-9]{12}$/i;
const fail=():never=>{throw Error('J1_RESOLVE_READ_FAILED');};
type Row=Record<string,unknown>;
function rows(result:{data:unknown;error:unknown}):Row[]{
 const data:unknown=result.data;
 if(result.error||!Array.isArray(data))fail();
 const list=data as unknown[];
 if(list.length>=LIMIT)fail();
 if(list.some((v:unknown)=>!v||typeof v!=='object'||Array.isArray(v)))fail();
 return list as Row[];
}
function str(x:unknown):string{
 if(typeof x!=='string'||!x)return fail();return x;
}
function uuid(x:unknown):string{
 const v=str(x);if(!UUID.test(v))fail();return v;
}
export async function POST(req:NextRequest){
 if(!isHostedJ1RunFactoryEnabled(process.env))
  return NextResponse.json({ok:false,error:'SHADOW_ONLY'},{status:404,headers});
 try{await requireAdminSession();}
 catch{return NextResponse.json({ok:false,error:'UNAUTHORIZED'},{status:401,headers});}
 if(req.headers.get('origin')!==new URL(req.url).origin)
  return NextResponse.json({ok:false,error:'ORIGIN_MISMATCH'},{status:403,headers});
 if(req.headers.get('content-type')?.split(';')[0]?.trim()!=='application/json')
  return NextResponse.json({ok:false,error:'REQUEST_INVALID'},{status:400,headers});
 try{
  const raw=await req.text();
  if(Buffer.byteLength(raw,'utf8')>MAX_INPUT_BYTES)return NextResponse.json(
   {ok:false,error:'REQUEST_INVALID'},{status:400,headers});
  const data:unknown=JSON.parse(raw);
  if(!data||typeof data!=='object'||Array.isArray(data))throw Error('REQUEST_INVALID');
  const query=(data as Record<string,unknown>).query;
  if(typeof query!=='string')throw Error('REQUEST_INVALID');
  const normalized=query.trim().normalize('NFKC');
  if(normalized.length<2||normalized.length>100||
    !/^[\p{L}\p{N} .,&'’()+\/-]+$/u.test(normalized))
   throw Error('J1_COMPANY_QUERY_INVALID');
  const client=getJ1RunFactoryShadowClient();
  const isIsin=/^[A-Z]{2}[A-Z0-9]{9}[0-9]$/.test(normalized.toUpperCase());
  const isTicker=/^[A-Z0-9._-]{2,15}$/.test(normalized.toUpperCase());
  let securityRows:Row[]=[];
  let issuerRows:Row[]=[];
  if(isIsin||isTicker){
   const result=await client.from('securities').select(
    'security_id,issuer_id,isin,ticker,exchange'
   )[isIsin?'eq':'ilike'](isIsin?'isin':'ticker',normalized.toUpperCase())
     .limit(LIMIT);
   securityRows=rows(result);
  }
  if(!isIsin){
   const [names,legal]=await Promise.all([
    client.from('issuers').select('issuer_id,display_name')
      .ilike('display_name',normalized+'%').limit(LIMIT),
    client.from('issuers').select('issuer_id,display_name')
      .ilike('legal_name',normalized+'%').limit(LIMIT),
   ]);
   const map=new Map<string,Row>();
   for(const issuer of [...rows(names),...rows(legal)])
    map.set(uuid(issuer.issuer_id),issuer);
   issuerRows=[...map.values()];
   if(issuerRows.length){
    const related=rows(await client.from('securities').select(
      'security_id,issuer_id,isin,ticker,exchange'
    ).in('issuer_id',issuerRows.map(i=>uuid(i.issuer_id))).limit(LIMIT));
    const existing=new Set(securityRows.map(x=>uuid(x.security_id)));
    for(const sec of related){
     const id=uuid(sec.security_id);
     if(!existing.has(id)){securityRows.push(sec);existing.add(id);}
    }
   }
  }
  const ids=[...new Set(securityRows.map(s=>uuid(s.issuer_id)))];
  if(ids.length){
   // Re-read complete issuer identities. Never infer a legal entity by ticker.
   issuerRows=rows(await client.from('issuers').select('issuer_id,display_name')
    .in('issuer_id',ids).limit(LIMIT));
  }
  const issuers=new Map(issuerRows.map(r=>[uuid(r.issuer_id),str(r.display_name)]));
  const observed:J1CompanyIdentity[]=securityRows.map(r=>{
   const issuerId=uuid(r.issuer_id);
   const issuerName=issuers.get(issuerId);
   if(!issuerName)fail();
   return {securityId:uuid(r.security_id),issuerId,
    issuerName:str(issuerName),isin:str(r.isin),ticker:str(r.ticker),exchange:str(r.exchange)};
  });
  const result=resolveJ1CompanyIdentity(normalized,observed);
  return NextResponse.json({ok:true,...result,readOnly:true,environment:'SHADOW_ONLY'},
   {headers});
 }catch(e){
  const code=e instanceof Error?e.message:'';
  return NextResponse.json({ok:false,
   error:code==='J1_COMPANY_QUERY_INVALID'||code==='REQUEST_INVALID'?code:
    'J1_RESOLVE_READ_FAILED'},
   {status:422,headers});
 }
}
