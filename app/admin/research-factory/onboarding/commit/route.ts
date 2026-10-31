import {NextResponse,type NextRequest} from 'next/server';
import {requireAdminSession} from '@/lib/auth/admin-session';
import {isHostedShadowResearchEnabled} from
 '@/lib/orotitan-equity/product/hosted-shadow-research-gate';
import {getHostedShadowRecoveryClient} from
 '@/lib/orotitan-equity/product/research-inbox-pilot-server';
import {buildP0DOnboardingCommand,type P0DInput} from
 '@/lib/orotitan-equity/product/runfactory-p0d-onboarding-command-v1';

export const runtime='nodejs';
export const dynamic='force-dynamic';
export const maxDuration=30;
const headers={'Cache-Control':'private, no-store'};
const UUID=/^[a-f0-9]{8}-(?:[a-f0-9]{4}-){3}[a-f0-9]{12}$/i;
const OUTCOMES=new Set(['CREATED','REUSED','LISTING_ADDED','REPLAYED']);
const fail=():never=>{throw Error('P0D_POSTCOMMIT_READBACK_FAILED');};
function object(x:unknown):Record<string,unknown>{
 if(!x||typeof x!=='object'||Array.isArray(x))return fail();
 return x as Record<string,unknown>;
}
function str(x:unknown):string{
 if(typeof x!=='string')return fail();return x;
}
function uuid(x:unknown):string{
 const y=str(x);if(!UUID.test(y))return fail();return y;
}
export async function POST(req:NextRequest){
 if(!isHostedShadowResearchEnabled(process.env))
  return NextResponse.json({ok:false,error:'SHADOW_ONLY'},{status:404,headers});
 try{await requireAdminSession();}
 catch{return NextResponse.json({ok:false,error:'UNAUTHORIZED'},{status:401,headers});}
 if(req.headers.get('origin')!==new URL(req.url).origin)
  return NextResponse.json({ok:false,error:'ORIGIN_MISMATCH'},{status:403,headers});
 if(req.headers.get('content-type')?.split(';')[0].trim()!=='application/json')
  return NextResponse.json({ok:false,error:'REQUEST_INVALID'},{status:400,headers});
 let body:unknown;
 try{
  const raw=await req.text();
  if(Buffer.byteLength(raw,'utf8')>6000||raw.length<10)
   return NextResponse.json({ok:false,error:'REQUEST_INVALID'},{status:400,headers});
  body=JSON.parse(raw) as unknown;
 }catch{return NextResponse.json({ok:false,error:'REQUEST_INVALID'},{status:400,headers});}
 try{
  const args=buildP0DOnboardingCommand(body as P0DInput);
  const client=getHostedShadowRecoveryClient();
  // A single PostgREST RPC invocation is a single PostgreSQL transaction.
  // SQL independently validates checksum, source attestation and all
  // identity collisions under two ordered transaction advisory locks.
  const result=await client.rpc('orotitan_onboard_identity_shadow_v1',args);
  if(result.error){
   const message=String(result.error.message??'');
   const code=(message.match(/OROTITAN_[A-Z0-9_]+/)??[])[0];
   return NextResponse.json({ok:false,error:code??'P0D_RPC_FAILED',
    rpcMayHaveBeenExecuted:true,checkIdempotencyBeforeRetry:true},
    {status:code?409:503,headers});
  }
  const receipt=object(result.data);
  const issuerId=uuid(receipt.issuer_id);
  const securityId=uuid(receipt.security_id);
  const dossierId=uuid(receipt.dossier_id);
  const status=str(receipt.outcome);
  const requestSha=str(receipt.request_sha256);
  if(!OUTCOMES.has(status)||receipt.run_created!==false||
     receipt.registry_stage_advanced!==false||
     receipt.source_automatically_verified!==false||
     !/^[a-f0-9]{64}$/.test(requestSha))
   fail();
  // Independent readback: do not equate an RPC HTTP 200 with persistence proof.
  const [issuerResult,securityResult,dossierResult,auditResult]=await Promise.all([
   client.from('issuers')
    .select('issuer_id,display_name,legal_name,country,reporting_currency')
    .eq('issuer_id',issuerId).single(),
   client.from('securities')
    .select('security_id,issuer_id,isin,ticker,exchange,trading_currency,primary_listing,quote_unit,price_decimals')
    .eq('security_id',securityId).single(),
   client.from('research_dossiers')
    .select('dossier_id,issuer_id,active,current_snapshot_id')
    .eq('dossier_id',dossierId).single(),
   client.from('orotitan_identity_onboarding_receipts_v1')
    .select('request_key,request_sha256,issuer_id,security_id,dossier_id')
    .eq('request_key',args.p_idempotency_key).single(),
  ]);
  if(issuerResult.error||securityResult.error||dossierResult.error||
     auditResult.error)fail();
  const issuer=object(issuerResult.data),security=object(securityResult.data),
   dossier=object(dossierResult.data),audit=object(auditResult.data);
  if(issuer.issuer_id!==issuerId||
     issuer.display_name!==args.p_display_name||
     (issuer.legal_name!==null&&issuer.legal_name!==args.p_legal_name)||
     (issuer.country!==null&&issuer.country!==args.p_issuer_country)||
     (issuer.reporting_currency!==null&&
      issuer.reporting_currency!==args.p_reporting_currency)||
     security.security_id!==securityId||
     security.issuer_id!==issuerId||
     security.isin!==args.p_isin||security.ticker!==args.p_ticker||
     security.exchange!==args.p_exchange||
     security.trading_currency!==args.p_trading_currency||
     security.primary_listing!==args.p_primary_listing||
     security.quote_unit!==args.p_quote_unit||
     security.price_decimals!==args.p_price_decimals||
     dossier.dossier_id!==dossierId||dossier.issuer_id!==issuerId||
     dossier.active!==true||audit.request_key!==args.p_idempotency_key||
     audit.request_sha256!==requestSha||
     audit.issuer_id!==issuerId||audit.security_id!==securityId||
     audit.dossier_id!==dossierId)fail();
  return NextResponse.json({ok:true,environment:'SHADOW_ONLY',
   outcome:status,issuerId,securityId,dossierId,
   idempotencyKey:args.p_idempotency_key,
   requestSha256:requestSha,
   persistedAndReadBack:true,
   operatorAttestationRecorded:true,
   officialSourceBytesAutomaticallyAuthenticated:false,
   runCreated:false,stageAdvanced:false,productionTouched:false,
  },{headers});
 }catch(e){
  const code=e instanceof Error?e.message:'';
  const clientError=code.startsWith('P0D_')||code.startsWith('INVALID_ISIN');
  return NextResponse.json({ok:false,
   error:clientError?code:'P0D_POSTCOMMIT_READBACK_OR_SERVICE_FAILURE',
   checkIdempotencyBeforeRetry:true,
  },{status:clientError?400:503,headers});
 }
}
