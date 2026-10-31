import {NextResponse,type NextRequest} from 'next/server';
import {requireAdminSession} from '@/lib/auth/admin-session';
import {isHostedShadowResearchEnabled} from '@/lib/orotitan-equity/product/hosted-shadow-research-gate';
import {getHostedShadowRecoveryClient} from '@/lib/orotitan-equity/product/research-inbox-pilot-server';
import {independentAuditAll,independentAuditOne,extractBoundedPages,assertValidSourceId}
 from '@/lib/orotitan-equity/product/asml-research-private-pdf-reaudit-v34';

export const runtime='nodejs';
export const dynamic='force-dynamic';
export const maxDuration=60;

const SAFE_ERRORS=new Set([
 'V34_SOURCE_DENIED','V34_SIZE_MISMATCH','V34_SHA_MISMATCH',
 'V34_PDF_STRUCTURE_INVALID','V34_RECEIPT_INVALID','V34_RECEIPT_MISMATCH',
 'V34_PRIVATE_STORAGE_READ_FAILED','V34_PRIVATE_STORAGE_SIZE_DENIED',
 'V34_PRIVATE_BUCKET_REQUIRED','V34_PAGE_RANGE_DENIED','V34_PDF_PAGE_COUNT_DENIED',
]);
function failure(error:unknown){
 const code=error instanceof Error&&SAFE_ERRORS.has(error.message)?
  error.message:'V34_AUDIT_UNAVAILABLE';
 // Never reveal third-party SDK errors, storage keys, source bytes or credentials.
 console.warn('OROTITAN_PRIVATE_REAUDIT_FAILED',{category:code});
 return NextResponse.json({ok:false,error:code,registryAdmitted:false},
   {status:422,headers:{'Cache-Control':'no-store'}});
}
export async function POST(req:NextRequest){
 if(!isHostedShadowResearchEnabled(process.env))
  return NextResponse.json({error:'SHADOW_ONLY'},{status:404});
 try{await requireAdminSession();}
 catch{return NextResponse.json({error:'UNAUTHORIZED'},{status:401});}
 if(req.headers.get('origin')!==new URL(req.url).origin)
  return NextResponse.json({error:'ORIGIN_MISMATCH'},{status:403});
 const contentType=(req.headers.get('content-type')??'').split(';')[0].trim();
 const contentLength=Number(req.headers.get('content-length')??'NaN');
 if(contentType!=='application/json'||!Number.isSafeInteger(contentLength)||
   contentLength<5||contentLength>1024)
  return NextResponse.json({error:'REQUEST_INVALID'},{status:413});
 try{
  const payload=await req.json() as unknown;
  if(!payload||typeof payload!=='object'||Array.isArray(payload))
   return NextResponse.json({error:'REQUEST_INVALID'},{status:400});
  const data=payload as Record<string,unknown>;
  const client=getHostedShadowRecoveryClient();
  const headers={'Cache-Control':'no-store'};
  if(data.mode==='rehash_all'){
   const sources=await independentAuditAll(client);
   return NextResponse.json({
    ok:true,mode:'rehash_all',checkedAt:new Date().toISOString(),
    sources,independentByteRehashCount:sources.length,
    independenceGroups:['ASML_ISSUER','TSMC_ISSUER'],
    pdfPageClaimsValidated:0,
    publisherAuthenticityVerified:false,
    historicalCutoffBytesCertified:false,registryAdmitted:false,
   },{headers});
  }
  if(data.mode==='scan_pages'){
   assertValidSourceId(data.sourceId);
   if(!Number.isSafeInteger(data.startPage)||!Number.isSafeInteger(data.pageCount)||
      Number(data.startPage)<1||Number(data.startPage)>2000||
      Number(data.pageCount)<1||Number(data.pageCount)>20)
    throw Error('V34_PAGE_RANGE_DENIED');
   const archived=await independentAuditOne(client,data.sourceId);
   const scan=await extractBoundedPages(data.sourceId,archived.bytes,
     Number(data.startPage),Number(data.pageCount));
   return NextResponse.json({
    ok:true,mode:'scan_pages',checkedAt:new Date().toISOString(),
    source:archived.summary,scan,
    numericMatchesDoNotValidateClaims:true,
    visualPageReviewStillRequired:true,
    issuerOriginAuthenticated:false,registryAdmitted:false,
   },{headers});
  }
  return NextResponse.json({error:'ACTION_DENIED'},{status:400});
 }catch(error){return failure(error);}
}
