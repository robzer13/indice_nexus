import {NextResponse,type NextRequest} from 'next/server';
import {requireAdminSession} from '@/lib/auth/admin-session';
import {isHostedShadowResearchEnabled} from '@/lib/orotitan-equity/product/hosted-shadow-research-gate';
import {getHostedShadowRecoveryClient} from '@/lib/orotitan-equity/product/research-inbox-pilot-server';
import {captureAsmlOfficialPdf,selectSource,type AsmlV29SourceId}
 from '@/lib/orotitan-equity/product/asml-official-pdf-capture-v29';
import {classifyAsmlCaptureError}
 from '@/lib/orotitan-equity/product/asml-capture-error-diagnostics-v291';
export const runtime='nodejs';
export const dynamic='force-dynamic';
export const maxDuration=60;

function next(req:NextRequest,key:'result'|'error',value:string,sourceId:AsmlV29SourceId|null=null){
  const url=new URL('/admin/research-primary-sources',req.url);
  url.searchParams.set(key,value);
  if(sourceId)url.searchParams.set('source',sourceId);
  return NextResponse.redirect(url,{status:303});
}

export async function POST(req:NextRequest){
  // Production and non-isolated previews must not reach source download/storage.
  if(!isHostedShadowResearchEnabled(process.env))
    return NextResponse.json({error:'SHADOW_ONLY'},{status:404});
  try{await requireAdminSession();}
  catch{return NextResponse.json({error:'UNAUTHORIZED'},{status:401});}
  if(req.headers.get('origin')!==new URL(req.url).origin)
    return NextResponse.json({error:'ORIGIN_MISMATCH'},{status:403});
  const len=Number(req.headers.get('content-length')??'NaN');
  if(!Number.isSafeInteger(len)||len<1||len>2000)
    return NextResponse.json({error:'REQUEST_SIZE_INVALID'},{status:413});
  const mime=(req.headers.get('content-type')??'').split(';')[0].trim();
  if(!['application/x-www-form-urlencoded','multipart/form-data'].includes(mime))
    return NextResponse.json({error:'UNSUPPORTED_CONTENT_TYPE'},{status:415});
  let selectedSourceId:AsmlV29SourceId|null=null;
  try{
    const form=await req.formData();
    const selected=form.get('sourceId');
    if(typeof selected!=='string')return next(req,'error','SOURCE_SELECTION_INVALID');
    const source=selectSource(selected);
    selectedSourceId=source.id as AsmlV29SourceId;
    const receipt=await captureAsmlOfficialPdf(getHostedShadowRecoveryClient(),selectedSourceId);
    return next(req,'result',`${receipt.sourceId}:${receipt.sha256}`,selectedSourceId);
  }catch(error){
    const code=classifyAsmlCaptureError(error);
    // Log only enumerated source and category; never log raw exceptions, URLs, cookies or secrets.
    console.warn('ASML_V29_CAPTURE_FAILED',{sourceId:selectedSourceId??'UNVERIFIED',category:code});
    return next(req,'error',code,selectedSourceId);
  }
}
