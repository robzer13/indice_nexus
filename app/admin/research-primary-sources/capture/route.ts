import {NextResponse,type NextRequest} from 'next/server';
import {requireAdminSession} from '@/lib/auth/admin-session';
import {isHostedShadowResearchEnabled} from '@/lib/orotitan-equity/product/hosted-shadow-research-gate';
import {getHostedShadowRecoveryClient} from '@/lib/orotitan-equity/product/research-inbox-pilot-server';
import {captureAsmlOfficialPdf,selectSource,type AsmlV29SourceId}
 from '@/lib/orotitan-equity/product/asml-official-pdf-capture-v29';
export const runtime='nodejs';
export const dynamic='force-dynamic';
export const maxDuration=60;

function next(req:NextRequest,key:'result'|'error',value:string){
  const url=new URL('/admin/research-primary-sources',req.url);
  url.searchParams.set(key,value);
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
  try{
    const form=await req.formData();
    const selected=form.get('sourceId');
    if(typeof selected!=='string')return next(req,'error','source_invalid');
    const source=selectSource(selected);
    const receipt=await captureAsmlOfficialPdf(getHostedShadowRecoveryClient(),
      source.id as AsmlV29SourceId);
    return next(req,'result',`${receipt.sourceId}:${receipt.sha256}`);
  }catch{
    // No internal URLs, credentials, fetch details, or storage paths leaked.
    return next(req,'error','capture_unavailable_or_failed');
  }
}
