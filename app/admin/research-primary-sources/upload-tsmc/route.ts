import {NextResponse,type NextRequest} from 'next/server';
import {requireAdminSession} from '@/lib/auth/admin-session';
import {isHostedShadowResearchEnabled} from '@/lib/orotitan-equity/product/hosted-shadow-research-gate';
import {getHostedShadowRecoveryClient} from '@/lib/orotitan-equity/product/research-inbox-pilot-server';
import {archiveBrowserProvidedTsmcQ3,TSMC_BROWSER_MAX_BYTES}
 from '@/lib/orotitan-equity/product/tsmc-browser-pdf-upload-v1';

export const runtime='nodejs';
export const dynamic='force-dynamic';
export const maxDuration=60;
const MAX_REQUEST=15_800_000;
const RETURN_PATH='/admin/research-primary-sources';
type UploadQuery='tsmc_sha'|'tsmc_error';
function redirectTo(req:NextRequest,key:UploadQuery,value:string){
 const url=new URL(RETURN_PATH,req.url);
 url.searchParams.set('source','TSMC_2026_Q3_CALL');
 url.searchParams.set(key,value);
 return NextResponse.redirect(url,{status:303});
}
export async function POST(req:NextRequest){
 // Only the isolated SHADOW Preview may archive manually transported bytes.
 if(!isHostedShadowResearchEnabled(process.env))
  return NextResponse.json({error:'SHADOW_ONLY'},{status:404});
 try{await requireAdminSession();}
 catch{return NextResponse.json({error:'UNAUTHORIZED'},{status:401});}
 if(req.headers.get('origin')!==new URL(req.url).origin)
  return NextResponse.json({error:'ORIGIN_MISMATCH'},{status:403});
 const len=Number(req.headers.get('content-length')??'NaN');
 if(!Number.isSafeInteger(len)||len<10_000||len>MAX_REQUEST)
  return NextResponse.json({error:'UPLOAD_SIZE_INVALID'},{status:413});
 const mime=req.headers.get('content-type')??'';
 if(!/^multipart\/form-data;\s*boundary=/i.test(mime))
  return NextResponse.json({error:'INVALID_MULTIPART'},{status:415});
 try{
  const form=await req.formData();
  if(form.get('documentId')!=='TSMC_2026_Q3_CALL')
   return redirectTo(req,'tsmc_error','source_invalid');
  const file=form.get('document');
  if(!(file instanceof File)||file.size<10_000||file.size>TSMC_BROWSER_MAX_BYTES||
     !/\.pdf$/i.test(file.name)||
     !['application/pdf','application/octet-stream'].includes(file.type))
   return redirectTo(req,'tsmc_error','invalid_pdf');
  const bytes=Buffer.from(await file.arrayBuffer());
  const receipt=await archiveBrowserProvidedTsmcQ3(getHostedShadowRecoveryClient(),bytes);
  return redirectTo(req,'tsmc_sha',receipt.sha256);
 }catch(error){
  // Never include raw errors, cookies, names, file contents or issuer URLs in logs.
  const cause=error instanceof Error?error.message:'';
  const code=cause==='TSMC_BROWSER_INVALID_PDF'?'invalid_pdf':
    cause.startsWith('TSMC_BROWSER_PRIVATE_')||cause.startsWith('TSMC_BROWSER_STORAGE_')
      ?'storage_failure':'capture_failed';
  console.warn('TSMC_BROWSER_SHADOW_UPLOAD_FAILURE',{category:code});
  return redirectTo(req,'tsmc_error',code);
 }
}
