import {NextResponse,type NextRequest} from 'next/server';
import {requireAdminSession} from '@/lib/auth/admin-session';
import {isHostedShadowResearchEnabled} from '@/lib/orotitan-equity/product/hosted-shadow-research-gate';
import {getHostedShadowRecoveryClient} from '@/lib/orotitan-equity/product/research-inbox-pilot-server';
import {archiveBrowserProvidedAsmlQ3Pdf} from '@/lib/orotitan-equity/product/asml-q3-browser-upload-v1';
export const runtime='nodejs';
export const dynamic='force-dynamic';
const MAX_REQUEST=25_000_000;
function redirectTo(req:NextRequest,key:'result'|'error',value:string){
 const url=new URL('/admin/research-q3-sources',req.url);
 url.searchParams.set(key,value);
 return NextResponse.redirect(url,{status:303});
}
export async function POST(req:NextRequest) {
 // No production route is permitted to perform source archival.
 if(!isHostedShadowResearchEnabled(process.env))
  return NextResponse.json({error:'SHADOW_ONLY'},{status:404});
 try{await requireAdminSession();}
 catch{return NextResponse.json({error:'UNAUTHORIZED'},{status:401});}
 const origin=req.headers.get('origin');
 const expected=new URL(req.url).origin;
 if(origin!==expected)return NextResponse.json({error:'ORIGIN_MISMATCH'},{status:403});
 const len=Number(req.headers.get('content-length')??'NaN');
 if(!Number.isSafeInteger(len)||len<10_000||len>MAX_REQUEST)
  return NextResponse.json({error:'UPLOAD_SIZE_INVALID'},{status:413});
 const mime=req.headers.get('content-type')??'';
 if(!/^multipart\/form-data;\s*boundary=/i.test(mime))
  return NextResponse.json({error:'INVALID_MULTIPART'},{status:415});
 try {
  const form=await req.formData();
  const file=form.get('document');
  if(!(file instanceof File)||file.size<10_000||file.size>24_000_000||
     !/\.pdf$/i.test(file.name) ||
     !['application/pdf','application/octet-stream'].includes(file.type))
   return redirectTo(req,'error','invalid_pdf');
  const bytes=Buffer.from(await file.arrayBuffer());
  await archiveBrowserProvidedAsmlQ3Pdf(getHostedShadowRecoveryClient(),bytes);
  return redirectTo(req,'result','stored_sha_verified');
 } catch {
  return redirectTo(req,'error','capture_failed');
 }
}
