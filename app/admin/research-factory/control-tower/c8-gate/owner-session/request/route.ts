import {NextRequest,NextResponse} from 'next/server';
import {requireAdminSession} from '@/lib/auth/admin-session';
import {createSoloOwnerAuthClient} from '@/lib/auth/c8-solo-owner';
import {isHostedC8ReadOnlyEnabled,isHostedShadowResearchEnabled} from '@/lib/orotitan-equity/product/hosted-shadow-research-gate';

export const runtime='nodejs';export const dynamic='force-dynamic';
const headers={'Cache-Control':'private, no-store'};
const fail=(error:string,status=400)=>NextResponse.json({ok:false,error},{status,headers});
export async function POST(req:NextRequest){
 if(!(isHostedC8ReadOnlyEnabled(process.env)||isHostedShadowResearchEnabled(process.env)))return fail('SHADOW_ONLY',404);
 try{await requireAdminSession();}catch{return fail('UNAUTHORIZED',401);}
 if(req.headers.get('origin')!==new URL(req.url).origin)return fail('ORIGIN_MISMATCH',403);
 if(req.headers.get('content-type')?.split(';')[0]?.trim()!=='application/json')return fail('REQUEST_INVALID');
 let email='';try{const body=await req.json() as {email?:unknown};
  if(Object.keys(body).length!==1||typeof body.email!=='string')return fail('REQUEST_INVALID');
  email=body.email.trim();}catch{return fail('REQUEST_INVALID');}
 if(email.length>254||!/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email))return fail('REQUEST_INVALID');
 try{
  const callback=new URL(req.url);
  // Only Vercel SHADOW Preview hostnames; reject arbitrary Host-derived redirects.
  if(callback.protocol!=='https:'||
     !/^orotitan-research-shadow-preview-[a-z0-9-]+-orotitan[.]vercel[.]app$/.test(callback.hostname)||
     callback.port)
    return fail('OWNER_REDIRECT_DENIED',403);
  const emailRedirectTo=callback.origin+'/admin/research-factory/control-tower/c8-owner';
  const {client}=createSoloOwnerAuthClient();
  const {error}=await client.auth.signInWithOtp({
    email,options:{shouldCreateUser:false,emailRedirectTo},
  });
  if(error)return fail('OWNER_OTP_REQUEST_REJECTED',403);
  return NextResponse.json({ok:true,status:'OTP_REQUESTED'},{headers});
 }catch{return fail('OWNER_AUTH_UNAVAILABLE',503);}
}
