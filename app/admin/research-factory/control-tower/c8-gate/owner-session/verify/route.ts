import {NextRequest,NextResponse} from 'next/server';
import {requireAdminSession} from '@/lib/auth/admin-session';
import {assertSoloOwnerUser,C8_OWNER_COOKIE,createSoloOwnerAuthClient} from '@/lib/auth/c8-solo-owner';
import {isHostedShadowResearchEnabled} from '@/lib/orotitan-equity/product/hosted-shadow-research-gate';

export const runtime='nodejs';export const dynamic='force-dynamic';
const headers={'Cache-Control':'private, no-store'};
const fail=(error:string,status=400)=>NextResponse.json({ok:false,error},{status,headers});
export async function POST(req:NextRequest){
 if(!isHostedShadowResearchEnabled(process.env))return fail('SHADOW_ONLY',404);
 try{await requireAdminSession();}catch{return fail('UNAUTHORIZED',401);}
 if(req.headers.get('origin')!==new URL(req.url).origin)return fail('ORIGIN_MISMATCH',403);
 if(req.headers.get('content-type')?.split(';')[0]?.trim()!=='application/json')return fail('REQUEST_INVALID');
 let email='',token='';try{const body=await req.json() as {email?:unknown;token?:unknown};
  if(Object.keys(body).length!==2||typeof body.email!=='string'||typeof body.token!=='string')return fail('REQUEST_INVALID');
  email=body.email.trim();token=body.token.trim();}catch{return fail('REQUEST_INVALID');}
 if(email.length>254||!/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email)||!/^[0-9]{6,10}$/.test(token))return fail('REQUEST_INVALID');
 try{const {client,config}=createSoloOwnerAuthClient();
  const {data,error}=await client.auth.verifyOtp({email,token,type:'email'});
  if(error||!data.session||!data.user)return fail('OWNER_OTP_INVALID',401);
  assertSoloOwnerUser(data.user,config);
  const maxAge=Math.max(1,Math.min(3600,(data.session.expires_at??0)-Math.floor(Date.now()/1000)));
  const response=NextResponse.json({ok:true,status:'OWNER_ATTESTED'},{headers});
  response.cookies.set(C8_OWNER_COOKIE,data.session.access_token,{httpOnly:true,secure:true,
   sameSite:'strict',path:'/admin/research-factory/control-tower/c8-gate',maxAge});
  return response;
 }catch{return fail('OWNER_ATTESTATION_DENIED',403);}
}
