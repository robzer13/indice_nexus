import {NextRequest,NextResponse} from 'next/server';
import {requireAdminSession} from '@/lib/auth/admin-session';
import {assertSoloOwnerUser,C8_OWNER_COOKIE,createSoloOwnerAuthClient} from '@/lib/auth/c8-solo-owner';
import {isHostedC8ReadOnlyEnabled,isHostedShadowResearchEnabled} from '@/lib/orotitan-equity/product/hosted-shadow-research-gate';

export const runtime='nodejs';
export const dynamic='force-dynamic';
const headers={'Cache-Control':'private, no-store'};
const fail=(error:string,status=400)=>
 NextResponse.json({ok:false,error},{status,headers});

// Exchanges only the short-lived Auth access JWT from a Supabase magic-link fragment.
// The refresh token stays client-side and is never requested, stored, or transmitted.
export async function POST(req:NextRequest){
 if(!(isHostedC8ReadOnlyEnabled(process.env)||isHostedShadowResearchEnabled(process.env)))
  return fail('SHADOW_ONLY',404);
 try{await requireAdminSession();}catch{return fail('UNAUTHORIZED',401);}
 if(req.headers.get('origin')!==new URL(req.url).origin)return fail('ORIGIN_MISMATCH',403);
 if(req.headers.get('content-type')?.split(';')[0]?.trim()!=='application/json')
  return fail('REQUEST_INVALID');
 let accessToken='';
 try{
  const body=await req.json() as {accessToken?:unknown};
  if(!body||typeof body!=='object'||Array.isArray(body)||
     Object.keys(body).length!==1||typeof body.accessToken!=='string')
    return fail('REQUEST_INVALID');
  accessToken=body.accessToken;
 }catch{return fail('REQUEST_INVALID');}
 if(accessToken.length<50||accessToken.length>4096||
    !/^[A-Za-z0-9_-]+[.][A-Za-z0-9_-]+[.][A-Za-z0-9_-]+$/.test(accessToken))
  return fail('OWNER_LINK_INVALID',401);
 try{
  const {client,config}=createSoloOwnerAuthClient();
  const {data,error}=await client.auth.getUser(accessToken);
  if(error||!data.user)return fail('OWNER_LINK_INVALID',401);
  assertSoloOwnerUser(data.user,config);

  // The cookie may never outlive the verified access JWT.
  const claims=JSON.parse(Buffer.from(accessToken.split('.')[1],'base64url').toString('utf8')) as {exp?:unknown};
  const now=Math.floor(Date.now()/1000);
  if(typeof claims.exp!=='number'||!Number.isSafeInteger(claims.exp)||claims.exp<=now+30)
   return fail('OWNER_LINK_EXPIRED',401);
  const maxAge=Math.min(3600,claims.exp-now);
  const response=NextResponse.json({ok:true,status:'OWNER_ATTESTED'},{headers});
  response.cookies.set(C8_OWNER_COOKIE,accessToken,{
   httpOnly:true,secure:true,sameSite:'strict',
   path:'/admin/research-factory/control-tower/c8-gate',maxAge,
  });
  return response;
 }catch{return fail('OWNER_ATTESTATION_DENIED',403);}
}
