import {NextRequest,NextResponse} from 'next/server';
import {requireAdminSession} from '@/lib/auth/admin-session';
import {C8_OWNER_COOKIE} from '@/lib/auth/c8-solo-owner';
import {isHostedShadowResearchEnabled} from '@/lib/orotitan-equity/product/hosted-shadow-research-gate';
export const runtime='nodejs';export const dynamic='force-dynamic';
const headers={'Cache-Control':'private, no-store'};
export async function POST(req:NextRequest){
 if(!isHostedShadowResearchEnabled(process.env))return NextResponse.json({ok:false,error:'SHADOW_ONLY'},{status:404,headers});
 try{await requireAdminSession();}catch{return NextResponse.json({ok:false,error:'UNAUTHORIZED'},{status:401,headers});}
 if(req.headers.get('origin')!==new URL(req.url).origin)return NextResponse.json({ok:false,error:'ORIGIN_MISMATCH'},{status:403,headers});
 const response=NextResponse.json({ok:true,status:'SIGNED_OUT'},{headers});
 response.cookies.set(C8_OWNER_COOKIE,'',{httpOnly:true,secure:true,sameSite:'strict',path:'/admin/research-factory/control-tower/c8-gate',maxAge:0});
 return response;
}
