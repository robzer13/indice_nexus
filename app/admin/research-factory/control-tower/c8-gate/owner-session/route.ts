import {NextResponse} from 'next/server';
import {requireAdminSession} from '@/lib/auth/admin-session';
import {requireSoloOwnerAttestation} from '@/lib/auth/c8-solo-owner';
import {isHostedC8ReadOnlyEnabled,isHostedShadowResearchEnabled} from '@/lib/orotitan-equity/product/hosted-shadow-research-gate';
export const runtime='nodejs';export const dynamic='force-dynamic';
const headers={'Cache-Control':'private, no-store'};
export async function GET(){
 if(!(isHostedC8ReadOnlyEnabled(process.env)||isHostedShadowResearchEnabled(process.env)))return NextResponse.json({ok:false,error:'SHADOW_ONLY'},{status:404,headers});
 try{await requireAdminSession();await requireSoloOwnerAttestation();
  return NextResponse.json({ok:true,authenticated:true,provider:'SUPABASE_AUTH'},{headers});
 }catch{return NextResponse.json({ok:true,authenticated:false},{headers});}
}
