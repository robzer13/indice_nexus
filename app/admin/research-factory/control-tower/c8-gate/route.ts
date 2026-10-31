import {NextRequest,NextResponse} from 'next/server';
import {requireAdminSession} from '@/lib/auth/admin-session';
import {isHostedShadowResearchEnabled} from
 '@/lib/orotitan-equity/product/hosted-shadow-research-gate';
import {getHostedShadowRecoveryClient} from
 '@/lib/orotitan-equity/product/research-inbox-pilot-server';

export const runtime='nodejs';
export const dynamic='force-dynamic';
const headers={'Cache-Control':'private, no-store'};
const fail=(error:string,status=409)=>NextResponse.json({
 ok:false,error,accepted:false,stageStarted:false,published:false,
},{status,headers});

export async function GET(req:NextRequest){
 if(!isHostedShadowResearchEnabled(process.env))return fail('SHADOW_ONLY',404);
 try{await requireAdminSession();}catch{return fail('UNAUTHORIZED',401);}
 const milestone=new URL(req.url).searchParams.get('milestone')??'J0';
 if(!/^J[0-8]$/.test(milestone))return fail('J0_C8_MILESTONE_INVALID',400);
 try{
  const client=getHostedShadowRecoveryClient();
  const [check,control,reviews,decisions]=await Promise.all([
   client.rpc('orotitan_j0_c8_read_gate_v1',{p_milestone:milestone}),
   client.from('orotitan_j0_c8_acceptance_control_v1')
    .select('enabled,activation_authority_ref').eq('lock_name','GLOBAL_SHADOW').single(),
   client.from('orotitan_j0_c8_reviews_v1')
    .select('review_id',{count:'exact',head:true}).eq('milestone_code',milestone),
   client.from('orotitan_j0_c8_decisions_v1')
    .select('decision_id',{count:'exact',head:true}).eq('milestone_code',milestone),
  ]);
  if(check.error||!check.data||control.error||!control.data||
     reviews.error||reviews.count===null||
     decisions.error||decisions.count===null||
     check.data.milestone!==milestone||
     check.data.preparedReviews!==reviews.count||
     check.data.accepted!==(decisions.count>0)||
     check.data.acceptanceEnabled!==control.data.enabled)
   return fail('J0_C8_READBACK_MISMATCH',503);
  const lockActive=control.data.enabled===true;
  return NextResponse.json({
   ok:true,scope:'SHADOW_ONLY',milestone,
   gate:check.data,
   activationLocked:!lockActive,
   preparedReviews:reviews.count,
   persistedDecisions:decisions.count,
   accepted:decisions.count>0,
   // Never report future/implicit approval from a missing decision.
   humanGoRequired:true,stageStarted:false,published:false,
  },{headers});
 }catch{return fail('J0_C8_GATE_UNAVAILABLE',503);}
}
