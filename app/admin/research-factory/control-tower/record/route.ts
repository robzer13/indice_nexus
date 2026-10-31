import {NextResponse,type NextRequest} from 'next/server';
import {requireAdminSession} from '@/lib/auth/admin-session';
import {isHostedShadowResearchEnabled} from
 '@/lib/orotitan-equity/product/hosted-shadow-research-gate';
import {getHostedShadowRecoveryClient} from
 '@/lib/orotitan-equity/product/research-inbox-pilot-server';

export const runtime='nodejs';
export const dynamic='force-dynamic';
const headers={'Cache-Control':'private, no-store'};
const bad=(code:string,status=422)=>NextResponse.json({
 ok:false,error:code,accepted:false,stageStarted:false,
},{status,headers});
export async function POST(req:NextRequest){
 if(!isHostedShadowResearchEnabled(process.env))
  return bad('SHADOW_ONLY',404);
 try{await requireAdminSession();}catch{return bad('UNAUTHORIZED',401);}
 if(req.headers.get('origin')!==new URL(req.url).origin)
  return bad('ORIGIN_MISMATCH',403);
 if(req.headers.get('content-type')?.split(';')[0]?.trim()!=='application/json')
  return bad('REQUEST_INVALID',400);
 let data:unknown;
 try{
  const raw=await req.text();
  if(raw.length<5||Buffer.byteLength(raw,'utf8')>2600)
   return bad('REQUEST_INVALID',400);
  data=JSON.parse(raw);
 }catch{return bad('REQUEST_INVALID',400);}
 if(!data||typeof data!=='object'||Array.isArray(data))
  return bad('REQUEST_INVALID',400);
 const v=data as Record<string,unknown>;
 const permitted=new Set(['eventKey','milestoneCode','eventType',
  'title','evidenceRef','runId','details']);
 if(Object.keys(v).some(k=>!permitted.has(k)))
  return bad('J0_UNEXPECTED_FIELDS',400);
 if(typeof v.eventKey!=='string'||
    !/^orotitan:j0:[a-zA-Z0-9:_-]{3,145}$/.test(v.eventKey)||
    typeof v.milestoneCode!=='string'||!/^J[0-8]$/.test(v.milestoneCode)||
    typeof v.eventType!=='string'||![
     'EVIDENCE_RECORDED','BLOCKER_OPENED','BLOCKER_RESOLVED',
     'GATE_REVIEW_REQUESTED'].includes(v.eventType)||
    typeof v.title!=='string'||v.title.trim().length<5||
    v.title.length>200||
    typeof v.evidenceRef!=='string'||v.evidenceRef.trim().length<8||
    v.evidenceRef.length>400||/[\r\n\u0000-\u001f]/.test(v.evidenceRef)||
    (v.runId!==null&&v.runId!==undefined&&
     (typeof v.runId!=='string'||
      !/^[a-f0-9]{8}-(?:[a-f0-9]{4}-){3}[a-f0-9]{12}$/i.test(v.runId)))||
    !v.details||typeof v.details!=='object'||Array.isArray(v.details)||
    JSON.stringify(v.details).length>3000)
  return bad('J0_INPUT_INVALID',400);
 try{
  const client=getHostedShadowRecoveryClient();
  if(v.eventType==='EVIDENCE_RECORDED'){
   const idx=(v.details as Record<string,unknown>).criterionIndex;
   const {data:milestone,error:gateError}=await client
    .from('orotitan_j0_milestones_v1')
    .select('gate_criteria')
    .eq('milestone_code',v.milestoneCode).single();
   if(gateError||!milestone||!Array.isArray(milestone.gate_criteria))
    return bad('J0_GATE_CRITERIA_UNAVAILABLE',503);
   if(typeof idx!=='number'||!Number.isSafeInteger(idx)||
      idx<1||idx>milestone.gate_criteria.length)
    return bad('J0_EVIDENCE_CRITERION_INVALID',400);
  }
  const {data:reply,error}=await client.rpc('orotitan_j0_record_event_v1',{
   p_event_key:v.eventKey,p_milestone_code:v.milestoneCode,
   p_event_type:v.eventType,p_title:v.title.trim(),
   p_evidence_ref:v.evidenceRef.trim(),p_run_id:v.runId??null,
   p_operator_details:v.details,
  });
  if(error){
   const safe=(error.message.match(/J0_[A-Z0-9_]+/)??[])[0];
   return bad(safe??'J0_EVENT_WRITE_FAILED',409);
  }
  if(!reply||typeof reply!=='object'||Array.isArray(reply)||
     typeof reply.event_id!=='string'||![
      'CREATED','REPLAYED'].includes(reply.status)||
     !['REPORTED','VERIFIED_DIRECT_REGISTRY_READBACK']
      .includes(reply.verification_state))
   return bad('J0_RECEIPT_INVALID',503);
  // Independent verification, not a blind relay of a success message.
  const {data:stored,error:readError}=await client
    .from('orotitan_j0_events_v1')
    .select('event_id,event_key,milestone_code,event_type,title,evidence_ref,run_id,operator_details,verification_state,verified_snapshot,created_at')
    .eq('event_id',reply.event_id).single();
  if(readError||!stored||stored.event_key!==v.eventKey||
     stored.milestone_code!==v.milestoneCode||
     stored.event_type!==v.eventType||stored.title!==v.title.trim()||
     stored.evidence_ref!==v.evidenceRef.trim()||
     stored.run_id!==(v.runId??null)||
     stored.verification_state!==reply.verification_state)
   return bad('J0_PERSISTED_READBACK_MISMATCH',503);
  return NextResponse.json({
   ok:true,eventId:reply.event_id,status:reply.status,
   verificationState:reply.verification_state,
   persistedReadbackVerified:true,accepted:false,
   stageStarted:false,published:false,scope:'SHADOW_ONLY',
  },{headers});
 }catch{return bad('J0_SHADOW_GATE_FAILED',503);}
}
