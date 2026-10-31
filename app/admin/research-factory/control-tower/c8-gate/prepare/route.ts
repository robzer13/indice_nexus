import {NextRequest,NextResponse} from 'next/server';
import {requireAdminSession} from '@/lib/auth/admin-session';
import {isHostedShadowResearchEnabled} from
 '@/lib/orotitan-equity/product/hosted-shadow-research-gate';
import {getHostedShadowRecoveryClient} from
 '@/lib/orotitan-equity/product/research-inbox-pilot-server';

export const runtime='nodejs';
export const dynamic='force-dynamic';
const headers={'Cache-Control':'private, no-store'};
const fail=(error:string,status=409)=>NextResponse.json(
 {ok:false,error,accepted:false,stageStarted:false,published:false},
 {status,headers});
export async function POST(req:NextRequest){
 if(!isHostedShadowResearchEnabled(process.env))return fail('SHADOW_ONLY',404);
 try{await requireAdminSession();}catch{return fail('UNAUTHORIZED',401);}
 if(req.headers.get('origin')!==new URL(req.url).origin)
  return fail('ORIGIN_MISMATCH',403);
 if(req.headers.get('content-type')?.split(';')[0]?.trim()!=='application/json')
  return fail('REQUEST_INVALID',400);
 let input:unknown;
 try{const raw=await req.text();
  if(Buffer.byteLength(raw,'utf8')>3500)return fail('REQUEST_TOO_LARGE',400);
  input=JSON.parse(raw);
 }catch{return fail('REQUEST_INVALID',400);}
 if(!input||typeof input!=='object'||Array.isArray(input))
  return fail('J0_C8_PREPARE_INVALID',400);
 const v=input as Record<string,unknown>;
 const keys=['reviewKey','milestone','auditRef','auditSha256','reviewerLabel','evidenceEventIds'];
 if(Object.keys(v).length!==keys.length||keys.some(k=>!(k in v)))
  return fail('J0_C8_PREPARE_INVALID',400);
 if(typeof v.reviewKey!=='string'||
    !/^orotitan:c8:review:[a-zA-Z0-9:_-]{8,124}$/.test(v.reviewKey)||
    typeof v.milestone!=='string'||!/^J[0-8]$/.test(v.milestone)||
    typeof v.auditRef!=='string'||!/^https:\/\/github\.com\/[\w-]+\/[\w.-]+\/blob\/[a-f0-9]{40}\/docs\/audits\/[\w.-]+\.md$/.test(v.auditRef)||
    typeof v.auditSha256!=='string'||!/^[a-f0-9]{64}$/.test(v.auditSha256)||
    typeof v.reviewerLabel!=='string'||v.reviewerLabel.trim().length<4||
    v.reviewerLabel.length>100||
    !Array.isArray(v.evidenceEventIds)||v.evidenceEventIds.length!==7||
    v.evidenceEventIds.some(id=>typeof id!=='string'||
     !/^[a-f0-9]{8}-(?:[a-f0-9]{4}-){3}[a-f0-9]{12}$/i.test(id)))
   return fail('J0_C8_PREPARE_INVALID',400);
 try{
  const client=getHostedShadowRecoveryClient();
  const rpc=await client.rpc('orotitan_j0_c8_prepare_review_v1',{
   p_review_key:v.reviewKey,p_milestone:v.milestone,
   p_audit_ref:v.auditRef,p_audit_sha256:v.auditSha256,
   p_reviewer_label:v.reviewerLabel.trim(),
   p_evidence_ids:v.evidenceEventIds,
  });
  if(rpc.error){
   const code=(rpc.error.message.match(/J0_C8_[A-Z0-9_]+/)??[])[0];
   return fail(code??'J0_C8_PREPARE_REJECTED',409);
  }
  const reply=rpc.data as Record<string,unknown>|null;
  if(!reply||typeof reply.reviewId!=='string'||
     typeof reply.reviewSha256!=='string'||
     !['PREPARED','REPLAYED'].includes(String(reply.status))||
     reply.accepted!==false)return fail('J0_C8_REVIEW_RECEIPT_INVALID',503);
  const stored=await client.from('orotitan_j0_c8_reviews_v1')
   .select('review_id,review_key,milestone_code,audit_ref,audit_sha256,reviewer_label,evidence_event_ids,review_sha256,journal_watermark')
   .eq('review_id',reply.reviewId).single();
  if(stored.error||!stored.data||
     stored.data.review_key!==v.reviewKey||
     stored.data.milestone_code!==v.milestone||
     stored.data.audit_ref!==v.auditRef||
     stored.data.audit_sha256!==v.auditSha256||
     stored.data.reviewer_label!==v.reviewerLabel.trim()||
     stored.data.review_sha256!==reply.reviewSha256||
     JSON.stringify(stored.data.evidence_event_ids)!==JSON.stringify(v.evidenceEventIds))
   return fail('J0_C8_REVIEW_READBACK_MISMATCH',503);
  return NextResponse.json({
   ok:true,status:reply.status,reviewId:reply.reviewId,
   reviewSha256:reply.reviewSha256,
   journalWatermark:stored.data.journal_watermark,
   persistedReadbackVerified:true,accepted:false,
   stageStarted:false,published:false,scope:'SHADOW_ONLY',
  },{headers});
 }catch{return fail('J0_C8_PREPARE_UNAVAILABLE',503);}
}
