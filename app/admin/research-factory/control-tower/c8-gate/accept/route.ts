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
/**
 * This route is deliberately dormant in the deployed SHADOW Preview.
 * A SEPARATE, later user GO must authorize both a runtime flag and a
 * privileged DB activation. Neither is set by the C8 remediation GO.
 */
export async function POST(req:NextRequest){
 if(!isHostedShadowResearchEnabled(process.env))return fail('SHADOW_ONLY',404);
 try{await requireAdminSession();}catch{return fail('UNAUTHORIZED',401);}
 if(req.headers.get('origin')!==new URL(req.url).origin)
  return fail('ORIGIN_MISMATCH',403);
 if(process.env.OROTITAN_C8_ACCEPTANCE_GO_ENABLED!=='1')
  return fail('J0_C8_ACCEPTANCE_LOCKED',423);
 if(req.headers.get('content-type')?.split(';')[0]?.trim()!=='application/json')
  return fail('REQUEST_INVALID',400);
 let parsed:unknown;
 try{const raw=await req.text();
  if(Buffer.byteLength(raw,'utf8')>2100)return fail('REQUEST_TOO_LARGE',400);
  parsed=JSON.parse(raw);
 }catch{return fail('REQUEST_INVALID',400);}
 if(!parsed||typeof parsed!=='object'||Array.isArray(parsed))
  return fail('J0_C8_DECISION_INVALID',400);
 const p=parsed as Record<string,unknown>;
 const fields=['decisionKey','reviewId','reviewSha256','approverLabel',
  'consentPhrase','authorityRef'];
 if(Object.keys(p).length!==fields.length||
    fields.some(f=>!(f in p))||
    typeof p.decisionKey!=='string'||
    !/^orotitan:c8:decision:[a-zA-Z0-9:_-]{8,124}$/.test(p.decisionKey)||
    typeof p.reviewId!=='string'||
    !/^[a-f0-9]{8}-(?:[a-f0-9]{4}-){3}[a-f0-9]{12}$/i.test(p.reviewId)||
    typeof p.reviewSha256!=='string'||!/^[a-f0-9]{64}$/.test(p.reviewSha256)||
    typeof p.approverLabel!=='string'||p.approverLabel.trim().length<4||
    p.approverLabel.length>100||
    typeof p.consentPhrase!=='string'||!/^ACCEPTER J[0-8] SHADOW$/.test(p.consentPhrase)||
    typeof p.authorityRef!=='string'||
    !/^GO_ACCEPT_J0_[A-Z0-9_]{8,120}$/.test(p.authorityRef))
   return fail('J0_C8_DECISION_INVALID',400);
 try{
  const client=getHostedShadowRecoveryClient();
  // Fail closed on the actual DB activation gate as a separate safeguard.
  const control=await client.from('orotitan_j0_c8_acceptance_control_v1')
    .select('enabled,activation_authority_ref')
    .eq('lock_name','GLOBAL_SHADOW').single();
  if(control.error||!control.data||control.data.enabled!==true||
     control.data.activation_authority_ref!==p.authorityRef)
   return fail('J0_C8_ACCEPTANCE_LOCKED',423);
  const rpc=await client.rpc('orotitan_j0_c8_accept_review_v1',{
   p_decision_key:p.decisionKey,
   p_review_id:p.reviewId,p_review_sha256:p.reviewSha256,
   p_approver_label:p.approverLabel.trim(),
   p_confirmation:p.consentPhrase,p_authority_ref:p.authorityRef,
  });
  if(rpc.error){
   const code=(rpc.error.message.match(/J0_C8_[A-Z0-9_]+/)??[])[0];
   return fail(code??'J0_C8_DECISION_REJECTED',409);
  }
  const receipt=rpc.data as Record<string,unknown>|null;
  if(!receipt||receipt.accepted!==true||
    !['ACCEPTED','REPLAYED'].includes(String(receipt.status))||
    typeof receipt.decisionId!=='string')
   return fail('J0_C8_DECISION_RECEIPT_INVALID',503);
  const stored=await client.from('orotitan_j0_c8_decisions_v1')
   .select('decision_id,decision_key,review_id,review_sha256,approver_label,consent_phrase,authority_ref,decision_status,scope')
   .eq('decision_id',receipt.decisionId).single();
  if(stored.error||!stored.data||stored.data.decision_key!==p.decisionKey||
     stored.data.review_id!==p.reviewId||
     stored.data.review_sha256!==p.reviewSha256||
     stored.data.approver_label!==p.approverLabel.trim()||
     stored.data.consent_phrase!==p.consentPhrase||
     stored.data.authority_ref!==p.authorityRef||
     stored.data.decision_status!=='ACCEPTED'||stored.data.scope!=='SHADOW_ONLY')
   return fail('J0_C8_DECISION_READBACK_MISMATCH',503);
  return NextResponse.json({
   ok:true,status:receipt.status,decisionId:receipt.decisionId,
   milestone:receipt.milestone,persistedReadbackVerified:true,
   accepted:true,stageStarted:false,published:false,scope:'SHADOW_ONLY',
  },{headers});
 }catch{return fail('J0_C8_DECISION_UNAVAILABLE',503);}
}
