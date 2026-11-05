import {NextRequest,NextResponse} from 'next/server';
import {requireAdminSession} from '@/lib/auth/admin-session';
import {requireSoloOwnerAttestation,type SoloOwnerAttestation} from
 '@/lib/auth/c8-solo-owner';
import {isHostedC8ReadOnlyEnabled,isHostedShadowResearchEnabled} from
 '@/lib/orotitan-equity/product/hosted-shadow-research-gate';
import {getC8GateReviewClient} from
 '@/lib/orotitan-equity/product/c8-gate-read-server';
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
 const isolatedC8=isHostedC8ReadOnlyEnabled(process.env);
 if(!(isolatedC8||isHostedShadowResearchEnabled(process.env)))
  return fail('SHADOW_ONLY',404);
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
 const mode=p.governanceMode??'STANDARD_TWO_PERSON';
 const allowed=new Set([...fields,'governanceMode']);
 if(Object.keys(p).some(k=>!allowed.has(k))||fields.some(f=>!(f in p))||
    !['STANDARD_TWO_PERSON','SOLO_OWNER'].includes(String(mode))||
    typeof p.decisionKey!=='string'||
    !/^orotitan:c8:decision:[a-zA-Z0-9:_-]{8,124}$/.test(p.decisionKey)||
    typeof p.reviewId!=='string'||
    !/^[a-f0-9]{8}-(?:[a-f0-9]{4}-){3}[a-f0-9]{12}$/i.test(p.reviewId)||
    typeof p.reviewSha256!=='string'||!/^[a-f0-9]{64}$/.test(p.reviewSha256)||
    typeof p.approverLabel!=='string'||p.approverLabel.trim().length<4||
    p.approverLabel.length>100||
    typeof p.consentPhrase!=='string'||
    !/^ACCEPTER J[0-8] SHADOW(?: EN MODE SOLO)?$/.test(p.consentPhrase)||
    typeof p.authorityRef!=='string'||
    !/^GO_ACCEPT_J0_[A-Z0-9_]{8,120}$/.test(p.authorityRef))
   return fail('J0_C8_DECISION_INVALID',400);
 if(isolatedC8&&mode!=='SOLO_OWNER')return fail('J0_C8_SOLO_ONLY',403);
 if(mode==='STANDARD_TWO_PERSON'&&
    /^GO_ACCEPT_J0_SOLO_/.test(p.authorityRef as string))
  return fail('J0_C8_GOVERNANCE_MODE_MISMATCH',409);
 let owner:SoloOwnerAttestation|null=null;
 if(mode==='SOLO_OWNER'){
  try{owner=await requireSoloOwnerAttestation();}
  catch{return fail('OWNER_ATTESTATION_REQUIRED',401);}
  if(!/^GO_ACCEPT_J0_SOLO_[A-Z0-9_]{8,115}$/.test(p.authorityRef as string))
   return fail('J0_C8_SOLO_AUTHORITY_INVALID',400);
 }
 try{
  const client=isolatedC8?getC8GateReviewClient():getHostedShadowRecoveryClient();
  // Fail closed on the actual DB activation gate as a separate safeguard.
  const control=await client.from('orotitan_j0_c8_acceptance_control_v1')
    .select('enabled,activation_authority_ref,solo_owner_enabled,solo_amendment_ref,solo_amendment_sha256')
    .eq('lock_name','GLOBAL_SHADOW').single();
  if(control.error||!control.data||control.data.enabled!==true||
     control.data.activation_authority_ref!==p.authorityRef)
   return fail('J0_C8_ACCEPTANCE_LOCKED',423);
  if(owner&&(control.data.solo_owner_enabled!==true||
    control.data.solo_amendment_ref!==owner.amendmentRef||
    control.data.solo_amendment_sha256!==owner.amendmentSha256))
   return fail('J0_C8_SOLO_AMENDMENT_NOT_RATIFIED',423);
  if(!owner&&control.data.solo_owner_enabled!==false)
   return fail('J0_C8_GOVERNANCE_MODE_MISMATCH',409);
  // A future GO must explicitly unlock THIS milestone, not all J0..J8.
  const review=await client.from('orotitan_j0_c8_reviews_v1')
   .select('milestone_code,review_sha256,governance_mode,auth_provider,owner_user_id,amendment_ref,amendment_sha256')
   .eq('review_id',p.reviewId).single();
  if(review.error||!review.data||
     review.data.review_sha256!==p.reviewSha256||review.data.governance_mode!==mode)
   return fail('J0_C8_REVIEW_NOT_FOUND_OR_HASH_MISMATCH',409);
  if(owner&&(review.data.milestone_code!=='J0'||
    review.data.auth_provider!==owner.authProvider||
    review.data.owner_user_id!==owner.ownerUserId||
    review.data.amendment_ref!==owner.amendmentRef||
    review.data.amendment_sha256!==owner.amendmentSha256))
   return fail('OWNER_ATTESTATION_MISMATCH',403);
  const milestoneControl=await client
   .from('orotitan_j0_c8_milestone_activation_v1')
   .select('enabled,authority_ref')
   .eq('milestone_code',review.data.milestone_code).single();
  if(milestoneControl.error||!milestoneControl.data||
     milestoneControl.data.enabled!==true||
     milestoneControl.data.authority_ref!==p.authorityRef)
   return fail('J0_C8_MILESTONE_ACCEPTANCE_LOCKED',423);
  const expectedConsent=owner?'ACCEPTER J0 SHADOW EN MODE SOLO':
   'ACCEPTER '+review.data.milestone_code+' SHADOW';
  if(p.consentPhrase!==expectedConsent)
   return fail('J0_C8_EXPLICIT_CONFIRMATION_REQUIRED',400);
  const approverLabel=owner?'supabase-auth:'+owner.ownerUserId:
   (p.approverLabel as string).trim();
  const rpcArgs:Record<string,unknown>={
   p_decision_key:p.decisionKey,
   p_review_id:p.reviewId,p_review_sha256:p.reviewSha256,
   p_approver_label:approverLabel,
   p_confirmation:p.consentPhrase,p_authority_ref:p.authorityRef,
  };
  if(owner)Object.assign(rpcArgs,{p_governance_mode:owner.governanceMode,
   p_auth_provider:owner.authProvider,p_owner_user_id:owner.ownerUserId,
   p_amendment_ref:owner.amendmentRef,p_amendment_sha256:owner.amendmentSha256});
  const rpc=await client.rpc('orotitan_j0_c8_accept_review_v1',rpcArgs);
  if(rpc.error){
   const code=(rpc.error.message.match(/J0_C8_[A-Z0-9_]+/)??[])[0];
   return fail(code??'J0_C8_DECISION_REJECTED',409);
  }
  const receipt=rpc.data as Record<string,unknown>|null;
  if(!receipt||receipt.accepted!==true||
    !['ACCEPTED','REPLAYED'].includes(String(receipt.status))||
    typeof receipt.decisionId!=='string'||
    (owner&&typeof receipt.decisionSha256!=='string'))
   return fail('J0_C8_DECISION_RECEIPT_INVALID',503);
  const stored=await client.from('orotitan_j0_c8_decisions_v1')
   .select('decision_id,decision_key,review_id,review_sha256,approver_label,consent_phrase,authority_ref,decision_status,scope,governance_mode,auth_provider,owner_user_id,amendment_ref,amendment_sha256,decision_sha256')
   .eq('decision_id',receipt.decisionId).single();
  if(stored.error||!stored.data||stored.data.decision_key!==p.decisionKey||
     stored.data.review_id!==p.reviewId||
     stored.data.review_sha256!==p.reviewSha256||
     stored.data.approver_label!==approverLabel||
     stored.data.consent_phrase!==p.consentPhrase||
     stored.data.authority_ref!==p.authorityRef||
     stored.data.decision_status!=='ACCEPTED'||stored.data.scope!=='SHADOW_ONLY'||
     stored.data.governance_mode!==mode||
     (owner&&(stored.data.auth_provider!==owner.authProvider||
      stored.data.owner_user_id!==owner.ownerUserId||
      stored.data.amendment_ref!==owner.amendmentRef||
      stored.data.amendment_sha256!==owner.amendmentSha256||
      stored.data.decision_sha256!==receipt.decisionSha256)))
   return fail('J0_C8_DECISION_READBACK_MISMATCH',503);
  return NextResponse.json({
   ok:true,status:receipt.status,decisionId:receipt.decisionId,
   milestone:receipt.milestone,persistedReadbackVerified:true,
   governanceMode:stored.data.governance_mode,
   accepted:true,stageStarted:false,published:false,scope:'SHADOW_ONLY',
  },{headers});
 }catch{return fail('J0_C8_DECISION_UNAVAILABLE',503);}
}
