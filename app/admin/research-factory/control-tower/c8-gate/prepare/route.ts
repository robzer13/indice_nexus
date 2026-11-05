import {NextRequest,NextResponse} from 'next/server';
import {createHash} from 'node:crypto';
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
const fail=(error:string,status=409)=>NextResponse.json(
 {ok:false,error,accepted:false,stageStarted:false,published:false},
 {status,headers});
export async function POST(req:NextRequest){
 const isolatedC8=isHostedC8ReadOnlyEnabled(process.env);
 if(!(isolatedC8||isHostedShadowResearchEnabled(process.env)))
  return fail('SHADOW_ONLY',404);
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
 const baseKeys=['reviewKey','milestone','auditRef','auditSha256','reviewerLabel','evidenceEventIds'];
 const mode=v.governanceMode??'STANDARD_TWO_PERSON';
 const allowed=new Set([...baseKeys,'governanceMode']);
 if(baseKeys.some(k=>!(k in v))||Object.keys(v).some(k=>!allowed.has(k))||
    !['STANDARD_TWO_PERSON','SOLO_OWNER'].includes(String(mode)))
  return fail('J0_C8_PREPARE_INVALID',400);
 if(typeof v.reviewKey!=='string'||
    !/^orotitan:c8:review:[a-zA-Z0-9:_-]{8,124}$/.test(v.reviewKey)||
    typeof v.milestone!=='string'||!/^J[0-8]$/.test(v.milestone)||
    typeof v.auditRef!=='string'||!/^https:\/\/github\.com\/robzer13\/indice_nexus\/blob\/[a-f0-9]{40}\/docs\/audits\/[\w.-]+\.md$/.test(v.auditRef)||
    typeof v.auditSha256!=='string'||!/^[a-f0-9]{64}$/.test(v.auditSha256)||
    typeof v.reviewerLabel!=='string'||v.reviewerLabel.trim().length<4||
    v.reviewerLabel.length>100||
    !Array.isArray(v.evidenceEventIds)||v.evidenceEventIds.length!==7||
    v.evidenceEventIds.some(id=>typeof id!=='string'||
     !/^[a-f0-9]{8}-(?:[a-f0-9]{4}-){3}[a-f0-9]{12}$/i.test(id)))
   return fail('J0_C8_PREPARE_INVALID',400);
 if(isolatedC8&&mode!=='SOLO_OWNER')return fail('J0_C8_SOLO_ONLY',403);
 if(mode==='SOLO_OWNER'&&v.milestone!=='J0')return fail('J0_C8_SOLO_J0_ONLY',403);
 let owner:SoloOwnerAttestation|null=null;
 if(mode==='SOLO_OWNER'){
  try{owner=await requireSoloOwnerAttestation();}
  catch{return fail('OWNER_ATTESTATION_REQUIRED',401);}
 }
 try{
  // Audit bytes are fetched from a FIXED official repository at an
  // immutable commit. No caller-controlled hostname, redirects or URL query.
  const match=(v.auditRef as string).match(
   /^https:\/\/github\.com\/robzer13\/indice_nexus\/blob\/([a-f0-9]{40})\/docs\/audits\/([\w.-]+\.md)$/);
  if(!match)return fail('J0_C8_AUDIT_REFERENCE_INVALID',400);
  const rawUrl='https://raw.githubusercontent.com/robzer13/indice_nexus/'+
   match[1]+'/docs/audits/'+match[2];
  let source:Response;
  try{
   source=await fetch(rawUrl,{
    cache:'no-store',redirect:'error',
    headers:{Accept:'text/plain'},
    signal:AbortSignal.timeout(15000),
   });
  }catch{return fail('J0_C8_AUDIT_SOURCE_UNAVAILABLE',503);}
  if(!source.ok)return fail('J0_C8_AUDIT_SOURCE_UNAVAILABLE',503);
  const expectedLength=Number(source.headers.get('content-length')??'0');
  if(expectedLength>262144)return fail('J0_C8_AUDIT_TOO_LARGE',413);
  const bytes=Buffer.from(await source.arrayBuffer());
  if(bytes.length===0||bytes.length>262144)
   return fail('J0_C8_AUDIT_TOO_LARGE',413);
  const observedHash=createHash('sha256').update(bytes).digest('hex');
  if(observedHash!==v.auditSha256)
   return fail('J0_C8_AUDIT_HASH_MISMATCH',409);
  const client=isolatedC8?getC8GateReviewClient():getHostedShadowRecoveryClient();
  const reviewerLabel=owner?'supabase-auth:'+owner.ownerUserId:
   (v.reviewerLabel as string).trim();
  const rpcArgs:Record<string,unknown>={
   p_review_key:v.reviewKey,p_milestone:v.milestone,p_audit_ref:v.auditRef,
   p_audit_sha256:v.auditSha256,p_reviewer_label:reviewerLabel,
   p_evidence_ids:v.evidenceEventIds,
  };
  if(owner)Object.assign(rpcArgs,{p_governance_mode:owner.governanceMode,
   p_auth_provider:owner.authProvider,p_owner_user_id:owner.ownerUserId,
   p_amendment_ref:owner.amendmentRef,p_amendment_sha256:owner.amendmentSha256});
  const rpc=await client.rpc('orotitan_j0_c8_prepare_review_v1',rpcArgs);
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
   .select('review_id,review_key,milestone_code,audit_ref,audit_sha256,reviewer_label,evidence_event_ids,review_sha256,journal_watermark,governance_mode,auth_provider,owner_user_id,amendment_ref,amendment_sha256')
   .eq('review_id',reply.reviewId).single();
  if(stored.error||!stored.data||
     stored.data.review_key!==v.reviewKey||
     stored.data.milestone_code!==v.milestone||
     stored.data.audit_ref!==v.auditRef||
     stored.data.audit_sha256!==v.auditSha256||
     stored.data.reviewer_label!==reviewerLabel||
     stored.data.review_sha256!==reply.reviewSha256||
     JSON.stringify(stored.data.evidence_event_ids)!==JSON.stringify(v.evidenceEventIds)||
     stored.data.governance_mode!==mode||
     (owner&&(stored.data.auth_provider!==owner.authProvider||
      stored.data.owner_user_id!==owner.ownerUserId||
      stored.data.amendment_ref!==owner.amendmentRef||
      stored.data.amendment_sha256!==owner.amendmentSha256)))
   return fail('J0_C8_REVIEW_READBACK_MISMATCH',503);
  return NextResponse.json({
   ok:true,status:reply.status,reviewId:reply.reviewId,
   reviewSha256:reply.reviewSha256,
   journalWatermark:stored.data.journal_watermark,
   governanceMode:stored.data.governance_mode,
   persistedReadbackVerified:true,accepted:false,
   stageStarted:false,published:false,scope:'SHADOW_ONLY',
  },{headers});
 }catch{return fail('J0_C8_PREPARE_UNAVAILABLE',503);}
}
