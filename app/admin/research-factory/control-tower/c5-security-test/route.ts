import {NextRequest,NextResponse} from 'next/server';
import {requireAdminSession} from '@/lib/auth/admin-session';
import {isHostedShadowResearchEnabled} from
 '@/lib/orotitan-equity/product/hosted-shadow-research-gate';
import {getHostedShadowRecoveryClient} from
 '@/lib/orotitan-equity/product/research-inbox-pilot-server';
import {
 verifyHostedJ0C5HttpMatrix,type J0C5HttpObservation,
 type J0C5HttpProbeName,
} from '@/lib/orotitan-equity/product/runfactory-j0-c5-http-verification-v1';

export const runtime='nodejs';
export const dynamic='force-dynamic';
export const maxDuration=90;
const headers={'Cache-Control':'private, no-store'};
const RUN_ID='204390a1-5cba-4b9c-9821-ca0c7df43eca';
const DOSSIER_ID='e30bd3e4-618c-41a2-93b5-015fc2ea708e';
const TARGET='/admin/research-factory/control-tower/record';
const fail=(error:string,status=409)=>NextResponse.json({
 ok:false,error,accepted:false,stageStarted:false,
 blocker:'issue:j0-c5-preview-http-sso-20261031',
},{status,headers});
type HttpProbe=Readonly<{name:J0C5HttpProbeName;origin:string;
 contentType:string;body:string}>;
async function countJournal():Promise<number>{
 const client=getHostedShadowRecoveryClient();
 const r=await client.from('orotitan_j0_events_v1')
  .select('event_id',{count:'exact',head:true});
 if(r.error||r.count===null)throw Error('J0_C5_JOURNAL_READ_FAILED');
 return r.count;
}
async function readRunInvariants():Promise<Readonly<{
 stateVersion:number;status:string;stageCount:number;
 artifactCount:number;snapshotCount:number;
}>>{
 const client=getHostedShadowRecoveryClient();
 const [run,stages,artifacts,snapshots]=await Promise.all([
  client.from('orotitan_runs').select(
   'run_id,run_status,current_stage,state_version'
  ).eq('run_id',RUN_ID).single(),
  client.from('orotitan_run_stages')
   .select('run_id',{count:'exact',head:true}).eq('run_id',RUN_ID),
  client.from('orotitan_artifacts')
   .select('artifact_id',{count:'exact',head:true}).eq('run_id',RUN_ID),
  client.from('research_snapshots')
   .select('snapshot_id',{count:'exact',head:true}).eq('dossier_id',DOSSIER_ID),
 ]);
 if(run.error||!run.data||stages.error||artifacts.error||
    snapshots.error||run.data.run_status!=='CREATED'||
    run.data.current_stage!==null||
    stages.count!==0||artifacts.count!==0||snapshots.count!==0||
    run.data.state_version!==2)throw Error('J0_C5_RUN_INVARIANT_FAILED');
 return {stateVersion:run.data.state_version,status:run.data.run_status,
  stageCount:stages.count,artifactCount:artifacts.count,
  snapshotCount:snapshots.count};
}
async function probe(origin:string,cookie:string,item:HttpProbe)
 :Promise<J0C5HttpObservation>{
 // Fixed same-origin HTTPS destination; never accept a caller-supplied URL.
 const response=await fetch(origin+TARGET,{
  method:'POST',
  headers:{
   'Content-Type':item.contentType,'Accept':'application/json',
   'Origin':item.origin,'Cookie':cookie,
  },
  body:item.body,cache:'no-store',redirect:'manual',
  signal:AbortSignal.timeout(16000),
 });
 const type=response.headers.get('content-type')??'';
 if(!type.toLowerCase().includes('application/json'))
  throw Error('J0_C5_VERCEL_SSO_OR_NONJSON_RESPONSE');
 const raw=await response.text();
 if(raw.length>1500)throw Error('J0_C5_OVERSIZED_RESPONSE');
 let data:unknown;
 try{data=JSON.parse(raw);}catch{throw Error('J0_C5_INVALID_RESPONSE_JSON');}
 if(!data||typeof data!=='object'||Array.isArray(data))
  throw Error('J0_C5_INVALID_RESPONSE_SHAPE');
 const value=data as Record<string,unknown>;
 return {name:item.name,status:response.status,
  error:typeof value.error==='string'?value.error:'',
  accepted:value.accepted===false?false:true,
  stageStarted:value.stageStarted===false?false:true,
  cacheControl:response.headers.get('cache-control')??'',
 };
}
export async function POST(req:NextRequest){
 if(!isHostedShadowResearchEnabled(process.env))return fail('SHADOW_ONLY',404);
 try{await requireAdminSession();}catch{return fail('UNAUTHORIZED',401);}
 if(req.headers.get('content-type')?.split(';')[0]?.trim()!=='application/json')
  return fail('J0_C5_REQUEST_INVALID',400);
 const fixedHostname=process.env.VERCEL_URL??'';
 const gitSha=process.env.VERCEL_GIT_COMMIT_SHA??'';
 if(!/^orotitan-research-shadow-preview-[a-z0-9-]+-orotitan\.vercel\.app$/.test(fixedHostname)||
    !/^[a-f0-9]{40}$/.test(gitSha))
  return fail('J0_C5_CANONICAL_PREVIEW_UNAVAILABLE',409);
 const fixedOrigin='https://'+fixedHostname;
 if(new URL(req.url).origin!==fixedOrigin||
    req.headers.get('origin')!==fixedOrigin)
  return fail('J0_C5_CANONICAL_ORIGIN_REQUIRED',403);
 const cookie=req.headers.get('cookie')??'';
 if(!cookie.includes('orotitan_admin_session='))
  return fail('J0_C5_ADMIN_COOKIE_NOT_PRESENT',401);
 try{
  const input=await req.text();
  if(input.trim()!=='{}')return fail('J0_C5_REQUEST_INVALID',400);
  const start=await countJournal();
  const baseline=await readRunInvariants();
  // Intentionally invalid bodies: no test request could insert a J0 event
  // even if an upstream authorization guard regressed.
  const malformed=JSON.stringify({unexpected:'negative-only'});
  const fakeGo=JSON.stringify({
   eventKey:'orotitan:j0:c5-negative-forged-acceptance',
   milestoneCode:'J0',eventType:'MILESTONE_ACCEPTED',
   title:'Unapproved milestone acceptance must be denied',
   evidenceRef:'test:j0-c5-fake-go',runId:null,details:{criterionIndex:5},
  });
  const tests:HttpProbe[]=[
   {name:'CROSS_ORIGIN',origin:'https://untrusted.invalid',
    contentType:'application/json',body:malformed},
   {name:'UNSUPPORTED_CONTENT_TYPE',origin:fixedOrigin,
    contentType:'text/plain',body:malformed},
   {name:'UNEXPECTED_FIELDS',origin:fixedOrigin,
    contentType:'application/json',body:malformed},
   {name:'FORGED_ACCEPTANCE',origin:fixedOrigin,
    contentType:'application/json',body:fakeGo},
  ];
  const observed: J0C5HttpObservation[]=[];
  for(const item of tests)
   observed.push(await probe(fixedOrigin,cookie,item));
  const verified=verifyHostedJ0C5HttpMatrix(observed);
  const end=await countJournal();
  const after=await readRunInvariants();
  if(start!==end||
     baseline.stateVersion!==after.stateVersion||
     baseline.status!==after.status||
     after.stageCount!==0||after.artifactCount!==0||
     after.snapshotCount!==0)
   throw Error('J0_C5_UNEXPECTED_DB_MUTATION');
  // After independent negative-request checks, persist one immutable
  // C5 proof in SHADOW. This is a claim with exact diagnostic data, NOT an
  // automatic verification state, gate acceptance, or blocker resolution.
  const fingerprint=gitSha.slice(0,12);
  const eventKey='orotitan:j0:c5-hosted-http-'+fingerprint;
  const evidenceRef='test:j0-c5-hosted-http-'+fingerprint;
  const details={
   criterionIndex:5,scope:'SHADOW_ONLY',
   testKind:'REAL_HTTP_SAME_PREVIEW_WITH_USER_AUTH_SESSION',
   deploymentCommit:gitSha,applicationResponseMatrix:
     verified.results.map(x=>({name:x.name,status:x.status,error:x.error})),
   all4ApplicationDenialsVerified:true,
   noUnintendedWrites:true,vercelSSOStayedEnabled:true,
   runRemainedCreated:true,stageStarted:false,published:false,
   acceptance:'NOT_GRANTED',
  };
  const client=getHostedShadowRecoveryClient();
  const write=await client.rpc('orotitan_j0_record_event_v1',{
   p_event_key:eventKey,p_milestone_code:'J0',
   p_event_type:'EVIDENCE_RECORDED',
   p_title:'C5 refus HTTP applicatifs vérifiés derrière SSO',
   p_evidence_ref:evidenceRef,p_run_id:null,
   p_operator_details:details,
  });
  if(write.error||!write.data||
     typeof write.data.event_id!=='string'||
     !['CREATED','REPLAYED'].includes(write.data.status))
   throw Error('J0_C5_EVIDENCE_WRITE_FAILED');
  const receipt=await client.from('orotitan_j0_events_v1')
   .select('event_id,event_key,milestone_code,event_type,evidence_ref,operator_details,verification_state')
   .eq('event_id',write.data.event_id).single();
  if(receipt.error||!receipt.data||
     receipt.data.event_key!==eventKey||
     receipt.data.milestone_code!=='J0'||
     receipt.data.event_type!=='EVIDENCE_RECORDED'||
     receipt.data.evidence_ref!==evidenceRef||
     receipt.data.operator_details?.all4ApplicationDenialsVerified!==true||
     receipt.data.operator_details?.noUnintendedWrites!==true||
     receipt.data.verification_state!=='REPORTED')
   throw Error('J0_C5_EVIDENCE_READBACK_FAILED');
  return NextResponse.json({
   ok:true,applicationHttpMatrixVerified:true,
   cases:verified.results.map(x=>({name:x.name,
    status:x.status,error:x.error})),
   eventId:receipt.data.event_id,
   persistedReadbackVerified:true,
   evidenceState:'REPORTED',accepted:false,stageStarted:false,
   published:false,blockerClosure:'REQUIRES_INDEPENDENT_REVIEW',
   scope:'SHADOW_ONLY',
  },{headers});
 }catch(e){
  const code=e instanceof Error?e.message:'';
  const safe=/^J0_C5_[A-Z0-9_:]+$/.test(code)?
   code:'J0_C5_HOSTED_TEST_NOT_VERIFIED';
  return fail(safe,409);
 }
}
