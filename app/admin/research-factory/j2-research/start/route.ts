import {NextRequest,NextResponse} from 'next/server';
import {requireAdminSession} from '@/lib/auth/admin-session';
import {isHostedJ2ResearchStageStartEnabled} from
 '@/lib/orotitan-equity/product/hosted-shadow-research-gate';
import {getJ2ResearchStageStartShadowClient} from
 '@/lib/orotitan-equity/product/runfactory-j2-stage-start-shadow-server';
import {buildJ2ResearchStartRpcPlan} from
 '@/lib/orotitan-equity/product/runfactory-j2-research-start-plan-v1';
import {GET as verifiedResearchPreflight} from
 '@/app/admin/research-factory/j2-research/preflight/route';
import type {J2ResearchStartDecision} from
 '@/lib/orotitan-equity/product/runfactory-j2-research-admission-v1';

/** Explicit controlled start ONLY in a separately isolated J2 Preview.
 * Never deploy with the C8/J1 environment. No Research execution or artifacts
 * are created by this route. Unknown responses require read-only reconciliation
 * and never automatically repeat the canonical RPC.
 */
export const runtime='nodejs';
export const dynamic='force-dynamic';
export const maxDuration=60;
const headers={'Cache-Control':'private, no-store','X-Content-Type-Options':'nosniff'};
const UUID=/^[0-9a-f]{8}-(?:[0-9a-f]{4}-){3}[0-9a-f]{12}$/i;
const fail=(reason:string):never=>{throw Error(reason);};
function object(value:unknown):Record<string,unknown>{
 if(!value||typeof value!=='object'||Array.isArray(value))
  return fail('J2_STAGE_START_INVALID_RECORD');
 return value as Record<string,unknown>;
}
function string(value:unknown):string{
 if(typeof value!=='string'||!value)return fail('J2_STAGE_START_INVALID_FIELD');
 return value;
}
function rows(value:unknown,max:number):Record<string,unknown>[]{
 if(!Array.isArray(value)||value.length>max)
  return fail('J2_STAGE_START_INVALID_READBACK');
 return (value as unknown[]).map(object);
}
const blocked=(code:string,status=409,maybeMutated=false)=>NextResponse.json({
 ok:false,error:code,stageStartAuthorized:false,
 registryMayHaveBeenMutated:maybeMutated,
 checkPersistedRunBeforeAnyRetry:maybeMutated,
 environment:'SHADOW_ONLY',
},{status,headers});

export async function POST(req:NextRequest){
 if(!isHostedJ2ResearchStageStartEnabled(process.env))
  return blocked('J2_STAGE_START_NOT_ENABLED',404);
 try{await requireAdminSession();}
 catch{return blocked('UNAUTHORIZED',401);}
 if(req.headers.get('origin')!==new URL(req.url).origin)
  return blocked('ORIGIN_MISMATCH',403);
 if(req.headers.get('content-type')?.split(';')[0]?.trim()!=='application/json')
  return blocked('REQUEST_INVALID',400);
 let body:Record<string,unknown>;
 try{
  const raw=await req.text();
  if(Buffer.byteLength(raw,'utf8')>1100||raw.length<5)
   return blocked('REQUEST_INVALID',400);
  body=object(JSON.parse(raw) as unknown);
  const expectedFields=['runId','expectedRunStateVersion',
   'contractSetSha256','stageContractSha256','confirmation'];
  if(Object.keys(body).length!==expectedFields.length||
     Object.keys(body).some(k=>!expectedFields.includes(k)))
   return blocked('REQUEST_INVALID',400);
 }catch{return blocked('REQUEST_INVALID',400);}
 try{
  const runId=string(body.runId);
  if(!UUID.test(runId)||!Number.isSafeInteger(body.expectedRunStateVersion)||
     typeof body.expectedRunStateVersion!=='number')
   return blocked('J2_STAGE_START_REQUEST_INVALID');
  const target=new URL(req.url);
  target.pathname='/admin/research-factory/j2-research/preflight';
  target.search='?runId='+encodeURIComponent(runId);
  // Independent authenticated Registry read before every stage-start attempt.
  const pre=await verifiedResearchPreflight(new NextRequest(target,{
   method:'GET',headers:{cookie:req.headers.get('cookie')??''},
  }));
  if(!pre.ok)return blocked('J2_STAGE_START_PREFLIGHT_BLOCKED');
  const verdict=object(await pre.json() as unknown);
  if(verdict.ok!==true||verdict.eligible!==true||
     verdict.stageAdmissionPreflight!=='J2_RESEARCH_START_PREFLIGHT_PASS'||
     verdict.reasonCode!=='J2_RESEARCH_START_PREFLIGHT_PASS'||
     verdict.runId!==runId||verdict.environment!=='SHADOW_ONLY'||
     verdict.readOnly!==true||
     verdict.stageStartAuthorized!==false||verdict.rpcExecuted!==false||
     verdict.researchExecuted!==false||verdict.publicationAuthorized!==false||
     verdict.expectedRunStateVersion!==body.expectedRunStateVersion||
     verdict.contractSetSha256!==body.contractSetSha256||
     verdict.stageContractSha256!==body.stageContractSha256)
   return blocked('J2_STAGE_START_PREFLIGHT_MISMATCH');
  const plan=buildJ2ResearchStartRpcPlan(
   verdict as unknown as J2ResearchStartDecision);
  if(body.confirmation!==plan.confirmation)
   return blocked('J2_EXPLICIT_GO_REQUIRED',403);

  const client=getJ2ResearchStageStartShadowClient();
  // Only one RPC per approved request; no transport-level POST retry.
  const rpc=await client.rpc(plan.rpc,plan.args);
  if(rpc.error)
   return blocked('J2_STAGE_START_RPC_FAILED',503,true);
  const reply=object(rpc.data);
  if(reply.run_id!==runId||reply.stage!=='RESEARCH'||
     reply.idempotent_replay!==false||
     reply.stage_revision!==1||reply.stage_state_version!==1)
   return blocked('J2_STAGE_START_RPC_RESULT_UNVERIFIED',503,true);

  const [runQ,stageQ,eventsQ,snapshotQ]=await Promise.all([
   client.from('orotitan_runs').select(
    'run_id,issuer_id,security_id,dossier_id,run_status,run_type,current_stage,state_version,contract_set_sha256,data_cutoff,baseline_snapshot_id,published_at'
   ).eq('run_id',runId).single(),
   client.from('orotitan_run_stages').select(
    'run_id,stage_code,stage_contract_name,stage_contract_version,stage_contract_sha256,lifecycle_status,stage_revision,state_version,handoff_gate_name,handoff_gate_state,active_manifest_artifact_id,active_manifest_version,active_manifest_kind,started_at'
   ).eq('run_id',runId).eq('stage_code','RESEARCH').single(),
   client.from('orotitan_run_events').select(
    'run_id,stage_code,event_type,idempotency_key,request_fingerprint_sha256'
   ).eq('run_id',runId).eq('event_type','STAGE_STARTED').limit(2),
   client.from('research_snapshots').select('snapshot_id')
    .eq('dossier_id',string(object(verdict.runContext).dossierId)).limit(1),
  ]);
  if(runQ.error||stageQ.error||eventsQ.error||snapshotQ.error)
   return blocked('J2_STAGE_START_POSTCOMMIT_READ_FAILED',503,true);
  const run=object(runQ.data),stage=object(stageQ.data),
    events=rows(eventsQ.data,2),snapshots=rows(snapshotQ.data,1),
    identity=object(verdict.runContext);
  if(run.run_id!==runId||run.issuer_id!==identity.issuerId||
     run.security_id!==identity.securityId||run.dossier_id!==identity.dossierId||
     run.data_cutoff!==identity.dataCutoff||
     run.contract_set_sha256!==verdict.contractSetSha256||
     run.run_status!=='ACTIVE'||run.run_type!=='INITIAL'||
     run.current_stage!=='RESEARCH'||
     run.state_version!==Number(verdict.expectedRunStateVersion)+1||
     run.baseline_snapshot_id!==null||run.published_at!==null||
     stage.run_id!==runId||stage.stage_code!=='RESEARCH'||
     stage.stage_contract_name!==plan.args.p_stage_contract_name||
     stage.stage_contract_version!==plan.args.p_stage_contract_version||
     stage.stage_contract_sha256!==plan.args.p_stage_contract_sha256||
     stage.lifecycle_status!=='IN_PROGRESS'||
     stage.stage_revision!==1||stage.state_version!==1||
     stage.handoff_gate_name!=='READY_FOR_DEEP_DIVE'||
     stage.handoff_gate_state!=='NOT_EVALUATED'||
     stage.active_manifest_artifact_id!==null||
     stage.active_manifest_version!==null||
     stage.active_manifest_kind!==null||
     typeof stage.started_at!=='string'||!stage.started_at||
     events.length!==1||
     events[0].idempotency_key!==plan.args.p_idempotency_key||
     events[0].request_fingerprint_sha256!==
       plan.args.p_request_fingerprint_sha256||
     snapshots.length!==0)
   return blocked('J2_STAGE_START_POSTCOMMIT_MISMATCH',503,true);
  return NextResponse.json({
   ok:true,status:'RESEARCH_STARTED',
   runId,stage:'RESEARCH',stageLifecycle:'IN_PROGRESS',
   runStatus:'ACTIVE',runStateVersion:run.state_version,
   stageStateVersion:stage.state_version,
   contractSetSha256:verdict.contractSetSha256,
   stageContractSha256:plan.args.p_stage_contract_sha256,
   registryReadbackVerified:true,
   researchExecuted:false,artifactsPersisted:0,
   snapshotCreated:false,publicationAuthorized:false,
   environment:'SHADOW_ONLY',productionTouched:false,
  },{headers});
 }catch{
  return blocked('J2_STAGE_START_UNEXPECTED_FAILURE',503,true);
 }
}
