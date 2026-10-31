import {NextResponse,type NextRequest} from 'next/server';
import {requireAdminSession} from '@/lib/auth/admin-session';
import {isHostedShadowResearchEnabled} from
 '@/lib/orotitan-equity/product/hosted-shadow-research-gate';
import {getHostedShadowRecoveryClient} from
 '@/lib/orotitan-equity/product/research-inbox-pilot-server';
import {verifyAllRunFactoryPinnedBytes} from
 '@/lib/orotitan-equity/product/runfactory-p0e-byte-gate';
import {buildP0EShadowInitialPlan,buildP0EShadowRpcArgs} from
 '@/lib/orotitan-equity/product/runfactory-p0e-shadow-plan-v1';

export const runtime='nodejs';
export const dynamic='force-dynamic';
export const maxDuration=120;
const headers={'Cache-Control':'private, no-store'};
const UUID=/^[a-f0-9]{8}-(?:[a-f0-9]{4}-){3}[a-f0-9]{12}$/i;
const fail=(s:string):never=>{throw Error(s);};
function obj(x:unknown):Record<string,unknown>{
 if(!x||typeof x!=='object'||Array.isArray(x))
  fail('P0E_READBACK_INVALID');
 return x as Record<string,unknown>;
}
function str(x:unknown):string{
 if(typeof x!=='string')fail('P0E_READBACK_INVALID');
 return x as string;
}
function checkedUuid(x:unknown):string{
 const v=str(x);
 if(!UUID.test(v))fail('P0E_READBACK_INVALID');
 return v;
}
export async function POST(req:NextRequest){
 if(!isHostedShadowResearchEnabled(process.env))
  return NextResponse.json({ok:false,error:'SHADOW_ONLY'},{status:404,headers});
 try{await requireAdminSession();}
 catch{return NextResponse.json({ok:false,error:'UNAUTHORIZED'},{status:401,headers});}
 if(req.headers.get('origin')!==new URL(req.url).origin)
  return NextResponse.json({ok:false,error:'ORIGIN_MISMATCH'},{status:403,headers});
 if(req.headers.get('content-type')?.split(';')[0]?.trim()!=='application/json')
  return NextResponse.json({ok:false,error:'REQUEST_INVALID'},{status:400,headers});
 let data:unknown;
 try{
  const raw=await req.text();
  if(raw.length<5||Buffer.byteLength(raw,'utf8')>1600)
   throw Error('REQUEST_INVALID');
  data=JSON.parse(raw) as unknown;
 }catch{
  return NextResponse.json({ok:false,error:'REQUEST_INVALID'},{status:400,headers});
 }
 try{
  const p=obj(data);
  if(p.profileId!=='OROTITAN_V3_1'||p.intent!=='INITIAL_RUN_SHADOW_ONLY')
   fail('P0E_EXPLICIT_PROFILE_OR_INTENT_REQUIRED');
  const issuerId=checkedUuid(p.issuerId),
   securityId=checkedUuid(p.securityId),
   dossierId=checkedUuid(p.dossierId);
  const isin=str(p.isin),exchange=str(p.exchange),
   dataCutoff=str(p.dataCutoff);
  const confirmation=str(p.confirmation);
  if(confirmation!=='GO P0-E SHADOW '+isin.toUpperCase()+' '+dataCutoff)
   fail('P0E_EXPLICIT_GO_REQUIRED');
  const client=getHostedShadowRecoveryClient();
  // Do not trust previous P0-B readings: identity and lineage are re-read.
  const [issuerQ,securityQ,dossierQ,runsQ]=await Promise.all([
   client.from('issuers').select('issuer_id,display_name').eq('issuer_id',issuerId).single(),
   client.from('securities')
    .select('security_id,issuer_id,isin,exchange,primary_listing')
    .eq('security_id',securityId).single(),
   client.from('research_dossiers')
    .select('dossier_id,issuer_id,active,current_snapshot_id')
    .eq('dossier_id',dossierId).single(),
   client.from('orotitan_runs')
    .select('run_id,creation_idempotency_key,contract_set_sha256,run_status')
    .eq('issuer_id',issuerId).eq('data_cutoff',dataCutoff).limit(20),
  ]);
  if(issuerQ.error||securityQ.error||dossierQ.error||runsQ.error)
   fail('P0E_SHADOW_IDENTITY_READ_FAILED');
  const rawRuns=runsQ.data;
  if(!Array.isArray(rawRuns)||rawRuns.length>=20)
   fail('P0E_SHADOW_IDENTITY_READ_FAILED');
  const issuer=obj(issuerQ.data),security=obj(securityQ.data),
   dossier=obj(dossierQ.data);
  if(issuer.issuer_id!==issuerId||!str(issuer.display_name).trim()||
     security.security_id!==securityId||
     security.issuer_id!==issuerId||
     security.isin!==isin.toUpperCase()||
     security.exchange!==exchange.toUpperCase()||
     security.primary_listing!==true||
     dossier.dossier_id!==dossierId||
     dossier.issuer_id!==issuerId||
     dossier.active!==true||dossier.current_snapshot_id!==null)
   fail('P0E_SHADOW_IDENTITY_LINEAGE_MISMATCH');
  const plan=buildP0EShadowInitialPlan({
   issuerId,securityId,dossierId,isin,exchange,dataCutoff,
   currentSnapshotId:null,
  });
  const nonCancelled=rawRuns.filter(r=>r.run_status!=='CANCELLED');
  if(nonCancelled.length>1||
    nonCancelled.some(r=>r.creation_idempotency_key!==plan.creationIdempotencyKey||
     r.contract_set_sha256!==plan.contractSetSha256))
   fail('P0E_EXISTING_RUN_CONFLICT');
  // Mandatory original content bytes and Git blob hashes. No cached
  // structural-only check is accepted as sufficient admission proof.
  const verified=await verifyAllRunFactoryPinnedBytes('OROTITAN_V3_1');
  const args=buildP0EShadowRpcArgs(plan,verified);
  const rpc=await client.rpc('orotitan_shadow_create_initial_run_v3_1',args);
  if(rpc.error){
   const error=String(rpc.error.message??'');
   const safe=(error.match(/P0E_[A-Z0-9_]+/)??[])[0];
   return NextResponse.json({ok:false,error:safe??'P0E_RPC_FAILED',
    registryMayHaveBeenMutated:true,
    checkIdempotencyBeforeRetry:true},{status:safe?409:503,headers});
  }
  const result=obj(rpc.data);
  const runId=checkedUuid(result.run_id);
  if(!['CREATED','REPLAYED'].includes(str(result.status))||
     result.issuer_id!==issuerId||result.security_id!==securityId||
     result.dossier_id!==dossierId||
     result.contract_set_sha256!==plan.contractSetSha256||
     result.atomic_creation_and_binding!==true||
     result.run_stage_started!==false||
     result.snapshot_created!==false||
     result.publication_authorized!==false)
    fail('P0E_RPC_RESULT_INVARIANT_MISMATCH');
  const [runQ,auditQ,eventsQ,stagesQ,snapshotsQ]=await Promise.all([
   client.from('orotitan_runs').select(
    'run_id,issuer_id,security_id,dossier_id,run_type,run_status,current_stage,state_version,data_cutoff,contract_set_sha256,process_version,pilotage_contract_version,baseline_snapshot_id,published_at'
   ).eq('run_id',runId).single(),
   client.from('orotitan_p0e_run_admission_receipts_v1').select(
    'run_id,creation_idempotency_key,profile_id,contract_set_sha256,publication_authorized'
   ).eq('run_id',runId).single(),
   client.from('orotitan_run_events').select('event_type')
    .eq('run_id',runId).limit(20),
   client.from('orotitan_run_stages').select('stage_id')
    .eq('run_id',runId).limit(1),
   client.from('research_snapshots').select('snapshot_id')
    .eq('dossier_id',dossierId).limit(1),
  ]);
  if(runQ.error||auditQ.error||eventsQ.error||stagesQ.error||
     snapshotsQ.error)fail('P0E_POSTCOMMIT_READBACK_FAILED');
  const events=eventsQ.data,stages=stagesQ.data,snapshots=snapshotsQ.data;
  if(!Array.isArray(events)||!Array.isArray(stages)||
     !Array.isArray(snapshots))fail('P0E_POSTCOMMIT_READBACK_FAILED');
  const run=obj(runQ.data),audit=obj(auditQ.data);
  const eventTypes=events.map(r=>r.event_type);
  if(run.run_id!==runId||run.issuer_id!==issuerId||
     run.security_id!==securityId||run.dossier_id!==dossierId||
     run.run_type!=='INITIAL'||run.run_status!=='CREATED'||
     run.current_stage!==null||Number(run.state_version)<2||
     run.data_cutoff!==dataCutoff||
     run.contract_set_sha256!==plan.contractSetSha256||
     run.process_version!=='3.1'||run.pilotage_contract_version!=='3.1'||
     run.baseline_snapshot_id!==null||run.published_at!==null||
     audit.run_id!==runId||audit.creation_idempotency_key!==plan.creationIdempotencyKey||
     audit.contract_set_sha256!==plan.contractSetSha256||
     audit.profile_id!=='OROTITAN_V3_1'||audit.publication_authorized!==false||
     eventTypes.length!==2||!eventTypes.includes('RUN_CREATED')||
     !eventTypes.includes('RUN_IDENTITY_BOUND')||
     stages.length!==0||snapshots.length!==0)
   fail('P0E_POSTCOMMIT_STATE_MISMATCH');
  return NextResponse.json({
   ok:true,status:result.status,runId,
   issuerId,securityId,dossierId,
   dataCutoff,runType:'INITIAL',runStatus:'CREATED',
   profileId:'OROTITAN_V3_1',
   contractSetSha256:plan.contractSetSha256,
   all15OriginalContractBytesVerified:true,
   receiptPersisted:true,runIdentityBound:true,
   stageStarted:false,snapshotCreated:false,published:false,
   persistedReadbackVerified:true,environment:'SHADOW_ONLY',
   productionTouched:false,
  },{headers});
 }catch(e){
  const code=e instanceof Error?e.message:'';
  const safe=/^(P0E_|INVALID_ISIN)/.test(code);
  return NextResponse.json({ok:false,
   error:safe?code:'P0E_GATE_OR_POSTCOMMIT_FAILURE',
   checkIdempotencyBeforeRetry:true,
  },{status:safe?409:503,headers});
 }
}
