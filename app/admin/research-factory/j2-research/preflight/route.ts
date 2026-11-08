import {NextResponse,type NextRequest} from 'next/server';
import {GET as readVerifiedJ1Handoff} from
 '@/app/admin/research-factory/run-preflight/handoff/route';
import {getJ1RunFactoryShadowClient} from
 '@/lib/orotitan-equity/product/runfactory-j1-shadow-server';
import {getJ2ResearchStageStartShadowClient} from
 '@/lib/orotitan-equity/product/runfactory-j2-stage-start-shadow-server';
import {isHostedJ1RunFactoryEnabled} from
 '@/lib/orotitan-equity/product/hosted-shadow-research-gate';
import {assessJ2ResearchStart} from
 '@/lib/orotitan-equity/product/runfactory-j2-research-admission-v1';

/** J2 preliminary admission: reuses J1's authenticated, fail-closed Registry
 * proof, then independently checks the fresh run state and exact pinned
 * Research contract. Does not create a stage, RPC or Research artifact.
 */
export const runtime='nodejs';
export const dynamic='force-dynamic';
const headers={'Cache-Control':'private, no-store','X-Content-Type-Options':'nosniff'};
function asRecord(x:unknown):Record<string,unknown>{
 if(!x||typeof x!=='object'||Array.isArray(x))throw Error('J2_BAD_RECORD');
 return x as Record<string,unknown>;
}
function items(x:unknown):Record<string,unknown>[]{
 if(!Array.isArray(x)||x.length>1)throw Error('J2_BAD_COLLECTION');
 return (x as unknown[]).map(asRecord);
}
function str(x:unknown):string{
 if(typeof x!=='string'||!x)throw Error('J2_BAD_STRING');
 return x;
}
export async function GET(req:NextRequest){
 const original=await readVerifiedJ1Handoff(req);
 // Preserve authentication, wrong-environment, bad-ID and failed-proof stops.
 if(!original.ok)return original;
 try{
  const trusted=asRecord(await original.json());
  if(trusted.ok!==true||
     trusted.routingStatus!=='RUN_READY_FOR_RESEARCH_PREFLIGHT'||
     trusted.registryReadbackVerified!==true||
     trusted.contractReceiptVerified!==true||
     trusted.researchStageAdmission!==false||
     trusted.researchStageStarted!==false||
     trusted.snapshotCreated!==false||trusted.published!==false||
     trusted.environment!=='SHADOW_ONLY'||trusted.productionTouched!==false)
    throw Error('J2_J1_HANDOFF_INVARIANTS');
  const runId=str(trusted.runId);
  const client=isHostedJ1RunFactoryEnabled(process.env)?
   getJ1RunFactoryShadowClient():getJ2ResearchStageStartShadowClient();
  // Fresh post-handoff read. Every later stage-start RPC must still perform
  // its own transactional state-version compare-and-swap.
  const [runQ,stageQ,snapshotQ]=await Promise.all([
   client.from('orotitan_runs').select(
    'run_id,issuer_id,security_id,dossier_id,entry_path,canonical_mode,run_type,run_status,current_stage,baseline_snapshot_id,published_at,state_version,data_cutoff,process_version,pilotage_contract_version,contract_set_sha256,contract_pins'
   ).eq('run_id',runId).single(),
   client.from('orotitan_run_stages').select('stage_code').eq('run_id',runId).limit(1),
   client.from('research_snapshots').select('snapshot_id')
    .eq('dossier_id',str(trusted.dossierId)).limit(1),
  ]);
  if(runQ.error||stageQ.error||snapshotQ.error)
   throw Error('J2_SHADOW_READ_FAILED');
  const run=asRecord(runQ.data),stages=items(stageQ.data),
   snapshots=items(snapshotQ.data);
  const identityVerified=run.run_id===runId&&
   run.issuer_id===trusted.issuerId&&
   run.security_id===trusted.securityId&&
   run.dossier_id===trusted.dossierId&&
   run.data_cutoff===trusted.dataCutoff&&
   run.contract_set_sha256===trusted.contractSetSha256;
  const result=assessJ2ResearchStart({
   runId,issuerId:str(run.issuer_id),securityId:str(run.security_id),
   dossierId:str(run.dossier_id),
   dataCutoff:str(run.data_cutoff),runType:str(run.run_type),
   runStatus:str(run.run_status),
   currentStage:typeof run.current_stage==='string'?run.current_stage:null,
   baselineSnapshotId:typeof run.baseline_snapshot_id==='string'?
    run.baseline_snapshot_id:null,
   publishedAt:typeof run.published_at==='string'?run.published_at:null,
   stateVersion:Number(run.state_version),
   processVersion:str(run.process_version),
   pilotageVersion:str(run.pilotage_contract_version),
   contractSetSha256:str(run.contract_set_sha256),
   entryPath:str(run.entry_path),canonicalMode:str(run.canonical_mode),
   contractPins:run.contract_pins,stageCount:stages.length,
   receiptVerified:true,identityVerified,zeroSnapshots:snapshots.length===0,
  });
  return NextResponse.json({
   ok:result.eligible,stageAdmissionPreflight:result.reasonCode,
   ...result,
   runContext:{runId,issuerId:trusted.issuerId,
    securityId:trusted.securityId,dossierId:trusted.dossierId,
    dataCutoff:trusted.dataCutoff,isin:trusted.isin,exchange:trusted.exchange},
   environment:'SHADOW_ONLY',readOnly:true,
   nextAction:result.eligible?
    'Configure a separate isolated J2 SHADOW stage-start gate, then obtain explicit GO. Stage RPC is not authorized by this preflight.':
    'Stop. Resolve exact admission blocker without attempting any stage RPC.',
  },{status:result.eligible?200:409,headers});
 }catch{
  return NextResponse.json({
   ok:false,error:'J2_RESEARCH_PREFLIGHT_INCOMPLETE',
   readOnly:true,stageStartAuthorized:false,rpcExecuted:false,
  },{status:409,headers});
 }
}
