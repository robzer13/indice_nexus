import {NextResponse,type NextRequest} from 'next/server';
import {requireAdminSession} from '@/lib/auth/admin-session';
import {isHostedJ1RunFactoryEnabled} from
 '@/lib/orotitan-equity/product/hosted-shadow-research-gate';
import {getJ1RunFactoryShadowClient} from
 '@/lib/orotitan-equity/product/runfactory-j1-shadow-server';
import {buildJ1ResearchHandoff} from
 '@/lib/orotitan-equity/product/runfactory-j1-research-handoff-v1';

/** Read-only Registry handoff: Research can retrieve a Run by canonical ID.
 * Never writes a handoff record, starts a stage or grants Research admission.
 * The next stage must independently enforce its own contract admission.
 */
export const runtime='nodejs';
export const dynamic='force-dynamic';
export const maxDuration=30;
const headers={'Cache-Control':'private, no-store','X-Content-Type-Options':'nosniff'};
const UUID=/^[a-f0-9]{8}-(?:[a-f0-9]{4}-){3}[a-f0-9]{12}$/i;
const SHA=/^[a-f0-9]{64}$/i;
const fail=():never=>{throw Error('J1_HANDOFF_NOT_READY');};
function record(x:unknown):Record<string,unknown>{
 if(!x||typeof x!=='object'||Array.isArray(x))fail();
 return x as Record<string,unknown>;
}
function text(x:unknown):string{
 if(typeof x!=='string')return fail();
 if(!x.trim())return fail();
 return x;
}
function uuid(x:unknown):string{
 const v=text(x);if(!UUID.test(v))fail();return v;
}
function list(x:unknown):Record<string,unknown>[]{
 if(!Array.isArray(x))return fail();
 const values=x as unknown[];
 if(values.length>=20)return fail();
 return values.map(record);
}
function value<T>(data:{data:T|null;error:unknown}):T{
 if(data.error||data.data===null)fail();
 return data.data as T;
}
export async function GET(req:NextRequest){
 if(!isHostedJ1RunFactoryEnabled(process.env))
  return NextResponse.json({ok:false,error:'SHADOW_ONLY'},{status:404,headers});
 try{await requireAdminSession();}
 catch{return NextResponse.json({ok:false,error:'UNAUTHORIZED'},{status:401,headers});}
 const params=new URL(req.url).searchParams;
 const runId=params.get('runId');
 if(!runId||!UUID.test(runId)||[...params.keys()].some(k=>k!=='runId')||
    params.getAll('runId').length!==1)
  return NextResponse.json({ok:false,error:'J1_HANDOFF_RUN_ID_INVALID'},
   {status:422,headers});
 try{
  const client=getJ1RunFactoryShadowClient();
  const run=record(value(await client.from('orotitan_runs').select(
   'run_id,issuer_id,security_id,dossier_id,entry_path,canonical_mode,run_type,run_status,current_stage,state_version,data_cutoff,baseline_snapshot_id,contract_pins,contract_set_sha256,creation_idempotency_key,process_version,pilotage_contract_version,published_at'
  ).eq('run_id',runId).single()));
  if(run.run_id!==runId||run.run_type!=='INITIAL'||
     run.run_status!=='CREATED'||run.current_stage!==null||
     Number(run.state_version)<2||run.baseline_snapshot_id!==null||
     run.entry_path!=='IMPOSED_COMPANY'||run.canonical_mode!=='ANALYZE'||
     run.process_version!=='3.1'||run.pilotage_contract_version!=='3.1'||
     run.published_at!==null)fail();
  const issuerId=uuid(run.issuer_id),securityId=uuid(run.security_id),
    dossierId=uuid(run.dossier_id);
  const dataCutoff=text(run.data_cutoff);
  const hash=text(run.contract_set_sha256);
  if(!SHA.test(hash)||!/^\d{4}-\d{2}-\d{2}$/.test(dataCutoff))fail();
  const pins=record(run.contract_pins);
  if(Object.keys(pins).length!==15)fail();
  const process=record(pins.process),pilotage=record(pins.pilotage),
    research=record(pins.research_stage);
  if(process.name!=='OROTITAN_EXECUTION_PROCESS_V3_1_FREEZE_V3.1'||
     process.version!=='3.1'||
     pilotage.name!=='OROTITAN_PILOTAGE_ORCHESTRATION_CONTRACT_V3_1_FREEZE_V3.1'||
     pilotage.version!=='3.1'||
     research.name!=='OROTITAN_RESEARCH_STAGE_CONTRACT_V2_FREEZE_V2.0'||
     research.version!=='2.0')fail();
  const [issuerQ,securityQ,dossierQ,receiptQ,eventsQ,stagesQ,snapshotsQ,peersQ]=
   await Promise.all([
    client.from('issuers').select('issuer_id,display_name')
     .eq('issuer_id',issuerId).single(),
    client.from('securities').select('security_id,issuer_id,isin,exchange,ticker')
     .eq('security_id',securityId).single(),
    client.from('research_dossiers')
     .select('dossier_id,issuer_id,active,current_snapshot_id')
     .eq('dossier_id',dossierId).single(),
    client.from('orotitan_p0e_run_admission_receipts_v1')
     .select('run_id,creation_idempotency_key,profile_id,contract_set_sha256,verification_receipt,publication_authorized')
     .eq('run_id',runId).single(),
    client.from('orotitan_run_events').select('event_type').eq('run_id',runId).limit(20),
    client.from('orotitan_run_stages').select('stage_code').eq('run_id',runId).limit(1),
    client.from('research_snapshots').select('snapshot_id').eq('dossier_id',dossierId).limit(1),
    client.from('orotitan_runs').select('run_id')
     .eq('issuer_id',issuerId).eq('data_cutoff',dataCutoff).limit(2),
   ]);
  const issuer=record(value(issuerQ)),security=record(value(securityQ)),
   dossier=record(value(dossierQ)),receipt=record(value(receiptQ));
  const events=list(value(eventsQ)).map(r=>text(r.event_type)),
   stages=list(value(stagesQ)),snapshots=list(value(snapshotsQ)),
   peers=list(value(peersQ));
  const verification=record(receipt.verification_receipt);
  if(issuer.issuer_id!==issuerId||security.security_id!==securityId||
     security.issuer_id!==issuerId||dossier.dossier_id!==dossierId||
     dossier.issuer_id!==issuerId||dossier.active!==true||
     dossier.current_snapshot_id!==null||receipt.run_id!==runId||
     receipt.creation_idempotency_key!==run.creation_idempotency_key||
     receipt.contract_set_sha256!==hash||
     receipt.profile_id!=='OROTITAN_V3_1'||
     receipt.publication_authorized!==false||
     verification.allOriginalBytesVerified!==true||
     verification.pinCount!==15||verification.passedPinCount!==15||
     verification.contractSetSha256!==hash||
     peers.length!==1||peers[0].run_id!==runId||
     events.length!==2||!events.includes('RUN_CREATED')||
     !events.includes('RUN_IDENTITY_BOUND')||
     stages.length!==0||snapshots.length!==0)fail();
  const issuerName=text(issuer.display_name),
   isin=text(security.isin),exchange=text(security.exchange);
  const handoff=buildJ1ResearchHandoff({
   runId,issuerId,securityId,dossierId,issuerName,isin,exchange,
   dataCutoff,profileId:'OROTITAN_V3_1',contractSetSha256:hash,
   status:'REPLAYED',persistedReadbackVerified:true,
   receiptPersisted:true,stageStarted:false,snapshotCreated:false,
   published:false,environment:'SHADOW_ONLY',productionTouched:false,
  });
  return NextResponse.json({
   ok:true,schema:'OROTITAN_J1_READ_ONLY_ROUTING_HANDOFF_V1',
   routingStatus:'RUN_READY_FOR_RESEARCH_PREFLIGHT',
   runId,issuerId,securityId,dossierId,isin,exchange,dataCutoff,
   contractSetSha256:hash,profileId:'OROTITAN_V3_1',
   handoff,registryReadbackVerified:true,contractReceiptVerified:true,
   researchStageAdmission:false,researchStageStarted:false,
   snapshotCreated:false,published:false,
   environment:'SHADOW_ONLY',productionTouched:false,
  },{headers});
 }catch{
  return NextResponse.json({ok:false,error:'J1_HANDOFF_NOT_READY'},
   {status:409,headers});
 }
}
