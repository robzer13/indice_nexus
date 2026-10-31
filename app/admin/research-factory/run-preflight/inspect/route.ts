import {NextResponse,type NextRequest} from 'next/server';
import {requireAdminSession} from '@/lib/auth/admin-session';
import {isHostedShadowResearchEnabled} from
 '@/lib/orotitan-equity/product/hosted-shadow-research-gate';
import {getHostedShadowRecoveryClient} from
 '@/lib/orotitan-equity/product/research-inbox-pilot-server';
import {inspectRunFactoryIdentity,assertIsin,type RunFactoryInventory} from
 '@/lib/orotitan-equity/product/runfactory-identity-preflight-v1';
import {requireExplicitProfile} from
 '@/lib/orotitan-equity/product/runfactory-contract-profile-v1';

export const runtime='nodejs';
export const dynamic='force-dynamic';
export const maxDuration=30;
const headers={'Cache-Control':'private, no-store'};
const UUID=/^[a-f0-9]{8}-(?:[a-f0-9]{4}-){3}[a-f0-9]{12}$/i;
const invalid=():never=>{throw Error('REGISTRY_INVENTORY_INVALID');};
const string=(x:unknown):string=>typeof x==='string'?x:invalid();
const uuid=(x:unknown):string=>UUID.test(string(x))?x as string:invalid();
const nullableUuid=(x:unknown):string|null=>x===null?null:uuid(x);
function checkedRows(
 result:{data:unknown;error:unknown},limit:number,
):Record<string,unknown>[]{
 if(result.error||!Array.isArray(result.data)||result.data.length>=limit)invalid();
 if(result.data.some((x:unknown)=>!x||typeof x!=='object'||Array.isArray(x)))
  invalid();
 return result.data as Record<string,unknown>[];
}

export async function POST(req:NextRequest){
 if(!isHostedShadowResearchEnabled(process.env))
  return NextResponse.json({ok:false,error:'SHADOW_ONLY'},{status:404,headers});
 try{await requireAdminSession();}
 catch{return NextResponse.json({ok:false,error:'UNAUTHORIZED'},{status:401,headers});}
 if(req.headers.get('origin')!==new URL(req.url).origin)
  return NextResponse.json({ok:false,error:'ORIGIN_MISMATCH'},{status:403,headers});
 const length=Number(req.headers.get('content-length')??'NaN');
 if(req.headers.get('content-type')?.split(';')[0]?.trim()!=='application/json'||
    !Number.isSafeInteger(length)||length<5||length>512)
  return NextResponse.json({ok:false,error:'REQUEST_INVALID'},{status:400,headers});
 try{
  const body:unknown=await req.json();
  if(!body||typeof body!=='object'||Array.isArray(body))
   throw Error('REQUEST_INVALID');
  const obj=body as Record<string,unknown>;
  const isin=assertIsin(string(obj.isin));
  const profileId=requireExplicitProfile(obj.profileId);
  if(obj.exchange!==null&&typeof obj.exchange!=='string')invalid();
  const exchange=(obj.exchange as string|null)?.trim().toUpperCase()||null;
  if(typeof obj.dataCutoff!=='string')invalid();
  const dataCutoff=obj.dataCutoff;
  // No writes, no RPC. Every SELECT has a hard cap and fails closed.
  const client=getHostedShadowRecoveryClient();
  const securities=checkedRows(await client.from('securities')
   .select('security_id,issuer_id,isin,exchange,primary_listing')
   .eq('isin',isin).limit(100),100).map(row=>({
    securityId:uuid(row.security_id),issuerId:uuid(row.issuer_id),
    isin:string(row.isin),exchange:string(row.exchange),
    primaryListing:row.primary_listing===true,
   }));
  const issuerIds=[...new Set(securities.map(s=>s.issuerId))];
  let issuers:RunFactoryInventory['issuers']=[];
  let dossiers:RunFactoryInventory['dossiers']=[];
  let snapshots:RunFactoryInventory['snapshots']=[];
  let runs:RunFactoryInventory['runs']=[];
  if(issuerIds.length>0){
   const [a,b,c]=await Promise.all([
    client.from('issuers').select('issuer_id,display_name')
     .in('issuer_id',issuerIds).limit(100),
    client.from('research_dossiers')
     .select('dossier_id,issuer_id,active,current_snapshot_id')
     .in('issuer_id',issuerIds).limit(100),
    client.from('orotitan_runs')
     .select('run_id,issuer_id,security_id,dossier_id,data_cutoff,contract_set_sha256,run_status,current_stage,run_type,creation_idempotency_key')
     .in('issuer_id',issuerIds).eq('data_cutoff',dataCutoff)
     .limit(100),
   ]);
   issuers=checkedRows(a,100).map(row=>({
    issuerId:uuid(row.issuer_id),displayName:string(row.display_name),
   }));
   dossiers=checkedRows(b,100).map(row=>({
    dossierId:uuid(row.dossier_id),issuerId:uuid(row.issuer_id),
    active:row.active===true,currentSnapshotId:nullableUuid(row.current_snapshot_id),
   }));
   const snapshotIds=[...new Set(dossiers.map(d=>d.currentSnapshotId)
    .filter((v):v is string=>v!==null))];
   if(snapshotIds.length>0){
    snapshots=checkedRows(await client.from('research_snapshots')
     .select('snapshot_id,issuer_id,security_id,dossier_id,contract_version')
     .in('snapshot_id',snapshotIds).limit(100),100).map(row=>({
      snapshotId:uuid(row.snapshot_id),issuerId:uuid(row.issuer_id),
      securityId:uuid(row.security_id),dossierId:uuid(row.dossier_id),
      contractVersion:row.contract_version===null?null:string(row.contract_version),
     }));
   }
   runs=checkedRows(c,100).map(row=>({
    runId:uuid(row.run_id),issuerId:uuid(row.issuer_id),
    securityId:nullableUuid(row.security_id),
    dossierId:nullableUuid(row.dossier_id),
    dataCutoff:string(row.data_cutoff),
    contractSetSha256:string(row.contract_set_sha256),
    runStatus:string(row.run_status),currentStage:
     row.current_stage===null?null:string(row.current_stage),
    runType:string(row.run_type),
    creationIdempotencyKey:string(row.creation_idempotency_key),
   }));
  }
  const result=inspectRunFactoryIdentity(
   {isin,exchange,dataCutoff,profileId},
   {securities,issuers,dossiers,snapshots,runs},
  );
  return NextResponse.json({
   ok:true,mode:'DRY_RUN_ONLY',checkedAt:new Date().toISOString(),
   result,authoritativeProductionEnvironmentVerified:false,
   contractOriginalBytesVerifiedForThisSession:false,
   humanGOAttached:false,rpcExecuted:false,registryAdmitted:false,
  },{headers});
 }catch(e){
  const code=e instanceof Error?e.message:'';
  const safe=new Set(['INVALID_ISIN','INVALID_ISIN_CHECK_DIGIT','INVALID_DATA_CUTOFF',
   'INVALID_EXCHANGE','EXPLICIT_CONTRACT_PROFILE_REQUIRED','REQUEST_INVALID']);
  return NextResponse.json({ok:false,
   error:safe.has(code)?code:'REGISTRY_READ_OR_PREFLIGHT_FAILED',
   registryAdmitted:false,rpcExecuted:false},
   {status:422,headers});
 }
}
