/**
 * OroTitan Run Factory P0-B — deterministic, issuer-neutral, READ-ONLY preflight.
 * Inputs are DB observations, not commands. No identifiers or authorities invented.
 * The returned decision NEVER authorizes a Registry mutation.
 */
import {
 getContractProfileSummary,requireExplicitProfile,type ContractProfileId,
} from './runfactory-contract-profile-v1';

export type SecurityObservation=Readonly<{
 securityId:string;issuerId:string;isin:string;exchange:string;
 primaryListing:boolean;
}>;
export type IssuerObservation=Readonly<{issuerId:string;displayName:string}>;
export type DossierObservation=Readonly<{
 dossierId:string;issuerId:string;active:boolean;
 currentSnapshotId:string|null;
}>;
export type SnapshotObservation=Readonly<{
 snapshotId:string;dossierId:string;issuerId:string;securityId:string;
 contractVersion:string|null;
}>;
export type RunObservation=Readonly<{
 runId:string;issuerId:string;securityId:string|null;
 dossierId:string|null;dataCutoff:string;contractSetSha256:string;
 runStatus:string;currentStage:string|null;runType:string;
 creationIdempotencyKey:string;
}>;
export type RunFactoryInventory=Readonly<{
 securities:readonly SecurityObservation[];
 issuers:readonly IssuerObservation[];
 dossiers:readonly DossierObservation[];
 snapshots:readonly SnapshotObservation[];
 runs:readonly RunObservation[];
}>;
export type RunFactoryInput=Readonly<{
 isin:string;exchange:string|null;dataCutoff:string;
 profileId:ContractProfileId;
}>;
export type RunFactoryDecisionStatus=
 'BLOCKED_SECURITY_NOT_FOUND'|'BLOCKED_SECURITY_AMBIGUOUS'|
 'BLOCKED_ISSUER_INCOMPLETE'|'BLOCKED_DOSSIER_NOT_FOUND'|
 'BLOCKED_DOSSIER_AMBIGUOUS'|'BLOCKED_SNAPSHOT_MISMATCH'|
 'BLOCKED_EXISTING_METHODOLOGY_CONFLICT'|
 'BLOCKED_EXISTING_RUN_AMBIGUOUS'|'BLOCKED_EXISTING_RUN_IDENTITY_MISMATCH'|
 'EXISTING_RUN_REUSE_CANDIDATE'|'NEW_RUN_PLAN_CANDIDATE';
export type RunFactoryDecision=Readonly<{
 status:RunFactoryDecisionStatus;
 profileId:ContractProfileId;contractSetSha256:string;
 isin:string;exchange:string|null;dataCutoff:string;
 issuerId:string|null;issuerName:string|null;securityId:string|null;
 dossierId:string|null;baselineSnapshotId:string|null;
 proposedRunType:'INITIAL'|'REFRESH'|null;
 existingRunId:string|null;
 existingRunStatus:string|null;
 candidateCreationIdempotencyKey:null;
 nextAction:string;
 evidenceStageAdmitted:false;runCreated:false;
 rpcExecuted:false;goAuthorized:false;registryAdmitted:false;
 planExecutableInShadow:false;
}>;

function validDate(s:string):boolean{
 if(!/^\d{4}-\d{2}-\d{2}$/.test(s))return false;
 const d=new Date(s+'T00:00:00.000Z');
 return Number.isFinite(d.getTime())&&d.toISOString().slice(0,10)===s;
}
/** ISO 6166 / Luhn format plus check digit. */
export function assertIsin(raw:string):string{
 if(typeof raw!=='string')throw Error('INVALID_ISIN');
 const value=raw.trim().toUpperCase();
 if(!/^[A-Z]{2}[A-Z0-9]{9}[0-9]$/.test(value))
  throw Error('INVALID_ISIN');
 let expanded='';
 for(const ch of value){
  const code=ch.charCodeAt(0);
  expanded+=code>=65&&code<=90?String(code-55):ch;
 }
 let sum=0;
 for(let i=expanded.length-1,offset=0;i>=0;i--,offset++){
  let n=Number(expanded[i]);
  if(offset%2===1){n*=2;if(n>9)n-=9;}
  sum+=n;
 }
 if(sum%10!==0)throw Error('INVALID_ISIN_CHECK_DIGIT');
 return value;
}
export function inspectRunFactoryIdentity(
 input:RunFactoryInput,inventory:RunFactoryInventory,
):RunFactoryDecision{
 const profileId=requireExplicitProfile(input.profileId);
 const profile=getContractProfileSummary(profileId);
 const isin=assertIsin(input.isin);
 if(!validDate(input.dataCutoff))throw Error('INVALID_DATA_CUTOFF');
 const exchange=input.exchange?.trim().toUpperCase()||null;
 if(exchange!==null&&(!/^[A-Z0-9_.-]{2,30}$/.test(exchange)))
  throw Error('INVALID_EXCHANGE');
 if(!inventory||!Array.isArray(inventory.securities)||
    !Array.isArray(inventory.issuers)||
    !Array.isArray(inventory.dossiers)||
    !Array.isArray(inventory.snapshots)||!Array.isArray(inventory.runs))
  throw Error('REGISTRY_INVENTORY_INVALID');
 const base={
  profileId,contractSetSha256:profile.contractSetSha256,
  isin,exchange,dataCutoff:input.dataCutoff,
  issuerId:null,issuerName:null,securityId:null,
  dossierId:null,baselineSnapshotId:null,proposedRunType:null,
  existingRunId:null,existingRunStatus:null,
  candidateCreationIdempotencyKey:null,
  evidenceStageAdmitted:false,runCreated:false,
  rpcExecuted:false,goAuthorized:false,registryAdmitted:false,
  planExecutableInShadow:false,
 } as const;
 type Info=Pick<RunFactoryDecision,'issuerId'|'issuerName'|'securityId'|
  'dossierId'|'baselineSnapshotId'|'proposedRunType'|
  'existingRunId'|'existingRunStatus'>;
 const fail=(status:RunFactoryDecisionStatus,nextAction:string,
  info:Partial<Info>={}):RunFactoryDecision=>({...base,...info,status,nextAction});
 const found=inventory.securities.filter(s=>s.isin.toUpperCase()===isin&&
  (exchange===null||s.exchange.toUpperCase()===exchange));
 if(found.length===0)return fail('BLOCKED_SECURITY_NOT_FOUND',
  'Resolve/onboard exact ISIN and listing in governed identity registry.');
 if(found.length!==1)return fail('BLOCKED_SECURITY_AMBIGUOUS',
  'Multiple security identities; specify unique exchange or resolve duplicate records.');
 const security=found[0];
 const issuerRows=inventory.issuers.filter(i=>i.issuerId===security.issuerId);
 const partial={issuerId:security.issuerId,securityId:security.securityId};
 if(issuerRows.length!==1||!issuerRows[0].displayName.trim())
  return fail('BLOCKED_ISSUER_INCOMPLETE',
   'Issuer identity missing or ambiguous; require authoritative issuer record.',partial);
 const issuerName=issuerRows[0].displayName;
 const associated={...partial,issuerName};
 const dossiers=inventory.dossiers.filter(d=>d.issuerId===security.issuerId&&d.active);
 if(dossiers.length===0)return fail('BLOCKED_DOSSIER_NOT_FOUND',
  'Active research dossier is missing; governed onboarding required.',associated);
 if(dossiers.length!==1)return fail('BLOCKED_DOSSIER_AMBIGUOUS',
  'Multiple active dossiers detected; resolve lineage before new run.',associated);
 const dossier=dossiers[0];
 const withDossier={...associated,dossierId:dossier.dossierId};
 let baselineSnapshotId:null|string=null;
 let proposedRunType:'INITIAL'|'REFRESH'='INITIAL';
 if(dossier.currentSnapshotId!==null){
  const snapshots=inventory.snapshots.filter(s=>
   s.snapshotId===dossier.currentSnapshotId);
  if(snapshots.length!==1||
     snapshots[0].dossierId!==dossier.dossierId||
     snapshots[0].issuerId!==security.issuerId||
     snapshots[0].securityId!==security.securityId||
     !snapshots[0].contractVersion?.trim())
   return fail('BLOCKED_SNAPSHOT_MISMATCH',
    'Canonical snapshot identity / contract is not consistent with dossier and security.',
    withDossier);
  baselineSnapshotId=dossier.currentSnapshotId;
  proposedRunType='REFRESH';
 }
 const baseInfo={...withDossier,baselineSnapshotId,proposedRunType};
 const existing=inventory.runs.filter(r=>
  r.issuerId===security.issuerId&&r.dataCutoff===input.dataCutoff&&
  r.runStatus!=='CANCELLED');
 const incompatible=existing.filter(r=>r.contractSetSha256!==profile.contractSetSha256);
 if(incompatible.length>0)return fail('BLOCKED_EXISTING_METHODOLOGY_CONFLICT',
  'Same issuer/cutoff already has different immutable Contract Set; review successor governance.',
  baseInfo);
 const same=existing.filter(r=>r.contractSetSha256===profile.contractSetSha256);
 if(same.length>1)return fail('BLOCKED_EXISTING_RUN_AMBIGUOUS',
  'Multiple non-cancelled runs under same issuer, cutoff and Contract Set.',
  baseInfo);
 if(same.length===1){
  const run=same[0];
  if((run.securityId!==null&&run.securityId!==security.securityId)||
     (run.dossierId!==null&&run.dossierId!==dossier.dossierId)||
     !run.runId.trim())
   return fail('BLOCKED_EXISTING_RUN_IDENTITY_MISMATCH',
    'Existing run identity disagrees with exact security/dossier.',
    baseInfo);
  return fail('EXISTING_RUN_REUSE_CANDIDATE',
   'Re-query persisted run and state_version; do not create a duplicate.',
   {...baseInfo,existingRunId:run.runId,existingRunStatus:run.runStatus});
 }
 return fail('NEW_RUN_PLAN_CANDIDATE',
  'Identity and lineage inputs found. Obtain GO and an authorized environment, then delegate to the existing versioned bootstrap and controlled RPC. SHADOW dry-run never executes.',
  baseInfo);
}
