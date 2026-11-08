/** J2 Stage Research admission candidate, pure and read-only.
 * It is NOT the Research execution nor authorization to call start_orotitan_stage.
 * Pin check must be against the *persisted* V3.1 Run, never an exported handoff.
 */
import pinPack from '../../../contracts/orotitan-equity/v3/contract-pin-pack-v3/OROTITAN_CONTRACT_PIN_PACK_V3_0_2.json';

export type J2ResearchStartInputs=Readonly<{
 runId:string;issuerId:string;securityId:string;dossierId:string;
 dataCutoff:string;runType:string;runStatus:string;currentStage:string|null;
 baselineSnapshotId:string|null;publishedAt:string|null;
 stateVersion:number;processVersion:string;pilotageVersion:string;
 contractSetSha256:string;entryPath:string;canonicalMode:string;
 contractPins:unknown;stageCount:number;receiptVerified:boolean;
 identityVerified:boolean;zeroSnapshots:boolean;
}>;
export type J2ResearchStartDecision=Readonly<{
 eligible:boolean;reasonCode:string;
 runId:string;expectedRunStateVersion:number|null;
 stage:'RESEARCH';stageContractName:string;
 stageContractVersion:string;stageContractSha256:string;
 contractSetSha256:string;stageStartAuthorized:false;
 rpcExecuted:false;researchExecuted:false;publicationAuthorized:false;
}>;

const UUID=/^[0-9a-f]{8}-(?:[0-9a-f]{4}-){3}[0-9a-f]{12}$/i;
const SHA=/^[0-9a-f]{64}$/i;
type Pin={name:string;version:string;content_sha256:string};
function asRecord(x:unknown):Record<string,unknown>|null{
 return x&&typeof x==='object'&&!Array.isArray(x)?x as Record<string,unknown>:null;
}
function exactPin(value:unknown,expected:Pin):boolean{
 const p=asRecord(value);
 return !!p&&p.name===expected.name&&p.version===expected.version&&
  p.content_sha256===expected.content_sha256;
}
export function assessJ2ResearchStart(
 input:J2ResearchStartInputs,
):J2ResearchStartDecision{
 const research=pinPack.contract_pins.research_stage;
 const base={
  runId:input.runId,stage:'RESEARCH' as const,
  stageContractName:research.name,stageContractVersion:research.version,
  stageContractSha256:research.content_sha256,
  contractSetSha256:pinPack.contract_set_sha256,
  stageStartAuthorized:false as const,rpcExecuted:false as const,
  researchExecuted:false as const,publicationAuthorized:false as const,
 };
 const decision=(reasonCode:string):J2ResearchStartDecision=>({
  ...base,eligible:reasonCode==='J2_RESEARCH_START_PREFLIGHT_PASS',reasonCode,
  expectedRunStateVersion:reasonCode==='J2_RESEARCH_START_PREFLIGHT_PASS'?
   input.stateVersion:null,
 });
 const pins=asRecord(input.contractPins);
 if(!UUID.test(input.runId)||!UUID.test(input.issuerId)||
    !UUID.test(input.securityId)||!UUID.test(input.dossierId))
  return decision('J2_IDENTITY_INVALID');
 if(!/^\d{4}-\d{2}-\d{2}$/.test(input.dataCutoff)||
    !Number.isSafeInteger(input.stateVersion)||input.stateVersion<2)
  return decision('J2_DATA_CUTOFF_OR_CAS_INVALID');
 if(input.runType!=='INITIAL'||input.runStatus!=='CREATED'||
    input.currentStage!==null||input.baselineSnapshotId!==null||
    input.publishedAt!==null||input.entryPath!=='IMPOSED_COMPANY'||
    input.canonicalMode!=='ANALYZE')
  return decision('J2_RUN_STATE_INELIGIBLE');
 if(input.stageCount!==0||!input.zeroSnapshots)
  return decision('J2_RESEARCH_STAGE_ALREADY_STARTED_OR_SNAPSHOT_PRESENT');
 if(input.receiptVerified!==true||input.identityVerified!==true)
  return decision('J2_PRIOR_HANDOFF_NOT_VERIFIED');
 if(input.processVersion!=='3.1'||input.pilotageVersion!=='3.1'||
    !SHA.test(input.contractSetSha256)||
    input.contractSetSha256!==pinPack.contract_set_sha256||
    !pins||Object.keys(pins).length!==15||
    !exactPin(pins.research_stage,research)||
    !exactPin(pins.process,pinPack.contract_pins.process)||
    !exactPin(pins.pilotage,pinPack.contract_pins.pilotage))
  return decision('J2_PINNED_CONTRACT_MISMATCH');
 return decision('J2_RESEARCH_START_PREFLIGHT_PASS');
}
