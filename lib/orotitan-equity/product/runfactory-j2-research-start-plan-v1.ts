/** Exact future Research stage-start RPC plan.
 *
 * Pure construction ONLY. No transport, no GO grant, and no persistent mutation.
 * The caller must re-read persisted state and obtain a fresh explicit GO before
 * using a separately isolated J2 SHADOW transport. The underlying canonical
 * start_orotitan_stage RPC enforces the run state-version CAS atomically.
 */
import {createHash} from 'node:crypto';
import pack from '../../../contracts/orotitan-equity/v3/contract-pin-pack-v3/OROTITAN_CONTRACT_PIN_PACK_V3_0_2.json';
import type {J2ResearchStartDecision} from './runfactory-j2-research-admission-v1';

const UUID=/^[0-9a-f]{8}-(?:[0-9a-f]{4}-){3}[0-9a-f]{12}$/i;
export type J2ResearchStartRpcPlan=Readonly<{
 rpc:'start_orotitan_stage';
 args:Readonly<{
  p_run_id:string;p_stage_code:'RESEARCH';
  p_expected_run_state_version:number;
  p_stage_contract_name:string;p_stage_contract_version:string;
  p_stage_contract_sha256:string;p_idempotency_key:string;
  p_request_fingerprint_sha256:string;p_actor_type:'PILOTAGE';
 }>;
 confirmation:string;
 environment:'SHADOW_ONLY';authorized:false;executed:false;
}>;
export function buildJ2ResearchStartRpcPlan(
 d:J2ResearchStartDecision,
):J2ResearchStartRpcPlan{
 const pin=pack.contract_pins.research_stage;
 if(d.eligible!==true||d.reasonCode!=='J2_RESEARCH_START_PREFLIGHT_PASS'||
    d.stageStartAuthorized!==false||d.rpcExecuted!==false||
    d.researchExecuted!==false||d.publicationAuthorized!==false||
    d.stage!=='RESEARCH'||!UUID.test(d.runId)||
    !Number.isSafeInteger(d.expectedRunStateVersion)||
    (d.expectedRunStateVersion??0)<2||
    d.stageContractName!==pin.name||
    d.stageContractVersion!==pin.version||
    d.stageContractSha256!==pin.content_sha256||
    d.contractSetSha256!==pack.contract_set_sha256)
  throw Error('J2_STAGE_START_PLAN_NOT_ADMITTED');
 const key='orotitan:j2:research:start:v1:'+d.runId.toLowerCase();
 const content=[
  'OROTITAN_J2_RESEARCH_STAGE_START_V1',d.runId.toLowerCase(),'RESEARCH',
  String(d.expectedRunStateVersion),pin.name,pin.version,
  pin.content_sha256,pack.contract_set_sha256,'PILOTAGE',
 ].join('\n');
 return {
  rpc:'start_orotitan_stage',
  args:{
   p_run_id:d.runId.toLowerCase(),
   p_stage_code:'RESEARCH',
   p_expected_run_state_version:d.expectedRunStateVersion,
   p_stage_contract_name:pin.name,p_stage_contract_version:pin.version,
   p_stage_contract_sha256:pin.content_sha256,p_idempotency_key:key,
   p_request_fingerprint_sha256:createHash('sha256')
    .update(content,'utf8').digest('hex'),
   p_actor_type:'PILOTAGE',
  },
  confirmation:'GO J2 RESEARCH SHADOW '+d.runId.toLowerCase(),
  environment:'SHADOW_ONLY',authorized:false,executed:false,
 };
}
