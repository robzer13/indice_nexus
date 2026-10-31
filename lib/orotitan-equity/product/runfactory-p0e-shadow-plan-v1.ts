/**
 * P0-E SHADOW adapter for V3.1 INITIAL run authority.
 *
 * The canonical V3.0.3 bootstrap deliberately asserts PRODUCTION. This
 * adapter never spoofs its environment or calls assertProductionEnvironment.
 * It reuses frozen V3.0.2 pins and checks runtime authority integrity,
 * faithfully reproducing the deterministic V3 creation descriptor/hash.
 * The Registry schema-generation label (V1/V2) is not the analytic method.
 */
import {createHash} from 'node:crypto';
import {
 activeContractPinPack,assertRuntimeBootstrapIntegrity,
 OROTITAN_RUNTIME_BOOTSTRAP_VERSION,
} from '../v3/runtime-bootstrap-v3-0-3';
import {assertIsin} from './runfactory-identity-preflight-v1';
import type {P0EFullByteReport} from './runfactory-p0e-byte-gate';

export type P0EIdentity=Readonly<{
 issuerId:string;securityId:string;dossierId:string;
 isin:string;exchange:string;dataCutoff:string;
 currentSnapshotId:null;
}>;
export type P0EPlan=Readonly<{
 schema:'OROTITAN_P0E_SHADOW_INITIAL_V3_1_PLAN';
 issuerId:string;securityId:string;dossierId:string;
 isin:string;exchange:string;dataCutoff:string;
 creationIdempotencyKey:string;requestFingerprintSha256:string;
 contractSetSha256:string;
 processVersion:'3.1';pilotageContractVersion:'3.1';
 runType:'INITIAL';parentRunId:null;baselineSnapshotId:null;
 environment:'SHADOW_ONLY';productionEnvironmentValidated:false;
 goAuthorizedForShadow:true;publicationAuthorized:false;
}>;
const UUID=/^[a-f0-9]{8}-(?:[a-f0-9]{4}-){3}[a-f0-9]{12}$/i;
const CUTOFF=/^\d{4}-\d{2}-\d{2}$/;
type Json=string|number|boolean|null|Json[]|{[key:string]:Json};
function canonical(v:Json):string{
 if(v===null||typeof v!=='object')return JSON.stringify(v);
 if(Array.isArray(v))return '['+v.map(canonical).join(',')+']';
 return '{'+Object.entries(v).sort(([a],[b])=>a.localeCompare(b))
  .map(([k,x])=>JSON.stringify(k)+':'+canonical(x)).join(',')+'}';
}
export function buildP0EShadowInitialPlan(identity:P0EIdentity):P0EPlan{
 assertRuntimeBootstrapIntegrity();
 if(!identity||!UUID.test(identity.issuerId)||!UUID.test(identity.securityId)||
    !UUID.test(identity.dossierId)||identity.currentSnapshotId!==null)
  throw Error('P0E_CANONICAL_IDENTITY_INVALID');
 const isin=assertIsin(identity.isin);
 const exchange=identity.exchange.trim().toUpperCase();
 if(!/^[A-Z0-9._-]{2,30}$/.test(exchange))
  throw Error('P0E_EXCHANGE_INVALID');
 const cutoff=identity.dataCutoff;
 const date=new Date(cutoff+'T00:00:00Z');
 if(!CUTOFF.test(cutoff)||!Number.isFinite(date.getTime())||
    date.toISOString().slice(0,10)!==cutoff)
  throw Error('P0E_CUTOFF_INVALID');
 const sha=activeContractPinPack.contract_set_sha256;
 if(sha!=='3644e501909326af04d66730fa30b1ac3da0d82fb6b717202af6d948f3211fe2')
  throw Error('P0E_CONTRACT_SET_UNEXPECTED');
 const descriptor={
  runtime_bootstrap_version:OROTITAN_RUNTIME_BOOTSTRAP_VERSION,
  issuer_id:identity.issuerId,
  entry_path:'IMPOSED_COMPANY',
  canonical_mode:'ANALYZE',
  run_type:'INITIAL',
  data_cutoff:cutoff,
  parent_run_id:null,
  baseline_snapshot_id:null,
  process_version:activeContractPinPack.contract_pins.process.version,
  pilotage_contract_version:activeContractPinPack.contract_pins.pilotage.version,
  contract_set_sha256:sha,
  creation_reason:'NORMAL',
  parent_expected_state_version:null,
  parent_expected_run_status:null,
  parent_expected_current_stage:null,
  parent_expected_contract_set_sha256:null,
 } as const;
 const digest=createHash('sha256').update(canonical(descriptor),'utf8').digest('hex');
 return {
  schema:'OROTITAN_P0E_SHADOW_INITIAL_V3_1_PLAN',
  issuerId:identity.issuerId,securityId:identity.securityId,
  dossierId:identity.dossierId,isin,exchange,dataCutoff:cutoff,
  creationIdempotencyKey:'orotitan:v3:create:'+digest,
  requestFingerprintSha256:digest,
  contractSetSha256:sha,
  processVersion:'3.1',pilotageContractVersion:'3.1',
  runType:'INITIAL',parentRunId:null,baselineSnapshotId:null,
  environment:'SHADOW_ONLY',productionEnvironmentValidated:false,
  goAuthorizedForShadow:true,publicationAuthorized:false,
 };
}
export function buildP0EShadowRpcArgs(plan:P0EPlan,proof:P0EFullByteReport){
 if(!proof||proof.allOriginalBytesVerified!==true||
    proof.pinCount!==15||proof.passedPinCount!==15||
    proof.contractSetSha256!==plan.contractSetSha256||
    proof.profileId!=='OROTITAN_V3_1')
  throw Error('P0E_ORIGINAL_BYTE_GATE_FAILED');
 const receipt={
  ...proof,checkedAt:new Date().toISOString(),
 };
 return {
  p_issuer_id:plan.issuerId,
  p_security_id:plan.securityId,
  p_dossier_id:plan.dossierId,
  p_isin:plan.isin,p_exchange:plan.exchange,
  p_data_cutoff:plan.dataCutoff,
  p_contract_pins:activeContractPinPack.contract_pins,
  p_contract_set_sha256:plan.contractSetSha256,
  p_creation_idempotency_key:plan.creationIdempotencyKey,
  p_request_fingerprint_sha256:plan.requestFingerprintSha256,
  p_verification_receipt:receipt,
  p_operator_confirmation:'GO P0-E SHADOW '+plan.isin+' '+plan.dataCutoff,
 };
}
