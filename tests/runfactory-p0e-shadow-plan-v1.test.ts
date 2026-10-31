import test from 'node:test';
import assert from 'node:assert/strict';
import {
 buildP0EShadowInitialPlan,buildP0EShadowRpcArgs,
} from '../lib/orotitan-equity/product/runfactory-p0e-shadow-plan-v1';
import {
 activeContractPinPack,
 buildRunCreationPlan,runtimeBootstrap,
} from '../lib/orotitan-equity/v3/runtime-bootstrap-v3-0-3';

const identity={
 issuerId:'c431698b-3b5f-40c4-bf24-8bf4612db876',
 securityId:'44bf0acd-9420-42f3-bdd1-1e725740154d',
 dossierId:'e30bd3e4-618c-41a2-93b5-015fc2ea708e',
 isin:'FR0000120321',exchange:'XPAR',dataCutoff:'2026-10-30',
 currentSnapshotId:null,
} as const;
test('SHADOW V3.1 INITIAL derives exact official V3.0.3 fingerprint offline, not in production runtime',()=>{
 const p=buildP0EShadowInitialPlan(identity);
 // Test-only reference fixture: NO Supabase connection and NO production
 // environment assertion in the SHADOW adapter.
 const canonicalReference=buildRunCreationPlan({
  environment:{
   projectRef:runtimeBootstrap.production_environment.project_ref,
   projectName:runtimeBootstrap.production_environment.project_name,
   region:runtimeBootstrap.production_environment.region,
   status:runtimeBootstrap.production_environment.required_status,
  },
  identity:{
   company:'L’Oréal',companyCommandName:'LOREAL',
   issuerId:identity.issuerId,securityId:identity.securityId,
   dossierId:identity.dossierId,currentSnapshotId:null,
   currentSnapshotContractVersion:null,
  },
  dataCutoff:identity.dataCutoff,
 });
 assert.equal(p.creationIdempotencyKey,
  canonicalReference.rpcArgs.p_creation_idempotency_key);
 assert.equal(p.requestFingerprintSha256,
  canonicalReference.rpcArgs.p_request_fingerprint_sha256);
 assert.equal(p.contractSetSha256,activeContractPinPack.contract_set_sha256);
 assert.equal(p.environment,'SHADOW_ONLY');
 assert.equal(p.productionEnvironmentValidated,false);
 assert.equal(p.publicationAuthorized,false);
 assert.equal(canonicalReference.runType,'INITIAL');
});
test('Explicit pinned byte proof required to build P0-E SHADOW RPC',()=>{
 const p=buildP0EShadowInitialPlan(identity);
 assert.throws(()=>buildP0EShadowRpcArgs(p,undefined as never),
  /P0E_ORIGINAL_BYTE_GATE_FAILED/);
 assert.throws(()=>buildP0EShadowRpcArgs(p,{
  profileId:'OROTITAN_V3_1',contractSetSha256:p.contractSetSha256,
  pinCount:15,passedPinCount:14,canonicalBytes:100,
  checks:[],allOriginalBytesVerified:true,registryMutated:false,
 }),/P0E_ORIGINAL_BYTE_GATE_FAILED/);
});
test('P0-E excludes refresh and invalid identities',()=>{
 assert.throws(()=>buildP0EShadowInitialPlan({
  ...identity,currentSnapshotId:'some' as never,
 }),/P0E_CANONICAL_IDENTITY_INVALID/);
 assert.throws(()=>buildP0EShadowInitialPlan({
  ...identity,isin:'FR0000120322',
 }),/INVALID_ISIN_CHECK_DIGIT/);
 assert.throws(()=>buildP0EShadowInitialPlan({
  ...identity,dataCutoff:'2026-02-30',
 }),/P0E_CUTOFF_INVALID/);
});
