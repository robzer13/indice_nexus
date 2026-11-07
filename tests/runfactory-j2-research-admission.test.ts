import assert from 'node:assert/strict';
import test from 'node:test';
import pack from '../contracts/orotitan-equity/v3/contract-pin-pack-v3/OROTITAN_CONTRACT_PIN_PACK_V3_0_2.json';
import {isHostedJ2ResearchStageStartEnabled,SHADOW_RESEARCH_ORIGIN,SHADOW_RESEARCH_PROJECT_REF} from
 '../lib/orotitan-equity/product/hosted-shadow-research-gate';
import {buildJ2ResearchStartRpcPlan} from
 '../lib/orotitan-equity/product/runfactory-j2-research-start-plan-v1';
import {assessJ2ResearchStart,type J2ResearchStartInputs} from
 '../lib/orotitan-equity/product/runfactory-j2-research-admission-v1';

const source:J2ResearchStartInputs={
 runId:'db00ce77-04eb-492b-a76b-164ecbe9ccfe',
 issuerId:'81736c80-9d6a-4f4e-bbc1-1b418912d94a',
 securityId:'e60d26ba-925c-4e02-95b0-6ffd53a73d42',
 dossierId:'09815f6c-d1dd-47a0-82f3-f69c0e32860a',
 dataCutoff:'2026-10-30',
 runType:'INITIAL',runStatus:'CREATED',currentStage:null,
 baselineSnapshotId:null,publishedAt:null,
 stateVersion:2,processVersion:'3.1',pilotageVersion:'3.1',
 contractSetSha256:pack.contract_set_sha256,
 entryPath:'IMPOSED_COMPANY',canonicalMode:'ANALYZE',
 contractPins:pack.contract_pins,stageCount:0,receiptVerified:true,
 identityVerified:true,zeroSnapshots:true,
};
test('J2 pure preflight admits only the exact verified J1 Run for Research',()=>{
 const p=assessJ2ResearchStart(source);
 assert.equal(p.eligible,true);
 assert.equal(p.reasonCode,'J2_RESEARCH_START_PREFLIGHT_PASS');
 assert.equal(p.expectedRunStateVersion,2);
 assert.equal(p.stage,'RESEARCH');
 assert.equal(p.stageContractName,pack.contract_pins.research_stage.name);
 assert.equal(p.stageContractVersion,'2.0');
 assert.equal(p.stageContractSha256,pack.contract_pins.research_stage.content_sha256);
 assert.equal(p.contractSetSha256,pack.contract_set_sha256);
 assert.equal(p.stageStartAuthorized,false);
 assert.equal(p.rpcExecuted,false);
 assert.equal(p.researchExecuted,false);
 assert.equal(p.publicationAuthorized,false);
});
test('J2 fails closed on ambiguous identity or stale/invalid CAS',()=>{
 for(const bad of [
  {...source,issuerId:'invalid'},
  {...source,dossierId:''},
  {...source,dataCutoff:'2026-11-07T23:59:00'},
  {...source,stateVersion:1},
  {...source,stateVersion:2.5},
 ]){
  const p=assessJ2ResearchStart(bad);
  assert.equal(p.eligible,false);
  assert.equal(p.expectedRunStateVersion,null);
 }
});
test('J2 refuses stage reuse, blocking run statuses and baseline mismatch',()=>{
 for(const bad of [
  {...source,runStatus:'ACTIVE'},
  {...source,runStatus:'BLOCKED'},
  {...source,runStatus:'CANCELLED'},
  {...source,currentStage:'RESEARCH'},
  {...source,stageCount:1},
  {...source,zeroSnapshots:false},
  {...source,baselineSnapshotId:source.runId},
  {...source,publishedAt:'2026-11-07T15:00:00Z'},
  {...source,runType:'REFRESH'},
  {...source,entryPath:'DISCOVERY'},
  {...source,canonicalMode:'DISCOVER'},
 ]){
  assert.equal(assessJ2ResearchStart(bad).eligible,false);
 }
});
test('J2 requires real persisted V3.1 pins and J1 readback, not handoff prose',()=>{
 for(const bad of [
  {...source,receiptVerified:false},
  {...source,identityVerified:false},
  {...source,contractSetSha256:'0'.repeat(64)},
  {...source,processVersion:'2.0'},
  {...source,pilotageVersion:'2.0'},
  {...source,contractPins:{}},
  {...source,contractPins:{...pack.contract_pins,
   research_stage:{...pack.contract_pins.research_stage,content_sha256:'0'.repeat(64)}}},
  {...source,contractPins:{...pack.contract_pins,
   pilotage:{...pack.contract_pins.pilotage,version:'3.0'}}},
 ]){
  const p=assessJ2ResearchStart(bad);
  assert.equal(p.eligible,false);
  assert.equal(p.stageStartAuthorized,false);
  assert.equal(p.rpcExecuted,false);
 }
});

test('J2 stage-start plan pins canonical RPC and deterministic CAS fingerprint without effects',()=>{
 const result=assessJ2ResearchStart(source);
 const p=buildJ2ResearchStartRpcPlan(result);
 assert.equal(p.rpc,'start_orotitan_stage');
 assert.equal(p.args.p_run_id,source.runId);
 assert.equal(p.args.p_stage_code,'RESEARCH');
 assert.equal(p.args.p_expected_run_state_version,2);
 assert.equal(p.args.p_stage_contract_name,pack.contract_pins.research_stage.name);
 assert.equal(p.args.p_stage_contract_version,'2.0');
 assert.equal(p.args.p_stage_contract_sha256,pack.contract_pins.research_stage.content_sha256);
 assert.match(p.args.p_request_fingerprint_sha256,/^[a-f0-9]{64}$/);
 assert.match(p.args.p_idempotency_key,/orotitan:j2:research:start:v1:/);
 assert.equal(p.args.p_actor_type,'PILOTAGE');
 assert.equal(p.confirmation,'GO J2 RESEARCH SHADOW '+source.runId);
 assert.equal(p.authorized,false);
 assert.equal(p.executed,false);
 assert.deepEqual(buildJ2ResearchStartRpcPlan(result),p);
});
test('J2 RPC plan refuses any ineligible or tampered admission decision',()=>{
 const accepted=assessJ2ResearchStart(source);
 for(const tampered of [
  {...accepted,eligible:false},
  {...accepted,reasonCode:'OTHER'},
  {...accepted,stageStartAuthorized:true},
  {...accepted,rpcExecuted:true},
  {...accepted,researchExecuted:true},
  {...accepted,publicationAuthorized:true},
  {...accepted,expectedRunStateVersion:null},
  {...accepted,expectedRunStateVersion:1},
  {...accepted,stageContractSha256:'0'.repeat(64)},
  {...accepted,contractSetSha256:'0'.repeat(64)},
 ]){
  assert.throws(()=>buildJ2ResearchStartRpcPlan(
   tampered as typeof accepted),/J2_STAGE_START_PLAN_NOT_ADMITTED/);
 }
});

test('J2 isolated SHADOW stage-start configuration cannot inherit J1/C8 access',()=>{
 const dummy='synthetic-only-not-a-key-aaaaaaaaaaaaaaaaaaaaaaaaaaaa';
 const isolatedProject='prj_111111111111111111111111';
 const valid={
  NODE_ENV:'production',VERCEL_ENV:'preview',
  VERCEL_PROJECT_ID:isolatedProject,
  OROTITAN_J2_STAGE_START_PROJECT_ID:isolatedProject,
  OROTITAN_J2_RESEARCH_STAGE_START_ENABLED:'1',
  OROTITAN_J2_STAGE_START_RUNTIME_GO:'1',
  OROTITAN_SHARED_INBOX_TEST_PROJECT_REF:SHADOW_RESEARCH_PROJECT_REF,
  OROTITAN_EXPECTED_SUPABASE_PROJECT_REF:SHADOW_RESEARCH_PROJECT_REF,
  NEXT_PUBLIC_SUPABASE_URL:SHADOW_RESEARCH_ORIGIN,
  OROTITAN_SHARED_INBOX_TEST_URL:SHADOW_RESEARCH_ORIGIN,
  OROTITAN_SHARED_INBOX_TEST_SERVICE_KEY:dummy,
  SUPABASE_SERVICE_ROLE_KEY:dummy,
  ADMIN_PASSWORD:'synthetic-password-for-tests',
  ADMIN_SESSION_SECRET:'synthetic-session-secret-at-least-32-character',
 };
 assert.equal(isHostedJ2ResearchStageStartEnabled(valid),true);
 for(const patch of [
  {OROTITAN_C8_READ_ONLY_ENABLED:'1'},
  {OROTITAN_C8_READ_ONLY_ENABLED:'0'},
  {OROTITAN_J1_RUN_FACTORY_ENABLED:'1'},
  {OROTITAN_HOSTED_SHADOW_RESEARCH_ENABLED:'1'},
  {OROTITAN_SHARED_INBOX_TEST_ENABLED:'1'},
  {OROTITAN_J2_RESEARCH_STAGE_START_ENABLED:'0'},
  {OROTITAN_J2_STAGE_START_RUNTIME_GO:'0'},
  {VERCEL_ENV:'production'},
  {VERCEL_PROJECT_ID:'prj_RBi6Sw0XcL0AbxSHVfoTwDkyQAHE'},
  {OROTITAN_J2_STAGE_START_PROJECT_ID:'prj_RBi6Sw0XcL0AbxSHVfoTwDkyQAHE'},
  {OROTITAN_J2_STAGE_START_PROJECT_ID:'prj_fake'},
  {NEXT_PUBLIC_SUPABASE_URL:'https://cugpgtzygqqlxetyeven.supabase.co'},
  {OROTITAN_SHARED_INBOX_TEST_URL:'https://cugpgtzygqqlxetyeven.supabase.co'},
  {SUPABASE_SERVICE_ROLE_KEY:'different-from-safe-test-key'},
 ]){
  assert.equal(isHostedJ2ResearchStageStartEnabled({...valid,...patch}),false,
   JSON.stringify(patch));
 }
});
