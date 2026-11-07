import assert from 'node:assert/strict';
import test from 'node:test';
import pack from '../contracts/orotitan-equity/v3/contract-pin-pack-v3/OROTITAN_CONTRACT_PIN_PACK_V3_0_2.json';
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
