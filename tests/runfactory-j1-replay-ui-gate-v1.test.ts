import test from 'node:test';
import assert from 'node:assert/strict';
import {
 isJ1InitialAdmissionCandidate,matchesJ1ExpectedPersistedRun,
 matchesJ1RunOutcome,type RunFactoryDecision,
} from '../lib/orotitan-equity/product/runfactory-identity-preflight-v1';

const validExisting:RunFactoryDecision={
  status:'EXISTING_RUN_REUSE_CANDIDATE',
  profileId:'OROTITAN_V3_1',
  contractSetSha256:'3644e501909326af04d66730fa30b1ac3da0d82fb6b717202af6d948f3211fe2',
  isin:'FR0000120321',exchange:'XPAR',dataCutoff:'2026-10-30',
  issuerId:'c431698b-3b5f-40c4-bf24-8bf4612db876',
  issuerName:'L’Oréal',
  securityId:'44bf0acd-9420-42f3-bdd1-1e725740154d',
  dossierId:'e30bd3e4-618c-41a2-93b5-015fc2ea708e',
  baselineSnapshotId:null,proposedRunType:'INITIAL',
  existingRunId:'204390a1-5cba-4b9c-9821-ca0c7df43eca',
  existingRunStatus:'CREATED',
  candidateCreationIdempotencyKey:null,
  nextAction:'Re-query persisted run and state_version; do not create a duplicate.',
  evidenceStageAdmitted:false,runCreated:false,rpcExecuted:false,
  goAuthorized:false,registryAdmitted:false,planExecutableInShadow:false,
};
test('J1 explicitly permits a persisted CREATED INITIAL run replay candidate',()=>{
  assert.equal(isJ1InitialAdmissionCandidate(validExisting),true);
});
test('J1 still permits a new correctly qualified INITIAL candidate',()=>{
  assert.equal(isJ1InitialAdmissionCandidate({
    ...validExisting,status:'NEW_RUN_PLAN_CANDIDATE',
    existingRunId:null,existingRunStatus:null,
  }),true);
});
test('J1 rejects stale, changed, ambiguous and non-INITIAL replay candidates',()=>{
  const denials:RunFactoryDecision[]=[
    {...validExisting,existingRunStatus:'ACTIVE'},
    {...validExisting,existingRunStatus:'CANCELLED'},
    {...validExisting,existingRunId:null},
    {...validExisting,proposedRunType:'REFRESH'},
    {...validExisting,baselineSnapshotId:'e30bd3e4-618c-41a2-93b5-015fc2ea708e'},
    {...validExisting,profileId:'OROTITAN_V2_1_1' as RunFactoryDecision['profileId']},
    {...validExisting,issuerId:null},
    {...validExisting,securityId:null},
    {...validExisting,dossierId:null},
    {...validExisting,runCreated:true as never},
    {...validExisting,registryAdmitted:true as never},
    {...validExisting,status:'BLOCKED_EXISTING_METHODOLOGY_CONFLICT'},
    {...validExisting,status:'NEW_RUN_PLAN_CANDIDATE'},
  ];
  for(const decision of denials)
    assert.equal(isJ1InitialAdmissionCandidate(decision),false,decision.status);
});

test('J1 CREATE and REPLAY expectations fail closed before RPC on changed registry',()=>{
 const id='204390a1-5cba-4b9c-9821-ca0c7df43eca';
 const existing=[{run_id:id,run_status:'CREATED'}];
 assert.equal(matchesJ1ExpectedPersistedRun(id,existing),true);
 assert.equal(matchesJ1ExpectedPersistedRun(id,[]),false);
 assert.equal(matchesJ1ExpectedPersistedRun(id,[{...existing[0],run_id:'other'}]),false);
 assert.equal(matchesJ1ExpectedPersistedRun(id,[{...existing[0],run_status:'ACTIVE'}]),false);
 assert.equal(matchesJ1ExpectedPersistedRun(id,[...existing,...existing]),false);
 assert.equal(matchesJ1ExpectedPersistedRun(null,[]),true);
 assert.equal(matchesJ1ExpectedPersistedRun(null,existing),false);
});
test('J1 reply must match explicit replayed ID, not a newly created Run',()=>{
 const id='204390a1-5cba-4b9c-9821-ca0c7df43eca';
 assert.equal(matchesJ1RunOutcome(id,id,'REPLAYED'),true);
 assert.equal(matchesJ1RunOutcome(id,id,'CREATED'),false);
 assert.equal(matchesJ1RunOutcome(id,'different','REPLAYED'),false);
 assert.equal(matchesJ1RunOutcome(null,id,'CREATED'),true);
 assert.equal(matchesJ1RunOutcome(null,id,'REPLAYED'),false);
});
