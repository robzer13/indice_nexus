import assert from 'node:assert/strict';
import pinPack from '../contracts/orotitan-equity/v3/contract-pin-pack-v3/OROTITAN_CONTRACT_PIN_PACK_V3_0_2.json';
import test from 'node:test';
import {resolveJ1CompanyIdentity,type J1CompanyIdentity} from
 '../lib/orotitan-equity/product/runfactory-j1-request-resolution-v1';
import {buildJ1ResearchHandoff,type J1VerifiedRun} from
 '../lib/orotitan-equity/product/runfactory-j1-research-handoff-v1';

const schneider:J1CompanyIdentity={
 isin:'FR0000121972',exchange:'XPAR',ticker:'SU',
 issuerId:'81736c80-9d6a-4f4e-bbc1-1b418912d94a',
 securityId:'e60d26ba-925c-4e02-95b0-6ffd53a73d42',
 issuerName:'Schneider Electric SE',
};
const loreal:J1CompanyIdentity={
 isin:'FR0000120321',exchange:'XPAR',ticker:'OR',
 issuerId:'aaa12345-aaaa-aaaa-aaaa-aaaaaaaaaaaa',
 securityId:'bbb12345-bbbb-bbbb-bbbb-bbbbbbbbbbbb',
 issuerName:"L'Oréal S.A.",
};
test('J1 single request resolves exact ISIN, ticker and issuer name prefix',()=>{
 for(const query of ['FR0000121972','fr0000121972','SU',
                     'Schneider Electric','Schneider Electric SE']){
  const result=resolveJ1CompanyIdentity(query,[loreal,schneider]);
  assert.equal(result.status,'RESOLVED',query);
  assert.deepEqual(result.identity,schneider);
 }
});
test('J1 no company guessing: absent, competing listings and competing issuer names',()=>{
 assert.deepEqual(resolveJ1CompanyIdentity('Unknown Company',[schneider]),
  {status:'NOT_FOUND',identity:null});
 const other={...schneider,exchange:'XAMS',securityId:'11111111-1111-4111-8111-111111111111'};
 assert.deepEqual(resolveJ1CompanyIdentity('SU',[schneider,other]),
  {status:'AMBIGUOUS',identity:null});
 assert.deepEqual(resolveJ1CompanyIdentity('FR0000121972',[schneider,other]),
  {status:'AMBIGUOUS',identity:null});
 const second={...schneider,isin:'FR0000120321',issuerName:'Schneider Electric SA'};
 assert.deepEqual(resolveJ1CompanyIdentity('Schneider Electric',[schneider,second]),
  {status:'AMBIGUOUS',identity:null});
});
const verified:J1VerifiedRun={
 runId:'db00ce77-04eb-492b-a76b-164ecbe9ccfe',
 issuerName:'Schneider Electric SE',isin:'FR0000121972',exchange:'XPAR',
 dataCutoff:'2026-10-30',profileId:'OROTITAN_V3_1',
 contractSetSha256:'3644e501909326af04d66730fa30b1ac3da0d82fb6b717202af6d948f3211fe2',
 status:'CREATED',persistedReadbackVerified:true,receiptPersisted:true,
 stageStarted:false,snapshotCreated:false,published:false,
 environment:'SHADOW_ONLY',productionTouched:false,
};
test('J1 Research handoff is deterministic and never asserts Research admission',()=>{
 const text=buildJ1ResearchHandoff(verified);
 assert.match(text,/RUN_ID = db00ce77-04eb-492b-a76b-164ecbe9ccfe/);
 assert.match(text,/EXPECTED_STAGE_CONTRACT = OROTITAN_RESEARCH_STAGE_CONTRACT_V2_FREEZE_V2.0/);
 assert.match(text,/RESEARCH_STAGE_ADMISSION = NOT_YET_EXECUTED/);
 assert.match(text,/RESEARCH_STAGE_STARTED = NO/);
 assert.match(text,/PUBLICATION_AUTHORIZED = NO/);
 assert.equal(buildJ1ResearchHandoff(verified),text);
});
test('J1 handoff fails closed if server receipt is not a verified SHADOW Run',()=>{
 for(const invalid of [
  {...verified,receiptPersisted:false},
  {...verified,stageStarted:true},
  {...verified,snapshotCreated:true},
  {...verified,published:true},
  {...verified,environment:'PRODUCTION'},
  {...verified,productionTouched:true},
  {...verified,contractSetSha256:'unverified'},
 ]){
  assert.throws(()=>buildJ1ResearchHandoff(invalid as J1VerifiedRun),
   /J1_HANDOFF_INVARIANT_FAILED/);
 }
});

test('J1 Research handoff exact names match the immutable V3.1 contract pack',()=>{
 const h=buildJ1ResearchHandoff(verified);
 for(const [field,key] of [
  ['HIGHER_AUTHORITY_PROCESS','process'],
  ['PILOTAGE_CONTRACT','pilotage'],
  ['EXPECTED_STAGE_CONTRACT','research_stage'],
 ] as const){
  assert.match(h,new RegExp(field+' = '+pinPack.contract_pins[key].name));
 }
 assert.equal(pinPack.contract_set_sha256,verified.contractSetSha256);
});
