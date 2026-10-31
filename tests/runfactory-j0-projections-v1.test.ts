import assert from 'node:assert/strict';
import test from 'node:test';
import {unresolvedJ0Issues,j0DisplayedStatus,validateJ0EvidenceCriterion}
 from '../lib/orotitan-equity/product/runfactory-j0-projections-v1';

const events=[
 {event_id:'a',event_seq:1,milestone_code:'J0',event_type:'BLOCKER_OPENED',
  evidence_ref:'issue:j0-ui',created_at:'2026-10-31T12:00:00Z'},
 {event_id:'b',event_seq:2,milestone_code:'J0',event_type:'BLOCKER_RESOLVED',
  evidence_ref:'issue:j0-ui',created_at:'2026-10-31T12:03:00Z'},
 {event_id:'c',event_seq:3,milestone_code:'J1',event_type:'BLOCKER_OPENED',
  evidence_ref:'issue:j1-http',created_at:'2026-10-31T12:04:00Z'},
 {event_id:'d',event_seq:4,milestone_code:'J0',event_type:'BLOCKER_OPENED',
  evidence_ref:'issue:j0-registry',created_at:'2026-10-31T12:05:00Z'},
] as const;
test('J0 issues: open/close sequences are deterministic and milestone-scoped',()=>{
 assert.deepEqual(unresolvedJ0Issues([...events].reverse(),'J0')
  .map(x=>x.evidence_ref),['issue:j0-registry']);
 assert.deepEqual(unresolvedJ0Issues(events,'J1')
  .map(x=>x.evidence_ref),['issue:j1-http']);
});
test('J0 issue resolution remains closed on duplicate/out-of-order display',()=>{
 const extra=[...events,{...events[1],event_id:'e',event_seq:5,
  created_at:'2026-10-31T12:06:00Z'}];
 assert.equal(unresolvedJ0Issues(extra,'J0').length,1);
});
test('J0 blocker overrides any initial status; never auto-accept',()=>{
 assert.equal(j0DisplayedStatus({initialStatus:'TESTE',unresolvedBlockerCount:1}),'BLOQUE');
 assert.equal(j0DisplayedStatus({initialStatus:'EN_COURS',unresolvedBlockerCount:0}),'EN_COURS');
 assert.notEqual(j0DisplayedStatus({initialStatus:'TESTE',unresolvedBlockerCount:0}),'ACCEPTE');
});
test('J0 rejects invalid state or invalid incident count',()=>{
 assert.throws(()=>j0DisplayedStatus({initialStatus:'ACCEPTED',unresolvedBlockerCount:0}),/J0_UNKNOWN/);
 assert.throws(()=>j0DisplayedStatus({initialStatus:'TESTE',unresolvedBlockerCount:-1}),/J0_BLOCKER/);
});

test('J0 evidence can be linked to a valid existing acceptance criterion',()=>{
 assert.equal(validateJ0EvidenceCriterion(1,['One criterion','Second criterion']),1);
 assert.equal(validateJ0EvidenceCriterion(2,['One criterion','Second criterion']),2);
});
test('J0 rejects forged, missing, out-of-bounds and malformed criterion links',()=>{
 for(const bad of [undefined,null,'1',0,-1,3,1.2,NaN]){
  assert.throws(()=>validateJ0EvidenceCriterion(bad,['One criterion','Second criterion']),
   /J0_EVIDENCE_CRITERION_INVALID/);
 }
 assert.throws(()=>validateJ0EvidenceCriterion(1,[]),/J0_GATE_CRITERIA_INVALID/);
 assert.throws(()=>validateJ0EvidenceCriterion(1,[{}]),/J0_GATE_CRITERIA_INVALID/);
});
