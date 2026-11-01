import test from 'node:test';
import assert from 'node:assert/strict';
import {evaluatePreJalonReadinessV1, type PreJalonInput,
 type VerifiedDesignPlan} from '../lib/orotitan-equity/product/pre-jalon-readiness-v1.ts';
const SHA='a'.repeat(64);
const plan:VerifiedDesignPlan={
 immutableRef:'https://github.com/robzer13/indice_nexus/blob/'+
  'b'.repeat(40)+'/docs/J1_FUNCTIONALITY_FIRST_DESIGN.md',
 sha256:SHA,sourceHashVerified:true,exactEightCriteriaVerified:true,
 functionalOutcomeDefined:true,existingReuseConsidered:true,
 alternativesCompared:true,deliverySlicesAndProofsDefined:true,
 negativeTestsAndStopConditionsDefined:true,authorPrincipal:'author-id',
 reviewerPrincipal:'reviewer-id',reviewerIdentityVerified:true,
 independentReviewVerdict:'APPROVED',reviewedSha256:SHA,
};
const input:PreJalonInput={milestoneCode:'J1',
 criteria:Array.from({length:8},(_,i)=>'Criterion '+(i+1)),
 predecessorC8Accepted:false,predecessorOpenBlockers:1,
 plan:null,buildGo:null};
const evaluate=(overrides:Partial<PreJalonInput>={})=>
 evaluatePreJalonReadinessV1({...input,...overrides});
test('J1 stays draft-only while J0 C8 is unaccepted, even with apparent GO',()=>{
 const r=evaluate({plan,buildGo:{milestoneCode:'J1',
  authorityRef:'GO_BUILD_J1_SHADOW_20261101_OK',independentlyVerified:true}});
 assert.equal(r.buildAuthorized,false);
 assert.equal(r.planApproved,false);
 assert.equal(r.status,'DRAFT_ONLY');
 assert.ok(r.blockers.includes('PREDECESSOR_C8_NOT_ACCEPTED'));
 assert.ok(r.blockers.includes('PREDECESSOR_OPEN_BLOCKER'));
});
test('status fields from historical roadmap cannot replace C8 authority',()=>{
 const r=evaluate({predecessorOpenBlockers:0});
 assert.equal(r.buildAuthorized,false);
 assert.ok(r.blockers.includes('PREDECESSOR_C8_NOT_ACCEPTED'));
});
test('valid reviewed plan without separate GO cannot start build',()=>{
 const r=evaluate({predecessorC8Accepted:true,predecessorOpenBlockers:0,plan});
 assert.equal(r.planApproved,true);
 assert.equal(r.buildAuthorized,false);
 assert.equal(r.status,'PLAN_APPROVED');
 assert.ok(r.blockers.includes('DISTINCT_GO_BUILD_MISSING'));
});
test('approved reviewed plan + upstream C8 + distinct GO authorizes engineering build',()=>{
 const r=evaluate({predecessorC8Accepted:true,predecessorOpenBlockers:0,plan,
  buildGo:{milestoneCode:'J1',authorityRef:'GO_BUILD_J1_SHADOW_20261101_OK',
  independentlyVerified:true}});
 assert.equal(r.buildAuthorized,true);
 assert.equal(r.blockers.length,0);
 assert.equal(r.status,'BUILD_AUTHORIZED');
});
test('unverified reviewer identity, equal principals, changed plan hash all deny',()=>{
 for(const bad of [{reviewerIdentityVerified:false},
  {reviewerPrincipal:'author-id'}, {reviewedSha256:'c'.repeat(64)}]){
  const r=evaluate({predecessorC8Accepted:true,predecessorOpenBlockers:0,
   plan:{...plan,...bad}});
  assert.equal(r.planApproved,false);
  assert.ok(r.blockers.includes('INDEPENDENT_REVIEW_NOT_VERIFIED'));
 }
});
test('missing critical planning fields or verified source hash deny',()=>{
 for(const bad of [{deliverySlicesAndProofsDefined:false},
  {sourceHashVerified:false},{immutableRef:'https://evil.example/plan.md'},
  {negativeTestsAndStopConditionsDefined:false}]){
  const r=evaluate({predecessorC8Accepted:true,predecessorOpenBlockers:0,
   plan:{...plan,...bad}});
  assert.equal(r.buildAuthorized,false);
 }
});
test('wrong milestone GO and unverified GO deny',()=>{
 for(const bad of [{milestoneCode:'J2'}, {independentlyVerified:false},
  {authorityRef:'GO_PUBLISH_J1_SHADOW_20261101_OK'}]){
  const r=evaluate({predecessorC8Accepted:true,predecessorOpenBlockers:0,plan,
   buildGo:{milestoneCode:'J1',authorityRef:'GO_BUILD_J1_SHADOW_20261101_OK',
    independentlyVerified:true,...bad}});
  assert.equal(r.buildAuthorized,false);
  assert.ok(r.blockers.includes('GO_BUILD_NOT_VERIFIED'));
 }
});
test('fail closed for invalid criteria, unknown issue count, and non J1-J8',()=>{
 assert.equal(evaluate({criteria:['Only one']}).buildAuthorized,false);
 assert.equal(evaluate({predecessorOpenBlockers:-1}).buildAuthorized,false);
 assert.throws(()=>evaluate({milestoneCode:'J0'}),/PJDG_NOT/);
 assert.throws(()=>evaluate({milestoneCode:'J9'}),/PJDG_NOT/);
});
test('revision verdict means rework, not approval',()=>{
 const r=evaluate({predecessorC8Accepted:true,predecessorOpenBlockers:0,
  plan:{...plan,independentReviewVerdict:'REVISE'}});
 assert.equal(r.status,'REVISE_PLAN');
 assert.equal(r.planApproved,false);
});
