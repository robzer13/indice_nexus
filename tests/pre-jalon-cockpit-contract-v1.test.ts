import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {join} from 'node:path';

const read=(path:string)=>readFileSync(join(process.cwd(),path),'utf8');
const page=read('app/admin/research-factory/control-tower/page.tsx');
const panel=read('app/admin/research-factory/control-tower/pre-jalon-panel.tsx');
const policy=read('lib/orotitan-equity/product/pre-jalon-readiness-v1.ts');

test('pre-jalon is SHADOW authenticated and derives predecessor authority from C8 DB',()=>{
 assert.match(page,/isAdminAuthenticated\(\)/);
 assert.match(page,/isHostedShadowResearchEnabled\(process\.env\)/);
 assert.match(page,/from\('orotitan_j0_c8_decisions_v1'\)/);
 assert.match(page,/\.eq\('decision_status','ACCEPTED'\)/);
 assert.match(page,/d\.scope==='SHADOW_ONLY'/);
 assert.match(page,/predecessorC8Accepted:acceptedMilestones\.has\(predecessor\)/);
 assert.match(page,/predecessorOpenBlockers:unresolvedJ0Issues\(events,predecessor\)\.length/);
});
test('historical initial_status cannot render a false accepted milestone',()=>{
 assert.match(page,/const accepted=acceptedMilestones\.size/);
 assert.match(page,/acceptedMilestones\.has\(m\.milestone_code\)\?'ACCEPTE'/);
 assert.match(page,/observedStatus==='ACCEPTE'\?'EN_COURS'/);
 assert.doesNotMatch(page,/const accepted=milestones\.filter/);
});
test('J1-J8 view never accepts a client supplied plan or GO',()=>{
 assert.match(page,/plan:null,buildGo:null/);
 assert.match(page,/criteria:Array\.isArray\(m\.gate_criteria\)/);
 assert.match(page,/<PreJalonPanel entries=\{preJalonEntries\}/);
 assert.doesNotMatch(panel,/<button/);
 assert.doesNotMatch(panel,/fetch\(/);
 assert.doesNotMatch(panel,/supabase/);
});
test('build permission requires an exact, verified plan, separate review and GO',()=>{
 assert.match(policy,/exactEightCriteriaVerified/);
 assert.match(policy,/sourceHashVerified!==true/);
 assert.match(policy,/reviewerIdentityVerified/);
 assert.match(policy,/plan\.reviewedSha256!==plan\.sha256/);
 assert.match(policy,/goPattern/);
 assert.match(policy,/independentlyVerified!==true/);
 assert.match(policy,/predecessorC8Accepted!==true/);
});
test('draft J1 plan contains exact eight official exit criteria, no authorization',()=>{
 const plan=read('docs/pre-jalons/J1_FUNCTIONALITY_FIRST_DESIGN_DRAFT_20261101.md');
 for(const criterion of [
  'L’Oréal HTTP REPLAYED','Second émetteur CREATED',
  '401/403 et RLS','Concurrence et doublons','Reprise d’erreur',
  'Aucun stage ou snapshot','Lecture Registry indépendante',
  'Acceptation humaine']){
  assert.ok(plan.includes(criterion),criterion);
 }
 assert.match(plan,/DRAFT_ONLY \/ NO-GO BUILD/);
 assert.match(plan,/GO BUILD J1 :\*\* absent/);
});
