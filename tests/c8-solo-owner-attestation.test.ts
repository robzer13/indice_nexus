import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {join} from 'node:path';
import {assertSoloOwnerUser,getSoloOwnerConfig} from '../lib/auth/c8-solo-owner-core';

const owner='11111111-1111-4111-8111-111111111111';
const env={OROTITAN_C8_OWNER_AUTH_ENABLED:'1',
 OROTITAN_C8_OWNER_AUTH_URL:'https://awgsurdyvsyolcgpnygh.supabase.co',
 OROTITAN_EXPECTED_SUPABASE_PROJECT_REF:'awgsurdyvsyolcgpnygh',
 OROTITAN_C8_OWNER_AUTH_PUBLISHABLE_KEY:'sb_publishable_fictional_test_key',
 OROTITAN_C8_SOLO_OWNER_USER_ID:owner,OROTITAN_C8_SOLO_AMENDMENT_RATIFIED:'1',
 OROTITAN_C8_SOLO_AMENDMENT_REF:'https://github.com/robzer13/indice_nexus/blob/'+
  'a'.repeat(40)+'/docs/audits/solo.md',
 OROTITAN_C8_SOLO_AMENDMENT_SHA256:'b'.repeat(64)};

test('SOLO owner config is exact-SHADOW, ratified, UUID allowlisted and never secret-keyed',()=>{
 const config=getSoloOwnerConfig(env);assert.equal(config.ownerUserId,owner);
 for(const override of [
  {OROTITAN_C8_SOLO_AMENDMENT_RATIFIED:'0'},
  {OROTITAN_EXPECTED_SUPABASE_PROJECT_REF:'cugpgtzygqqlxetyeven'},
  {OROTITAN_C8_OWNER_AUTH_URL:'https://cugpgtzygqqlxetyeven.supabase.co'},
  {OROTITAN_C8_OWNER_AUTH_PUBLISHABLE_KEY:'sb_secret_fictional_test_key'},
  {OROTITAN_C8_SOLO_OWNER_USER_ID:'not-a-uuid'},
 ])assert.throws(()=>getSoloOwnerConfig({...env,...override}),/OWNER_AUTH_CONFIGURATION_DENIED/);
});

test('mocked Auth identity requires exact UUID, confirmed email and non-anonymous account',()=>{
 const config=getSoloOwnerConfig(env);
 const verified={id:owner,email_confirmed_at:'2026-11-05T08:00:00Z',is_anonymous:false};
 assert.deepEqual(assertSoloOwnerUser(verified,config),{
  authProvider:'SUPABASE_AUTH',ownerUserId:owner,governanceMode:'SOLO_OWNER',
  amendmentRef:env.OROTITAN_C8_SOLO_AMENDMENT_REF,
  amendmentSha256:env.OROTITAN_C8_SOLO_AMENDMENT_SHA256});
 assert.throws(()=>assertSoloOwnerUser({...verified,id:'22222222-2222-4222-8222-222222222222'},config));
 assert.throws(()=>assertSoloOwnerUser({...verified,email_confirmed_at:undefined},config));
 assert.throws(()=>assertSoloOwnerUser({...verified,is_anonymous:true},config));
});

test('SOLO HTTP operations attest server-side and seal no email or client identity',()=>{
 const prepare=readFileSync(join(process.cwd(),'app/admin/research-factory/control-tower/c8-gate/prepare/route.ts'),'utf8');
 const accept=readFileSync(join(process.cwd(),'app/admin/research-factory/control-tower/c8-gate/accept/route.ts'),'utf8');
 for(const source of [prepare,accept]){
  assert.match(source,/requireAdminSession\(\)/);assert.match(source,/requireSoloOwnerAttestation\(\)/);
  assert.match(source,/ORIGIN_MISMATCH/);assert.match(source,/supabase-auth:/);
  assert.doesNotMatch(source,/p_owner_user_id:p\./);assert.doesNotMatch(source,/p_auth_provider:p\./);
 }
 assert.match(accept,/ACCEPTER J0 SHADOW EN MODE SOLO/);
 assert.match(accept,/OROTITAN_C8_ACCEPTANCE_GO_ENABLED/);
 assert.match(accept,/J0_C8_GOVERNANCE_MODE_MISMATCH/);
 assert.match(accept,/\^GO_ACCEPT_J0_SOLO_/);
});

test('additive SQL keeps STANDARD and SOLO distinct and all SOLO locks fail closed',()=>{
 const sql=readFileSync(join(process.cwd(),'migrations/20261105_orotitan_j0_c8_solo_owner_attestation_shadow.sql'),'utf8');
 assert.match(sql,/DEFAULT 'STANDARD_TWO_PERSON'/);assert.match(sql,/solo_owner_enabled boolean NOT NULL DEFAULT false/);
 assert.match(sql,/J0_C8_SOLO_AMENDMENT_NOT_RATIFIED/);assert.match(sql,/p_milestone IS DISTINCT FROM 'J0'/);
 assert.match(sql,/ACCEPTER J0 SHADOW EN MODE SOLO/);assert.match(sql,/GO_ACCEPT_J0_SOLO_/);
 assert.match(sql,/CREATE OR REPLACE FUNCTION public[.]orotitan_j0_c8_accept_review_v1\(/);
 assert.match(sql,/v_lock[.]solo_owner_enabled IS DISTINCT FROM false/);
 assert.match(sql,/J0_C8_GOVERNANCE_MODE_MISMATCH/);
 for(const field of ['p_auth_provider','p_owner_user_id','p_amendment_ref','p_amendment_sha256'])
  assert.match(sql,new RegExp(field+' (?:IS DISTINCT FROM|IS NULL)'));
 assert.match(sql,/solo_amendment_ref IS NOT NULL AND solo_amendment_sha256 IS NOT NULL/);
 assert.match(sql,/decision_sha256/);assert.match(sql,/'stageStarted',false,'published',false/);
 assert.doesNotMatch(sql,/UPDATE\s+public[.]orotitan_runs/i);
 assert.doesNotMatch(sql,/UPDATE\s+public[.]orotitan_j0_milestones_v1/i);
});
