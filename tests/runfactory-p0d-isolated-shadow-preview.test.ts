import test from 'node:test';
import assert from 'node:assert/strict';
import {pgrstClockSkewRetryFetch} from '../lib/supabase/pgrst-clock-retry';
import {
  SHADOW_RESEARCH_ORIGIN,SHADOW_RESEARCH_PROJECT_REF,
  isHostedShadowOnboardingOnlyEnabled,isHostedShadowResearchEnabled,
  isHostedJ1RunFactoryEnabled,isP0DOnboardingTransportAllowed,
  isHostedShadowOnboardingIsolationRequested,isP0DOnboardingHttpPathAllowed,
  assertOutsideOnboardingOnly,
} from '../lib/orotitan-equity/product/hosted-shadow-research-gate';

const configuration={
  NODE_ENV:'production',
  VERCEL_ENV:'preview',
  NEXT_PUBLIC_SUPABASE_URL:SHADOW_RESEARCH_ORIGIN,
  OROTITAN_SHARED_INBOX_TEST_URL:SHADOW_RESEARCH_ORIGIN,
  OROTITAN_SHARED_INBOX_TEST_PROJECT_REF:SHADOW_RESEARCH_PROJECT_REF,
  OROTITAN_EXPECTED_SUPABASE_PROJECT_REF:SHADOW_RESEARCH_PROJECT_REF,
  OROTITAN_SHARED_INBOX_TEST_SERVICE_KEY:'s'.repeat(64),
  SUPABASE_SERVICE_ROLE_KEY:'s'.repeat(64),
  ADMIN_PASSWORD:'p'.repeat(20),
  ADMIN_SESSION_SECRET:'a'.repeat(48),
  OROTITAN_HOSTED_SHADOW_ONBOARDING_ONLY_ENABLED:'1',
  OROTITAN_HOSTED_SHADOW_RESEARCH_ENABLED:'0',
  OROTITAN_SHARED_INBOX_TEST_ENABLED:'0',
} as const;

test('P0-D dedicated SHADOW Preview excludes generic Research, J1 and C8',()=>{
  assert.equal(isHostedShadowOnboardingOnlyEnabled(configuration),true);
  assert.equal(isHostedShadowResearchEnabled(configuration),false);
  assert.equal(isHostedJ1RunFactoryEnabled(configuration),false);
  for(const change of [
    {VERCEL_ENV:'production'},
    {OROTITAN_EXPECTED_SUPABASE_PROJECT_REF:'cugpgtzygqqlxetyeven'},
    {OROTITAN_HOSTED_SHADOW_RESEARCH_ENABLED:'1'},
    {OROTITAN_SHARED_INBOX_TEST_ENABLED:'1'},
    {OROTITAN_J1_RUN_FACTORY_ENABLED:'1'},
    {OROTITAN_C8_READ_ONLY_ENABLED:'1'},
    {SUPABASE_SERVICE_ROLE_KEY:'wrong-secret'},
    {OROTITAN_HOSTED_SHADOW_ONBOARDING_ONLY_ENABLED:'0'},
  ]){
    assert.equal(isHostedShadowOnboardingOnlyEnabled({...configuration,...change}),false);
  }
});

test('P0-D transport allows identity reads and only the governed onboarding RPC',()=>{
  const allows=(path:string,method:string,base=SHADOW_RESEARCH_ORIGIN)=>
    isP0DOnboardingTransportAllowed(new URL(path,base),method);
  for(const path of [
    '/rest/v1/issuers?select=issuer_id',
    '/rest/v1/securities?select=isin',
    '/rest/v1/research_dossiers?select=dossier_id',
    '/rest/v1/companies?select=id',
    '/rest/v1/legacy_company_identity_map?select=issuer_id',
    '/rest/v1/orotitan_identity_onboarding_receipts_v1?select=request_key',
  ])assert.equal(allows(path,'GET'),true,path);
  assert.equal(allows('/rest/v1/rpc/orotitan_onboard_identity_shadow_v1','POST'),true);
  for(const [path,method] of [
    ['/rest/v1/orotitan_runs','GET'],
    ['/rest/v1/orotitan_run_stages','GET'],
    ['/rest/v1/rpc/orotitan_shadow_create_initial_run_v3_1','POST'],
    ['/rest/v1/rpc/orotitan_j0_c8_accept_review_v1','POST'],
    ['/rest/v1/securities','POST'],
    ['/rest/v1/issuers','PATCH'],
    ['/rest/v1/rpc/orotitan_onboard_identity_shadow_v1?unexpected=1','POST'],
    ['/rest/v1/rpc/orotitan_onboard_identity_shadow_v1','GET'],
  ])assert.equal(allows(path,method),false,path);
  assert.equal(allows('/rest/v1/securities','GET','https://cugpgtzygqqlxetyeven.supabase.co'),false);
  assert.equal(allows('/rest/v1/securities#x','GET'),false);
});

test('onboarding-only reservation blocks ordinary application APIs and DB clients',()=>{
 assert.equal(isHostedShadowOnboardingIsolationRequested(configuration),true);
 assert.equal(isHostedShadowOnboardingIsolationRequested({}),false);
 assert.throws(()=>assertOutsideOnboardingOnly(configuration),/P0D_GENERIC_SUPABASE_CLIENT_DENIED/);
 assert.doesNotThrow(()=>assertOutsideOnboardingOnly({}));
 for(const path of [
  '/admin','/admin/research-factory/onboarding',
  '/admin/research-factory/onboarding/inspect',
  '/admin/research-factory/onboarding/execute',
  '/admin/research-factory/onboarding/commit',
 ])assert.equal(isP0DOnboardingHttpPathAllowed(path),true,path);
 for(const path of [
  '/', '/api/sync','/api/health','/admin/companies/new',
  '/admin/snapshots/new','/admin/research-factory/run-preflight',
  '/admin/research-factory/onboarding-malicious',
  '/admin/research-factory/onboarding2',
 ])assert.equal(isP0DOnboardingHttpPathAllowed(path),false,path);
});

test('P0-C clock-skew retry recovers only SHADOW identity GETs',async()=>{
 const url=SHADOW_RESEARCH_ORIGIN+'/rest/v1/securities?select=isin&ticker=ilike.SU';
 const clockError=()=>Response.json(
  {code:'PGRST303',message:'JWT issued at future'}, {status:401},
 );
 let attempts=0;
 const guarded=pgrstClockSkewRetryFetch(async()=>{
  attempts++;
  return attempts===1?clockError():Response.json([],{status:200});
 },SHADOW_RESEARCH_ORIGIN,async()=>{});
 assert.equal(isP0DOnboardingTransportAllowed(new URL(url),'GET'),true);
 const response=await guarded(url,{method:'GET'});
 assert.equal(response.status,200);
 assert.equal(attempts,2);
 // The same retry wrapper must never retry a failed POST, even if its
 // response happens to carry the otherwise retryable clock-skew code.
 attempts=0;
 const notRetried=pgrstClockSkewRetryFetch(async()=>{
  attempts++;return clockError();
 },SHADOW_RESEARCH_ORIGIN,async()=>{});
 const commitPath=SHADOW_RESEARCH_ORIGIN+'/rest/v1/rpc/orotitan_onboard_identity_shadow_v1';
 assert.equal(isP0DOnboardingTransportAllowed(new URL(commitPath),'POST'),true);
 assert.equal((await notRetried(commitPath,{method:'POST'})).status,401);
 assert.equal(attempts,1);
});
