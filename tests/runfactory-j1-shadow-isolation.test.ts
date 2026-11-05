import assert from 'node:assert/strict';
import {createRequire,register} from 'node:module';
import test from 'node:test';
import type {NextRequest} from 'next/server';
import {
  isHostedJ1RunFactoryEnabled,isHostedC8ReadOnlyEnabled,
  isHostedShadowResearchEnabled,SHADOW_RESEARCH_ORIGIN,
  SHADOW_RESEARCH_PROJECT_REF,
} from '../lib/orotitan-equity/product/hosted-shadow-research-gate';

createRequire(import.meta.url)('./helpers/deny-network.cjs');
const state={cookie:''};
(globalThis as typeof globalThis&{__j1Test:typeof state}).__j1Test=state;
register('./module-mock-hooks.mjs',import.meta.url,{data:{
  resolutions:[
    ['server-only','mock:j1-server-only'],
    ['next/headers','mock:j1-headers'],
  ],
  sources:[
    ['mock:j1-server-only','export {};'],
    ['mock:j1-headers',`export async function cookies(){return {
      get:()=>globalThis.__j1Test.cookie?{value:globalThis.__j1Test.cookie}:undefined,
      set:(_name,value)=>{globalThis.__j1Test.cookie=value;},
      delete:()=>{globalThis.__j1Test.cookie='';},
    };}`],
  ],
  modulePrefixes:[
    new URL('../app/',import.meta.url).href,
    new URL('../lib/',import.meta.url).href,
  ],
}});

const config={
  NODE_ENV:'production',VERCEL_ENV:'preview',
  OROTITAN_C8_READ_ONLY_ENABLED:'1',
  OROTITAN_C8_ACCEPTANCE_GO_ENABLED:'0',
  OROTITAN_J1_RUN_FACTORY_ENABLED:'1',
  OROTITAN_HOSTED_SHADOW_RESEARCH_ENABLED:'1',
  OROTITAN_SHARED_INBOX_TEST_ENABLED:'1',
  OROTITAN_SHARED_INBOX_TEST_PROJECT_REF:SHADOW_RESEARCH_PROJECT_REF,
  OROTITAN_EXPECTED_SUPABASE_PROJECT_REF:SHADOW_RESEARCH_PROJECT_REF,
  OROTITAN_SHARED_INBOX_TEST_URL:SHADOW_RESEARCH_ORIGIN,
  NEXT_PUBLIC_SUPABASE_URL:SHADOW_RESEARCH_ORIGIN,
  OROTITAN_SHARED_INBOX_TEST_SERVICE_KEY:'synthetic-key-not-a-secret-aaaaaaaaaaaaaaaaaaaaaaaa',
  SUPABASE_SERVICE_ROLE_KEY:'synthetic-key-not-a-secret-aaaaaaaaaaaaaaaaaaaaaaaa',
  ADMIN_PASSWORD:'synthetic-admin-password-longer-than-16',
  ADMIN_SESSION_SECRET:'synthetic-admin-session-secret-longer-than-32-chars',
};
const oldEnv=new Map(Object.keys(config).map(k=>[k,process.env[k]]));
const oldFetch=globalThis.fetch;
const calls:{method:string;pathname:string}[]=[];
const fetchMock:typeof fetch=async(input,init)=>{
  const url=new URL(typeof input==='string'?input:
    input instanceof URL?input.href:input.url);
  const method=(init?.method??(input instanceof Request?input.method:'GET')).toUpperCase();
  calls.push({method,pathname:url.pathname});
  assert.equal(url.origin,SHADOW_RESEARCH_ORIGIN);
  assert.equal(init?.redirect,'error');
  return new Response(JSON.stringify(method==='POST'?
    {status:'REPLAYED',run_id:'204390a1-5cba-4b9c-9821-ca0c7df43eca'}:[]),{
      status:200,headers:{'Content-Type':'application/json'},
    });
};
test.beforeEach(()=>{
  Object.assign(process.env,config);
  state.cookie='';calls.length=0;
  globalThis.fetch=fetchMock;
});
test.after(()=>{
  for(const [k,v] of oldEnv){
    if(v===undefined)delete process.env[k];else process.env[k]=v;
  }
  globalThis.fetch=oldFetch;
});

test('J1-only gate keeps C8 read-only and Research disabled',()=>{
  assert.equal(isHostedJ1RunFactoryEnabled(process.env),true);
  assert.equal(isHostedC8ReadOnlyEnabled(process.env),true);
  assert.equal(isHostedShadowResearchEnabled(process.env),false);
  for(const [key,bad] of [
    ['OROTITAN_J1_RUN_FACTORY_ENABLED','0'],
    ['OROTITAN_C8_READ_ONLY_ENABLED','0'],
    ['OROTITAN_C8_ACCEPTANCE_GO_ENABLED','1'],
    ['VERCEL_ENV','production'],
    ['OROTITAN_EXPECTED_SUPABASE_PROJECT_REF','cugpgtzygqqlxetyeven'],
    ['NEXT_PUBLIC_SUPABASE_URL','https://cugpgtzygqqlxetyeven.supabase.co'],
  ]){
    const baseline=process.env[key];process.env[key]=bad;
    assert.equal(isHostedJ1RunFactoryEnabled(process.env),false,key);
    process.env[key]=baseline;
  }
});

test('J1 transport allows only read Registry and one atomic run RPC',async()=>{
  const {getJ1RunFactoryShadowClient}=await import(
    '../lib/orotitan-equity/product/runfactory-j1-shadow-server');
  const client=getJ1RunFactoryShadowClient();
  const allowedRead=await client.from('issuers').select('issuer_id').limit(1);
  assert.equal(allowedRead.error,null);
  const allowedRpc=await client.rpc('orotitan_shadow_create_initial_run_v3_1',{});
  assert.equal(allowedRpc.error,null);
  assert.deepEqual(calls,[
    {method:'GET',pathname:'/rest/v1/issuers'},
    {method:'POST',pathname:'/rest/v1/rpc/orotitan_shadow_create_initial_run_v3_1'},
  ]);
  calls.length=0;
  for(const mutation of [
    ()=>client.from('orotitan_runs').insert({}),
    ()=>client.from('orotitan_runs').update({}).eq('run_id','fake'),
    ()=>client.from('orotitan_run_stages').insert({}),
    ()=>client.from('research_snapshots').insert({}),
    ()=>client.rpc('orotitan_j0_c8_accept_review_v1',{}),
    ()=>client.rpc('orotitan_onboard_identity_shadow_v1',{}),
    ()=>client.storage.from('private').upload('blocked',new Uint8Array([1])),
    ()=>client.from('companies').select('*'),
  ]){
    const r=await mutation();
    assert.ok(r.error,'request must fail closed');
  }
  assert.equal(calls.length,0);
});

const request=(path:string,origin='https://qualification.invalid',body='{}')=>
  new Request('https://qualification.invalid/admin/research-factory/run-preflight/'+path,{
    method:'POST',
    headers:{origin,'content-type':'application/json'},
    body,
  }) as NextRequest;

test('J1 POST endpoints refuse unauthenticated and cross-origin requests before I/O',async()=>{
  const inspect=await import('../app/admin/research-factory/run-preflight/inspect/route');
  const commit=await import('../app/admin/research-factory/run-preflight/commit/route');
  for(const handler of [inspect.POST,commit.POST]){
    const unauthorized=await handler(request('inspect'));
    assert.equal(unauthorized.status,401);
  }
  const auth=await import('../lib/auth/admin-session');
  await auth.createAdminSession();
  for(const handler of [inspect.POST,commit.POST]){
    const crossOrigin=await handler(request('inspect','https://other.invalid'));
    assert.equal(crossOrigin.status,403);
  }
  assert.equal(calls.length,0);
});

test('run admission route blocked with missing J1 permission',async()=>{
  process.env.OROTITAN_J1_RUN_FACTORY_ENABLED='0';
  const commit=await import('../app/admin/research-factory/run-preflight/commit/route');
  const r=await commit.POST(request('commit'));
  assert.equal(r.status,404);
  assert.equal(calls.length,0);
});
