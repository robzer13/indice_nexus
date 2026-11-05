import assert from 'node:assert/strict';
import {createRequire, register} from 'node:module';
import dns, {resolveTxt as callbackResolveTxt} from 'node:dns';
import dnsPromises, {resolveTxt as promiseResolveTxt} from 'node:dns/promises';
import {runInNewContext} from 'node:vm';
import {readFileSync, readdirSync} from 'node:fs';
import {join, relative} from 'node:path';
import test from 'node:test';
import type {NextRequest} from 'next/server';
import {
  isHostedC8ReadOnlyEnabled, isHostedShadowResearchEnabled,
  SHADOW_RESEARCH_ORIGIN, SHADOW_RESEARCH_PROJECT_REF,
} from '../lib/orotitan-equity/product/hosted-shadow-research-gate';

createRequire(import.meta.url)('./helpers/deny-network.cjs');

const state = {cookie:'', effects:[] as string[]};
(globalThis as typeof globalThis & {__c8Test:typeof state}).__c8Test = state;
const sources = new Map([
  ['mock:c8-server-only','export {};'],
  ['mock:c8-headers', `export async function cookies() { return {
    get:()=>globalThis.__c8Test.cookie ? {value:globalThis.__c8Test.cookie}:undefined,
    set:(_name,value)=>{globalThis.__c8Test.cookie=value;},
    delete:()=>{globalThis.__c8Test.cookie='';}
  }; }`],
  ['mock:c8-navigation', `export function redirect(path) {throw Error('REDIRECT:'+path);}`],
  ['mock:c8-cache', `export function revalidatePath() {globalThis.__c8Test.effects.push('revalidate');}`],
  ['mock:c8-effects', `const effect=(name)=>async()=>{
    globalThis.__c8Test.effects.push(name);throw Error('UNEXPECTED_EFFECT:'+name);
  };
  export const executeShadowRecoveryCycle=effect('recovery');
  export const createCompany=effect('company');
  export const updateCompanyMetadata=effect('company-update');
  export const getAllCompanies=effect('company-read');
  export const getRecentMarketSyncRuns=effect('market-read');
  export const createImmutableSnapshot=effect('snapshot');
  export const refreshMarketPrices=effect('prices');`],
]);
register('./module-mock-hooks.mjs', import.meta.url, {data:{
  resolutions:[
    ['server-only','mock:c8-server-only'], ['next/headers','mock:c8-headers'],
    ['next/navigation','mock:c8-navigation'], ['next/cache','mock:c8-cache'],
    ...['@/lib/orotitan-equity/product/research-recovery-shadow-server',
      '@/lib/data/company-admin','@/lib/data/market-prices','@/lib/data/snapshots',
      '@/lib/market/refresh-prices'].map(name=>[name,'mock:c8-effects']),
  ], sources:[...sources], modulePrefixes:[
    new URL('../app/',import.meta.url).href, new URL('../lib/',import.meta.url).href,
  ],
}});

const synthetic = {
  NODE_ENV:'production', VERCEL_ENV:'preview', OROTITAN_C8_READ_ONLY_ENABLED:'1',
  // Intentionally also enable Research: C8 must override it.
  OROTITAN_HOSTED_SHADOW_RESEARCH_ENABLED:'1', OROTITAN_SHARED_INBOX_TEST_ENABLED:'1',
  OROTITAN_SHARED_INBOX_TEST_PROJECT_REF:SHADOW_RESEARCH_PROJECT_REF,
  OROTITAN_EXPECTED_SUPABASE_PROJECT_REF:SHADOW_RESEARCH_PROJECT_REF,
  OROTITAN_SHARED_INBOX_TEST_URL:SHADOW_RESEARCH_ORIGIN,
  NEXT_PUBLIC_SUPABASE_URL:SHADOW_RESEARCH_ORIGIN,
  OROTITAN_SHARED_INBOX_TEST_SERVICE_KEY:'synthetic-only-not-a-key-aaaaaaaaaaaaaaaaaaaaaaaaaaaa',
  SUPABASE_SERVICE_ROLE_KEY:'synthetic-only-not-a-key-aaaaaaaaaaaaaaaaaaaaaaaaaaaa',
  ADMIN_PASSWORD:'synthetic-admin-password-32characters',
  ADMIN_SESSION_SECRET:'synthetic-session-secret-for-shadow-only-32characters',
  CRON_SECRET:'synthetic-cron',
};
const originals = new Map(Object.keys(synthetic).map(key=>[key,process.env[key]]));
const originalFetch = globalThis.fetch;
const calls: {method:string;path:string}[] = [];
let prepared = 0, accepted = false, dbEnabled = false, mismatch = false;

const fetchMock:typeof fetch = async (input, init) => {
  const url = new URL(typeof input==='string'?input:input instanceof URL?input.href:input.url);
  const method = init?.method ?? 'GET';
  calls.push({method,path:url.pathname});
  assert.equal(url.origin,SHADOW_RESEARCH_ORIGIN);
  assert.ok(['GET','HEAD'].includes(method), 'no write transport allowed');
  assert.equal(init?.redirect,'error');
  let data:unknown = null;
  const headers:Record<string,string> = {'Content-Type':'application/json'};
  if(url.pathname.endsWith('/rpc/orotitan_j0_c8_read_gate_v1')) {
    data={milestone:url.searchParams.get('p_milestone'),preparedReviews:prepared,
      accepted,acceptanceEnabled:dbEnabled,acceptanceAllowedNow:true};
  } else if(url.pathname.endsWith('/orotitan_j0_c8_acceptance_control_v1')) {
    data={enabled:dbEnabled,activation_authority_ref:null};
  } else if(url.pathname.endsWith('/orotitan_j0_c8_milestone_activation_v1')) {
    data={enabled:dbEnabled};
  } else if(url.pathname.endsWith('/orotitan_j0_c8_reviews_v1')) {
    headers['content-range']=`*/${mismatch?prepared+1:prepared}`;
  } else if(url.pathname.endsWith('/orotitan_j0_c8_decisions_v1')) {
    headers['content-range']=`*/${accepted?1:0}`;
  } else {throw Error('UNEXPECTED_REQUEST');}
  return new Response(method==='HEAD'?null:JSON.stringify(data),{status:200,headers});
};

test.beforeEach(()=>{
  Object.assign(process.env,synthetic);
  delete process.env.OROTITAN_C8_ACCEPTANCE_GO_ENABLED;
  state.cookie=''; state.effects.length=0; calls.length=0;
  prepared=0; accepted=false; dbEnabled=false; mismatch=false;
  globalThis.fetch=fetchMock;
});
test.afterEach(()=>{
  assert.deepEqual(state.effects,[], 'no business side effect executed');
});
test.after(()=>{
  for(const [key,value] of originals) {
    if(value===undefined)delete process.env[key]; else process.env[key]=value;
  }
  delete process.env.OROTITAN_C8_ACCEPTANCE_GO_ENABLED;
  globalThis.fetch=originalFetch;
});

const request = (method='GET', path='/admin/research-factory/control-tower/c8-gate') =>
  new Request('https://qualification.invalid'+path,{method,
    headers:{origin:'https://qualification.invalid','content-type':'application/json',
      authorization:'Bearer synthetic-cron'},
    ...(method==='GET'?{}:{body:'{}'}),
  }) as NextRequest;
async function session() {
  const auth=await import('../lib/auth/admin-session');
  await auth.createAdminSession();
  assert.equal(await auth.isAdminAuthenticated(),true);
}

test('network helper blocks callback TXT resolution through CJS and ESM without callback delegation',()=>{
  let callbacks=0;
  for(const resolve of [dns.resolveTxt, callbackResolveTxt]) {
    assert.throws(()=>resolve('qualification.invalid',()=>{callbacks++;}),
      /LOCAL_QUALIFICATION_NETWORK_DENIED/);
  }
  assert.equal(callbacks,0);
});

test('network helper rejects Promise TXT resolution through both Node entrypoints',async()=>{
  for(const resolve of [dns.promises.resolveTxt, dnsPromises.resolveTxt, promiseResolveTxt]) {
    await assert.rejects(resolve('qualification.invalid'),/LOCAL_QUALIFICATION_NETWORK_DENIED/);
  }
});

test('network helper blocks callback Resolver instances before native TXT resolution',()=>{
  const resolver=new dns.Resolver();
  let callbacks=0;
  assert.throws(()=>resolver.resolveTxt('qualification.invalid',()=>{callbacks++;}),
    /LOCAL_QUALIFICATION_NETWORK_DENIED/);
  assert.equal(callbacks,0);
});

test('network helper rejects Promise Resolver instances before native TXT resolution',async()=>{
  for(const resolver of [new dns.promises.Resolver(), new dnsPromises.Resolver()]) {
    await assert.rejects(resolver.resolveTxt('qualification.invalid'),
      /LOCAL_QUALIFICATION_NETWORK_DENIED/);
  }
});

test('network helper never delegates DNS or existing denied transports to mocked native backends',async()=>{
  // Execute the actual helper with inert backends: a regression cannot send a
  // DNS request from this test, even if its monkeypatch stops working.
  let backendCalls=0;
  const backend=()=>{backendCalls++;throw Error('MOCK_NATIVE_BACKEND_REACHED');};
  const methods=['lookup','lookupService','resolve','resolve4','resolve6','resolveAny',
    'resolveCaa','resolveCname','resolveMx','resolveNaptr','resolveNs','resolvePtr',
    'resolveSoa','resolveSrv','resolveTlsa','resolveTxt','reverse'];
  // Pin the actual Node 22 DNS surface: any new resolver API needs review.
  assert.deepEqual(Object.keys(dns).filter(name=>/^(lookup|resolve|reverse)/.test(name)).sort(),
    [...methods].sort());
  assert.deepEqual(Object.keys(dnsPromises).filter(name=>/^(lookup|resolve|reverse)/.test(name)).sort(),
    [...methods].sort());
  class CallbackResolver {}
  class PromiseResolver {}
  const callbackApi:Record<string,unknown>={Resolver:CallbackResolver};
  const promiseApi:Record<string,unknown>={Resolver:PromiseResolver};
  for(const method of methods) {
    callbackApi[method]=backend;promiseApi[method]=backend;
    Object.assign(CallbackResolver.prototype,{[method]:backend});
    Object.assign(PromiseResolver.prototype,{[method]:backend});
  }
  callbackApi.promises=promiseApi;
  const net={Socket:{prototype:{connect:backend}}};
  const tls={connect:backend}, dgram={Socket:{prototype:{send:backend}}};
  const http={request:backend,get:backend}, https={request:backend,get:backend};
  const modules:Record<string,unknown>={
    'node:net':net,'node:tls':tls,'node:dgram':dgram,'node:http':http,'node:https':https,
    'node:dns':callbackApi,'node:dns/promises':promiseApi,
    'node:module':{syncBuiltinESMExports:()=>{}},
  };
  const context={require:(name:string)=>{
    assert.ok(name in modules);return modules[name];
  },fetch:async()=>backend()};
  runInNewContext(readFileSync(join(process.cwd(),'tests/helpers/deny-network.cjs'),'utf8'),context);
  for(const api of [callbackApi, new CallbackResolver() as Record<string,unknown>]) {
    for(const method of methods) {
      assert.throws(()=>(api[method] as (...args:unknown[])=>unknown)('qualification.invalid',()=>{}),
        /LOCAL_QUALIFICATION_NETWORK_DENIED/);
    }
  }
  for(const api of [promiseApi, new PromiseResolver() as Record<string,unknown>]) {
    for(const method of methods) {
      await assert.rejects((api[method] as (...args:unknown[])=>Promise<unknown>)('qualification.invalid'),
        /LOCAL_QUALIFICATION_NETWORK_DENIED/);
    }
  }
  for(const op of [()=>net.Socket.prototype.connect(),
    ()=>tls.connect(),()=>dgram.Socket.prototype.send(),()=>http.request(),()=>http.get(),
    ()=>https.request(),()=>https.get()]) {
    assert.throws(op,/LOCAL_QUALIFICATION_NETWORK_DENIED/);
  }
  await assert.rejects(context.fetch(),/LOCAL_QUALIFICATION_NETWORK_DENIED/);
  assert.equal(backendCalls,0);
});

test('C8 strict gate is independent of Research flags and absent mode adds no grant',()=>{
  assert.equal(isHostedC8ReadOnlyEnabled(synthetic),true);
  assert.equal(isHostedShadowResearchEnabled(synthetic),false);
  const env:Record<string,string|undefined>={...synthetic};
  delete env.OROTITAN_HOSTED_SHADOW_RESEARCH_ENABLED;
  delete env.OROTITAN_SHARED_INBOX_TEST_ENABLED;
  assert.equal(isHostedC8ReadOnlyEnabled(env),true);
  delete env.OROTITAN_C8_READ_ONLY_ENABLED;
  assert.equal(isHostedC8ReadOnlyEnabled(env),false);
  assert.equal(isHostedShadowResearchEnabled(env),false);
});

test('login creates a real signed admin cookie and redirects to owner verification only',async()=>{
  const {loginAction,logoutAction}=await import('../app/admin/actions');
  const form=new FormData();form.set('password',synthetic.ADMIN_PASSWORD);
  await assert.rejects(loginAction(form),/REDIRECT:\/admin\/research-factory\/control-tower\/c8-owner$/);
  assert.equal(await (await import('../lib/auth/admin-session')).isAdminAuthenticated(),true);
  await assert.rejects(logoutAction(),/REDIRECT:\/admin$/);
  assert.equal(state.cookie,'');
  assert.equal(calls.length,0);
});

test('GET C8 with a signed session rereads persisted state using five read calls',async()=>{
  await session(); prepared=2; accepted=true; dbEnabled=true;
  const {GET}=await import('../app/admin/research-factory/control-tower/c8-gate/route');
  const response=await GET(request());
  assert.equal(response.status,200);
  const body=await response.json();
  assert.equal(body.preparedReviews,2);assert.equal(body.persistedDecisions,1);
  assert.equal(body.accepted,true);assert.equal(body.activationLocked,true);
  assert.equal(body.gate.acceptanceAllowedNow,false);
  assert.equal(body.stageStarted,false);assert.equal(body.published,false);
  assert.equal(response.headers.get('cache-control'),'private, no-store');
  assert.equal(calls.length,5);
  assert.deepEqual(calls.map(c=>c.method).sort(),['GET','GET','GET','HEAD','HEAD']);
});

test('GET C8 without session or with a forged cookie refuses before any client request',async()=>{
  const {GET}=await import('../app/admin/research-factory/control-tower/c8-gate/route');
  for(const cookie of ['', 'forged.signature']) {
    state.cookie=cookie;
    assert.equal((await GET(request())).status,401);
    assert.equal(calls.length,0);
  }
});

const invalid: [string,string|undefined][] = [
  ['OROTITAN_C8_READ_ONLY_ENABLED','0'],['OROTITAN_C8_READ_ONLY_ENABLED','true'],
  ['OROTITAN_C8_READ_ONLY_ENABLED',''],['VERCEL_ENV','production'],
  ['VERCEL_ENV',undefined],['NODE_ENV','development'],
  ['OROTITAN_SHARED_INBOX_TEST_PROJECT_REF','cugpgtzygqqlxetyeven'],
  ['OROTITAN_EXPECTED_SUPABASE_PROJECT_REF','cugpgtzygqqlxetyeven'],
  ['NEXT_PUBLIC_SUPABASE_URL','https://cugpgtzygqqlxetyeven.supabase.co'],
  ['OROTITAN_SHARED_INBOX_TEST_URL','https://cugpgtzygqqlxetyeven.supabase.co'],
  ['OROTITAN_SHARED_INBOX_TEST_URL',SHADOW_RESEARCH_ORIGIN+'.attacker.invalid'],
  ['SUPABASE_SERVICE_ROLE_KEY','synthetic-split-key'],
  ['ADMIN_SESSION_SECRET','short'],
];
for(const [key,value] of invalid) test(`C8 refuses invalid ${key}=${value ?? 'absent'} without I/O`,async()=>{
  await session();
  if(value===undefined)delete process.env[key];else process.env[key]=value;
  assert.equal(isHostedC8ReadOnlyEnabled(process.env),false);
  assert.equal(isHostedShadowResearchEnabled(process.env),false);
  const {GET}=await import('../app/admin/research-factory/control-tower/c8-gate/route');
  assert.equal((await GET(request())).status,404);
  const {loginAction}=await import('../app/admin/actions');
  await assert.rejects(loginAction(new FormData()),/C8_READ_ONLY_CONFIG_DENIED/);
  assert.equal(calls.length,0);
});

test('C8 refuses inconsistent persisted readback and invalid milestones',async()=>{
  await session();mismatch=true;
  const {GET}=await import('../app/admin/research-factory/control-tower/c8-gate/route');
  assert.equal((await GET(request())).status,503);
  calls.length=0;
  assert.equal((await GET(request('GET','/admin/research-factory/control-tower/c8-gate?milestone=J9'))).status,400);
  assert.equal(calls.length,0);
});

// Explicit reviewed inventory: adding any server entry forces this test to be updated.
const postRoutes=[
  'research-q3-sources/upload','research-primary-sources/upload-tsmc',
  'research-primary-sources/capture','research-primary-sources/audit/run',
  'research-factory/onboarding/inspect','research-factory/onboarding/commit',
  'research-factory/run-preflight/inspect','research-factory/run-preflight/commit',
  'research-factory/contracts/verify','research-factory/control-tower/record',
  'research-factory/control-tower/c5-security-test',
  'research-factory/control-tower/c8-gate/prepare','research-factory/control-tower/c8-gate/accept',
  'research-factory/control-tower/c8-gate/owner-session/request',
  'research-factory/control-tower/c8-gate/owner-session/verify',
  'research-factory/control-tower/c8-gate/owner-session/logout',
];
const ownerOnlyPosts=new Set([
  'research-factory/control-tower/c8-gate/owner-session/request',
  'research-factory/control-tower/c8-gate/owner-session/verify',
  'research-factory/control-tower/c8-gate/owner-session/logout',
]);
for(const path of postRoutes) test(`direct POST ${path} never reaches Research/Registry mutation`,async()=>{
  await session();
  const mod=await import(`../app/admin/${path}/route.ts`) as {POST:(req:NextRequest)=>Promise<Response>};
  for(const value of ['1','0','invalid','']) {
    process.env.OROTITAN_C8_READ_ONLY_ENABLED=value;
    const response=await mod.POST(request('POST','/admin/'+path));
    const isC8Owner=ownerOnlyPosts.has(path)&&value==='1';
    const expected=isC8Owner?(path.endsWith('/logout')?200:400):404;
    assert.equal(response.status,expected);
    const payload=await response.json() as {error?:string;status?:string};
    if(expected===404)assert.equal(payload.error,'SHADOW_ONLY');
    if(expected===400)assert.equal(payload.error,'REQUEST_INVALID');
    if(expected===200)assert.equal(payload.status,'SIGNED_OUT');
  }
  assert.equal(calls.length,0);
});

test('C8 owner status route is reachable in read-only but does not attest a missing owner session',async()=>{
  await session();
  const {GET}=await import('../app/admin/research-factory/control-tower/c8-gate/owner-session/route');
  const response=await GET();
  assert.equal(response.status,200);
  assert.deepEqual(await response.json(),{ok:true,authenticated:false});
  for(const value of ['0','invalid','']) {
    process.env.OROTITAN_C8_READ_ONLY_ENABLED=value;
    assert.equal((await GET()).status,404);
  }
  assert.equal(calls.length,0);
});

const actionModules:Record<string,string[]>={
  'actions':['createCompanyAction','updateCompanyAction','createSnapshotAction',
    'createSnapshotJsonAction','refreshPricesAction'],
  'paper/actions':['createPaperPortfolioAction','createPaperPolicyAction',
    'evaluatePaperSignalAction','executeVirtualFillAction'],
  'research-requests/actions':['requestResearchPilotAction'],
  'research-recovery/actions':['runResearchRecoveryAction'],
};
for(const [path,names] of Object.entries(actionModules)) for(const name of names)
  test(`direct Server Action ${name} refuses with valid session before business effects`,async()=>{
    await session();
    const mod=await import(`../app/admin/${path}.ts`) as Record<string,(form:FormData)=>Promise<void>>;
    const form=new FormData();form.set('source','asml-observation');
    form.set('company_id','00000000-0000-4000-8000-000000000001');
    for(const value of ['1','invalid','0','']) {
      process.env.OROTITAN_C8_READ_ONLY_ENABLED=value;
      await assert.rejects(mod[name](form),/C8_READ_ONLY_MUTATION_DENIED/);
    }
    assert.equal(calls.length,0);
  });

test('C8 denies the writing GET cron even with valid cron authorization',async()=>{
  const {GET}=await import('../app/api/cron/prices/route');
  assert.equal((await GET(request('GET','/api/cron/prices'))).status,403);
  assert.equal(calls.length,0);
});

test('generic and Research clients cannot bypass C8, including development fixture fallback',async()=>{
  const generic=await import('../lib/supabase/server');
  const research=await import('../lib/orotitan-equity/product/research-inbox-pilot-server');
  for(const nodeEnv of ['production','development']) {
    Object.assign(process.env,{NODE_ENV:nodeEnv,CI:'true',OROTITAN_PRODUCT_BROWSER_FIXTURE:'1'});
    assert.throws(()=>generic.createServerSupabaseClient(),/C8_READ_ONLY_MUTATION_DENIED/);
    assert.throws(()=>research.getHostedShadowRecoveryClient(),/C8_READ_ONLY_MUTATION_DENIED/);
    assert.equal(research.researchInboxPilotMode(),'disabled');
    await assert.rejects(research.requestPilotResearch('synthetic-id'),/C8_READ_ONLY_MUTATION_DENIED/);
  }
  delete process.env.OROTITAN_PRODUCT_BROWSER_FIXTURE;
  delete process.env.CI;
  assert.equal(calls.length,0);
});

test('restricted C8 client rejects insert/RPC writes and unrelated reads before transport',async()=>{
  const {getC8GateReadClient}=await import('../lib/orotitan-equity/product/c8-gate-read-server');
  const c=getC8GateReadClient();
  for(const op of [
    ()=>c.from('orotitan_j0_c8_reviews_v1').insert({}),
    ()=>c.from('orotitan_j0_c8_decisions_v1').update({}).eq('decision_id','synthetic'),
    ()=>c.from('orotitan_j0_c8_reviews_v1').delete().eq('review_id','synthetic'),
    ()=>c.rpc('orotitan_j0_record_event_v1',{}),
    ()=>c.rpc('orotitan_j0_c8_read_gate_v1',{p_milestone:'J0'}), // POST forbidden even for read RPC
    ()=>c.from('orotitan_runs').select('*'),
    ()=>c.storage.from('private').upload('test',Buffer.from('synthetic')),
  ]) {
    const result=await op(); assert.ok(result.error);
  }
  assert.equal(calls.length,0);
});

test('outside C8 explicitly enabled Research retains its prior gate and client behavior',async()=>{
  delete process.env.OROTITAN_C8_READ_ONLY_ENABLED;
  assert.equal(isHostedShadowResearchEnabled(process.env),true);
  const research=await import('../lib/orotitan-equity/product/research-inbox-pilot-server');
  assert.equal(research.researchInboxPilotMode(),'shared-test');
  assert.ok(research.getHostedShadowRecoveryClient());
  const {POST}=await import('../app/admin/research-factory/control-tower/c8-gate/accept/route');
  await session();
  const response=await POST(request('POST'));
  assert.equal(response.status,423);
  assert.equal((await response.json()).error,'J0_C8_ACCEPTANCE_LOCKED');
  assert.equal(calls.length,0);
});

test('accept HTTP refuses STANDARD, including its default, under a SOLO GO before any client call',async()=>{
  delete process.env.OROTITAN_C8_READ_ONLY_ENABLED;
  process.env.OROTITAN_C8_ACCEPTANCE_GO_ENABLED='1';
  await session();
  const {POST}=await import('../app/admin/research-factory/control-tower/c8-gate/accept/route');
  const base={
    decisionKey:'orotitan:c8:decision:standard-under-solo-go',
    reviewId:'11111111-1111-4111-8111-111111111111',reviewSha256:'a'.repeat(64),
    approverLabel:'Independent Standard Approver',consentPhrase:'ACCEPTER J0 SHADOW',
    authorityRef:'GO_ACCEPT_J0_SOLO_LOCAL_TEST_20261105',
  };
  for(const governanceMode of [undefined,'STANDARD_TWO_PERSON'] as const) {
    const body=governanceMode===undefined?base:{...base,governanceMode};
    const req=new Request('https://qualification.invalid/admin/research-factory/control-tower/c8-gate/accept',{
      method:'POST',headers:{origin:'https://qualification.invalid','content-type':'application/json'},
      body:JSON.stringify(body),
    }) as NextRequest;
    const response=await POST(req);
    assert.equal(response.status,409);
    assert.equal((await response.json()).error,'J0_C8_GOVERNANCE_MODE_MISMATCH');
  }
  assert.equal(calls.length,0);
});

test('inventory covers every HTTP handler and exported Server Action in app',()=>{
  function files(dir:string):string[] {
    return readdirSync(dir,{withFileTypes:true}).flatMap(e=>
      e.isDirectory()?files(join(dir,e.name)):[join(dir,e.name)]);
  }
  const routes:string[]=[], actions:string[]=[];
  for(const path of files(join(process.cwd(),'app'))) {
    if(!/\.tsx?$/.test(path))continue;
    const source=readFileSync(path,'utf8');
    const rel=relative(join(process.cwd(),'app'),path).replaceAll('\\','/');
    if(rel.endsWith('/route.ts')) {
      for(const m of source.matchAll(/export\s+(?:async\s+)?function\s+(GET|POST|PUT|PATCH|DELETE|HEAD|OPTIONS)\s*\(/g))
        routes.push(`${m[1]} ${rel}`);
      assert.doesNotMatch(source,/export\s*\{|export\s+const\s+(GET|POST|PUT|PATCH|DELETE|HEAD|OPTIONS)\s*=/);
    }
    if(/['"]use server['"]/.test(source)) {
      assert.ok(rel.endsWith('actions.ts'),`unreviewed Server Action module ${rel}`);
      for(const m of source.matchAll(/export\s+async\s+function\s+(\w+)/g))actions.push(`${rel}:${m[1]}`);
      assert.doesNotMatch(source,/export\s*\{|export\s+const/);
    }
  }
  assert.deepEqual(routes.sort(),[
    ...postRoutes.map(p=>`POST admin/${p}/route.ts`),
    'GET admin/research-factory/control-tower/c8-gate/route.ts',
    'GET admin/research-factory/control-tower/c8-gate/owner-session/route.ts',
    'GET api/cron/prices/route.ts','GET api/orotitan/runtime/route.ts',
  ].sort());
  assert.deepEqual(actions.sort(),[
    ...Object.entries(actionModules).flatMap(([p,names])=>names.map(n=>`admin/${p}.ts:${n}`)),
    'admin/actions.ts:loginAction','admin/actions.ts:logoutAction',
  ].sort());
});
