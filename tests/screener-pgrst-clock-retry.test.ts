import assert from 'node:assert/strict';
import test from 'node:test';
import { pgrstClockSkewRetryFetch } from '../lib/supabase/pgrst-clock-retry';

const root = 'https://cugpgtzygqqlxetyeven.supabase.co';
const api = root + '/rest/v1/research_dossiers?select=dossier_id';
const clockError = () => Response.json(
  {code:'PGRST303',message:'JWT issued at future',details:null,hint:null},
  {status:401},
);
const ok = () => Response.json([{dossier_id:'fixture'}],{status:200});
const calls: number[] = [];
const sleep = async (ms:number) => {calls.push(ms);};

test('P0 PostgREST: one scoped GET retry recovers a confirmed future-JWT 401',async()=>{
  calls.length=0;
  let attempts=0;
  const wrapped=pgrstClockSkewRetryFetch(async()=>{
    attempts++;
    return attempts===1?clockError():ok();
  },root,sleep);
  const result=await wrapped(api,{method:'GET'});
  assert.equal(result.status,200);
  assert.equal(attempts,2);
  assert.deepEqual(calls,[400]);
});

test('P0 PostgREST: bounded attempt budget fails closed on a persistent future-JWT error',async()=>{
  calls.length=0;
  let attempts=0;
  const wrapped=pgrstClockSkewRetryFetch(async()=>{
    attempts++;
    return clockError();
  },root,sleep);
  const response=await wrapped(api);
  assert.equal(response.status,401);
  assert.equal((await response.json()).code,'PGRST303');
  assert.equal(attempts,3);
  assert.deepEqual(calls,[400,1600]);
});

test('P0 PostgREST: ordinary 401s, 403s, and unrelated PGRST303 do not retry',async()=>{
  for(const response of [
    Response.json({code:'PGRST303',message:'JWT expired'}, {status:401}),
    Response.json({code:'PGRST301',message:'JWT issued at future'}, {status:401}),
    Response.json({code:'PGRST303',message:'JWT issued at future'}, {status:403}),
    Response.json({error:'not authorized'}, {status:401}),
  ]){
    let attempts=0;
    const wrapped=pgrstClockSkewRetryFetch(async()=>{attempts++;return response;},root,sleep);
    const actual=await wrapped(api);
    assert.equal(actual,response);
    assert.equal(attempts,1);
  }
});

test('P0 PostgREST: never retry auth, writes, RPC POST, foreign host, or similar path',async()=>{
  for(const [url,method] of [
    [root+'/rest/v1/research_dossiers','POST'],
    [root+'/rest/v1/rpc/privileged_upload','POST'],
    [root+'/rest/v1/research_dossiers','PATCH'],
    [root+'/rest/v1/research_dossiers','DELETE'],
    [root+'/auth/v1/user','GET'],
    [root+'/rest/v12/research_dossiers','GET'],
    ['https://attacker.invalid/rest/v1/research_dossiers','GET'],
    ['https://cugpgtzygqqlxetyeven.supabase.co.attacker.invalid/rest/v1/research_dossiers','GET'],
  ]){
    let attempts=0;
    const wrapped=pgrstClockSkewRetryFetch(async()=>{attempts++;return clockError();},root,sleep);
    const response=await wrapped(url,{method});
    assert.equal(response.status,401);
    assert.equal(attempts,1,method+' '+url);
  }
});

test('P0 PostgREST: abort, non-JSON errors and upstream errors never become empty results',async()=>{
  const signal=new AbortController();
  signal.abort();
  let attempts=0;
  const wrapped=pgrstClockSkewRetryFetch(async()=>{attempts++;return clockError();},root,sleep);
  const result=await wrapped(api,{signal:signal.signal});
  assert.equal(result.status,401);
  assert.equal(attempts,1);
  const textError=pgrstClockSkewRetryFetch(async()=>new Response('JWT issued at future',{
    status:401,headers:{'Content-Type':'text/plain'},
  }),root,sleep);
  assert.equal((await (await textError(api)).text()),'JWT issued at future');
  const boom=pgrstClockSkewRetryFetch(async()=>{throw new Error('network unavailable');},root,sleep);
  await assert.rejects(()=>boom(api),/network unavailable/);
});
