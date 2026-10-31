import test from 'node:test';
import assert from 'node:assert/strict';
import {
 verifyHostedJ0C5HttpMatrix,type J0C5HttpObservation,
} from '../lib/orotitan-equity/product/runfactory-j0-c5-http-verification-v1';

const valid:J0C5HttpObservation[]=[
 {name:'CROSS_ORIGIN',status:403,error:'ORIGIN_MISMATCH',accepted:false,stageStarted:false,cacheControl:'private, no-store'},
 {name:'UNSUPPORTED_CONTENT_TYPE',status:400,error:'REQUEST_INVALID',accepted:false,stageStarted:false,cacheControl:'private, no-store'},
 {name:'UNEXPECTED_FIELDS',status:400,error:'J0_UNEXPECTED_FIELDS',accepted:false,stageStarted:false,cacheControl:'private, no-store'},
 {name:'FORGED_ACCEPTANCE',status:400,error:'J0_INPUT_INVALID',accepted:false,stageStarted:false,cacheControl:'private, no-store'},
];
test('C5 exact matrix accepts four application generated denial responses',()=>{
 const result=verifyHostedJ0C5HttpMatrix([...valid].reverse());
 assert.equal(result.passed,true);
 assert.equal(result.cases,4);
});
test('C5 Vercel SSO edge 401 can never count as OroTitan 403',()=>{
 assert.throws(()=>verifyHostedJ0C5HttpMatrix(
  valid.map(v=>v.name==='CROSS_ORIGIN'?
   {...v,status:401,error:'VERCEL_AUTHENTICATION'}:v)),/J0_C5_HTTP_APPLICATION_DENIAL_MISSING/);
});
test('C5 denies wrong response types, flags or absent no-store',()=>{
 const variants=[
  {...valid[2],status:200},
  {...valid[2],error:'INTERNAL_ERROR'},
  {...valid[2],accepted:true},
  {...valid[2],stageStarted:true},
  {...valid[2],cacheControl:'public, max-age=300'},
 ];
 for(const changed of variants)
  assert.throws(()=>verifyHostedJ0C5HttpMatrix(
   valid.map(v=>v.name===changed.name?changed:v)),
   /J0_C5_HTTP_APPLICATION_DENIAL_MISSING/);
});
test('C5 rejects missing, extra, duplicated and unknown cases',()=>{
 assert.throws(()=>verifyHostedJ0C5HttpMatrix(valid.slice(0,3)),
  /J0_C5_HTTP_CASE_COUNT/);
 assert.throws(()=>verifyHostedJ0C5HttpMatrix([...valid,...valid[0]]),
  /J0_C5_HTTP_CASE_COUNT/);
 assert.throws(()=>verifyHostedJ0C5HttpMatrix([...valid.slice(0,3),valid[1]]),
  /J0_C5_HTTP_DUPLICATE_OR_UNKNOWN/);
 assert.throws(()=>verifyHostedJ0C5HttpMatrix(
  valid.map((v,i)=>i===0?{...v,name:'UNEXPECTED_FOURTH' as never}:v)),
  /J0_C5_HTTP_DUPLICATE_OR_UNKNOWN/);
});
