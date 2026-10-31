import test from 'node:test';
import assert from 'node:assert/strict';
import {buildV35Matrix,validateV35Chunk,type V35Source} from
 '../lib/orotitan-equity/product/asml-research-claim-matrix-v35';

const HASH='add58be9d9822ca12584b63c55c089b36e1ae5bf7a1825ebf1d1572628f2c52c';
const PAGE_HASH='ca40c61b1cd77c9b06b51ca7eb39ef7f2d7098ac7a8b4148d2dffe3fc18ef692';
const RECEIPT='b513ee1bdffc34200ba87255bebda63eb6b3e88324a67d3a696dae321e16ddfb';
const src:V35Source={id:'ASML_2025_US_GAAP',title:'ASML',
 sourceSha256:HASH,probes:[
  {claimId:'C-010',numeric:'6212.5',derived:false,contextHint:'factoring'},
  {claimId:'C-012',numeric:'29.1',derived:false,contextHint:'China'},
  {claimId:'C-015',numeric:'24.9',derived:false,contextHint:'ZEISS'},
  {claimId:'C-009',numeric:'11027.3',derived:true,contextHint:null},
 ]};
function reply(pages:Array<{pdfPage:number;textLength:number;textSha256:string;
 candidateMatches:Array<{claimId:string;excerpt:string;match:'NUMERIC_TOKEN_ONLY'}>}>,
 start=1,total=pages.length,hash=HASH){
 return {ok:true,mode:'scan_pages',checkedAt:'2026-10-31T06:47:00.277Z',
  registryAdmitted:false,issuerOriginAuthenticated:false,
  source:{id:src.id,sourceOriginalHash:hash,integrity:'STORED_BYTES_SHA256_MATCH',
    receiptHash:RECEIPT},
  scan:{id:src.id,pdfPageCount:total,startPage:start,
    endPage:start+pages.length-1,pageObservations:pages}};
}
test('v3.5 extracts traceable page observations without admission',()=>{
 const chunk=validateV35Chunk(reply([
  {pdfPage:1,textLength:100,textSha256:PAGE_HASH,candidateMatches:[
   {claimId:'C-010',excerpt:'In 2025 €6212.5 million sold through factoring',
    match:'NUMERIC_TOKEN_ONLY'},
   {claimId:'C-012',excerpt:'Effect of changes in exchange rates (29.1)',
    match:'NUMERIC_TOKEN_ONLY'},
   {claimId:'C-015',excerpt:'24.9% equity interest in Carl Zeiss SMT',
    match:'NUMERIC_TOKEN_ONLY'}]},
 ]),src,1);
 const matrix=buildV35Matrix([src],[chunk]);
 assert.equal(matrix.complete,true);
 assert.equal(matrix.claimCount,4);
 assert.equal(matrix.validatedClaimCount,0);
 assert.equal(matrix.registryAdmitted,false);
 assert.equal(matrix.claims[0].observations[0].pdfPage,1);
 assert.equal(matrix.claims[0].observations[0].contextHintObservation,'HINT_PRESENT');
 assert.equal(matrix.claims[1].observations[0].contextHintObservation,'HINT_ABSENT');
 assert.equal(matrix.claims[2].observations[0].contextHintObservation,'HINT_PRESENT');
 assert.equal(matrix.claims[3].searchStatus,'DERIVED_RECONCILIATION_REQUIRED');
});
test('partial scan is not a negative evidence conclusion',()=>{
 const chunk=validateV35Chunk(reply([
  {pdfPage:1,textLength:0,textSha256:PAGE_HASH,candidateMatches:[]},
 ],1,350),src,1);
 const matrix=buildV35Matrix([src],[chunk]);
 assert.equal(matrix.complete,false);
 assert.equal(matrix.claims[0].searchStatus,'PARTIAL_SCAN');
 assert.equal(matrix.coverage[0].unextractableTextPages,1);
});
test('reject changed hash, unexpected claim, missing page, and unauthorized response',()=>{
 const base=[{pdfPage:1,textLength:50,textSha256:PAGE_HASH,candidateMatches:[]}];
 assert.throws(()=>validateV35Chunk(reply(base,1,1,'f'.repeat(64)),src,1),/V35_SCAN_RESPONSE_INVALID/);
 assert.throws(()=>validateV35Chunk(reply(base,1,1),src,2),/V35_SCAN_RESPONSE_INVALID/);
 assert.throws(()=>validateV35Chunk(reply([
  {pdfPage:1,textLength:1,textSha256:PAGE_HASH,candidateMatches:[
   {claimId:'C-404',excerpt:'no',match:'NUMERIC_TOKEN_ONLY'}]},
 ]),src,1),/V35_SCAN_RESPONSE_INVALID/);
 const bad={...reply(base),registryAdmitted:true};
 assert.throws(()=>validateV35Chunk(bad,src,1),/V35_SCAN_RESPONSE_INVALID/);
});
test('reject discontinuous chunk coverage rather than falsely mark complete',()=>{
 const first=validateV35Chunk(reply([
  {pdfPage:1,textLength:10,textSha256:PAGE_HASH,candidateMatches:[]}
 ],1,3),src,1);
 const third=validateV35Chunk(reply([
  {pdfPage:3,textLength:10,textSha256:PAGE_HASH,candidateMatches:[]}
 ],3,3),src,3);
 assert.throws(()=>buildV35Matrix([src],[first,third]),/V35_SCAN_RESPONSE_INVALID/);
});
