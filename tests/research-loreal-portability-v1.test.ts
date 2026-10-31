import test from 'node:test';
import assert from 'node:assert/strict';
import {findNumericCandidates,inspectNumericContext} from
 '../lib/orotitan-equity/product/research-pdf-numeric-core-v1';
import {checkReportedArithmetic,runArithmeticChecks} from
 '../lib/orotitan-equity/product/research-financial-arithmetic-qa-v1';
import {assessPrivatePdfSource,type PrivatePdfCandidate} from
 '../lib/orotitan-equity/product/research-private-source-readiness-v1';
import {LOREAL_PILOT2_QA,LOREAL_PILOT2_PDF_PROBES,LOREAL_PILOT2_STATUS} from
 '../lib/orotitan-equity/product/loreal-pilot2-candidate-config-v1';

test('L’Oreal pilot2 seven bridges: four exact and three rounding-compatible',()=>{
 const qa=runArithmeticChecks(LOREAL_PILOT2_QA);
 assert.equal(qa.length,7);
 assert.equal(qa.filter(q=>q.verdict==='EXACT_MATCH').length,4);
 assert.equal(qa.filter(q=>q.verdict==='ROUNDING_COMPATIBLE').length,3);
 assert.equal(qa.every(q=>q.passed&&!q.claimValidated&&!q.registryAdmitted),true);
 const quarterly=qa.find(q=>q.id==='QA-Q3-DIV');
 assert.equal(quarterly?.calculatedSum,'11561.6');
 assert.equal(quarterly?.reportedComparator,'11561.4');
 assert.equal(quarterly?.difference,'0.2');
 assert.equal(quarterly?.roundingTolerance,'0.25');
 const fy=qa.find(q=>q.id==='QA-FY25-BRIDGE');
 assert.equal(fy?.calculatedSum,'44052.0');
 assert.equal(fy?.roundingTolerance,'0.15');
 assert.equal(LOREAL_PILOT2_STATUS.researchStageAdmitted,false);
});
test('rounding detection is not an adjustable issuer-specific override',()=>{
 const base=LOREAL_PILOT2_QA[5];
 assert.equal(checkReportedArithmetic(base).verdict,'ROUNDING_COMPATIBLE');
 assert.equal(checkReportedArithmetic({...base,reportedComparator:'35338.0'}).verdict,
  'OUT_OF_TOLERANCE');
 assert.throws(()=>checkReportedArithmetic({...base,reportedComparator:'1.234'}));
 assert.throws(()=>checkReportedArithmetic({...base,sourceIds:[]}));
});
test('issuer-neutral numeric matching on L’Oreal and legacy ASML syntax',()=>{
 const loreal=LOREAL_PILOT2_PDF_PROBES['LR-S01'];
 const found=findNumericCandidates(
  'Sales amounted to EUR 44,052.0 million. Operating margin 20.2% and net cash flow 7,161.6.',
  loreal,100);
 assert.deepEqual(found.map(x=>x.claimId),['LR-C001','LR-C003','LR-C004']);
 assert.equal(inspectNumericContext(found[1],['operating margin']),'HINT_PRESENT');
 const tsmc=findNumericCandidates('The 2026 capital budget is USD60 billion to USD64 billion.',[
  {claimId:'C-022',numeric:'60',contextHints:['capital budget','capital expenditures']},
  {claimId:'C-023',numeric:'64'},
 ]);
 assert.equal(tsmc.length,2);
 assert.equal(inspectNumericContext(tsmc[0],['capital budget','capital expenditures']),
  'HINT_PRESENT');
 assert.equal(inspectNumericContext(tsmc[1],[]),'HINT_NOT_CONFIGURED');
 assert.throws(()=>findNumericCandidates('20',[
  {claimId:'A-1',numeric:'20'},{claimId:'A-1',numeric:'20'}]));
});
const base:PrivatePdfCandidate={
 id:'LR-S01',title:'L’Oreal FY2025',publicationDate:'2026-03-18',
 expectedSha256:null,expectedByteSize:null,privateStoragePath:null,
 receiptSha256:null,storedBytesIndependentlyRehashed:false,
 privateReceiptIndependentlyVerified:false,captureTimestamp:null,
};
test('uncaptured L’Oreal document blocks private scan and Registry admission',()=>{
 const result=assessPrivatePdfSource(base,'2026-10-30');
 assert.equal(result.state,'ORIGINAL_BYTES_MISSING');
 assert.equal(result.scanAllowed,false);
 assert.equal(result.registryAdmitted,false);
 assert.equal(result.claimValidated,false);
});
test('private scan requires verified bytes + receipt, and never authenticates issuer',()=>{
 const captured={...base,expectedSha256:'a'.repeat(64),expectedByteSize:10000,
  privateStoragePath:'research/lo-real/2026/fy25.pdf',
  receiptSha256:'b'.repeat(64),captureTimestamp:'2026-10-31T07:05:00Z'};
 assert.equal(assessPrivatePdfSource(captured,'2026-10-30').state,
  'INTEGRITY_NOT_VERIFIED');
 const ready=assessPrivatePdfSource({...captured,
  storedBytesIndependentlyRehashed:true,privateReceiptIndependentlyVerified:true},
  '2026-10-30');
 assert.equal(ready.state,'SCAN_ELIGIBLE_PRE_ADMISSION');
 assert.equal(ready.scanAllowed,true);
 assert.equal(ready.capturedAfterCutoff,true);
 assert.equal(ready.historicalBytesAtCutoffCertified,false);
 assert.equal(ready.issuerAuthenticated,false);
 assert.equal(ready.registryAdmitted,false);
 assert.equal(assessPrivatePdfSource({...captured,publicationDate:'2026-11-01'},
  '2026-10-30').state,'POST_CUTOFF_PUBLICATION');
});
