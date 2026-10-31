import test from 'node:test';
import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import type {SupabaseClient} from '@supabase/supabase-js';
import {
 V34_BUCKET,V34_CUTOFF,V34_RUN_ID,V34_SOURCES,
 assertValidSourceId,verifyExpectedPdfFingerprint,verifyStoredPdfBytes,
 validatePrivateReceipt,numericCandidatesFromPage,extractBoundedPages,
 independentAuditOne,
} from '../lib/orotitan-equity/product/asml-research-private-pdf-reaudit-v34';

const sha=(b:Buffer)=>createHash('sha256').update(b).digest('hex');
function minimalPdf(text:string):Buffer{
 // A minimal valid single-page PDF fixture, NOT issuer data.
 const escaped=text.replace(/([()\\])/g,'\\$1');
 const stream='BT /F1 12 Tf 60 710 Td ('+escaped+') Tj ET\n';
 const objects=[
  '<< /Type /Catalog /Pages 2 0 R >>',
  '<< /Type /Pages /Kids [3 0 R] /Count 1 >>',
  '<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>',
  '<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>',
  '<< /Length '+Buffer.byteLength(stream)+' >>\nstream\n'+stream+'endstream',
 ];
 let pdf='%PDF-1.4\n',offsets=[0];
 for(let i=0;i<objects.length;i++){
  offsets.push(Buffer.byteLength(pdf));
  pdf+=(i+1)+' 0 obj\n'+objects[i]+'\nendobj\n';
 }
 const xref=Buffer.byteLength(pdf);
 pdf+='xref\n0 '+(objects.length+1)+'\n0000000000 65535 f \n';
 for(const offset of offsets.slice(1))pdf+=String(offset).padStart(10,'0')+' 00000 n \n';
 pdf+='trailer\n<< /Size '+(objects.length+1)+' /Root 1 0 R >>\nstartxref\n'+xref+'\n%%EOF\n';
 return Buffer.from(pdf,'latin1');
}
test('pure byte verification validates a test fixture and rejects size/hash/PDF format errors',()=>{
 const good=minimalPdf('32667.3 ASML test only');
 assert.equal(verifyExpectedPdfFingerprint(good,sha(good),good.length),sha(good));
 assert.throws(()=>verifyExpectedPdfFingerprint(good,sha(good),good.length+1),/SIZE_MISMATCH/);
 assert.throws(()=>verifyExpectedPdfFingerprint(good,'a'.repeat(64),good.length),/SHA_MISMATCH/);
 const fake=Buffer.from('<html>'+good.toString('latin1'));
 assert.throws(()=>verifyExpectedPdfFingerprint(fake,sha(fake),fake.length),/PDF_STRUCTURE_INVALID/);
 assert.throws(()=>verifyStoredPdfBytes('ASML_2025_US_GAAP',good),/SIZE_MISMATCH/);
});
test('only exact three locked SHADOW source identifiers are permitted',()=>{
 for(const id of Object.keys(V34_SOURCES))assert.doesNotThrow(()=>assertValidSourceId(id));
 for(const id of ['__proto__','constructor','https://bad.invalid/a','ASML_2026_Q3_USER']){
  assert.throws(()=>assertValidSourceId(id),/SOURCE_DENIED/);
 }
 assert.equal(V34_CUTOFF,'2026-10-30');
});
test('all three receipt schemas are checked against immutable stored SHA, byte-size and pre-admission flags',()=>{
 for(const [id,source] of Object.entries(V34_SOURCES)){
  const date='2026-10-31T06:05:30.963Z';
  const base={
   schema:source.receiptSchema,sha256:source.sha256,byteSize:source.size,
   sourcePath:source.sourcePath,receiptPath:source.receiptPath,cutoff:V34_CUTOFF,
   registryAdmitted:false,storageReadbackVerified:true,
  };
  const specific=id==='ASML_2025_US_GAAP'?{
   runId:V34_RUN_ID,sourceId:id,bucket:V34_BUCKET,stageAdvanced:false,
   sourceOriginCryptographicallyAttested:false,documentAuthenticityIndependentlyVerified:false,
   historicalBytesAtCutoffIndependentlyVerified:false,sourcePublicationDate:source.sourceDate,receivedAt:date,
  }:id==='ASML_2026_Q3_US_GAAP'?{
   documentId:source.documentId,publicationDate:source.sourceDate,
   sourceOriginAuthenticated:false,sourceDocumentAuthenticityVerified:false,
   uploadedAt:date,
  }:{
   runId:V34_RUN_ID,documentId:id,bucket:V34_BUCKET,publicationDate:source.sourceDate,
   stageAdvanced:false,sourceOriginAuthenticated:false,sourceDocumentAuthenticityVerified:false,
   historicalBytesAtCutoffIndependentlyVerified:false,uploadedAt:date,
  };
  const receipt={...base,...specific};
  const valid=Buffer.from(JSON.stringify(receipt));
  const checked=validatePrivateReceipt(id as keyof typeof V34_SOURCES,valid);
  assert.equal(checked.receiptHash,sha(valid));assert.equal(checked.uploadedAt,date);
  for(const bad of [
   {...receipt,sha256:'0'.repeat(64)}, {...receipt,registryAdmitted:true},
   {...receipt,sourcePath:'other'}, {...receipt,storageReadbackVerified:false},
  ])assert.throws(()=>validatePrivateReceipt(id as keyof typeof V34_SOURCES,
   Buffer.from(JSON.stringify(bad))),/RECEIPT_MISMATCH/);
 }
});
test('numeric candidate matching does not promote calculated or unarchived claims',()=>{
 const m=numericCandidatesFromPage('ASML_2025_US_GAAP',
  'TOTAL NET SALES 32,667.3 million; Net service and field option sales 8,193.0.');
 assert.deepEqual(m.map(x=>x.claimId),['C-001','C-002']);
 assert.ok(m.every(x=>x.match==='NUMERIC_TOKEN_ONLY'));
 assert.ok(!m.some(x=>x.claimId==='C-003'||x.claimId==='C-009'));
 const t=numericCandidatesFromPage('TSMC_2026_Q3_CALL','CapEx forecast US$60 to US$64 billion.');
 assert.deepEqual(t.map(x=>x.claimId),['C-022','C-023']);
});
test('bounded text extraction detects a numeric token in a real one-page synthetic PDF',async()=>{
 const b=minimalPdf('Total net sales 32667.3');
 const r=await extractBoundedPages('ASML_2025_US_GAAP',b,1,1);
 assert.equal(r.pdfPageCount,1);assert.equal(r.startPage,1);assert.equal(r.endPage,1);
 assert.ok(r.pageObservations[0].textLength>10);
 assert.ok(r.pageObservations[0].candidateMatches.some(x=>x.claimId==='C-001'));
 await assert.rejects(extractBoundedPages('ASML_2025_US_GAAP',b,1,21),/PAGE_RANGE_DENIED/);
});
test('private bucket and SHA mismatch fail closed before a receipt is trusted',async()=>{
 const source=V34_SOURCES.ASML_2025_US_GAAP;
 let downloadCount=0;
 const client={storage:{
  async getBucket(){return {data:{id:V34_BUCKET,public:true},error:null};},
  from(){return {async download(){downloadCount++;return {data:null,error:{code:'missing'}};}};},
 }} as unknown as SupabaseClient;
 await assert.rejects(independentAuditOne(client,'ASML_2025_US_GAAP'),/PRIVATE_BUCKET_REQUIRED/);
 assert.equal(downloadCount,0);
 const f={storage:{
  async getBucket(){return {data:{id:V34_BUCKET,public:false},error:null};},
  from(){return {async download(path:string){
   assert.equal(path,source.sourcePath);
   return {data:new Blob([new Uint8Array(Buffer.alloc(source.size,0))]),error:null};
  }};},
 }} as unknown as SupabaseClient;
 await assert.rejects(independentAuditOne(f,'ASML_2025_US_GAAP'),/SHA_MISMATCH/);
});
