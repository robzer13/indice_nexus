import test from 'node:test';
import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import type {SupabaseClient} from '@supabase/supabase-js';
import {ASML_V29_BUCKET,ASML_V29_RUN} from '../lib/orotitan-equity/product/asml-official-pdf-capture-v29';
import {archiveBrowserProvidedTsmcQ3,TSMC_BROWSER_ROOT,TSMC_BROWSER_SOURCE}
 from '../lib/orotitan-equity/product/tsmc-browser-pdf-upload-v1';

const pdf=Buffer.concat([
 Buffer.from('%PDF-1.7\n'),
 Buffer.from('Synthetic test only, never an authentic TSMC transcript.\n'.repeat(240)),
 Buffer.from('%%EOF\n'),
]);
const digest=createHash('sha256').update(pdf).digest('hex');
function mockStorage(opts:{publicBucket?:boolean;readbackMismatch?:boolean}={}){
 const objects=new Map<string,Buffer>();
 let writes=0;
 const client={storage:{
  async getBucket(id:string){
   assert.equal(id,ASML_V29_BUCKET);
   return {data:{id,public:opts.publicBucket??false},error:null};
  },
  from(id:string){
   assert.equal(id,ASML_V29_BUCKET);
   return {
    async upload(path:string,body:Buffer,options:Record<string,unknown>){
     assert.equal(options.upsert,false);
     writes++;
     if(objects.has(path))return {error:{statusCode:'409',code:'KeyAlreadyExists'}};
     objects.set(path,Buffer.from(body));return {error:null};
    },
    async download(path:string){
     const original=objects.get(path);
     if(!original)return {data:null,error:{code:'NOT_FOUND'}};
     const bytes=opts.readbackMismatch?Buffer.from('tampered'):original;
     return {data:new Blob([new Uint8Array(bytes)]),error:null};
    }
   };
  },
 }} as unknown as SupabaseClient;
 return {client,objects,get writes(){return writes;}};
}
test('TSMC manual transfer creates two immutable private files and receipt with non-authenticated provenance',async()=>{
 const f=mockStorage();
 const receipt=await archiveBrowserProvidedTsmcQ3(f.client,pdf,()=>new Date('2026-10-30T16:00:00Z'));
 assert.equal(receipt.schema,'OROTITAN_TSMC_Q3_BROWSER_PDF_RECEIPT_V1');
 assert.equal(receipt.documentId,'TSMC_2026_Q3_CALL');
 assert.equal(receipt.runId,ASML_V29_RUN);
 assert.equal(receipt.sourceLink,TSMC_BROWSER_SOURCE.url);
 assert.equal(receipt.sha256,digest);
 assert.equal(receipt.byteSize,pdf.length);
 assert.equal(receipt.storageReadbackVerified,true);
 assert.equal(receipt.sourceOriginAuthenticated,false);
 assert.equal(receipt.sourceDocumentAuthenticityVerified,false);
 assert.equal(receipt.historicalBytesAtCutoffIndependentlyVerified,false);
 assert.equal(receipt.registryAdmitted,false);
 assert.equal(receipt.stageAdvanced,false);
 assert.ok(receipt.sourcePath.startsWith(TSMC_BROWSER_ROOT+'/pdf/'));
 assert.equal(f.objects.size,2);
 assert.ok(f.objects.get(receipt.sourcePath)?.equals(pdf));
 assert.equal(JSON.parse(f.objects.get(receipt.receiptPath)!.toString()).sha256,digest);
});
test('TSMC duplicate upload preserves original receipt and first timestamp',async()=>{
 const f=mockStorage();
 const a=await archiveBrowserProvidedTsmcQ3(f.client,pdf,()=>new Date('2026-10-30T16:00:00Z'));
 const b=await archiveBrowserProvidedTsmcQ3(f.client,pdf,()=>new Date('2026-10-31T16:00:00Z'));
 assert.equal(a.sourcePath,b.sourcePath);
 assert.equal(f.objects.size,2);
 assert.equal(b.uploadedAt,'2026-10-30T16:00:00.000Z');
});
test('TSMC invalid payload and public bucket fail closed',async()=>{
 const f=mockStorage();
 await assert.rejects(
  archiveBrowserProvidedTsmcQ3(f.client,Buffer.from('<html>'+pdf.toString('latin1')+'</html>')),
  /INVALID_PDF/
 );
 assert.equal(f.writes,0);
 const pub=mockStorage({publicBucket:true});
 await assert.rejects(archiveBrowserProvidedTsmcQ3(pub.client,pdf),/PRIVATE_BUCKET_REQUIRED/);
 assert.equal(pub.writes,0);
});
test('TSMC tampered storage readback cannot produce receipt',async()=>{
 const f=mockStorage({readbackMismatch:true});
 await assert.rejects(archiveBrowserProvidedTsmcQ3(f.client,pdf),/STORAGE_READBACK_MISMATCH/);
 assert.equal(f.objects.size,1);
});
