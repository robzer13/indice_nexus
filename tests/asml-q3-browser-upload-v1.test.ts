import test from 'node:test';
import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import type {SupabaseClient} from '@supabase/supabase-js';
import {archiveBrowserProvidedAsmlQ3Pdf,ASML_Q3_US_GAAP_PDF}
 from '../lib/orotitan-equity/product/asml-q3-browser-upload-v1';
const pdf=Buffer.concat([Buffer.from('%PDF-1.7\n'),
 Buffer.from('Synthetic financial PDF, no authentic ASML source.\n'.repeat(280)),
 Buffer.from('%%EOF\n')]);
const hash=createHash('sha256').update(pdf).digest('hex');
function fixture(opts:{public?:boolean;readbackMismatch?:boolean}={}) {
 const objects=new Map<string,Buffer>();
 let writes=0,downloads=0;
 const client={storage:{
  async getBucket(id:string){return {data:{id,public:opts.public??false},error:null};},
  from(_id:string){return {
   async upload(path:string,data:Buffer,options:Record<string,unknown>){
    assert.equal(options.upsert,false);writes++;
    if(objects.has(path))return {error:{statusCode:'409',code:'KeyAlreadyExists'}};
    objects.set(path,Buffer.from(data));return {error:null};
   },
   async download(path:string){
    downloads++;
    const original=objects.get(path);
    if(!original)return {data:null,error:{code:'NOT_FOUND'}};
    return {data:new Blob([new Uint8Array(
      opts.readbackMismatch ? Buffer.from('corrupted') : original)]),error:null};
   },
  };},
 }} as unknown as SupabaseClient;
 return {client,objects,get writes(){return writes;},get downloads(){return downloads;}};
}
test('private PDF capture is create-only, SHA verified and not provenance attested',async()=>{
 const f=fixture();
 const x=await archiveBrowserProvidedAsmlQ3Pdf(f.client,pdf,
  ()=>new Date('2026-10-30T13:00:00Z'));
 assert.equal(x.documentId,ASML_Q3_US_GAAP_PDF.id);
 assert.equal(x.sha256,hash);
 assert.equal(x.byteSize,pdf.length);
 assert.equal(x.storageReadbackVerified,true);
 assert.equal(x.sourceOriginAuthenticated,false);
 assert.equal(x.sourceDocumentAuthenticityVerified,false);
 assert.equal(x.registryAdmitted,false);
 assert.equal(f.objects.size,2);
 assert.ok(f.objects.has(x.sourcePath));
 assert.ok(f.objects.has(x.receiptPath));
 assert.equal(f.downloads,2);
});
test('same bytes retry preserves original immutable PDF and metadata',async()=>{
 const f=fixture();
 const a=await archiveBrowserProvidedAsmlQ3Pdf(f.client,pdf,
  ()=>new Date('2026-10-30T13:00:00Z'));
 const b=await archiveBrowserProvidedAsmlQ3Pdf(f.client,pdf,
  ()=>new Date('2026-10-30T13:10:00Z'));
 assert.equal(a.sourcePath,b.sourcePath);
 assert.equal(f.objects.size,2);
 const manifest=JSON.parse(f.objects.get(a.receiptPath)!.toString('utf8'));
 assert.equal(manifest.uploadedAt,'2026-10-30T13:00:00.000Z');
});
test('wrong mime bytes, public bucket, and tampered readback fail closed',async()=>{
 const f=fixture();
 await assert.rejects(archiveBrowserProvidedAsmlQ3Pdf(f.client,
  Buffer.from('<html>'+pdf.toString('latin1')+'</html>')),/INVALID_PDF/);
 assert.equal(f.writes,0);
 const pub=fixture({public:true});
 await assert.rejects(archiveBrowserProvidedAsmlQ3Pdf(pub.client,pdf),/PRIVATE_BUCKET_REQUIRED/);
 assert.equal(pub.writes,0);
 const mismatch=fixture({readbackMismatch:true});
 await assert.rejects(archiveBrowserProvidedAsmlQ3Pdf(mismatch.client,pdf),/READBACK_MISMATCH/);
 assert.equal(mismatch.objects.size,1);
});
