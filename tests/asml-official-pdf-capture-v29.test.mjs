import test from 'node:test';
import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import {
  ASML_V29_SOURCES,ASML_V29_BUCKET,
  captureAsmlOfficialPdf,selectSource,acquireOfficialPdf,
  validateSameOriginRedirect,checkPdfBytes,
} from '../lib/orotitan-equity/product/asml-official-pdf-capture-v29.ts';

const bytes=Buffer.concat([
  Buffer.from('%PDF-1.7\n'),
  Buffer.from('Test-only fixture; NOT an issuer source.\n'.repeat(140)),
  Buffer.from('%%EOF\n'),
]);
const sha=createHash('sha256').update(bytes).digest('hex');
function storageFixture({publicBucket=false,tamperReadback=false}={}){
  const objects=new Map(); let reads=0; let writes=0;
  const client={storage:{
    async getBucket(id){assert.equal(id,ASML_V29_BUCKET);return {data:{id,public:publicBucket},error:null};},
    from(id){assert.equal(id,ASML_V29_BUCKET);return {
      async upload(path,body,opts){
        assert.equal(opts.upsert,false);writes++;
        if(objects.has(path))return {error:{statusCode:'409',code:'KeyAlreadyExists'}};
        objects.set(path,Buffer.from(body));return {error:null};
      },
      async download(path){
        reads++;
        const original=objects.get(path);
        if(!original)return {data:null,error:{code:'NOT_FOUND'}};
        const body=tamperReadback?Buffer.from('bad readback'):original;
        return {data:new Blob([body]),error:null};
      },
    };},
  }};
  return {client,objects,get reads(){return reads;},get writes(){return writes;}};
}
function issuerFetch(targetId='ASML_2025_US_GAAP'){
  const s=ASML_V29_SOURCES[targetId];
  return async (url,init)=>{
    assert.equal(url,s.url);
    assert.equal(init.redirect,'manual');
    assert.equal(init.cache,'no-store');
    return new Response(bytes,{status:200,headers:{'content-type':'application/pdf'}});
  };
}
test('only exact, enumerated document IDs are permitted',()=>{
  assert.equal(selectSource('ASML_2025_US_GAAP').accountingBasis,'US_GAAP');
  assert.throws(()=>selectSource('https://evil.example/file.pdf'),/UNKNOWN_SOURCE/);
  assert.throws(()=>selectSource('__proto__'),/UNKNOWN_SOURCE/);
});
test('allowlisted server TLS PDF capture persists immutable bytes and verified receipt',async()=>{
  const s=storageFixture();
  const receipt=await captureAsmlOfficialPdf(s.client,'ASML_2025_US_GAAP',issuerFetch(),
    ()=>new Date('2026-10-30T20:00:00Z'));
  assert.equal(receipt.sha256,sha);
  assert.equal(receipt.byteSize,bytes.length);
  assert.equal(receipt.registryAdmitted,false);
  assert.equal(receipt.sourceOriginCryptographicallyAttested,false);
  assert.equal(receipt.documentAuthenticityIndependentlyVerified,false);
  assert.equal(receipt.storageReadbackVerified,true);
  assert.equal(receipt.accountingBasis,'US_GAAP');
  assert.equal(s.objects.size,2);
  assert.ok(s.objects.get(receipt.sourcePath).equals(bytes));
  assert.equal(s.reads,2);
  assert.equal(JSON.parse(s.objects.get(receipt.receiptPath).toString()).sha256,sha);
});
test('retry is idempotent, original receipt timestamp is preserved',async()=>{
  const s=storageFixture();
  const a=await captureAsmlOfficialPdf(s.client,'ASML_2025_US_GAAP',issuerFetch(),()=>new Date('2026-10-30T20:00:00Z'));
  const b=await captureAsmlOfficialPdf(s.client,'ASML_2025_US_GAAP',issuerFetch(),()=>new Date('2026-10-30T21:00:00Z'));
  assert.equal(a.sourcePath,b.sourcePath);
  assert.equal(s.objects.size,2);
  const persisted=JSON.parse(s.objects.get(a.receiptPath).toString());
  assert.equal(persisted.receivedAt,'2026-10-30T20:00:00.000Z');
});
test('safe same-host redirect is permitted and recorded',async()=>{
  const origin=ASML_V29_SOURCES.ASML_2025_US_GAAP;
  let calls=0;
  const moved='https://ourbrand.asml.com/m/71076aaad607de4d/original/Exhibit%252015.2.pdf';
  const f=async(url)=>{
    calls++;
    if(calls===1){assert.equal(url,origin.url);return new Response(null,{status:302,headers:{location:moved}});}
    assert.equal(url,moved);return new Response(bytes,{status:200,headers:{'content-type':'application/pdf'}});
  };
  const r=await acquireOfficialPdf('ASML_2025_US_GAAP',f);
  assert.equal(r.finalUrl,moved);assert.deepEqual(r.redirectChain,[moved]);assert.equal(calls,2);
});
test('off-host, http, and out-of-path redirects fail closed',async()=>{
  const source=selectSource('ASML_2025_US_GAAP');
  for(const bad of ['https://evil.example/a.pdf','http://ourbrand.asml.com/m/71076aaad607de4d/original/x.pdf',
    'https://ourbrand.asml.com/private/x.pdf']){
    assert.throws(()=>validateSameOriginRedirect(source,bad),/OUTSIDE_ALLOWLIST/);
  }
  const evil=async()=>new Response(null,{status:302,headers:{location:'https://evil.example/pdf'}});
  await assert.rejects(acquireOfficialPdf('ASML_2025_US_GAAP',evil),/REDIRECT_OUTSIDE_ALLOWLIST/);
});
test('public bucket never fetches or stores',async()=>{
  const s=storageFixture({publicBucket:true});let fetched=0;
  await assert.rejects(captureAsmlOfficialPdf(s.client,'ASML_2025_US_GAAP',async()=>{fetched++;return new Response(bytes);}),/PRIVATE_BUCKET_REQUIRED/);
  assert.equal(fetched,0);assert.equal(s.writes,0);
});
test('malformed PDF, MIME confusion, or oversized body fails closed',async()=>{
  assert.throws(()=>checkPdfBytes(Buffer.from('<html>fake</html>'),40_000_000),/INVALID_PDF/);
  const fake=async()=>new Response(Buffer.from('<html>'+bytes.toString('latin1')+'</html>'),
    {status:200,headers:{'content-type':'text/html'}});
  await assert.rejects(acquireOfficialPdf('ASML_2025_US_GAAP',fake),/UNEXPECTED_MIME/);
  const big=async()=>new Response(bytes,{status:200,headers:{'content-type':'application/pdf','content-length':'80000000'}});
  await assert.rejects(acquireOfficialPdf('ASML_2025_US_GAAP',big),/CONTENT_LENGTH_INVALID/);
});
test('storage readback mismatch never produces successful receipt',async()=>{
  const s=storageFixture({tamperReadback:true});
  await assert.rejects(captureAsmlOfficialPdf(s.client,'ASML_2025_US_GAAP',issuerFetch()),/PRIVATE_READBACK_MISMATCH/);
  assert.equal(s.objects.size,1);
});
