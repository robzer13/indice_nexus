import test from 'node:test';
import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import type {SupabaseClient} from '@supabase/supabase-js';
import type {OfficialSourceCandidate} from '../lib/orotitan-equity/product/research-recovery-official-source-v1';
import {ASML_Q2_2026_PUBLIC_RELEASE,ASML_Q2_2026_OFFICIAL_PRESENTATION} from '../lib/orotitan-equity/product/research-recovery-official-source-v1';
import {archiveAsmlCandidate,RECOVERY_PRIVATE_BUCKET} from '../lib/orotitan-equity/product/research-recovery-private-archive-v1';

const body=Buffer.from('<html><title>ASML Q2 2026</title>'+'source'.repeat(90)+'</html>','utf8');
const sha=createHash('sha256').update(body).digest('hex');
const candidate:OfficialSourceCandidate={
  documentId:ASML_Q2_2026_PUBLIC_RELEASE.documentId,
  url:ASML_Q2_2026_PUBLIC_RELEASE.url,
  publicationDate:'2026-07-15',retrievedAt:'2026-10-30T10:00:00.000Z',
  byteSize:body.length,sha256:sha,contentType:'text/html',
  admitted:false,historicalBytesCertified:false,
};

function fixture(opts:{public?:boolean;uploadCode?:string;uploadErrorCode?:string;stored?:Buffer;expectedBody?:Buffer;downloadFailure?:boolean}={}) {
  const log:{uploadPath?:string;uploadOptions?:Record<string,unknown>;downloads:number;uploads:number}={
    downloads:0,uploads:0,
  };
  const client={
    storage:{
      async getBucket(id:string){
        assert.equal(id,RECOVERY_PRIVATE_BUCKET);
        return {data:{id,public:opts.public??false},error:null};
      },
      from(id:string){
        assert.equal(id,RECOVERY_PRIVATE_BUCKET);
        return {
          async upload(path:string,data:Buffer,options:Record<string,unknown>){
            log.uploads++;
            log.uploadPath=path;
            log.uploadOptions=options;
            assert.deepEqual(data,opts.expectedBody??body);
            return {error:opts.uploadCode?{statusCode:opts.uploadCode,code:opts.uploadErrorCode}:null};
          },
          async download(path:string){
            log.downloads++;
            assert.equal(path,log.uploadPath);
            return opts.downloadFailure
              ? {data:null,error:{message:'readback failed'}}
              : {data:new Blob([new Uint8Array(opts.stored??opts.expectedBody??body)]),error:null};
          },
        };
      },
    },
  } as unknown as SupabaseClient;
  return {client,log};
}

test('private content-addressed object is create-only and READBACK hash verified',async()=>{
  const f=fixture();
  const result=await archiveAsmlCandidate(f.client,candidate,body);
  assert.equal(result.bucket,RECOVERY_PRIVATE_BUCKET);
  assert.equal(result.path,'research-recovery-v1/asml-observation/q2-2026/'+sha+'.html');
  assert.equal(result.sha256,sha);
  assert.equal(result.byteSize,body.length);
  assert.equal(result.verifiedReadback,true);
  assert.equal(result.registryAdmitted,false);
  assert.equal(f.log.uploadOptions?.upsert,false);
  assert.equal(f.log.uploadOptions?.contentType,'text/html');
  assert.equal(f.log.downloads,1);
});
test('existing SHA-addressed object is verified, never overwritten',async()=>{
  const f=fixture({uploadCode:'409',uploadErrorCode:'KeyAlreadyExists'});
  const receipt=await archiveAsmlCandidate(f.client,candidate,body);
  assert.equal(receipt.verifiedReadback,true);
  assert.equal(f.log.uploadOptions?.upsert,false);
  assert.equal(f.log.downloads,1);
});
test('legacy 400 exact duplicate is accepted only after readback',async()=>{
  const f=fixture({uploadCode:'400',uploadErrorCode:'already_exists'});
  const receipt=await archiveAsmlCandidate(f.client,candidate,body);
  assert.equal(receipt.verifiedReadback,true);
  assert.equal(f.log.downloads,1);
});
test('unrelated 400 and 409 responses fail closed, never triggering a download',async()=>{
  for(const code of ['400','409']) {
    const f=fixture({uploadCode:code,uploadErrorCode:'AccessDenied'});
    await assert.rejects(archiveAsmlCandidate(f.client,candidate,body),/CREATE_FAILED/);
    assert.equal(f.log.downloads,0);
  }
});
test('public bucket forbidden before any write',async()=>{
  const f=fixture({public:true});
  await assert.rejects(archiveAsmlCandidate(f.client,candidate,body),/BUCKET_NOT_PRIVATE/);
  assert.equal(f.log.uploads,0);
});
test('wrong hash and incorrect length forbidden before any backend request',async()=>{
  const f=fixture();
  await assert.rejects(archiveAsmlCandidate(f.client,{...candidate,sha256:'f'.repeat(64)},body),/INPUT_INTEGRITY/);
  await assert.rejects(archiveAsmlCandidate(f.client,{...candidate,byteSize:1000},body),/INPUT_INTEGRITY/);
  assert.equal(f.log.uploads,0);
});
test('unexpected write failure, readback failure and byte mismatch fail closed',async()=>{
  const upload=fixture({uploadCode:'500'});
  await assert.rejects(archiveAsmlCandidate(upload.client,candidate,body),/CREATE_FAILED/);
  assert.equal(upload.log.downloads,0);
  const missing=fixture({downloadFailure:true});
  await assert.rejects(archiveAsmlCandidate(missing.client,candidate,body),/READBACK_FAILED/);
  const corrupt=fixture({stored:Buffer.from('different'.repeat(100))});
  await assert.rejects(archiveAsmlCandidate(corrupt.client,candidate,body),/READBACK_MISMATCH/);
});

test('official ASML PDF is archived create-only under its own matching SHA filename',async()=>{
  const pdf=Buffer.concat([Buffer.from('%PDF-1.7\n'),Buffer.alloc(12000,0x20),Buffer.from('\n%%EOF\n')]);
  const digest=createHash('sha256').update(pdf).digest('hex');
  const fixturePdf=fixture({expectedBody:pdf});
  const receipt=await archiveAsmlCandidate(fixturePdf.client,{
    documentId:ASML_Q2_2026_OFFICIAL_PRESENTATION.documentId,
    url:ASML_Q2_2026_OFFICIAL_PRESENTATION.url,
    publicationDate:'2026-07-15',
    retrievedAt:'2026-10-30T12:00:00.000Z',
    byteSize:pdf.length,sha256:digest,
    contentType:'application/pdf',
    admitted:false,historicalBytesCertified:false,
  },pdf);
  assert.equal(receipt.path,'research-recovery-v1/asml-observation/q2-2026/'+digest+'.pdf');
  assert.equal(receipt.verifiedReadback,true);
  assert.equal(receipt.registryAdmitted,false);
  assert.equal(fixturePdf.log.uploadOptions?.contentType,'application/pdf');
  assert.equal(fixturePdf.log.uploadOptions?.upsert,false);
});
