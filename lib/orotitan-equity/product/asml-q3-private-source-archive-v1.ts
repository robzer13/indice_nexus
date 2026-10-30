/**
 * Private SHADOW-only pre-admission capture; DOES NOT mutate Registry.
 * Caller requires exact hosted SHADOW gate and server-side authenticated admin.
 * Source and receipt are both create-only and re-read with integrity checks.
 */
import {createHash} from 'node:crypto';
import type {SupabaseClient} from '@supabase/supabase-js';
import {ASML_Q3_SEC_SOURCES,ASML_Q3_CUTOFF,ASML_Q3_MAX_BYTES,
 type AsmlQ3Candidate,type AsmlQ3SourceId} from './asml-q3-official-sources-v1';

export const ASML_Q3_PRIVATE_BUCKET='orotitan-source-files-v1';
export const ASML_Q3_ARCHIVE_ROOT='research-acquisition-v1/asml/asof-2026-10-30/q3-2026';
const SHA=/^[0-9a-f]{64}$/;

export type AsmlQ3ArchiveReceipt=Readonly<{
  schema:'OROTITAN_Q3_PRIVATE_CAPTURE_RECEIPT_V1';
  documentId:AsmlQ3SourceId;sourceUrl:string;publicationDate:'2026-10-14';
  cutoff:'2026-10-30';retrievedAt:string;sha256:string;byteSize:number;
  mime:'text/html';sourcePath:string;receiptPath:string;
  includesLinkedMedia:false;readbackVerified:true;registryAdmitted:false;
}>;

function isDuplicate(error:unknown):boolean {
  if(!error || typeof error!=='object')return false;
  const e=error as Record<string,unknown>;
  return ['400','409'].includes(String(e.statusCode)) &&
    (['KeyAlreadyExists','ResourceAlreadyExists','already_exists'].includes(String(e.code??e.error??'')) ||
    /^(?:The resource already exists|Asset Already Exists|Duplicate)$/i.test(String(e.message??'').trim()));
}
const digest=(data:Buffer)=>createHash('sha256').update(data).digest('hex');

async function storeCreateOnlyAndVerify(
  client:SupabaseClient,path:string,body:Buffer,mime:'text/html'|'application/json',
):Promise<void>{
  const h=client.storage.from(ASML_Q3_PRIVATE_BUCKET);
  const {error}=await h.upload(path,body,{
    contentType:mime,upsert:false,cacheControl:'0',
  });
  if(error && !isDuplicate(error))throw Error('ASML_Q3_PRIVATE_UPLOAD_FAILED');
  const {data, error:downloadError}=await h.download(path);
  if(downloadError||!data)throw Error('ASML_Q3_PRIVATE_READBACK_FAILED');
  const verified=Buffer.from(await data.arrayBuffer());
  if(verified.length!==body.length || digest(verified)!==digest(body)) {
    // Receipt metadata has immutable first capture timestamp and must be
    // validated semantically, not replaced, on already-existing receipts.
    if(mime==='application/json' && error && isDuplicate(error)){
      try {
        const old=JSON.parse(verified.toString('utf8')) as AsmlQ3ArchiveReceipt;
        const current=JSON.parse(body.toString('utf8')) as AsmlQ3ArchiveReceipt;
        if(old.schema===current.schema && old.documentId===current.documentId &&
           old.sourceUrl===current.sourceUrl &&
           old.publicationDate===current.publicationDate &&
           old.cutoff===current.cutoff && old.sha256===current.sha256 &&
           old.byteSize===current.byteSize && old.mime===current.mime &&
           old.sourcePath===current.sourcePath && old.receiptPath===current.receiptPath &&
           old.readbackVerified===true && old.registryAdmitted===false &&
           old.includesLinkedMedia===false && !Number.isNaN(Date.parse(old.retrievedAt)))return;
      } catch { /* A corrupt manifest cannot be accepted. */ }
    }
    throw Error('ASML_Q3_PRIVATE_READBACK_MISMATCH');
  }
}

export async function archiveAsmlQ3Candidate(
  client:SupabaseClient,candidate:AsmlQ3Candidate,bytes:Buffer,
):Promise<AsmlQ3ArchiveReceipt>{
  const source=Object.hasOwn(ASML_Q3_SEC_SOURCES,candidate.documentId)
    ?ASML_Q3_SEC_SOURCES[candidate.documentId]:null;
  if(!source || candidate.url!==source.url ||
     candidate.publicationDate!=='2026-10-14' ||
     candidate.cutoff!==ASML_Q3_CUTOFF ||
     candidate.contentType!=='text/html' ||
     candidate.admitted!==false || candidate.historicalBytesCertified!==false ||
     candidate.includesLinkedMedia!==false ||
     !Buffer.isBuffer(bytes)||bytes.length<1000||bytes.length>ASML_Q3_MAX_BYTES ||
     bytes.length!==candidate.byteSize || !SHA.test(candidate.sha256) ||
     digest(bytes)!==candidate.sha256 || Number.isNaN(Date.parse(candidate.retrievedAt)))
     throw Error('ASML_Q3_PRIVATE_INPUT_INTEGRITY');
  const {data:bucket,error}=await client.storage.getBucket(ASML_Q3_PRIVATE_BUCKET);
  if(error||!bucket||bucket.id!==ASML_Q3_PRIVATE_BUCKET||bucket.public!==false)
    throw Error('ASML_Q3_PRIVATE_BUCKET_UNSAFE');
  const sourcePath=ASML_Q3_ARCHIVE_ROOT+'/'+source.documentId+'/'+candidate.sha256+'.html';
  const receiptPath=ASML_Q3_ARCHIVE_ROOT+'/receipts/'+source.documentId+'-'+candidate.sha256+'.json';
  await storeCreateOnlyAndVerify(client,sourcePath,bytes,'text/html');
  const receipt:AsmlQ3ArchiveReceipt={
    schema:'OROTITAN_Q3_PRIVATE_CAPTURE_RECEIPT_V1',
    documentId:candidate.documentId,
    sourceUrl:source.url,
    publicationDate:'2026-10-14',cutoff:'2026-10-30',
    retrievedAt:candidate.retrievedAt,sha256:candidate.sha256,
    byteSize:candidate.byteSize,mime:'text/html',
    sourcePath,receiptPath,
    includesLinkedMedia:false,readbackVerified:true,registryAdmitted:false,
  };
  await storeCreateOnlyAndVerify(client,receiptPath,
    Buffer.from(JSON.stringify(receipt)+'\n','utf8'),'application/json');
  return receipt;
}

export async function listAsmlQ3SourceReceipts(client:SupabaseClient):
  Promise<AsmlQ3ArchiveReceipt[]>{
  const dir=ASML_Q3_ARCHIVE_ROOT+'/receipts';
  const {data,error}=await client.storage.from(ASML_Q3_PRIVATE_BUCKET)
    .list(dir,{limit:50,sortBy:{column:'name',order:'asc'}});
  if(error||!data||data.length>=50)throw Error('ASML_Q3_RECEIPT_INDEX_FAILED');
  const result:AsmlQ3ArchiveReceipt[]=[];
  for(const row of data){
    if(!/^(?:SEC_EX_99_[123])-[0-9a-f]{64}\.json$/.test(row.name))
      throw Error('ASML_Q3_RECEIPT_NAME_INVALID');
    const {data:blob,error:dlError}=await client.storage.from(ASML_Q3_PRIVATE_BUCKET)
      .download(dir+'/'+row.name);
    if(dlError||!blob)throw Error('ASML_Q3_RECEIPT_READBACK_FAILED');
    const b=Buffer.from(await blob.arrayBuffer());
    if(b.length>4000)throw Error('ASML_Q3_RECEIPT_INVALID');
    const receipt=JSON.parse(b.toString('utf8')) as AsmlQ3ArchiveReceipt;
    if(receipt.schema!=='OROTITAN_Q3_PRIVATE_CAPTURE_RECEIPT_V1' ||
       !Object.hasOwn(ASML_Q3_SEC_SOURCES,receipt.documentId) ||
       receipt.sourceUrl!==ASML_Q3_SEC_SOURCES[receipt.documentId].url ||
       receipt.receiptPath!==dir+'/'+row.name ||
       !receipt.sourcePath.startsWith(ASML_Q3_ARCHIVE_ROOT+'/'+receipt.documentId+'/') ||
       receipt.sourcePath!==ASML_Q3_ARCHIVE_ROOT+'/'+receipt.documentId+'/'+receipt.sha256+'.html' ||
       receipt.publicationDate!=='2026-10-14'||receipt.cutoff!=='2026-10-30' ||
       !SHA.test(receipt.sha256) || receipt.byteSize<1000 || receipt.byteSize>ASML_Q3_MAX_BYTES ||
       receipt.mime!=='text/html' || receipt.registryAdmitted!==false ||
       receipt.includesLinkedMedia!==false || receipt.readbackVerified!==true)
      throw Error('ASML_Q3_RECEIPT_INTEGRITY_FAILED');
    result.push(receipt);
  }
  return result;
}
