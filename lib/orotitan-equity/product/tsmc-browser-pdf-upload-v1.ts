/**
 * OroTitan — manual browser-supplied TSMC Q3 2026 transcript.
 * SHADOW pre-admission ONLY: immutable private bytes, read-back SHA-256.
 * User-supplied files do NOT authenticate publisher origin or historical bytes.
 * NO Registry mutation, run stage transition, or production storage.
 */
import {createHash} from 'node:crypto';
import type {SupabaseClient} from '@supabase/supabase-js';
import {ASML_V29_BUCKET,ASML_V29_ROOT,ASML_V29_CUTOFF,ASML_V29_RUN,
 ASML_V29_SOURCES} from './asml-official-pdf-capture-v29';

export const TSMC_BROWSER_SOURCE=ASML_V29_SOURCES.TSMC_2026_Q3_CALL;
export const TSMC_BROWSER_ROOT=ASML_V29_ROOT+'/tsmc-browser-supplied-v1';
export const TSMC_BROWSER_MAX_BYTES=15_000_000;
const sha256=(bytes:Buffer)=>createHash('sha256').update(bytes).digest('hex');

export type TsmcBrowserReceipt=Readonly<{
 schema:'OROTITAN_TSMC_Q3_BROWSER_PDF_RECEIPT_V1';
 runId:typeof ASML_V29_RUN;
 documentId:'TSMC_2026_Q3_CALL';
 sourceLink:string;
 publicationDate:'2026-10-15';
 cutoff:typeof ASML_V29_CUTOFF;
 uploadedAt:string;
 sha256:string;
 byteSize:number;
 mediaType:'application/pdf';
 bucket:typeof ASML_V29_BUCKET;
 sourcePath:string;
 receiptPath:string;
 storageReadbackVerified:true;
 sourceOriginAuthenticated:false;
 sourceDocumentAuthenticityVerified:false;
 historicalBytesAtCutoffIndependentlyVerified:false;
 registryAdmitted:false;
 stageAdvanced:false;
}>;

function duplicated(e:unknown):boolean{
 if(!e||typeof e!=='object')return false;
 const x=e as Record<string,unknown>;
 return ['400','409'].includes(String(x.statusCode))&&
  (['KeyAlreadyExists','ResourceAlreadyExists','already_exists'].includes(String(x.code??x.error??''))||
   /^(?:The resource already exists|Asset Already Exists|Duplicate)$/i.test(String(x.message??'').trim()));
}
function checkPdf(data:Buffer):void{
 if(!Buffer.isBuffer(data)||data.length<10_000||data.length>TSMC_BROWSER_MAX_BYTES||
    data.toString('latin1',0,5)!=='%PDF-'||
    !data.toString('latin1',Math.max(0,data.length-4096)).includes('%%EOF'))
   throw Error('TSMC_BROWSER_INVALID_PDF');
}
async function storeAndVerify(client:SupabaseClient,path:string,bytes:Buffer,
 mime:'application/pdf'|'application/json',expected?:TsmcBrowserReceipt):Promise<Buffer>{
 const bucket=client.storage.from(ASML_V29_BUCKET);
 const {error}=await bucket.upload(path,bytes,{contentType:mime,upsert:false,cacheControl:'0'});
 if(error&&!duplicated(error))throw Error('TSMC_BROWSER_STORAGE_WRITE_FAILED');
 const {data,error:readError}=await bucket.download(path);
 if(readError||!data)throw Error('TSMC_BROWSER_STORAGE_READBACK_FAILED');
 const stored=Buffer.from(await data.arrayBuffer());
 if(stored.length===bytes.length&&sha256(stored)===sha256(bytes))return stored;
 if(error&&duplicated(error)&&expected&&mime==='application/json'){
   try{
    const previous=JSON.parse(stored.toString('utf8')) as TsmcBrowserReceipt;
    if(previous.schema===expected.schema&&
       previous.runId===expected.runId&&
       previous.documentId===expected.documentId&&
       previous.sourceLink===expected.sourceLink&&
       previous.publicationDate===expected.publicationDate&&
       previous.cutoff===expected.cutoff&&
       previous.sha256===expected.sha256&&
       previous.byteSize===expected.byteSize&&
       previous.sourcePath===expected.sourcePath&&
       previous.receiptPath===expected.receiptPath&&
       previous.bucket===expected.bucket&&
       previous.mediaType==='application/pdf'&&
       previous.storageReadbackVerified===true&&
       previous.sourceOriginAuthenticated===false&&
       previous.sourceDocumentAuthenticityVerified===false&&
       previous.historicalBytesAtCutoffIndependentlyVerified===false&&
       previous.registryAdmitted===false&&previous.stageAdvanced===false&&
       Number.isFinite(Date.parse(previous.uploadedAt)))return stored;
   }catch{/* Refuse corrupt or mismatching immutable receipt. */}
 }
 throw Error('TSMC_BROWSER_STORAGE_READBACK_MISMATCH');
}
export async function archiveBrowserProvidedTsmcQ3(
 client:SupabaseClient,bytes:Buffer,clock:()=>Date=()=>new Date()
):Promise<TsmcBrowserReceipt>{
 checkPdf(bytes);
 if(TSMC_BROWSER_SOURCE.publicationDate>ASML_V29_CUTOFF)
   throw Error('TSMC_BROWSER_POST_CUTOFF');
 const {data:bucket,error}=await client.storage.getBucket(ASML_V29_BUCKET);
 if(error||!bucket||bucket.id!==ASML_V29_BUCKET||bucket.public!==false)
   throw Error('TSMC_BROWSER_PRIVATE_BUCKET_REQUIRED');
 const digest=sha256(bytes);
 const sourcePath=TSMC_BROWSER_ROOT+'/pdf/'+digest+'.pdf';
 const receiptPath=TSMC_BROWSER_ROOT+'/receipts/'+digest+'.json';
 await storeAndVerify(client,sourcePath,bytes,'application/pdf');
 const receipt:TsmcBrowserReceipt={
   schema:'OROTITAN_TSMC_Q3_BROWSER_PDF_RECEIPT_V1',
   runId:ASML_V29_RUN,documentId:'TSMC_2026_Q3_CALL',
   sourceLink:TSMC_BROWSER_SOURCE.url,
   publicationDate:'2026-10-15',cutoff:ASML_V29_CUTOFF,
   uploadedAt:clock().toISOString(),sha256:digest,byteSize:bytes.length,
   mediaType:'application/pdf',bucket:ASML_V29_BUCKET,sourcePath,receiptPath,
   storageReadbackVerified:true,
   sourceOriginAuthenticated:false,sourceDocumentAuthenticityVerified:false,
   historicalBytesAtCutoffIndependentlyVerified:false,
   registryAdmitted:false,stageAdvanced:false,
 };
 const saved=await storeAndVerify(client,receiptPath,
   Buffer.from(JSON.stringify(receipt)+'\n','utf8'),'application/json',receipt);
 const previous=JSON.parse(saved.toString('utf8')) as TsmcBrowserReceipt;
 // Semantic receipt check is also required on the non-duplicate path.
 if(previous.schema!==receipt.schema||previous.sha256!==digest||
    previous.sourcePath!==sourcePath||previous.receiptPath!==receiptPath||
    previous.documentId!=='TSMC_2026_Q3_CALL'||
    previous.sourceOriginAuthenticated!==false||previous.registryAdmitted!==false)
  throw Error('TSMC_BROWSER_RECEIPT_INVALID');
 return previous;
}
