/**
 * PRE-ADMISSION ONLY: user-downloadable ASML official Q3 2026 financial PDF.
 * The browser transports supplied bytes; original host authenticity is not
 * established by a browser upload. Server verifies structure and storage hash.
 * Never writes Registry, never claims source-date byte certification.
 */
import {createHash} from 'node:crypto';
import type {SupabaseClient} from '@supabase/supabase-js';
import {ASML_Q3_PRIVATE_BUCKET,ASML_Q3_ARCHIVE_ROOT}
 from './asml-q3-private-source-archive-v1';

export const ASML_Q3_US_GAAP_PDF=Object.freeze({
  id:'ASML_Q3_US_GAAP_2026_PDF',
  name:'ASML Financial Statements US GAAP — Q3 2026',
  officialLink:'https://www.investor.asml.com/static-files/d310262b-d037-485d-83b5-92c93cc462f0',
  published:'2026-10-14',cutoff:'2026-10-30',
});
const MAX=24_000_000;
const sha=(b:Buffer)=>createHash('sha256').update(b).digest('hex');

export type UserSuppliedQ3PdfReceipt=Readonly<{
 schema:'ASML_Q3_BROWSER_SUPPLIED_PDF_RECEIPT_V1';
 documentId:typeof ASML_Q3_US_GAAP_PDF.id;
 sourceLink:string;
 publicationDate:'2026-10-14';
 cutoff:'2026-10-30';
 uploadedAt:string;
 sha256:string;byteSize:number;
 mediaType:'application/pdf';
 sourcePath:string;receiptPath:string;
 storageReadbackVerified:true;
 sourceOriginAuthenticated:false;
 sourceDocumentAuthenticityVerified:false;
 includesAllReferencedAssets:'SELF_CONTAINED_PDF_NOT_INDEPENDENTLY_AUDITED';
 registryAdmitted:false;
}>;

function duplicate(e:unknown):boolean{
 if(!e||typeof e!=='object')return false;
 const x=e as Record<string,unknown>;
 return ['400','409'].includes(String(x.statusCode)) &&
  (['KeyAlreadyExists','ResourceAlreadyExists','already_exists'].includes(String(x.code??x.error??'')) ||
    /^(?:The resource already exists|Asset Already Exists|Duplicate)$/i.test(String(x.message??'').trim()));
}
async function putImmutable(client:SupabaseClient,path:string,bytes:Buffer,mime:'application/pdf'|'application/json') {
 const bucket=client.storage.from(ASML_Q3_PRIVATE_BUCKET);
 const {error}=await bucket.upload(path,bytes,{contentType:mime,upsert:false,cacheControl:'0'});
 if(error&&!duplicate(error))throw Error('Q3_BROWSER_UPLOAD_STORAGE_FAILURE');
 const {data,error:readError}=await bucket.download(path);
 if(readError||!data)throw Error('Q3_BROWSER_UPLOAD_READBACK_FAILURE');
 const stored=Buffer.from(await data.arrayBuffer());
 if(stored.length!==bytes.length||sha(stored)!==sha(bytes)) {
  if(mime==='application/json'&&error&&duplicate(error)){
   // On retries the receipt keeps its original uploadedAt; verify all other
   // signed-by-server assertions rather than replace an immutable receipt.
   try{
    const old=JSON.parse(stored.toString('utf8')) as UserSuppliedQ3PdfReceipt;
    const next=JSON.parse(bytes.toString('utf8')) as UserSuppliedQ3PdfReceipt;
    if(old.schema===next.schema&&old.documentId===next.documentId&&
      old.sha256===next.sha256&&old.byteSize===next.byteSize&&
      old.sourcePath===next.sourcePath&&old.receiptPath===next.receiptPath&&
      old.sourceLink===next.sourceLink&&old.publicationDate===next.publicationDate&&
      old.cutoff===next.cutoff&&old.storageReadbackVerified===true&&
      old.sourceOriginAuthenticated===false&&old.sourceDocumentAuthenticityVerified===false&&
      old.registryAdmitted===false&&Number.isFinite(Date.parse(old.uploadedAt)))return;
   }catch{}
  }
  throw Error('Q3_BROWSER_UPLOAD_READBACK_MISMATCH');
 }
}
export async function archiveBrowserProvidedAsmlQ3Pdf(
 client:SupabaseClient,content:Buffer,clock:()=>Date=()=>new Date()
):Promise<UserSuppliedQ3PdfReceipt>{
 if(!Buffer.isBuffer(content)||content.length<10_000||content.length>MAX||
    content.toString('latin1',0,5)!=='%PDF-'||
    !content.toString('latin1',Math.max(0,content.length-4096)).includes('%%EOF'))
  throw Error('Q3_BROWSER_UPLOAD_INVALID_PDF');
 const {data:bucket,error}=await client.storage.getBucket(ASML_Q3_PRIVATE_BUCKET);
 if(error||!bucket||bucket.id!==ASML_Q3_PRIVATE_BUCKET||bucket.public!==false)
  throw Error('Q3_BROWSER_UPLOAD_PRIVATE_BUCKET_REQUIRED');
 const hash=sha(content);
 const sourcePath=ASML_Q3_ARCHIVE_ROOT+'/browser-supplied/'+hash+'.pdf';
 const receiptPath=ASML_Q3_ARCHIVE_ROOT+'/browser-receipts/'+hash+'.json';
 await putImmutable(client,sourcePath,content,'application/pdf');
 const receipt:UserSuppliedQ3PdfReceipt={
  schema:'ASML_Q3_BROWSER_SUPPLIED_PDF_RECEIPT_V1',
  documentId:ASML_Q3_US_GAAP_PDF.id,
  sourceLink:ASML_Q3_US_GAAP_PDF.officialLink,
  publicationDate:'2026-10-14',cutoff:'2026-10-30',
  uploadedAt:clock().toISOString(),sha256:hash,byteSize:content.length,
  mediaType:'application/pdf',sourcePath,receiptPath,
  storageReadbackVerified:true,
  sourceOriginAuthenticated:false,
  sourceDocumentAuthenticityVerified:false,
  includesAllReferencedAssets:'SELF_CONTAINED_PDF_NOT_INDEPENDENTLY_AUDITED',
  registryAdmitted:false,
 };
 await putImmutable(client,receiptPath,
  Buffer.from(JSON.stringify(receipt)+'\n','utf8'),'application/json');
 return receipt;
}
