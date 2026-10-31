/**
 * OroTitan ASML Research v3.4 — independent read-only SHADOW storage rehash.
 *
 * This auditor is separate from upload/capture implementations.
 * It obtains ORIGINAL stored PDF bytes via private Supabase storage and
 * independently checks SHA-256, size, receipt fields and PDF magic/EOF.
 * It does NOT authenticate issuer origin, data-cutoff byte identity, or claims.
 * No writes to SHADOW or production and NO Registry access or admission.
 */
import {createHash} from 'node:crypto';
import type {SupabaseClient} from '@supabase/supabase-js';

export const V34_RUN_ID='a1e59126-1ca9-4e55-8764-54a1e3d4dc24';
export const V34_CUTOFF='2026-10-30';
export const V34_BUCKET='orotitan-source-files-v1';
const ROOT='research-acquisition-v1/asml/asof-2026-10-30';
export const V34_SOURCES=Object.freeze({
 ASML_2025_US_GAAP: {
  display:'ASML FY2025 annual report (US GAAP)',
  sha256:'add58be9d9822ca12584b63c55c089b36e1ae5bf7a1825ebf1d1572628f2c52c',
  size:23928741,sourceDate:'2026-02-25',
  sourcePath:ROOT+'/official-pdf-v29/ASML_2025_US_GAAP/add58be9d9822ca12584b63c55c089b36e1ae5bf7a1825ebf1d1572628f2c52c.pdf',
  receiptPath:ROOT+'/official-pdf-v29/receipts/ASML_2025_US_GAAP-add58be9d9822ca12584b63c55c089b36e1ae5bf7a1825ebf1d1572628f2c52c.json',
  receiptSchema:'ASML_OFFICIAL_PDF_CAPTURE_RECEIPT_V2_9',
  documentId:'ASML_2025_US_GAAP',
  sourceOriginAuthenticated:false,
  evidenceClaims:['C-001','C-002','C-003','C-005','C-008','C-009','C-010','C-011','C-012','C-013','C-014','C-015','C-016','C-017','C-018'],
 },
 ASML_2026_Q3_US_GAAP: {
  display:'ASML Q3 2026 statements (US GAAP; unaudited)',
  sha256:'562ba8423b4cdb74f48c0d033f9c1571c314f525f252cffce245ad532b03e2a9',
  size:1369295,sourceDate:'2026-10-14',
  sourcePath:ROOT+'/q3-2026/browser-supplied/562ba8423b4cdb74f48c0d033f9c1571c314f525f252cffce245ad532b03e2a9.pdf',
  receiptPath:ROOT+'/q3-2026/browser-receipts/562ba8423b4cdb74f48c0d033f9c1571c314f525f252cffce245ad532b03e2a9.json',
  receiptSchema:'ASML_Q3_BROWSER_SUPPLIED_PDF_RECEIPT_V1',
  documentId:'ASML_Q3_US_GAAP_2026_PDF',
  sourceOriginAuthenticated:false,
  evidenceClaims:['C-006','C-021'],
 },
 TSMC_2026_Q3_CALL: {
  display:'TSMC Q3 2026 edited call transcript',
  sha256:'bb1d3aeadac9f44189d9d36aa410b0c1c458f9f8b3d8aa63ecc71e6813c5f030',
  size:261306,sourceDate:'2026-10-15',
  sourcePath:ROOT+'/official-pdf-v29/tsmc-browser-supplied-v1/pdf/bb1d3aeadac9f44189d9d36aa410b0c1c458f9f8b3d8aa63ecc71e6813c5f030.pdf',
  receiptPath:ROOT+'/official-pdf-v29/tsmc-browser-supplied-v1/receipts/bb1d3aeadac9f44189d9d36aa410b0c1c458f9f8b3d8aa63ecc71e6813c5f030.json',
  receiptSchema:'OROTITAN_TSMC_Q3_BROWSER_PDF_RECEIPT_V1',
  documentId:'TSMC_2026_Q3_CALL',
  sourceOriginAuthenticated:false,
  evidenceClaims:['C-022','C-023'],
 },
} as const);

export type V34SourceId=keyof typeof V34_SOURCES;
export type V34ReceiptSummary=Readonly<{
 id:V34SourceId;schema:string;sourceDate:string;captureTimestamp:string;
 sourceOriginalHash:string;receiptHash:string;byteSize:number;integrity:'STORED_BYTES_SHA256_MATCH';
 bucketPrivate:true;receiptStructureChecked:true;
 independentlyAuthenticatedPublisher:false;
 historicalBytesAtCutoffCertified:false;
 registryAdmitted:false;
 evidenceClaims:readonly string[];
}>;
export type V34NumericProbe=Readonly<{claimId:string;numeric:string;derived?:boolean;context?:string}>;
export const V34_PROBES:Readonly<Record<V34SourceId,readonly V34NumericProbe[]>>={
 ASML_2025_US_GAAP:[
  {claimId:'C-001',numeric:'32667.3'}, {claimId:'C-002',numeric:'8193.0'},
  {claimId:'C-003',numeric:'50.870255',derived:true},
  {claimId:'C-005',numeric:'46.5',context:'performance obligations'},
  {claimId:'C-008',numeric:'12658.5'}, {claimId:'C-009',numeric:'11027.3',derived:true},
  {claimId:'C-010',numeric:'6212.5'}, {claimId:'C-011',numeric:'3507.7'},
  {claimId:'C-012',numeric:'29.1',context:'China'},
  {claimId:'C-013',numeric:'9519.7'}, {claimId:'C-014',numeric:'61.2',context:'customers'},
  {claimId:'C-015',numeric:'24.9',context:'ZEISS'}, {claimId:'C-016',numeric:'4406.9'},
  {claimId:'C-017',numeric:'5950.0'}, {claimId:'C-018',numeric:'2550.3'},
 ],
 ASML_2026_Q3_US_GAAP:[
  {claimId:'C-006',numeric:'11662.0'},
  {claimId:'C-021',numeric:'9300.6'},
 ],
 TSMC_2026_Q3_CALL:[
  {claimId:'C-022',numeric:'60',context:'capital expenditures'},
  {claimId:'C-023',numeric:'64',context:'capital expenditures'},
 ],
};

const hash=(b:Buffer)=>createHash('sha256').update(b).digest('hex');
const isRecord=(v:unknown):v is Record<string,unknown>=>typeof v==='object'&&v!==null&&!Array.isArray(v);
export function assertValidSourceId(id:unknown):asserts id is V34SourceId{
 if(typeof id!=='string'||!Object.prototype.hasOwnProperty.call(V34_SOURCES,id))
  throw Error('V34_SOURCE_DENIED');
}
export function verifyExpectedPdfFingerprint(bytes:Buffer,expectedHash:string,expectedSize:number):string{
 if(bytes.length!==expectedSize)throw Error('V34_SIZE_MISMATCH');
 const sha=hash(bytes);
 if(sha!==expectedHash)throw Error('V34_SHA_MISMATCH');
 if(bytes.toString('latin1',0,5)!=='%PDF-'||
    !bytes.toString('latin1',Math.max(0,bytes.length-4096)).includes('%%EOF'))
  throw Error('V34_PDF_STRUCTURE_INVALID');
 return sha;
}
export function verifyStoredPdfBytes(id:V34SourceId,bytes:Buffer):string{
 const source=V34_SOURCES[id];
 return verifyExpectedPdfFingerprint(bytes,source.sha256,source.size);
}
export function validatePrivateReceipt(id:V34SourceId,body:Buffer):{receiptHash:string;uploadedAt:string} {
 const source=V34_SOURCES[id];
 if(body.length<100||body.length>8000)throw Error('V34_RECEIPT_INVALID');
 let v:unknown;try{v=JSON.parse(body.toString('utf8'));}catch{throw Error('V34_RECEIPT_INVALID');}
 if(!isRecord(v)||v.schema!==source.receiptSchema||
   v.sha256!==source.sha256||v.byteSize!==source.size||
   v.sourcePath!==source.sourcePath||v.receiptPath!==source.receiptPath||
   v.cutoff!==V34_CUTOFF||v.registryAdmitted!==false||
   v.storageReadbackVerified!==true)
  throw Error('V34_RECEIPT_MISMATCH');
 let timestamp:unknown;
 if(id==='ASML_2025_US_GAAP'){
  if(v.runId!==V34_RUN_ID||v.sourceId!==id||v.bucket!==V34_BUCKET||
     v.stageAdvanced!==false||v.sourceOriginCryptographicallyAttested!==false||
     v.documentAuthenticityIndependentlyVerified!==false||
     v.historicalBytesAtCutoffIndependentlyVerified!==false||
     v.sourcePublicationDate!==source.sourceDate)
   throw Error('V34_RECEIPT_MISMATCH');
  timestamp=v.receivedAt;
 }else if(id==='ASML_2026_Q3_US_GAAP'){
  if(v.documentId!==source.documentId||v.publicationDate!==source.sourceDate||
     v.sourceOriginAuthenticated!==false||v.sourceDocumentAuthenticityVerified!==false)
   throw Error('V34_RECEIPT_MISMATCH');
  timestamp=v.uploadedAt;
 }else{
  if(v.documentId!==id||v.runId!==V34_RUN_ID||v.bucket!==V34_BUCKET||
     v.publicationDate!==source.sourceDate||v.stageAdvanced!==false||
     v.sourceOriginAuthenticated!==false||v.sourceDocumentAuthenticityVerified!==false||
     v.historicalBytesAtCutoffIndependentlyVerified!==false)
   throw Error('V34_RECEIPT_MISMATCH');
  timestamp=v.uploadedAt;
 }
 if(typeof timestamp!=='string'||!Number.isFinite(Date.parse(timestamp)))
  throw Error('V34_RECEIPT_INVALID');
 return {receiptHash:hash(body),uploadedAt:timestamp};
}
async function retrieve(client:SupabaseClient,path:string,max:number):Promise<Buffer>{
 const {data,error}=await client.storage.from(V34_BUCKET).download(path);
 if(error||!data)throw Error('V34_PRIVATE_STORAGE_READ_FAILED');
 if(data.size>max)throw Error('V34_PRIVATE_STORAGE_SIZE_DENIED');
 return Buffer.from(await data.arrayBuffer());
}
export async function independentAuditOne(client:SupabaseClient,id:V34SourceId):
 Promise<{summary:V34ReceiptSummary;bytes:Buffer}> {
 const {data:bucket,error}=await client.storage.getBucket(V34_BUCKET);
 if(error||!bucket||bucket.id!==V34_BUCKET||bucket.public!==false)
  throw Error('V34_PRIVATE_BUCKET_REQUIRED');
 const source=V34_SOURCES[id];
 const bytes=await retrieve(client,source.sourcePath,40_000_000);
 const digest=verifyStoredPdfBytes(id,bytes);
 const receipt=await retrieve(client,source.receiptPath,8000);
 const verified=validatePrivateReceipt(id,receipt);
 return {
  summary:{
   id,schema:source.receiptSchema,sourceDate:source.sourceDate,
   captureTimestamp:verified.uploadedAt,
   sourceOriginalHash:digest,receiptHash:verified.receiptHash,
   byteSize:bytes.length,integrity:'STORED_BYTES_SHA256_MATCH',
   bucketPrivate:true,receiptStructureChecked:true,
   independentlyAuthenticatedPublisher:false,
   historicalBytesAtCutoffCertified:false,registryAdmitted:false,
   evidenceClaims:source.evidenceClaims,
  },
  bytes,
 };
}
export async function independentAuditAll(client:SupabaseClient):Promise<V34ReceiptSummary[]>{
 const ids=Object.keys(V34_SOURCES) as V34SourceId[];
 const result:V34ReceiptSummary[]=[];
 for(const id of ids)result.push((await independentAuditOne(client,id)).summary);
 return result;
}
export function numericCandidatesFromPage(id:V34SourceId,pageText:string):
 Array<{claimId:string;excerpt:string;match:'NUMERIC_TOKEN_ONLY'}> {
 const clean=pageText.replace(/\s+/g,' ');
 const noThousands=clean.replace(/(\d)[, ](?=\d{3}(?:\D|$))/g,'$1');
 const matches:Array<{claimId:string;excerpt:string;match:'NUMERIC_TOKEN_ONLY'}>=[];
 for(const probe of V34_PROBES[id]){
  if(probe.derived)continue;
  const needle=probe.numeric;
  const re=new RegExp('(?<!\\d)'+needle.replace(/[.*+?^${}()|[\]\\]/g,'\\$&')+'(?!\\d)');
  const found=re.exec(noThousands);
  if(!found)continue;
  // Context is informational, NOT a claim-fitness certification.
  const excerpt=noThousands.slice(Math.max(0,found.index-95),Math.min(noThousands.length,found.index+needle.length+150));
  matches.push({claimId:probe.claimId,excerpt:excerpt.slice(0,270),match:'NUMERIC_TOKEN_ONLY'});
 }
 return matches;
}
export type V34PageObservation=Readonly<{
 pdfPage:number;textLength:number;textSha256:string;
 candidateMatches:ReturnType<typeof numericCandidatesFromPage>;
}>;
export type V34PageScan=Readonly<{
 id:V34SourceId;pdfPageCount:number;startPage:number;endPage:number;
 textExtraction:'PDFJS_PAGE_TEXT_NONAUTHORITATIVE';
 pageObservations:V34PageObservation[];
 pagesNotScanned:number;
}>;
export async function extractBoundedPages(
 id:V34SourceId,bytes:Buffer,startPage:number,maxPages:number,
):Promise<V34PageScan>{
 if(!Number.isSafeInteger(startPage)||startPage<1||startPage>2000||
    !Number.isSafeInteger(maxPages)||maxPages<1||maxPages>20)
  throw Error('V34_PAGE_RANGE_DENIED');
 const {getDocumentProxy}=await import('unpdf');
 const pdf=await getDocumentProxy(new Uint8Array(bytes),{
   maxImageSize:8_388_608,disableFontFace:true,useSystemFonts:false,
 });
 try{
  if(pdf.numPages>2000)throw Error('V34_PDF_PAGE_COUNT_DENIED');
  if(startPage>pdf.numPages)throw Error('V34_PAGE_RANGE_DENIED');
  const endPage=Math.min(pdf.numPages,startPage+maxPages-1);
  const pages:V34PageObservation[]=[];
  for(let index=startPage;index<=endPage;index++){
   const page=await pdf.getPage(index);
   try{
    const text=await page.getTextContent();
    const raw=text.items.map(i=>('str' in i&&typeof i.str==='string')?i.str:'').join(' ');
    const pageText=raw.slice(0,160_000);
    pages.push({pdfPage:index,textLength:raw.length,
      textSha256:hash(Buffer.from(raw,'utf8')),
      candidateMatches:numericCandidatesFromPage(id,pageText).slice(0,12)});
   }finally{page.cleanup();}
  }
  return {id,pdfPageCount:pdf.numPages,startPage,endPage,
    textExtraction:'PDFJS_PAGE_TEXT_NONAUTHORITATIVE',
    pageObservations:pages,pagesNotScanned:pdf.numPages-pages.length};
 }finally{await pdf.destroy();}
}
