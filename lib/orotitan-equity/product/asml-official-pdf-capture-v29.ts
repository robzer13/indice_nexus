/**
 * OroTitan ASML v2.9 — SHADOW-only, non-authoritative, server-side ORIGINAL PDF capture.
 * Mandatory caller gates: isHostedShadowResearchEnabled(env) + requireAdminSession().
 * This module neither changes the Registry nor authenticates the publisher cryptographically.
 * A successful HTTPS download/readback proves byte identity and records its exact origin URL.
 */
import {createHash} from 'node:crypto';
import type {SupabaseClient} from '@supabase/supabase-js';

export const ASML_V29_BUCKET='orotitan-source-files-v1';
export const ASML_V29_ROOT='research-acquisition-v1/asml/asof-2026-10-30/official-pdf-v29';
export const ASML_V29_RUN='a1e59126-1ca9-4e55-8764-54a1e3d4dc24';
export const ASML_V29_CUTOFF='2026-10-30';
export const ASML_V29_PRIOR_Q3_SHADOW_SHA='562ba8423b4cdb74f48c0d033f9c1571c314f525f252cffce245ad532b03e2a9';

export const ASML_V29_SOURCES = Object.freeze({
  ASML_2025_US_GAAP: Object.freeze({
    id:'ASML_2025_US_GAAP',
    title:'ASML Annual Report 2025 based on US GAAP',
    url:'https://ourbrand.asml.com/m/71076aaad607de4d/original/asml-2025-annual-report-based-on-us-gaap.pdf',
    originHost:'ourbrand.asml.com',
    redirectPathPrefix:'/m/71076aaad607de4d/original/',
    publicationDate:'2026-02-25',
    accountingBasis:'US_GAAP',
    maxBytes:40_000_000,
  }),
  ASML_2026_Q3_US_GAAP: Object.freeze({
    id:'ASML_2026_Q3_US_GAAP',
    title:'ASML Financial statements US GAAP Q3 2026',
    url:'https://investor.asml.com/static-files/d310262b-d037-485d-83b5-92c93cc462f0',
    originHost:'investor.asml.com',
    redirectPathPrefix:'/static-files/',
    publicationDate:'2026-10-14',
    accountingBasis:'US_GAAP_UNAUDITED',
    maxBytes:24_000_000,
  }),
  TSMC_2026_Q3_CALL: Object.freeze({
    id:'TSMC_2026_Q3_CALL',
    title:'TSMC Q3 2026 earnings call, LSEG-edited transcript at issuer IR',
    url:'https://investor.tsmc.com/english/encrypt/files/encrypt_file/reports/2026-10/eba4fac72873bf546b669bd2b5d973df5c93215d/TSMC%203Q26%20Transcript.pdf',
    originHost:'investor.tsmc.com',
    redirectPathPrefix:'/english/encrypt/files/encrypt_file/reports/2026-10/',
    publicationDate:'2026-10-15',
    accountingBasis:'TSMC_NON_FINANCIAL_EARNINGS_CALL',
    maxBytes:15_000_000,
  }),
});

export type AsmlV29SourceId=keyof typeof ASML_V29_SOURCES;
export type AsmlV29Receipt=Readonly<{
  schema:'ASML_OFFICIAL_PDF_CAPTURE_RECEIPT_V2_9';
  runId:typeof ASML_V29_RUN;
  cutoff:typeof ASML_V29_CUTOFF;
  sourceId:AsmlV29SourceId;
  sourceTitle:string;
  expectedSourceUrl:string;
  finalSourceUrl:string;
  redirectChain:readonly string[];
  sourcePublicationDate:string;
  accountingBasis:string;
  retrievalMode:'ALLOWLISTED_SERVER_TLS_FETCH';
  receivedAt:string;
  sha256:string;
  byteSize:number;
  mediaType:'application/pdf';
  bucket:typeof ASML_V29_BUCKET;
  sourcePath:string;
  receiptPath:string;
  storageReadbackVerified:true;
  transportHttpsVerifiedByRuntime:true;
  sourceOriginCryptographicallyAttested:false;
  documentAuthenticityIndependentlyVerified:false;
  historicalBytesAtCutoffIndependentlyVerified:false;
  q3PreviousShadowHashMatch:boolean|null;
  registryAdmitted:false;
  stageAdvanced:false;
}>;

export function selectSource(id:string) {
  if(!Object.prototype.hasOwnProperty.call(ASML_V29_SOURCES,id))
    throw Error('ASML_V29_UNKNOWN_SOURCE');
  return ASML_V29_SOURCES[id as AsmlV29SourceId];
}

export function validateSameOriginRedirect(source:ReturnType<typeof selectSource>,url:string){
  let target:URL;
  try { target=new URL(url); }catch { throw Error('ASML_V29_BAD_REDIRECT'); }
  if(target.protocol!=='https:'||target.hostname!==source.originHost||
     target.port!==''||target.username!==''||target.password!==''||
     !target.pathname.startsWith(source.redirectPathPrefix)||
     target.search!==''||target.hash!=='')
    throw Error('ASML_V29_REDIRECT_OUTSIDE_ALLOWLIST');
  return target.href;
}

export function checkPdfBytes(bytes:Buffer,maxBytes:number):void {
  if(!Buffer.isBuffer(bytes)||bytes.length<1000||bytes.length>maxBytes||
     bytes.toString('latin1',0,5)!=='%PDF-'||
     !bytes.toString('latin1',Math.max(0,bytes.length-4096)).includes('%%EOF'))
    throw Error('ASML_V29_INVALID_PDF');
}

const hash=(b:Buffer)=>createHash('sha256').update(b).digest('hex');
const dup=(e:unknown):boolean => {
  if(!e||typeof e!=='object')return false;
  const x=e as Record<string,unknown>;
  return ['400','409'].includes(String(x.statusCode)) &&
    (['KeyAlreadyExists','ResourceAlreadyExists','already_exists'].includes(String(x.code??x.error??''))||
     /^(?:The resource already exists|Asset Already Exists|Duplicate)$/i.test(String(x.message??'').trim()));
};

async function readBoundedBody(res:Response,maxBytes:number):Promise<Buffer> {
  const header=res.headers.get('content-length');
  if(header!==null){
    const len=Number(header);
    if(!Number.isSafeInteger(len)||len<0||len>maxBytes)
      throw Error('ASML_V29_CONTENT_LENGTH_INVALID');
  }
  if(!res.body)throw Error('ASML_V29_EMPTY_RESPONSE_BODY');
  const chunks:Buffer[]=[];
  let total=0;
  const reader=res.body.getReader();
  try{
    for(;;){
      const {value,done}=await reader.read();
      if(done)break;
      if(value){
        total+=value.byteLength;
        if(total>maxBytes)throw Error('ASML_V29_BODY_EXCEEDS_LIMIT');
        chunks.push(Buffer.from(value));
      }
    }
  } finally {await reader.cancel().catch(()=>undefined);reader.releaseLock();}
  return Buffer.concat(chunks,total);
}

export async function acquireOfficialPdf(
 sourceId:AsmlV29SourceId,
 fetchImpl:typeof fetch=fetch,
):Promise<{bytes:Buffer;finalUrl:string;redirectChain:string[]}> {
  const source=selectSource(sourceId);
  if(source.publicationDate>ASML_V29_CUTOFF)throw Error('ASML_V29_POST_CUTOFF_SOURCE');
  let url=validateSameOriginRedirect(source,source.url);
  const redirectChain:string[]=[];
  // Redirects are permitted only within the EXACT issuer hostname and prefix.
  for(let hop=0;hop<3;hop++){
    const res=await fetchImpl(url,{
      method:'GET',redirect:'manual',cache:'no-store',
      headers:{accept:'application/pdf'},signal:AbortSignal.timeout(45_000),
    });
    if([301,302,303,307,308].includes(res.status)){
      const loc=res.headers.get('location');
      if(!loc)throw Error('ASML_V29_REDIRECT_WITHOUT_LOCATION');
      if(hop===2)throw Error('ASML_V29_TOO_MANY_REDIRECTS');
      url=validateSameOriginRedirect(source,new URL(loc,url).href);
      redirectChain.push(url);
      continue;
    }
    if(!res.ok)throw Error('ASML_V29_ISSUER_FETCH_HTTP_'+res.status);
    const mime=(res.headers.get('content-type')??'').split(';')[0].trim().toLowerCase();
    if(mime!==''&&!['application/pdf','application/octet-stream','binary/octet-stream'].includes(mime))
      throw Error('ASML_V29_UNEXPECTED_MIME');
    const bytes=await readBoundedBody(res,source.maxBytes);
    checkPdfBytes(bytes,source.maxBytes);
    return {bytes,finalUrl:url,redirectChain};
  }
  throw Error('ASML_V29_UNREACHABLE');
}

async function writeOnceAndReadback(client:SupabaseClient,path:string,bytes:Buffer,mime:'application/pdf'|'application/json',receipt?:AsmlV29Receipt) {
  const storage=client.storage.from(ASML_V29_BUCKET);
  const {error}=await storage.upload(path,bytes,{contentType:mime,upsert:false,cacheControl:'0'});
  if(error&&!dup(error))throw Error('ASML_V29_PRIVATE_STORAGE_WRITE_FAILED');
  const {data,error:downloadError}=await storage.download(path);
  if(downloadError||!data)throw Error('ASML_V29_PRIVATE_READBACK_FAILED');
  const back=Buffer.from(await data.arrayBuffer());
  if(back.length===bytes.length&&hash(back)===hash(bytes))return;
  if(receipt&&error&&dup(error)){
    try{
      const original=JSON.parse(back.toString('utf8')) as AsmlV29Receipt;
      const equal=original.schema===receipt.schema&&original.runId===receipt.runId&&
        original.cutoff===receipt.cutoff&&original.sourceId===receipt.sourceId&&
        original.expectedSourceUrl===receipt.expectedSourceUrl&&
        original.finalSourceUrl===receipt.finalSourceUrl&&
        JSON.stringify(original.redirectChain)===JSON.stringify(receipt.redirectChain)&&
        original.sha256===receipt.sha256&&original.byteSize===receipt.byteSize&&
        original.bucket===receipt.bucket&&original.sourcePath===receipt.sourcePath&&
        original.receiptPath===receipt.receiptPath&&original.registryAdmitted===false&&
        original.stageAdvanced===false&&original.storageReadbackVerified===true&&
        original.historicalBytesAtCutoffIndependentlyVerified===false&&
        original.q3PreviousShadowHashMatch===receipt.q3PreviousShadowHashMatch&&
        original.sourceOriginCryptographicallyAttested===false&&
        original.documentAuthenticityIndependentlyVerified===false&&
        Number.isFinite(Date.parse(original.receivedAt));
      if(equal)return;
    }catch{}
  }
  throw Error('ASML_V29_PRIVATE_READBACK_MISMATCH');
}

export async function captureAsmlOfficialPdf(
  client:SupabaseClient,sourceId:AsmlV29SourceId,
  fetchImpl:typeof fetch=fetch,now:()=>Date=()=>new Date(),
):Promise<AsmlV29Receipt>{
  const source=selectSource(sourceId);
  const {data:bucket,error}=await client.storage.getBucket(ASML_V29_BUCKET);
  if(error||!bucket||bucket.id!==ASML_V29_BUCKET||bucket.public!==false)
    throw Error('ASML_V29_PRIVATE_BUCKET_REQUIRED');
  const captured=await acquireOfficialPdf(sourceId,fetchImpl);
  const sha=hash(captured.bytes);
  const sourcePath=`${ASML_V29_ROOT}/${sourceId}/${sha}.pdf`;
  const receiptPath=`${ASML_V29_ROOT}/receipts/${sourceId}-${sha}.json`;
  await writeOnceAndReadback(client,sourcePath,captured.bytes,'application/pdf');
  const receipt:AsmlV29Receipt={
    schema:'ASML_OFFICIAL_PDF_CAPTURE_RECEIPT_V2_9',
    runId:ASML_V29_RUN,cutoff:ASML_V29_CUTOFF,sourceId,
    sourceTitle:source.title,expectedSourceUrl:source.url,
    finalSourceUrl:captured.finalUrl,redirectChain:captured.redirectChain,
    sourcePublicationDate:source.publicationDate,accountingBasis:source.accountingBasis,
    retrievalMode:'ALLOWLISTED_SERVER_TLS_FETCH',receivedAt:now().toISOString(),
    sha256:sha,byteSize:captured.bytes.length,mediaType:'application/pdf',
    bucket:ASML_V29_BUCKET,sourcePath,receiptPath,
    storageReadbackVerified:true,transportHttpsVerifiedByRuntime:true,
    sourceOriginCryptographicallyAttested:false,
    documentAuthenticityIndependentlyVerified:false,
    historicalBytesAtCutoffIndependentlyVerified:false,
    q3PreviousShadowHashMatch:sourceId==='ASML_2026_Q3_US_GAAP'?
      sha===ASML_V29_PRIOR_Q3_SHADOW_SHA:null,
    registryAdmitted:false,stageAdvanced:false,
  };
  await writeOnceAndReadback(client,receiptPath,Buffer.from(JSON.stringify(receipt)+'\n'),'application/json',receipt);
  return receipt;
}
