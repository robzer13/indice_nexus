/**
 * Strict read-only official ASML Q3 2026 SEC exhibit acquisition.
 * Candidate bytes only: not Registry evidence, no certified valuation inputs.
 * Frozen data cutoff: 2026-10-30. No user URLs, redirects or credentials.
 */
import {createHash} from 'node:crypto';

export const ASML_Q3_CUTOFF='2026-10-30';
export const ASML_Q3_FILING_DATE='2026-10-14';
const SEC_ROOT='https://www.sec.gov/Archives/edgar/data/937966/000162828026065813/';
export const ASML_Q3_SEC_SOURCES=Object.freeze({
  'SEC_EX_99_1':Object.freeze({
    url:SEC_ROOT+'pressreleasefinancialresul.htm',
    documentId:'SEC_EX_99_1',title:'ASML 2026 Q3 press release',
    publicationDate:ASML_Q3_FILING_DATE,
  }),
  'SEC_EX_99_2':Object.freeze({
    url:SEC_ROOT+'a2026_10x14xpresentation.htm',
    documentId:'SEC_EX_99_2',title:'ASML 2026 Q3 investor presentation',
    publicationDate:ASML_Q3_FILING_DATE,
  }),
  'SEC_EX_99_3':Object.freeze({
    url:SEC_ROOT+'financial_statementsxusx.htm',
    documentId:'SEC_EX_99_3',title:'ASML 2026 Q3 US GAAP financial statement wrapper',
    publicationDate:ASML_Q3_FILING_DATE,
  }),
});
export type AsmlQ3SourceId=keyof typeof ASML_Q3_SEC_SOURCES;
export const ASML_Q3_SOURCE_IDS=Object.keys(ASML_Q3_SEC_SOURCES) as AsmlQ3SourceId[];
export const ASML_Q3_MAX_BYTES=2_000_000;
export type AsmlQ3Candidate=Readonly<{
  documentId:AsmlQ3SourceId;url:string;publicationDate:string;cutoff:string;
  retrievedAt:string;sha256:string;byteSize:number;contentType:'text/html';
  admitted:false;historicalBytesCertified:false;includesLinkedMedia:false;
}>;
const DATE=/^\d{4}-\d{2}-\d{2}$/;

export async function retrieveAsmlQ3SecExhibit(
  id:AsmlQ3SourceId,fetcher:typeof fetch=fetch,
  cutoff:string=ASML_Q3_CUTOFF,now:()=>Date=()=>new Date(),
):Promise<Readonly<{candidate:AsmlQ3Candidate;bytes:Buffer}>> {
  if(!Object.hasOwn(ASML_Q3_SEC_SOURCES,id))throw Error('ASML_Q3_SOURCE_NOT_ALLOWLISTED');
  const source=ASML_Q3_SEC_SOURCES[id];
  if(!DATE.test(cutoff) || cutoff!==ASML_Q3_CUTOFF ||
     source.publicationDate>cutoff)throw Error('ASML_Q3_TEMPORAL_GATE');
  const response=await fetcher(source.url,{
    method:'GET',redirect:'error',credentials:'omit',cache:'no-store',
    headers:{accept:'text/html'},signal:AbortSignal.timeout(9_000),
  });
  if(response.status!==200 || response.redirected || response.url!==source.url)
    throw Error('ASML_Q3_ORIGIN_OR_STATUS_UNVERIFIED');
  const mime=(response.headers.get('content-type')??'').toLowerCase();
  if(!/^text\/html(?:\s*;|\s*$)/.test(mime))
    throw Error('ASML_Q3_CONTENT_TYPE_UNVERIFIED');
  const lengthHeader=response.headers.get('content-length');
  if(lengthHeader!==null) {
    const length=Number(lengthHeader);
    if(!Number.isSafeInteger(length) || length<0 || length>ASML_Q3_MAX_BYTES)
      throw Error('ASML_Q3_CONTENT_LENGTH_INVALID');
  }
  if(!response.body)throw Error('ASML_Q3_EMPTY_BODY');
  const chunks:Buffer[]=[];let size=0;
  const reader=response.body.getReader();
  try{
    while(true){
      const {value,done}=await reader.read();if(done)break;
      if(value){
        size+=value.byteLength;
        if(size>ASML_Q3_MAX_BYTES)throw Error('ASML_Q3_TOO_LARGE');
        chunks.push(Buffer.from(value));
      }
    }
  } finally {reader.releaseLock();}
  if(size<1_000)throw Error('ASML_Q3_TRUNCATED');
  const bytes=Buffer.concat(chunks,size);
  const body=new TextDecoder('utf-8',{fatal:true}).decode(bytes)
    .replace(/<[^>]*>/g,' ').replace(/&(?:nbsp|#160);/gi,' ')
    .replace(/\s+/g,' ');
  if(!/ASML/i.test(body) || !/2026/.test(body) ||
     !(id==='SEC_EX_99_1'
       ? /11\.7 billion|third.quarter|Q3 2026/i.test(body)
       : id==='SEC_EX_99_2'
       ? /investor|presentation|third.quarter/i.test(body)
       : /financial|consolidated|US GAAP/i.test(body)))
    throw Error('ASML_Q3_DOCUMENT_IDENTITY_UNVERIFIED');
  return {candidate:{
    documentId:id,url:source.url,publicationDate:source.publicationDate,
    cutoff,retrievedAt:now().toISOString(),byteSize:size,
    sha256:createHash('sha256').update(bytes).digest('hex'),
    contentType:'text/html',admitted:false,historicalBytesCertified:false,
    includesLinkedMedia:false,
  },bytes};
}
