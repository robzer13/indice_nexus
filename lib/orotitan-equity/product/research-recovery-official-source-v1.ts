/**
 * Public, allowlisted PRIMARY SOURCE probe, confined by the caller to the ASML
 * read-only mirror in a strictly gated SHADOW Preview.
 *
 * This fetches an official issuer release, validates transport/content/date,
 * and calculates the hash of the bytes observed TODAY. It does not claim that
 * those bytes existed unchanged on the valuation date, admit evidence, compute
 * shares or update the Registry. Raw bytes can be passed to the private archive.
 *
 * No user-selected URLs and no redirects (SSRF/host substitution fail-closed).
 */
import {createHash} from 'node:crypto';

export const ASML_Q2_2026_OFFICIAL_PRESENTATION = Object.freeze({
  url: 'https://ourbrand.asml.com/asset/9078cf4d-91fd-4dd9-a5d5-d1caab6dc046/2026_07_15_Presentation-Investor-Relations-Q2-2026.pdf',
  publicationDate: '2026-07-15',
  evaluationCutoff: '2026-09-21',
  documentId: 'ASML_OFFICIAL_Q2_2026_INVESTOR_PRESENTATION',
});

export const ASML_Q2_2026_PUBLIC_RELEASE = Object.freeze({
  url: 'https://www.investor.asml.com/news-releases/news-release-details/q2-2026-financial-results',
  publicationDate: '2026-07-15',
  evaluationCutoff: '2026-09-21',
  documentId: 'ASML_OFFICIAL_Q2_2026_RELEASE',
});
const BYTE_LIMIT=2_000_000;
const DATE=/^\d{4}-\d{2}-\d{2}$/;

export type OfficialSourceCandidate = Readonly<{
  documentId:string;
  url:string;
  publicationDate:string;
  retrievedAt:string;
  byteSize:number;
  sha256:string;
  contentType:'text/html'|'application/pdf';
  admitted:false;
  historicalBytesCertified:false;
}>;

export async function retrieveAsmlQ2OfficialRelease(
  fetcher: typeof fetch=fetch,
  evaluationDate:string=ASML_Q2_2026_PUBLIC_RELEASE.evaluationCutoff,
  now:()=>Date=()=>new Date(),
): Promise<Readonly<{candidate:OfficialSourceCandidate; bytes:Buffer}>> {
  const source=ASML_Q2_2026_PUBLIC_RELEASE;
  if (!DATE.test(evaluationDate) ||
      source.publicationDate > evaluationDate ||
      evaluationDate > source.evaluationCutoff) {
    throw new Error('RECOVERY_SOURCE_TEMPORAL_GATE');
  }
  const response=await fetcher(source.url,{
    method:'GET',
    redirect:'error',
    credentials:'omit',
    cache:'no-store',
    headers:{accept:'text/html'},
    signal:AbortSignal.timeout(8_500),
  });
  if (response.status !== 200 || response.redirected || response.url !== source.url) {
    throw new Error('RECOVERY_SOURCE_ORIGIN_UNVERIFIED');
  }
  const contentType=response.headers.get('content-type')?.toLowerCase()??'';
  if (!/^text\/html(?:\s*;|\s*$)/.test(contentType)) {
    throw new Error('RECOVERY_SOURCE_CONTENT_TYPE');
  }
  const advertisedSize=Number(response.headers.get('content-length'));
  if (Number.isFinite(advertisedSize) && advertisedSize > BYTE_LIMIT) {
    throw new Error('RECOVERY_SOURCE_TOO_LARGE');
  }
  if (!response.body) throw new Error('RECOVERY_SOURCE_EMPTY_BODY');
  const chunks:Uint8Array[]=[];
  let size=0;
  const reader=response.body.getReader();
  try {
    while (true) {
      const {done,value}=await reader.read();
      if (done) break;
      if (value) {
        size+=value.byteLength;
        if (size>BYTE_LIMIT) throw new Error('RECOVERY_SOURCE_TOO_LARGE');
        chunks.push(value);
      }
    }
  } finally {
    reader.releaseLock();
  }
  if (size<256) throw new Error('RECOVERY_SOURCE_EMPTY_BODY');
  const bytes=Buffer.concat(chunks.map(chunk=>Buffer.from(chunk)),size);
  const page=new TextDecoder('utf-8',{fatal:true}).decode(bytes)
    .replace(/<[^>]*>/g,' ').replace(/\s+/g,' ');
  if (!/\bASML\b/i.test(page) ||
      !/July 15,? 2026/i.test(page) ||
      !(/\bQ2\s+2026\b/i.test(page) || /2026\s+second-quarter\s+results/i.test(page))) {
    throw new Error('RECOVERY_SOURCE_IDENTITY_OR_DATE_UNVERIFIED');
  }
  return {candidate:{
    documentId:source.documentId,url:source.url,
    publicationDate:source.publicationDate,
    retrievedAt:now().toISOString(),byteSize:size,
    sha256:createHash('sha256').update(bytes).digest('hex'),
    contentType:'text/html',
    admitted:false,historicalBytesCertified:false,
  },bytes};
}

/** Metadata-only helper for read-only probes; never exposes source bytes to UI. */
export async function probeAsmlQ2OfficialRelease(
  fetcher: typeof fetch=fetch,
  evaluationDate:string=ASML_Q2_2026_PUBLIC_RELEASE.evaluationCutoff,
  now:()=>Date=()=>new Date(),
): Promise<OfficialSourceCandidate> {
  return (await retrieveAsmlQ2OfficialRelease(fetcher,evaluationDate,now)).candidate;
}

/**
 * Deterministic alternative primary document from ASML's own brand asset CDN.
 * The exact URL, MIME, date, PDF signature and size must all match.
 * PDF bytes are preserved, never treated as historic or admitted evidence.
 */
export async function retrieveAsmlQ2OfficialPresentation(
  fetcher:typeof fetch=fetch,
  evaluationDate:string=ASML_Q2_2026_OFFICIAL_PRESENTATION.evaluationCutoff,
  now:()=>Date=()=>new Date(),
):Promise<Readonly<{candidate:OfficialSourceCandidate;bytes:Buffer}>> {
  const source=ASML_Q2_2026_OFFICIAL_PRESENTATION;
  if (!DATE.test(evaluationDate) || source.publicationDate>evaluationDate ||
      evaluationDate>source.evaluationCutoff) throw Error('RECOVERY_SOURCE_TEMPORAL_GATE');
  const response=await fetcher(source.url,{
    method:'GET',redirect:'error',credentials:'omit',cache:'no-store',
    headers:{accept:'application/pdf'},signal:AbortSignal.timeout(8_500),
  });
  if(response.status!==200 || response.redirected || response.url!==source.url)
    throw Error('RECOVERY_SOURCE_PDF_ORIGIN_UNVERIFIED');
  const contentType=response.headers.get('content-type')?.toLowerCase()??'';
  if(!/^application\/pdf(?:\s*;|\s*$)/.test(contentType))
    throw Error('RECOVERY_SOURCE_PDF_CONTENT_TYPE');
  const advertisedSize=Number(response.headers.get('content-length'));
  if(Number.isFinite(advertisedSize) && advertisedSize>BYTE_LIMIT)
    throw Error('RECOVERY_SOURCE_TOO_LARGE');
  if(!response.body)throw Error('RECOVERY_SOURCE_EMPTY_BODY');
  const chunks:Uint8Array[]=[];
  let size=0;
  const reader=response.body.getReader();
  try {
    while(true) {
      const {done,value}=await reader.read();
      if(done)break;
      if(value){
        size+=value.byteLength;
        if(size>BYTE_LIMIT)throw Error('RECOVERY_SOURCE_TOO_LARGE');
        chunks.push(value);
      }
    }
  } finally {reader.releaseLock();}
  if(size<10_000)throw Error('RECOVERY_SOURCE_PDF_TRUNCATED');
  const bytes=Buffer.concat(chunks.map(v=>Buffer.from(v)),size);
  if(bytes.toString('latin1',0,5)!=='%PDF-' ||
     !bytes.toString('latin1',Math.max(0,size-2048)).includes('%%EOF'))
    throw Error('RECOVERY_SOURCE_PDF_SIGNATURE_INVALID');
  return {candidate:{
    documentId:source.documentId,url:source.url,
    publicationDate:source.publicationDate,
    retrievedAt:now().toISOString(),
    byteSize:size,
    sha256:createHash('sha256').update(bytes).digest('hex'),
    contentType:'application/pdf',
    admitted:false,historicalBytesCertified:false,
  },bytes};
}

/** At most two trusted issuer endpoints per cycle, no arbitrary search. */
export async function retrievePreferredAsmlQ2OfficialSource(
  fetcher:typeof fetch=fetch,
  evaluationDate:string=ASML_Q2_2026_PUBLIC_RELEASE.evaluationCutoff,
  now:()=>Date=()=>new Date(),
):Promise<Readonly<{candidate:OfficialSourceCandidate;bytes:Buffer}>> {
  // Enforce the frozen valuation clock BEFORE any network request.
  if(!DATE.test(evaluationDate) ||
     evaluationDate<ASML_Q2_2026_PUBLIC_RELEASE.publicationDate ||
     evaluationDate>ASML_Q2_2026_PUBLIC_RELEASE.evaluationCutoff)
    throw Error('RECOVERY_SOURCE_TEMPORAL_GATE');
  try {
    return await retrieveAsmlQ2OfficialRelease(fetcher,evaluationDate,now);
  } catch {
    // Site investor HTML may return 403 to datacenter IPs.
    // Distinct issuer-owned PDF is permitted only after its own strict checks.
    return await retrieveAsmlQ2OfficialPresentation(fetcher,evaluationDate,now);
  }
}
