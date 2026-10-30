/**
 * Public, allowlisted PRIMARY SOURCE probe, confined by the caller to the ASML
 * read-only mirror in a strictly gated SHADOW Preview.
 *
 * This fetches an official issuer release, validates transport/content/date,
 * and calculates the hash of the bytes observed TODAY. It does not claim that
 * those bytes existed unchanged on the valuation date, admit evidence, compute
 * shares, update the Registry, or archive the original document.
 *
 * No user-selected URLs and no redirects (SSRF/host substitution fail-closed).
 */
import {createHash} from 'node:crypto';

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
  contentType:'text/html';
  admitted:false;
  historicalBytesCertified:false;
}>;

export async function probeAsmlQ2OfficialRelease(
  fetcher: typeof fetch=fetch,
  evaluationDate:string=ASML_Q2_2026_PUBLIC_RELEASE.evaluationCutoff,
  now:()=>Date=()=>new Date(),
): Promise<OfficialSourceCandidate> {
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
  return {
    documentId:source.documentId,url:source.url,
    publicationDate:source.publicationDate,
    retrievedAt:now().toISOString(),byteSize:size,
    sha256:createHash('sha256').update(bytes).digest('hex'),
    contentType:'text/html',
    admitted:false,historicalBytesCertified:false,
  };
}
