/**
 * CI-only external issuer probe: at most TWO pinned official ASML endpoints.
 * No Supabase, Vercel credentials, persistence or Registry mutation.
 */
import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import {
  retrievePreferredAsmlQ2OfficialSource,
  ASML_Q2_2026_PUBLIC_RELEASE,
  ASML_Q2_2026_OFFICIAL_PRESENTATION,
} from '../../lib/orotitan-equity/product/research-recovery-official-source-v1';

if (process.env.CI !== 'true' || process.env.NODE_ENV === 'production') {
  throw new Error('RECOVERY_LIVE_PROBE_CI_ONLY');
}

async function main(){
  const attempts:string[]=[];
  const traceFetch=(async(url:string|URL|Request,options?:RequestInit)=>{
    const target=String(url);
    assert.ok([ASML_Q2_2026_PUBLIC_RELEASE.url,
      ASML_Q2_2026_OFFICIAL_PRESENTATION.url].includes(target));
    assert.equal(options?.redirect,'error');
    attempts.push(target);
    const response=await fetch(url,options);
    // Only safe transport metadata; no source document bodies in public CI.
    console.log(JSON.stringify({
      issuerHost:new URL(target).hostname,
      httpStatus:response.status,
      contentType:response.headers.get('content-type'),
    }));
    return response;
  }) as typeof fetch;
  const {candidate,bytes}=await retrievePreferredAsmlQ2OfficialSource(traceFetch);
  assert.equal(candidate.publicationDate,'2026-07-15');
  assert.equal(candidate.admitted,false);
  assert.equal(candidate.historicalBytesCertified,false);
  assert.ok(candidate.byteSize>=256 && candidate.byteSize<=2_000_000);
  assert.equal(createHash('sha256').update(bytes).digest('hex'),candidate.sha256);
  assert.ok(attempts.length===1||attempts.length===2);
  console.log(JSON.stringify({
    result:'PUBLIC_ASML_Q2_LIVE_SOURCE_VERIFIED',
    documentId:candidate.documentId,
    url:candidate.url,
    publicationDate:candidate.publicationDate,
    observedAt:candidate.retrievedAt,
    bytes:candidate.byteSize,
    sha256:candidate.sha256,
    registryAdmitted:false,
    archived:false,
    attempts:attempts.length,
  },null,2));
}
main().catch(error=>{
  console.error('RECOVERY_LIVE_SOURCE_PROBE_FAILED:',error);
  process.exitCode=1;
});
