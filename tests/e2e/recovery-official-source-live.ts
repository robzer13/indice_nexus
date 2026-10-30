/**
 * CI-only external read probe: real ASML primary document, no Supabase client,
 * no Vercel credential, no file persistence, no Registry or stage mutation.
 * Fail-closed: a blocked issuer request must produce FAILED, not success.
 */
import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import {retrieveAsmlQ2OfficialRelease,ASML_Q2_2026_PUBLIC_RELEASE}
  from '../../lib/orotitan-equity/product/research-recovery-official-source-v1';

if (process.env.CI !== 'true' || process.env.NODE_ENV === 'production') {
  throw new Error('RECOVERY_LIVE_PROBE_CI_ONLY');
}
async function main() {
const tracedFetcher=(async (input:string|URL|Request,options?:RequestInit)=>{
  const response=await fetch(input,options);
  const location=response.headers.get('location');
  let redirectHost:string|null=null;
  if (location) {
    try {redirectHost=new URL(location,String(input)).hostname;} catch {redirectHost='malformed';}
  }
  console.log(JSON.stringify({
    httpStatus:response.status,
    effectiveHost:new URL(response.url||String(input)).hostname,
    redirected:response.redirected,
    redirectionHost:redirectHost,
    contentType:response.headers.get('content-type'),
  }));
  return response;
}) as typeof fetch;
let result:Awaited<ReturnType<typeof retrieveAsmlQ2OfficialRelease>>;
try {
  result=await retrieveAsmlQ2OfficialRelease(tracedFetcher);
} catch (error) {
  // Diagnostic probes of explicitly pinned official ASML assets, no writes.
  // No redirect following, no arbitrary remote URLs and no downgrade to
  // unsourced content when the primary press-release endpoint is blocked.
  const alternatives=[
    'https://ourbrand.asml.com/asset/9078cf4d-91fd-4dd9-a5d5-d1caab6dc046/2026_07_15_Presentation-Investor-Relations-Q2-2026.pdf',
    'https://www.investor.asml.com/static-files/4fe3450c-f56c-473d-809a-3a7ae2b7fab3',
  ];
  for(const url of alternatives) {
    try {
      const response=await fetch(url,{redirect:'error',signal:AbortSignal.timeout(8000)});
      console.log(JSON.stringify({alternativeHost:new URL(url).hostname,status:response.status,
        contentType:response.headers.get('content-type'),redirected:response.redirected}));
      await response.body?.cancel();
    } catch (problem) {
      console.log(JSON.stringify({alternativeHost:new URL(url).hostname,
        failed:problem instanceof Error?problem.name:'unknown'}));
    }
  }
  throw error;
}
const {candidate,bytes}=result;
assert.equal(candidate.url,ASML_Q2_2026_PUBLIC_RELEASE.url);
assert.equal(candidate.publicationDate,'2026-07-15');
assert.equal(candidate.admitted,false);
assert.equal(candidate.historicalBytesCertified,false);
assert.ok(candidate.byteSize>=256 && candidate.byteSize<=2_000_000);
assert.equal(createHash('sha256').update(bytes).digest('hex'),candidate.sha256);
console.log(JSON.stringify({
  result:'PUBLIC_ASML_Q2_LIVE_SOURCE_VERIFIED',
  url:candidate.url,
  documentDate:candidate.publicationDate,
  retrievalTime:candidate.retrievedAt,
  size:candidate.byteSize,
  sha256:candidate.sha256,
  registryAdmitted:false,
  archived:false,
},null,2));

}
main().catch(error=>{
  console.error('RECOVERY_LIVE_SOURCE_PROBE_FAILED:',error);
  process.exitCode=1;
});
