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
const {candidate,bytes}=await retrieveAsmlQ2OfficialRelease();
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
