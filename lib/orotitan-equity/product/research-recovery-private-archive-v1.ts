/**
 * SHADOW-only private archive for publicly accessible primary-source bytes.
 * The caller must have passed the strict hosted SHADOW gate and must never
 * expose its service-role client to the browser.
 *
 * Content-addressed, create-only and re-downloaded for hash verification.
 * This is NOT an OroTitan Registry artifact, source admission or attestation.
 */
import {createHash} from 'node:crypto';
import type {SupabaseClient} from '@supabase/supabase-js';
import type {OfficialSourceCandidate} from './research-recovery-official-source-v1';
import {ASML_Q2_2026_PUBLIC_RELEASE} from './research-recovery-official-source-v1';

export const RECOVERY_PRIVATE_BUCKET='orotitan-source-files-v1';
const SHA=/^[a-f0-9]{64}$/;
const MAX_BYTES=2_000_000;

export type PrivateSourceReceipt=Readonly<{
  bucket:typeof RECOVERY_PRIVATE_BUCKET;
  path:string;
  sha256:string;
  byteSize:number;
  verifiedReadback:true;
  registryAdmitted:false;
}>;

export async function archiveAsmlCandidate(
  client:SupabaseClient,
  candidate:OfficialSourceCandidate,
  bytes:Buffer,
):Promise<PrivateSourceReceipt> {
  if (candidate.documentId !== ASML_Q2_2026_PUBLIC_RELEASE.documentId ||
      candidate.url !== ASML_Q2_2026_PUBLIC_RELEASE.url ||
      candidate.publicationDate !== ASML_Q2_2026_PUBLIC_RELEASE.publicationDate ||
      candidate.contentType !== 'text/html' ||
      candidate.admitted !== false ||
      candidate.historicalBytesCertified !== false ||
      !Buffer.isBuffer(bytes) ||
      bytes.length < 256 || bytes.length > MAX_BYTES ||
      bytes.length !== candidate.byteSize ||
      !SHA.test(candidate.sha256) ||
      createHash('sha256').update(bytes).digest('hex') !== candidate.sha256) {
    throw new Error('RECOVERY_ARCHIVE_INPUT_INTEGRITY');
  }
  const {data:bucket,error:bucketError}=await client.storage.getBucket(RECOVERY_PRIVATE_BUCKET);
  if (bucketError || !bucket || bucket.id !== RECOVERY_PRIVATE_BUCKET || bucket.public !== false) {
    throw new Error('RECOVERY_ARCHIVE_BUCKET_NOT_PRIVATE');
  }
  const path='research-recovery-v1/asml-observation/q2-2026/'+candidate.sha256+'.html';
  const handler=client.storage.from(RECOVERY_PRIVATE_BUCKET);
  const {error:uploadError}=await handler.upload(path,bytes,{
    contentType:'text/html',upsert:false,cacheControl:'0',
  });
  if (uploadError && String(uploadError.statusCode) !== '409') {
    throw new Error('RECOVERY_ARCHIVE_CREATE_FAILED');
  }
  // Readback check is mandatory on both new writes and 409 collisions.
  // A collision is not proof of byte equality, only matching bytes are.
  const {data:download,error:downloadError}=await handler.download(path);
  if (downloadError || !download) throw new Error('RECOVERY_ARCHIVE_READBACK_FAILED');
  const stored=Buffer.from(await download.arrayBuffer());
  if (stored.length !== bytes.length ||
      createHash('sha256').update(stored).digest('hex') !== candidate.sha256) {
    throw new Error('RECOVERY_ARCHIVE_READBACK_MISMATCH');
  }
  return {bucket:RECOVERY_PRIVATE_BUCKET,path,
    sha256:candidate.sha256,byteSize:stored.length,
    verifiedReadback:true,registryAdmitted:false};
}
