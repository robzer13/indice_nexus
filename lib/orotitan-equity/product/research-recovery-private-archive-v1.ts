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
import {ASML_Q2_2026_PUBLIC_RELEASE,ASML_Q2_2026_OFFICIAL_PRESENTATION} from './research-recovery-official-source-v1';

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
  const trusted=candidate.documentId===ASML_Q2_2026_PUBLIC_RELEASE.documentId
    ? {source:ASML_Q2_2026_PUBLIC_RELEASE,mime:'text/html',extension:'html'}
    : candidate.documentId===ASML_Q2_2026_OFFICIAL_PRESENTATION.documentId
      ? {source:ASML_Q2_2026_OFFICIAL_PRESENTATION,mime:'application/pdf',extension:'pdf'}
      : null;
  if (!trusted ||
      candidate.url !== trusted.source.url ||
      candidate.publicationDate !== trusted.source.publicationDate ||
      candidate.contentType !== trusted.mime ||
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
  const path='research-recovery-v1/asml-observation/q2-2026/'+candidate.sha256+'.'+trusted.extension;
  const handler=client.storage.from(RECOVERY_PRIVATE_BUCKET);
  const {error:uploadError}=await handler.upload(path,bytes,{
    contentType:trusted.mime,upsert:false,cacheControl:'0',
  });
  if (uploadError) {
    // Supabase Storage has returned both HTTP 409 and legacy HTTP 400
    // for the precise 'already exists' case. Never treat arbitrary 400/409
    // failures (authorization, bad MIME, malformed upload) as valid reuse.
    const err=uploadError as unknown as Record<string,unknown>;
    const code=String(err.code??err.error??'');
    const message=String(err.message??'');
    const isDuplicate=['KeyAlreadyExists','ResourceAlreadyExists','already_exists']
      .includes(code) ||
      /^(?:The resource already exists|Asset Already Exists|Duplicate)$/i.test(message.trim());
    if (!isDuplicate || !['400','409'].includes(String(uploadError.statusCode))) {
      throw new Error('RECOVERY_ARCHIVE_CREATE_FAILED');
    }
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
