import { createHash, randomUUID } from "node:crypto";
import type { SupabaseClient } from "@supabase/supabase-js";
import {
  ResearchIntakeError, type PreparedResearchDraft, type ResearchIntakePort,
  type IntakeActor, type TrustedSubmissionGrant, type VerifierActor,
  type TrustedVerifierGrant, type VerifierLookup, type UnverifiedReceipt,
} from "./research-evidence-intake-candidate-v1";

/**
 * D29b private-runtime proposal. NOTHING imports this from the public MCP
 * handler. Injection of a privileged server client does not grant OAuth rights.
 * The bucket MUST be private and the runtime MUST be independently authorized.
 */
export const RESEARCH_PRIVATE_BUCKET = "orotitan-research-evidence-quarantine-v1";
export const RESEARCH_PRIVATE_PREFIX = "research-intake/v1";
const HASH = /^[0-9a-f]{64}$/;
const UUID = /^[0-9a-f]{8}-[0-9a-f]{4}-[1-8][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i;

export type StoredPrivateBlob = Readonly<{
  objectPath: string;
  sha256: string;
  sizeBytes: number;
}>;

export type CommitReceiptInput = Readonly<{
  draft: PreparedResearchDraft;
  grant: TrustedSubmissionGrant;
  stored: StoredPrivateBlob;
}>;

/** Execute in ONE private SERIALIZABLE DB transaction; must not update Registry. */
export interface ResearchReceiptTransactionPort {
  /** A short read-only DB preflight (not sufficient for admission). */
  checkBeforeUpload(draft: PreparedResearchDraft, grant: TrustedSubmissionGrant): Promise<void>;
  /**
   * MUST recheck: live CAS run+stage, issuer, stage state, grant expiry and
   * revocation, actor/client/resource/pins, and any duplicate SHA/filename.
   * INSERT receipt and append-only audit in the SAME transaction, or reject.
   * Returned duplicate must reference pre-existing bytes, never the new blob.
   */
  commitReceiptAndAudit(input: CommitReceiptInput): Promise<{
    receipt: UnverifiedReceipt;
    authoritativeObjectPath: string;
  }>;
  /**
   * Private verifier-only transaction: independently check reviewer grant,
   * revocation, issuer, live CAS and exact run/sha; return metadata only.
   */
  readVerifierMetadata(lookup: VerifierLookup, grant: TrustedVerifierGrant):
    Promise<UnverifiedReceipt | null>;
}

export interface PrivateResearchBlobPort {
  /** Cannot overwrite; must fail if path already exists. */
  putImmutable(objectPath: string, bytes: Uint8Array, mediaType: string): Promise<void>;
  /** No bucket-wide enumeration, signed URL or public download. */
  readExact(objectPath: string): Promise<Uint8Array>;
}

export type ResearchScopedGrantReader = Pick<
  ResearchIntakePort, "resolveSubmissionGrant" | "resolveVerifierGrant"
>;

/**
 * Do not use filename as path or accept a caller-selected object key.
 * Each attempt gets a unique immutable object. This avoids an upload race on
 * the content hash; the database decides idempotency under a unique index.
 */
export function privateAttemptObjectPath(
  draft: PreparedResearchDraft, attemptId: string,
): string {
  if (!UUID.test(draft.runId) || draft.stage !== "RESEARCH" ||
    !HASH.test(draft.sha256) || !UUID.test(attemptId) ||
    !/^[A-Za-z0-9][A-Za-z0-9._-]{0,99}$/.test(draft.fileName) ||
    draft.fileName.includes("..")) {
    throw new ResearchIntakeError("INTAKE_PRIVATE_PATH_INVALID");
  }
  return [
    RESEARCH_PRIVATE_PREFIX, draft.runId, draft.stage, draft.sha256,
    attemptId.toLowerCase() + "." + draft.fileName.split(".").at(-1)!.toLowerCase(),
  ].join("/");
}

/**
 * Real Supabase Storage API calls, but PRIVATE runtime only. No service-key
 * construction or new roles, bucket creation, URL signing or MCP API here.
 * upsert=false: never replace immutable bytes. Re-read and SHA after upload.
 */
export function createSupabasePrivateResearchBlobPort(
  storageClient: Pick<SupabaseClient, "storage">,
  bucketName: string = RESEARCH_PRIVATE_BUCKET,
): PrivateResearchBlobPort {
  if (bucketName !== RESEARCH_PRIVATE_BUCKET || !storageClient?.storage) {
    throw new ResearchIntakeError("INTAKE_PRIVATE_BUCKET_NOT_CONFIGURED");
  }
  const bucket = storageClient.storage.from(bucketName);
  return {
    async putImmutable(path, bytes, mediaType) {
      const { error } = await bucket.upload(path, Buffer.from(bytes), {
        contentType: mediaType, cacheControl: "0", upsert: false,
      });
      if (error) throw new ResearchIntakeError("INTAKE_PRIVATE_UPLOAD_FAILED");
    },
    async readExact(path) {
      const { data, error } = await bucket.download(path);
      if (error || !data) throw new ResearchIntakeError("INTAKE_PRIVATE_REREAD_FAILED");
      return new Uint8Array(await data.arrayBuffer());
    },
  };
}

function assertOriginalBytes(draft: PreparedResearchDraft): void {
  const hash = createHash("sha256").update(draft.bytes).digest("hex");
  if (hash !== draft.sha256 || draft.bytes.byteLength !== draft.sizeBytes) {
    throw new ResearchIntakeError("INTAKE_PRIVATE_PREPARED_BYTES_CHANGED");
  }
}

function assertReceipt(
  result: { receipt: UnverifiedReceipt; authoritativeObjectPath: string },
  draft: PreparedResearchDraft, path: string,
): UnverifiedReceipt {
  const r = result?.receipt;
  const expectedPrefix = RESEARCH_PRIVATE_PREFIX + "/" + draft.runId + "/RESEARCH/" + draft.sha256 + "/";
  if (!r || !r.receiptId || !result.authoritativeObjectPath ||
    !result.authoritativeObjectPath.startsWith(expectedPrefix) ||
    (r.duplicate === false && result.authoritativeObjectPath !== path) ||
    r.runId !== draft.runId || r.issuerId !== draft.issuerId ||
    r.stage !== "RESEARCH" || r.sha256 !== draft.sha256 ||
    r.runStateVersion !== draft.runStateVersion ||
    r.stageStateVersion !== draft.stageStateVersion ||
    r.fileName !== draft.fileName || r.sizeBytes !== draft.sizeBytes ||
    r.submittedBySubject !== draft.submittedBySubject ||
    r.status !== "RECEIVED_NOT_VERIFIED" || r.authority !== "QUARANTINE_ONLY" ||
    r.admitted !== false || r.mayAdvance !== false || r.mayPublish !== false ||
    typeof r.duplicate !== "boolean") {
    throw new ResearchIntakeError("INTAKE_PRIVATE_RECEIPT_INVALID");
  }
  return r;
}

/**
 * Make the private port consumable by submitResearchDraftCandidate().
 *
 * Blob storage cannot participate in a PostgreSQL transaction. If storage
 * succeeded but the DB failed (including ambiguous commit), DO NOT delete
 * the object: the transaction might have committed. An independently reviewed
 * reconciler may later remove proven unreferenced objects after a safety TTL.
 */
export function createPrivateResearchIntakePort(args: {
  grants: ResearchScopedGrantReader;
  blobs: PrivateResearchBlobPort;
  receipts: ResearchReceiptTransactionPort;
  newAttemptId?: () => string;
}): ResearchIntakePort {
  const makeId = args.newAttemptId ?? randomUUID;
  return {
    resolveSubmissionGrant: (actor: IntakeActor, runId: string) =>
      args.grants.resolveSubmissionGrant(actor, runId),
    resolveVerifierGrant: (actor: VerifierActor, runId: string) =>
      args.grants.resolveVerifierGrant(actor, runId),

    async persistQuarantinedDraft(draft, grant) {
      assertOriginalBytes(draft);
      // Private DB preflight rejects already-revoked/mismatched grants before
      // allocating storage. DB must repeat every check in final transaction.
      await args.receipts.checkBeforeUpload(draft, grant);
      const path = privateAttemptObjectPath(draft, makeId());
      await args.blobs.putImmutable(path, Uint8Array.from(draft.bytes), draft.mediaType);
      const reread = await args.blobs.readExact(path);
      if (reread.byteLength !== draft.sizeBytes ||
        createHash("sha256").update(reread).digest("hex") !== draft.sha256) {
        throw new ResearchIntakeError("INTAKE_PRIVATE_STORED_HASH_MISMATCH");
      }
      const committed = await args.receipts.commitReceiptAndAudit({
        draft, grant, stored: { objectPath: path, sha256: draft.sha256, sizeBytes: reread.byteLength },
      });
      return assertReceipt(committed, draft, path);
    },

    findVerifierMetadata: (lookup, grant) => args.receipts.readVerifierMetadata(lookup, grant),
  };
}
