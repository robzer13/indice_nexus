/**
 * D29b PostgreSQL private receipt adapter — REVIEW ONLY.
 * SQL function names are DRAFT CONTRACTS, NOT DEPLOYED DATABASE OBJECTS.
 * The only permitted runtime principal is a separate private server role;
 * this module must never be imported by the public OAuth/MCP handler.
 */
import {
  ResearchIntakeError, type PreparedResearchDraft, type TrustedSubmissionGrant,
  type TrustedVerifierGrant, type VerifierLookup, type UnverifiedReceipt,
} from "./research-evidence-intake-candidate-v1";
import type {
  ResearchReceiptTransactionPort, CommitReceiptInput,
} from "./research-private-quarantine-adapter-v1";

export type PrivatePgQueryResult<T> = { rows: T[] };
export type PrivatePgClient = {
  query: <T>(query: string, params?: unknown[]) => Promise<PrivatePgQueryResult<T>>;
  release: () => void;
};
export type PrivatePgPool = { connect: () => Promise<PrivatePgClient> };
const PRIVATE_ROLE = "orotitan_research_intake_writer";
const RESOURCE = "https://orotitan-chatgpt-mcp-pilot.vercel.app/api/orotitan/mcp";
const HASH = /^[0-9a-f]{64}$/;

async function inTransaction<T>(
  pool: PrivatePgPool, write: boolean, task: (client: PrivatePgClient) => Promise<T>,
): Promise<T> {
  const client = await pool.connect();
  let active = false;
  try {
    await client.query(write
      ? "BEGIN ISOLATION LEVEL SERIALIZABLE"
      : "BEGIN ISOLATION LEVEL REPEATABLE READ READ ONLY");
    active = true;
    // This fails unless the independently provisioned private principal has
    // membership. NEVER grant the role to anon, authenticated or OAuth runtime.
    await client.query("SET LOCAL ROLE " + PRIVATE_ROLE);
    const result = await task(client);
    await client.query("COMMIT");
    active = false;
    return result;
  } catch (err) {
    if (active) try { await client.query("ROLLBACK"); } catch { /* fail closed */ }
    throw err;
  } finally {
    client.release();
  }
}

function assertPinnedDraft(d: PreparedResearchDraft, g: TrustedSubmissionGrant) {
  if (d.runId !== g.runId || d.issuerId !== g.issuerId ||
    d.stage !== "RESEARCH" || g.stage !== "RESEARCH" ||
    d.runStateVersion !== g.runStateVersion ||
    d.stageStateVersion !== g.stageStateVersion ||
    d.dataCutoff !== g.dataCutoff ||
    d.contractPinsSha256 !== g.contractPinsSha256 ||
    d.submittedBySubject !== g.subject || d.submittedByClientId !== g.clientId ||
    g.resource !== RESOURCE || !g.maySubmitResearchDraft ||
    !g.grantId || !HASH.test(g.contractPinsSha256)) {
    throw new ResearchIntakeError("INTAKE_PRIVATE_PG_SCOPE_DENIED");
  }
}
type StoredRow = {
  receipt_id: string;
  run_id: string;
  issuer_id: string;
  stage: string;
  run_state_version: number;
  stage_state_version: number;
  sha256: string;
  file_name: string;
  size_bytes: number;
  submitted_by_subject: string;
  duplicate: boolean;
  object_path: string;
};
function rowToReceipt(row: StoredRow): UnverifiedReceipt {
  if (!row || !row.receipt_id || !HASH.test(row.sha256) ||
    !Number.isSafeInteger(Number(row.run_state_version)) ||
    !Number.isSafeInteger(Number(row.stage_state_version)) ||
    typeof row.duplicate !== "boolean" ||
    !Number.isSafeInteger(Number(row.size_bytes))) {
    throw new ResearchIntakeError("INTAKE_PRIVATE_PG_INVALID_RECEIPT_ROW");
  }
  return {
    receiptId: row.receipt_id, runId: row.run_id, issuerId: row.issuer_id,
    stage: "RESEARCH", runStateVersion: Number(row.run_state_version),
    stageStateVersion: Number(row.stage_state_version),
    sha256: row.sha256, fileName: row.file_name,
    sizeBytes: Number(row.size_bytes),
    submittedBySubject: row.submitted_by_subject,
    status: "RECEIVED_NOT_VERIFIED", authority: "QUARANTINE_ONLY",
    duplicate: row.duplicate,
    mayAdvance: false, mayPublish: false, admitted: false,
  };
}

/**
 * Driver adapter only. The reviewed DB functions must independently enforce:
 * private schema and role, exact Besi run+issuer, live CAS, owner/client/resource,
 * grant expiry/revocation, append-only audit on same transaction, unique
 * idempotency and forbidden Registry writes. No SQL migration is applied here.
 */
export function createPrivateResearchPgTransactionPort(
  pool: PrivatePgPool,
): ResearchReceiptTransactionPort {
  return {
    async checkBeforeUpload(d, g) {
      assertPinnedDraft(d, g);
      const r = await inTransaction(pool, false, client =>
        client.query<{ allowed: boolean }>(
          "SELECT orotitan_research_intake_private.preflight_v1(" +
          "$1::uuid,$2::text,$3::text,$4::text,$5::text,$6::text," +
          "$7::bigint,$8::bigint,$9::date,$10::text) AS allowed",
          [d.runId, d.issuerId, g.grantId, g.subject, g.clientId, g.resource,
            d.runStateVersion, d.stageStateVersion, d.dataCutoff, d.contractPinsSha256],
        ));
      if (r.rows.length !== 1 || r.rows[0].allowed !== true) {
        throw new ResearchIntakeError("INTAKE_PRIVATE_PG_PREFLIGHT_DENIED");
      }
    },

    async commitReceiptAndAudit({ draft:d, grant:g, stored }: CommitReceiptInput) {
      assertPinnedDraft(d, g);
      if (stored.sha256 !== d.sha256 || stored.sizeBytes !== d.sizeBytes ||
        !stored.objectPath.startsWith("research-intake/v1/" + d.runId +
          "/RESEARCH/" + d.sha256 + "/")) {
        throw new ResearchIntakeError("INTAKE_PRIVATE_PG_BLOB_SCOPE_DENIED");
      }
      return inTransaction(pool, true, async client => {
        const r = await client.query<StoredRow>(
          "SELECT * FROM orotitan_research_intake_private.commit_receipt_v1(" +
          "$1::uuid,$2::text,$3::text,$4::text,$5::text,$6::text," +
          "$7::bigint,$8::bigint,$9::date,$10::text,$11::text,$12::text," +
          "$13::text,$14::text,$15::bigint,$16::text)",
          [d.runId,d.issuerId,g.grantId,g.subject,g.clientId,g.resource,
            d.runStateVersion,d.stageStateVersion,d.dataCutoff,
            d.contractPinsSha256,d.sha256,d.fileName,d.mediaType,
            stored.objectPath,d.sizeBytes,d.sourceDate],
        );
        const row = r.rows[0];
        const prefix = "research-intake/v1/" + d.runId +
          "/RESEARCH/" + d.sha256 + "/";
        // Every returned row and object reference must be validated INSIDE
        // the DB transaction. A bogus result must force ROLLBACK, not COMMIT.
        if (r.rows.length !== 1 || !row ||
          row.run_id !== d.runId || row.issuer_id !== d.issuerId ||
          row.stage !== "RESEARCH" || row.sha256 !== d.sha256 ||
          row.file_name !== d.fileName ||
          row.submitted_by_subject !== g.subject ||
          Number(row.run_state_version) !== d.runStateVersion ||
          Number(row.stage_state_version) !== d.stageStateVersion ||
          Number(row.size_bytes) !== d.sizeBytes ||
          typeof row.object_path !== "string" ||
          !row.object_path.startsWith(prefix) ||
          (row.duplicate === false && row.object_path !== stored.objectPath)) {
          throw new ResearchIntakeError("INTAKE_PRIVATE_PG_COMMIT_RESULT_INVALID");
        }
        const receipt = rowToReceipt(row);
        return { receipt, authoritativeObjectPath: row.object_path };
      });
    },

    async readVerifierMetadata(q: VerifierLookup, g: TrustedVerifierGrant) {
      if (q.runId !== g.runId || g.stage !== "RESEARCH" ||
        g.runStateVersion !== q.expectedRunStateVersion ||
        g.stageStateVersion !== q.expectedStageStateVersion ||
        !g.mayReadQuarantineMetadata || !HASH.test(q.sha256)) {
        throw new ResearchIntakeError("INTAKE_PRIVATE_PG_VERIFIER_DENIED");
      }
      const r = await inTransaction(pool, false, client =>
        client.query<StoredRow>(
          "SELECT * FROM orotitan_research_intake_private.verifier_metadata_v1(" +
          "$1::uuid,$2::text,$3::text,$4::bigint,$5::bigint,$6::text)",
          [q.runId,g.issuerId,g.reviewerId,
            q.expectedRunStateVersion,q.expectedStageStateVersion,q.sha256],
        ));
      if (r.rows.length > 1) {
        throw new ResearchIntakeError("INTAKE_PRIVATE_PG_DUPLICATE_METADATA");
      }
      const row=r.rows[0];
      if (!row) return null;
      if (row.run_id !== q.runId || row.issuer_id !== g.issuerId ||
        row.sha256 !== q.sha256 || row.stage !== "RESEARCH" ||
        Number(row.run_state_version) !== q.expectedRunStateVersion ||
        Number(row.stage_state_version) !== q.expectedStageStateVersion) {
        throw new ResearchIntakeError("INTAKE_PRIVATE_PG_VERIFIER_RESULT_INVALID");
      }
      return rowToReceipt(row);
    },
  };
}
