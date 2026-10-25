/* Opt-in only; never called by production route without V2.0.11 admission and exact GO PUBLISH. */
import {
  validateV2ResearchSnapshotForPersistenceWithLongThesis,
  type V2ResearchSnapshot,
} from "./research-snapshot-schema-thesis-compat";
import { ResearchSnapshotValidationError, type PersistenceResult } from "../v1/persistence-core";
import {
  verifyCertifiedThesisAndCandidate,
  type ExactThesisAdmissionInput,
} from "./certified-thesis-source-admission";

export type V2SnapshotRpc = (args: {
  p_dossier_id: string;
  p_expected_current_snapshot_id: string | null;
  p_canonical_payload: V2ResearchSnapshot;
}) => Promise<{ data: unknown; error: Error | null }>;

export type PersistValidatedV2ResearchSnapshotInput = {
  dossierId: string;
  expectedCurrentSnapshotId: string | null;
  canonicalPayload: unknown;
  certifiedSourceProof: Omit<ExactThesisAdmissionInput, "canonicalCandidate" | "dossierId">;
};

function isUuid(value: unknown): value is string {
  return typeof value === "string"
    && /^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i.test(value);
}

function parsePersistenceResult(data: unknown): PersistenceResult {
  if (typeof data !== "object" || data === null) throw new Error("V2 persistence RPC returned an invalid result");
  const result = data as Record<string, unknown>;
  if ((result.status !== "INSERTED" && result.status !== "IDEMPOTENT_SUCCESS")
      || typeof result.dossier_id !== "string"
      || typeof result.snapshot_id !== "string"
      || typeof result.current_snapshot_id !== "string") {
    throw new Error("V2 persistence RPC returned an invalid result");
  }
  return result as unknown as PersistenceResult;
}

export async function persistValidatedV2ResearchSnapshotWithLongThesis(
  input: PersistValidatedV2ResearchSnapshotInput,
  rpc: V2SnapshotRpc,
): Promise<PersistenceResult> {
  if (!isUuid(input.dossierId)) throw new Error("dossierId must be a UUID compatible with physical persistence");
  if (input.expectedCurrentSnapshotId !== null && !isUuid(input.expectedCurrentSnapshotId)) {
    throw new Error("expectedCurrentSnapshotId must be null or a UUID");
  }
  const sourceGate = verifyCertifiedThesisAndCandidate({
    ...input.certifiedSourceProof,
    dossierId: input.dossierId,
    canonicalCandidate: input.canonicalPayload,
  });
  if (!sourceGate.admittedForValidation) {
    throw new Error(`V2.0.11 SEALED SOURCE ADMISSION DENIED: ${sourceGate.errors.join("; ")}`);
  }
  const validation = validateV2ResearchSnapshotForPersistenceWithLongThesis(input.canonicalPayload, input.dossierId);
  if (!validation.ok) throw new ResearchSnapshotValidationError(validation);
  const response = await rpc({
    p_dossier_id: input.dossierId,
    p_expected_current_snapshot_id: input.expectedCurrentSnapshotId,
    p_canonical_payload: validation.snapshot,
  });
  if (response.error) throw response.error;
  return parsePersistenceResult(response.data);
}
