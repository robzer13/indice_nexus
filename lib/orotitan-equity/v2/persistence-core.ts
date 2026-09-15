import {
  validateV2ResearchSnapshotForPersistence,
  type V2ResearchSnapshot,
} from "./research-snapshot-schema";
import { ResearchSnapshotValidationError, type PersistenceResult } from "../v1/persistence-core";

export type V2SnapshotRpc = (args: {
  p_dossier_id: string;
  p_expected_current_snapshot_id: string | null;
  p_canonical_payload: V2ResearchSnapshot;
}) => Promise<{ data: unknown; error: Error | null }>;

export type PersistValidatedV2ResearchSnapshotInput = {
  dossierId: string;
  expectedCurrentSnapshotId: string | null;
  canonicalPayload: unknown;
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

export async function persistValidatedV2ResearchSnapshot(
  input: PersistValidatedV2ResearchSnapshotInput,
  rpc: V2SnapshotRpc,
): Promise<PersistenceResult> {
  if (!isUuid(input.dossierId)) throw new Error("dossierId must be a UUID compatible with physical persistence");
  if (input.expectedCurrentSnapshotId !== null && !isUuid(input.expectedCurrentSnapshotId)) {
    throw new Error("expectedCurrentSnapshotId must be null or a UUID");
  }
  const validation = validateV2ResearchSnapshotForPersistence(input.canonicalPayload, input.dossierId);
  if (!validation.ok) throw new ResearchSnapshotValidationError(validation);
  const response = await rpc({
    p_dossier_id: input.dossierId,
    p_expected_current_snapshot_id: input.expectedCurrentSnapshotId,
    p_canonical_payload: validation.snapshot,
  });
  if (response.error) throw response.error;
  return parsePersistenceResult(response.data);
}
