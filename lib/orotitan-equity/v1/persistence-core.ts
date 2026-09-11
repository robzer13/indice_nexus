import {
  validateResearchSnapshotForPersistence,
  type ResearchSnapshot,
  type ValidationFailure,
} from "./research-snapshot-schema";

export type PersistValidatedResearchSnapshotInput = {
  dossierId: string;
  expectedCurrentSnapshotId: string | null;
  canonicalPayload: unknown;
};

export type PersistenceResult = {
  status: "INSERTED" | "IDEMPOTENT_SUCCESS";
  dossier_id: string;
  snapshot_id: string;
  current_snapshot_id: string;
};

export type SnapshotRpc = (args: {
  p_dossier_id: string;
  p_expected_current_snapshot_id: string | null;
  p_canonical_payload: ResearchSnapshot;
}) => Promise<{ data: unknown; error: Error | null }>;

export class ResearchSnapshotValidationError extends Error {
  readonly validation: ValidationFailure;

  constructor(validation: ValidationFailure) {
    super(`Research snapshot validation failed at ${validation.stage}: ${validation.errors.join("; ")}`);
    this.name = "ResearchSnapshotValidationError";
    this.validation = validation;
  }
}

function isUuid(value: unknown): value is string {
  return typeof value === "string"
    && /^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i.test(value);
}

function parsePersistenceResult(data: unknown): PersistenceResult {
  if (typeof data !== "object" || data === null) throw new Error("Persistence RPC returned an invalid result");
  const result = data as Record<string, unknown>;
  if ((result.status !== "INSERTED" && result.status !== "IDEMPOTENT_SUCCESS")
    || typeof result.dossier_id !== "string"
    || typeof result.snapshot_id !== "string"
    || typeof result.current_snapshot_id !== "string") {
    throw new Error("Persistence RPC returned an invalid result");
  }
  return result as unknown as PersistenceResult;
}

export async function persistValidatedResearchSnapshot(
  input: PersistValidatedResearchSnapshotInput,
  rpc: SnapshotRpc,
): Promise<PersistenceResult> {
  if (!isUuid(input.dossierId)) throw new Error("dossierId must be a UUID compatible with physical persistence");
  if (input.expectedCurrentSnapshotId !== null && !isUuid(input.expectedCurrentSnapshotId)) {
    throw new Error("expectedCurrentSnapshotId must be null or a UUID");
  }
  const validation = validateResearchSnapshotForPersistence(input.canonicalPayload, input.dossierId);
  if (!validation.ok) throw new ResearchSnapshotValidationError(validation);
  const response = await rpc({
    p_dossier_id: input.dossierId,
    p_expected_current_snapshot_id: input.expectedCurrentSnapshotId,
    p_canonical_payload: validation.snapshot,
  });
  if (response.error) throw response.error;
  return parsePersistenceResult(response.data);
}