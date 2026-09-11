import "server-only";
import { createServerSupabaseClient } from "@/lib/supabase/server";
import { persistValidatedResearchSnapshot as persistCore } from "./persistence-core";
import type { PersistValidatedResearchSnapshotInput, PersistenceResult, SnapshotRpc } from "./persistence-core";

export type { PersistValidatedResearchSnapshotInput, PersistenceResult, SnapshotRpc } from "./persistence-core";
export { ResearchSnapshotValidationError } from "./persistence-core";

function defaultRpc(): SnapshotRpc {
  const supabase = createServerSupabaseClient();
  return async (args) => supabase.rpc("persist_orotitan_research_snapshot", args);
}

export async function persistValidatedResearchSnapshot(
  input: PersistValidatedResearchSnapshotInput,
  rpc?: SnapshotRpc,
): Promise<PersistenceResult> {
  return persistCore(input, rpc ?? defaultRpc());
}