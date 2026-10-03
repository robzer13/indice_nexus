import "server-only";

import { createServerSupabaseClient } from "../../supabase/server";
import {
  executeControlledOperation,
  type ArtifactRow,
  type ControlledBridgePort,
  type ControlledRequest,
  type ControlledResult,
  type DossierRow,
  type IssuerRow,
  type RpcResult,
  type RunRow,
  type SecurityRow,
  type StageCode,
  type StageRow,
} from "./chatgpt-supabase-bridge";

type ServerClient = ReturnType<typeof createServerSupabaseClient>;

function databaseError(prefix: string, error: { message: string } | null): Error {
  return new Error(`${prefix}: ${error?.message ?? "unknown Supabase error"}`);
}

function rows<T>(data: unknown): T[] {
  return Array.isArray(data) ? (data as T[]) : [];
}

function rpcObject(data: unknown): RpcResult {
  if (typeof data !== "object" || data === null || Array.isArray(data)) {
    throw new Error("Supabase RPC returned a non-object payload");
  }
  return data as RpcResult;
}

export function createServerControlledBridgePort(
  client: ServerClient = createServerSupabaseClient(),
): ControlledBridgePort {
  return {
    async listIssuers(): Promise<IssuerRow[]> {
      const { data, error } = await client
        .from("issuers")
        .select("issuer_id,display_name,legal_name");
      if (error) throw databaseError("LOAD issuers failed", error);
      return rows<IssuerRow>(data);
    },

    async listSecurities(): Promise<SecurityRow[]> {
      const { data, error } = await client
        .from("securities")
        .select("security_id,issuer_id,ticker,market_data_symbol,primary_listing,listing_status");
      if (error) throw databaseError("LOAD securities failed", error);
      return rows<SecurityRow>(data);
    },

    async listDossiers(issuerId: string): Promise<DossierRow[]> {
      const { data, error } = await client
        .from("research_dossiers")
        .select("dossier_id,issuer_id,current_snapshot_id,active")
        .eq("issuer_id", issuerId);
      if (error) throw databaseError("LOAD dossiers failed", error);
      return rows<DossierRow>(data);
    },

    async listRuns(issuerId: string): Promise<RunRow[]> {
      const { data, error } = await client
        .from("orotitan_runs")
        .select(
          "run_id,issuer_id,security_id,dossier_id,run_status,current_stage,run_type,canonical_mode,data_cutoff,contract_set_sha256,state_version,updated_at",
        )
        .eq("issuer_id", issuerId)
        .order("updated_at", { ascending: false });
      if (error) throw databaseError("LOAD runs failed", error);
      return rows<RunRow>(data);
    },

    async getRun(runId: string): Promise<RunRow | null> {
      const { data, error } = await client
        .from("orotitan_runs")
        .select(
          "run_id,issuer_id,security_id,dossier_id,run_status,current_stage,run_type,canonical_mode,data_cutoff,contract_set_sha256,state_version,updated_at",
        )
        .eq("run_id", runId)
        .maybeSingle();
      if (error) throw databaseError("LOAD run failed", error);
      return (data as RunRow | null) ?? null;
    },

    async getStage(runId: string, stage: StageCode): Promise<StageRow | null> {
      const { data, error } = await client
        .from("orotitan_run_stages")
        .select(
          "run_id,stage_code,stage_revision,lifecycle_status,handoff_gate_state,active_manifest_artifact_id,active_manifest_version,active_manifest_kind,blocker_summary,state_version",
        )
        .eq("run_id", runId)
        .eq("stage_code", stage)
        .maybeSingle();
      if (error) throw databaseError("LOAD stage failed", error);
      return (data as StageRow | null) ?? null;
    },

    async listArtifacts(runId: string): Promise<ArtifactRow[]> {
      const { data, error } = await client
        .from("orotitan_artifacts")
        .select(
          "artifact_id,version,run_id,stage_code,artifact_type,logical_name,authority_class,authority_state,artifact_status,availability_state,content_sha256,size_bytes,media_type,storage_backend,storage_uri,github_repository,github_path,github_commit_sha,github_blob_sha,supabase_bucket,supabase_object_path,manifest_artifact_id,manifest_version",
        )
        .eq("run_id", runId);
      if (error) throw databaseError("LOAD artifacts failed", error);
      return rows<ArtifactRow>(data);
    },

    async readSupabaseObject(bucket: string, objectPath: string): Promise<Uint8Array> {
      const { data, error } = await client.storage.from(bucket).download(objectPath);
      if (error || !data) throw databaseError("artifact download failed", error);
      return new Uint8Array(await data.arrayBuffer());
    },

    async resolveArtifact(args): Promise<RpcResult> {
      const { data, error } = await client.rpc("resolve_orotitan_artifact", args);
      if (error) throw databaseError("resolve_orotitan_artifact failed", error);
      return rpcObject(data);
    },

    async checkpointStage(args): Promise<RpcResult> {
      const { data, error } = await client.rpc("checkpoint_orotitan_stage", args);
      if (error) throw databaseError("checkpoint_orotitan_stage failed", error);
      return rpcObject(data);
    },

    async finalizeStage(args): Promise<RpcResult> {
      const { data, error } = await client.rpc("finalize_orotitan_stage", args);
      if (error) throw databaseError("finalize_orotitan_stage failed", error);
      return rpcObject(data);
    },

    async reopenStage(args): Promise<RpcResult> {
      const { data, error } = await client.rpc("reopen_orotitan_stage", args);
      if (error) throw databaseError("reopen_orotitan_stage failed", error);
      return rpcObject(data);
    },
  };
}

export async function executeServerControlledOperation(
  request: ControlledRequest,
): Promise<ControlledResult> {
  return executeControlledOperation(createServerControlledBridgePort(), request);
}
