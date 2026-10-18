import "server-only";

import { requireAdminSession } from "../../auth/admin-session";
import { createServerSupabaseClient } from "../../supabase/server";
import { runtimeBootstrap, type RuntimeEnvironmentObservation } from "./runtime-bootstrap-v3-0-3";
import type { D5Dossier, D5SnapshotRow } from "./pilotage-evidence-history-v3-1";
import {
  loadTrustedPilotageDecision,
  type ArtifactRow, type IdentityRow, type RunRow, type StageRow,
  type TrustedPilotagePort, TrustedLoaderError,
} from "./pilotage-trusted-loader-v3-1";

const MAX_RESULTS = 3000;
const MAX_BYTES = 35 * 1024 * 1024;
const UUID = /^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i;
const REPO = /^[a-zA-Z0-9_.-]+\/[a-zA-Z0-9_.-]+$/;

export function assertProductionUrl(): void {
  const value = process.env.NEXT_PUBLIC_SUPABASE_URL;
  if (!value) throw new TrustedLoaderError("SUPABASE_URL_MISSING");
  let url: URL;
  try { url = new URL(value); }
  catch { throw new TrustedLoaderError("SUPABASE_URL_INVALID"); }
  const expectedHost = runtimeBootstrap.production_environment.project_ref + ".supabase.co";
  if (url.protocol !== "https:" || url.hostname !== expectedHost ||
    url.port !== "" || url.username !== "" || url.password !== "" ||
    (url.pathname !== "/" && url.pathname !== "") ||
    url.search !== "" || url.hash !== "") {
    throw new TrustedLoaderError("SUPABASE_PRODUCTION_HOST_MISMATCH");
  }
}

function allowedArtifactRepositories(): Set<string> {
  // No implicit repository fallback. Allow-list and token must be configured server-side.
  const raw = process.env.OROTITAN_PRIVATE_GITHUB_REPOSITORIES;
  if (!raw) throw new TrustedLoaderError("IMMUTABLE_REPOSITORY_ALLOWLIST_MISSING");
  const list = raw.split(",").map(s => s.trim()).filter(Boolean);
  if (list.length === 0 || list.some(s => !REPO.test(s))) {
    throw new TrustedLoaderError("IMMUTABLE_REPOSITORY_ALLOWLIST_INVALID");
  }
  return new Set(list);
}

async function fetchImmutableGithubBytes(row: ArtifactRow): Promise<Uint8Array> {
  if (row.storage_backend !== "PRIVATE_GITHUB") {
    throw new TrustedLoaderError("UNSUPPORTED_IMMUTABLE_STORAGE_BACKEND");
  }
  const token = process.env.OROTITAN_PRIVATE_GITHUB_TOKEN;
  if (!token) throw new TrustedLoaderError("IMMUTABLE_READ_TOKEN_MISSING");
  const repository = row.github_repository;
  const path = row.github_path;
  const commit = row.github_commit_sha;
  if (!repository || !REPO.test(repository) || !allowedArtifactRepositories().has(repository)) {
    throw new TrustedLoaderError("IMMUTABLE_REPOSITORY_NOT_ALLOWED");
  }
  if (!path || path.length > 1024 || path.split("/").some(p =>
    !p || p === "." || p === ".." || p.includes("\\") || p.includes("\0"))) {
    throw new TrustedLoaderError("IMMUTABLE_PATH_INVALID");
  }
  if (!commit || !/^[a-f0-9]{40}$/.test(commit)) {
    throw new TrustedLoaderError("IMMUTABLE_COMMIT_INVALID");
  }
  const segments = repository.split("/");
  const address = "https://api.github.com/repos/" + segments.map(encodeURIComponent).join("/") +
    "/contents/" + path.split("/").map(encodeURIComponent).join("/") +
    "?ref=" + encodeURIComponent(commit);
  let response: Response;
  try {
    response = await fetch(address, {
      method: "GET", redirect: "error", cache: "no-store",
      signal: AbortSignal.timeout(15000),
      headers: {
        Accept: "application/vnd.github.raw+json",
        Authorization: "Bearer " + token,
        "X-GitHub-Api-Version": "2022-11-28",
      },
    });
  } catch { throw new TrustedLoaderError("IMMUTABLE_OBJECT_FETCH_FAILED"); }
  if (!response.ok) throw new TrustedLoaderError("IMMUTABLE_OBJECT_HTTP_ERROR", String(response.status));
  const length = Number(response.headers.get("content-length") ?? 0);
  if (length > MAX_BYTES) throw new TrustedLoaderError("IMMUTABLE_OBJECT_TOO_LARGE");
  const bytes = new Uint8Array(await response.arrayBuffer());
  if (bytes.byteLength > MAX_BYTES) throw new TrustedLoaderError("IMMUTABLE_OBJECT_TOO_LARGE");
  return bytes;
}

type DbClient = ReturnType<typeof createServerSupabaseClient>;

export function createServerPilotageReadPort(client: DbClient): TrustedPilotagePort {
  // No writes or RPCs are exposed through this port.
  return {
    async getRun(runId) {
      const { data, error } = await client.from("orotitan_runs")
        .select("run_id,issuer_id,security_id,dossier_id,baseline_snapshot_id,canonical_mode,run_type,run_status,current_stage,data_cutoff,process_version,pilotage_contract_version,contract_pins,contract_set_sha256,state_version")
        .eq("run_id", runId).maybeSingle();
      if (error) throw new TrustedLoaderError("REGISTRY_RUN_QUERY_FAILED");
      return data as RunRow | null;
    },
    async getStages(runId) {
      const { data, error } = await client.from("orotitan_run_stages")
        .select("run_id,stage_code,stage_revision,stage_contract_name,stage_contract_version,stage_contract_sha256,lifecycle_status,handoff_gate_name,handoff_gate_state,active_manifest_artifact_id,active_manifest_version,active_manifest_kind,blocker_summary,state_version")
        .eq("run_id", runId).order("stage_code").range(0, MAX_RESULTS);
      if (error) throw new TrustedLoaderError("REGISTRY_STAGES_QUERY_FAILED");
      return (data ?? []) as StageRow[];
    },
    async getArtifacts(runId) {
      const { data, error } = await client.from("orotitan_artifacts")
        .select("artifact_id,version,run_id,stage_code,artifact_type,authority_class,artifact_status,authority_state,availability_state,media_type,size_bytes,hash_algorithm,content_sha256,storage_backend,storage_uri,github_repository,github_path,github_commit_sha,github_blob_sha,manifest_artifact_id,manifest_version")
        .eq("run_id", runId).order("artifact_id").order("version").range(0, MAX_RESULTS);
      if (error) throw new TrustedLoaderError("REGISTRY_ARTIFACTS_QUERY_FAILED");
      return (data ?? []) as ArtifactRow[];
    },
    async getIssuer(issuerId) {
      const { data, error } = await client.from("issuers")
        .select("issuer_id,display_name,legal_name")
        .eq("issuer_id", issuerId).maybeSingle();
      if (error) throw new TrustedLoaderError("REGISTRY_ISSUER_QUERY_FAILED");
      return data as IdentityRow | null;
    },
    async getDossier(dossierId) {
      const { data, error } = await client.from("research_dossiers")
        .select("dossier_id,issuer_id,current_snapshot_id,active")
        .eq("dossier_id", dossierId).maybeSingle();
      if (error) throw new TrustedLoaderError("D5_DOSSIER_QUERY_FAILED");
      return data as D5Dossier | null;
    },
    async getSnapshot(snapshotId) {
      const { data, error } = await client.from("research_snapshots")
        .select("snapshot_id,dossier_id,issuer_id,security_id,report_id,execution_mode,data_cutoff,calculation_date,report_version,method_version,calculation_version,evidence_ledger_version,contract_version,schema_version,canonical_payload")
        .eq("snapshot_id", snapshotId).maybeSingle();
      if (error) throw new TrustedLoaderError("D5_SNAPSHOT_QUERY_FAILED");
      return data as D5SnapshotRow | null;
    },
    readImmutableBytes: fetchImmutableGithubBytes,
  };
}

/**
 * Server-only entrypoint. Caller must provide independently observed runtime
 * environment (name, region, health) from an authenticated infrastructure
 * source; the Supabase URL host is separately pinned here.
 * No route is exposed in D2, and no client-provided environment is trusted.
 */
export async function loadServerPilotageDecision(input: {
  runId: string;
  expectedRunStateVersion: number;
  expectedCurrentStageStateVersion: number;
  verifiedEnvironment: RuntimeEnvironmentObservation;
}) {
  await requireAdminSession();
  assertProductionUrl();
  if (!UUID.test(input.runId)) throw new TrustedLoaderError("RUN_ID_INVALID");
  if (!Number.isSafeInteger(input.expectedRunStateVersion) ||
    !Number.isSafeInteger(input.expectedCurrentStageStateVersion)) {
    throw new TrustedLoaderError("INVALID_EXPECTED_VERSIONS");
  }
  // This verifies the environment contract, not actual platform health.
  // The trusted caller must independently substantiate the observation.
  const port = createServerPilotageReadPort(createServerSupabaseClient());
  return loadTrustedPilotageDecision({
    port,
    environment: input.verifiedEnvironment,
    runId: input.runId,
    expectedRunStateVersion: input.expectedRunStateVersion,
    expectedCurrentStageStateVersion: input.expectedCurrentStageStateVersion,
  });
}
