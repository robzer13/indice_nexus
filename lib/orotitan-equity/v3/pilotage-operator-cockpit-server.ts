import "server-only";

import { requireAdminSession } from "../../auth/admin-session";
import { createServerSupabaseClient } from "../../supabase/server";
import {
  assertProductionEnvironment, runtimeBootstrap,
  type RuntimeEnvironmentObservation,
} from "./runtime-bootstrap-v3-0-3";
import {
  assertProductionUrl, createServerPilotageReadPort, loadServerPilotageDecision,
} from "./pilotage-trusted-loader-server";
import { TrustedLoaderError } from "./pilotage-trusted-loader-v3-1";
import {
  asOperatorDecision, asOperatorRun, safeVerifierReason,
  type OperatorDecision, type OperatorRun, type RunSummary,
} from "./pilotage-operator-view-v3-1";

const UUID = /^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i;
const LIMIT = 30;
const PROJECT_REF = runtimeBootstrap.production_environment.project_ref;

export type OperatorCockpitData = {
  runs: OperatorRun[];
  selectedRunId: string | null;
  preview: OperatorDecision | null;
  previewState: "NOT_SELECTED" | "VERIFIED" | "BLOCKED";
  blockerCode: string | null;
  publicationAuthorized: false;
  mutationPerformed: false;
};

/**
 * Requires an independently observed project status, not a caller-controlled
 * or locally hard-coded object. A scoped, read-only Management API token is
 * optional for Registry listing but required before a D2-D8 trust decision.
 */
export async function observeSupabaseEnvironment(): Promise<RuntimeEnvironmentObservation> {
  const token = process.env.SUPABASE_MANAGEMENT_READ_TOKEN;
  if (!token || token.trim().length < 20) {
    throw new TrustedLoaderError("ENVIRONMENT_OBSERVATION_TOKEN_MISSING");
  }
  let response: Response;
  try {
    response = await fetch("https://api.supabase.com/v1/projects/" + PROJECT_REF, {
      method: "GET",
      redirect: "error",
      cache: "no-store",
      signal: AbortSignal.timeout(8000),
      headers: {
        Authorization: "Bearer " + token,
        Accept: "application/json",
      },
    });
  } catch {
    throw new TrustedLoaderError("ENVIRONMENT_OBSERVATION_FETCH_FAILED");
  }
  if (!response.ok) {
    throw new TrustedLoaderError("ENVIRONMENT_OBSERVATION_HTTP_REJECTED");
  }
  let raw: unknown;
  try { raw = await response.json(); }
  catch { throw new TrustedLoaderError("ENVIRONMENT_OBSERVATION_INVALID_JSON"); }
  if (!raw || typeof raw !== "object" || Array.isArray(raw)) {
    throw new TrustedLoaderError("ENVIRONMENT_OBSERVATION_INVALID_JSON");
  }
  const project = raw as Record<string, unknown>;
  if (project.ref !== PROJECT_REF || typeof project.name !== "string" ||
    typeof project.region !== "string" || typeof project.status !== "string") {
    throw new TrustedLoaderError("ENVIRONMENT_OBSERVATION_INCOMPLETE");
  }
  const observed = {
    projectRef: project.ref, projectName: project.name,
    region: project.region, status: project.status,
  };
  try { assertProductionEnvironment(observed); }
  catch { throw new TrustedLoaderError("ENVIRONMENT_OBSERVATION_NOT_ADMITTED"); }
  return observed;
}

/**
 * Read-only admin product. Listing is raw Registry observation, never evidence
 * admission; selecting a run invokes the existing independent D2-D8 verifier.
 */
export async function loadAdminPilotageCockpit(selectedRunId: string | null): Promise<OperatorCockpitData> {
  await requireAdminSession();
  if (selectedRunId !== null && !UUID.test(selectedRunId)) {
    return { runs: [], selectedRunId: null, preview: null,
      previewState: "BLOCKED", blockerCode: "RUN_ID_INVALID",
      publicationAuthorized: false, mutationPerformed: false };
  }

  assertProductionUrl();
  const client = createServerSupabaseClient();
  const { data, error } = await client.from("orotitan_runs")
    .select("run_id,run_status,current_stage,canonical_mode,data_cutoff,issuer_id,state_version,created_at")
    .order("created_at", { ascending: false }).limit(LIMIT);
  if (error) {
    return { runs: [], selectedRunId, preview: null,
      previewState: "BLOCKED", blockerCode: "REGISTRY_LIST_QUERY_FAILED",
      publicationAuthorized: false, mutationPerformed: false };
  }
  const runs = ((data ?? []) as RunSummary[]).map(asOperatorRun);
  if (!selectedRunId) {
    return { runs, selectedRunId: null, preview: null,
      previewState: "NOT_SELECTED", blockerCode: null,
      publicationAuthorized: false, mutationPerformed: false };
  }
  try {
    const port = createServerPilotageReadPort(client);
    const [run, stages] = await Promise.all([
      port.getRun(selectedRunId), port.getStages(selectedRunId),
    ]);
    if (!run) throw new TrustedLoaderError("RUN_NOT_FOUND");
    const current = stages.find(s => s.stage_code === run.current_stage);
    if (!current) throw new TrustedLoaderError("CURRENT_STAGE_MISSING");
    // Management API observation is independent of Registry and of the URL pin.
    const verifiedEnvironment = await observeSupabaseEnvironment();
    const decision = await loadServerPilotageDecision({
      runId: selectedRunId, expectedRunStateVersion: run.state_version,
      expectedCurrentStageStateVersion: current.state_version,
      verifiedEnvironment,
    });
    const preview = asOperatorDecision(decision);
    return { runs, selectedRunId, preview, previewState: "VERIFIED",
      blockerCode: preview.outcome === "BLOCKED" ? preview.reasonCode : null,
      publicationAuthorized: false, mutationPerformed: false };
  } catch (error) {
    return { runs, selectedRunId, preview: null, previewState: "BLOCKED",
      blockerCode: safeVerifierReason(error), publicationAuthorized: false, mutationPerformed: false };
  }
}
