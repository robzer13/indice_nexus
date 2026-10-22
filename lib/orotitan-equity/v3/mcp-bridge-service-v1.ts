import "server-only";
import { createClient } from "@supabase/supabase-js";

/**
 * MCP service: deliberately narrow, server-side bridge to the existing
 * independently verified Registry/Pilotage read path and quarantine storage.
 * No direct GPT model invocation, Registry writer, or stage transition.
 */
import { createServerSupabaseClient } from "../../supabase/server";
import { assertProductionUrl, createServerPilotageReadPort } from "./pilotage-trusted-loader-server";
import { loadTrustedPilotageDecision } from "./pilotage-trusted-loader-v3-1";
import { observeSupabaseEnvironment } from "./pilotage-operator-cockpit-server";
import { asOperatorBlockers, asOperatorDecision } from "./pilotage-operator-view-v3-1";
import {
  intakeObjectPath, intakePrefix, MANUAL_INTAKE_ALLOWED_RUN_STATUSES,
  MANUAL_INTAKE_BUCKET, MANUAL_INTAKE_MAX_FILES_PER_STAGE,
  parseIntakeObject, validateIntakeBytes,
} from "./manual-intake-v3-1";
import { McpBridgeError, mcpWriteEnabled, requireMcpRun } from "./mcp-bridge-access-v1";
import { delegatedPilotageStatus } from "./mcp-delegated-pilotage-v1";
import { buildRecoveryWorklist } from "./pilotage-blocker-recovery-v3-1";
import type { OperatorRun } from "./pilotage-operator-view-v3-1";

type DbClient = ReturnType<typeof createServerSupabaseClient>;
type Run = {
  run_id: string;
  issuer_id: string | null;
  security_id: string | null;
  run_status: string;
  current_stage: string | null;
  state_version: number;
  data_cutoff: string;
  process_version: string;
  pilotage_contract_version: string;
  canonical_mode: string;
};

function client(): DbClient {
  assertProductionUrl();
  return createServerSupabaseClient();
}

/**
 * MCP OAuth callers must use their own short-lived Supabase JWT under RLS.
 * Never use service_role or a server-side privileged fallback for this path.
 * The Registry tables currently deny authenticated SELECT (verified
 * 2026-10-20); it will remain blocked until narrow RLS+GRANT review.
 */
function delegatedClient(jwt: string, env: NodeJS.ProcessEnv): DbClient {
  assertProductionUrl();
  const key=env.SUPABASE_PUBLISHABLE_KEY;
  if (!key || !/^sb_publishable_[a-zA-Z0-9_-]{12,}$/.test(key) ||
    !/^[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+$/.test(jwt) ||
    jwt.length > 9000) {
    throw new McpBridgeError("DELEGATED_DATABASE_ACCESS_NOT_CONFIGURED");
  }
  return createClient(env.NEXT_PUBLIC_SUPABASE_URL!,key,{
    global:{headers:{Authorization:"Bearer "+jwt}},
    auth:{persistSession:false,autoRefreshToken:false,detectSessionInUrl:false},
  });
}

async function getRun(db: DbClient, runId: string): Promise<Run> {
  const { data, error } = await db.from("orotitan_runs")
    .select("run_id,issuer_id,security_id,run_status,current_stage,state_version,data_cutoff,process_version,pilotage_contract_version,canonical_mode")
    .eq("run_id", runId).maybeSingle();
  if (error) throw new McpBridgeError("REGISTRY_UNAVAILABLE");
  if (!data) throw new McpBridgeError("RUN_NOT_FOUND");
  return data as Run;
}

async function labels(db: DbClient, run: Run) {
  const issuer = run.issuer_id ? await db.from("issuers")
    .select("issuer_id,display_name").eq("issuer_id",run.issuer_id).maybeSingle() : null;
  const security = run.security_id ? await db.from("securities")
    .select("security_id,issuer_id,ticker,isin").eq("security_id",run.security_id).maybeSingle() : null;
  const issuerName = issuer && !issuer.error && issuer.data?.issuer_id === run.issuer_id ?
    issuer.data.display_name : null;
  const matches = security && !security.error && security.data?.issuer_id === run.issuer_id;
  return {
    issuerName,
    ticker: matches ? security!.data!.ticker : null,
    isin: matches ? security!.data!.isin : null,
  };
}

/** Registry observation: identifiers, blockers and stage are NOT a verified prompt. */
export async function getMcpRunContext(runIdRaw: string, env: NodeJS.ProcessEnv) {
  const runId = requireMcpRun(runIdRaw,env);
  return observeRunContext(client(),runId);
}

/** JWT already signature/issuer/resource/client validated at the HTTP gate. */
export async function getMcpRunContextDelegated(
  runIdRaw: string,jwt: string,env: NodeJS.ProcessEnv,
) {
  const runId=requireMcpRun(runIdRaw,env);
  return observeRunContext(delegatedClient(jwt,env),runId);
}

async function observeRunContext(db: DbClient,runId: string) {
  const run=await getRun(db,runId);
  const [identity, stageResponse] = await Promise.all([
    labels(db,run),
    db.from("orotitan_run_stages")
      .select("stage_code,lifecycle_status,contract_status_code,handoff_gate_name,handoff_gate_state,active_manifest_kind,blocker_summary,state_version")
      .eq("run_id",runId).eq("stage_code",run.current_stage ?? "").maybeSingle(),
  ]);
  if(stageResponse.error) throw new McpBridgeError("STAGE_OBSERVATION_UNAVAILABLE");
  const stage=stageResponse.data;
  return {
    trust: "REGISTRY_OBSERVATION_NOT_ARTIFACT_VERIFICATION" as const,
    runId, ...identity,
    runStatus: run.run_status,
    registryStage: run.current_stage,
    stageLifecycle: stage?.lifecycle_status ?? "UNKNOWN",
    contractStatusCode: stage?.contract_status_code ?? null,
    handoffGateName: stage?.handoff_gate_name ?? null,
    handoffGateState: stage?.handoff_gate_state ?? "UNKNOWN",
    activeManifestKind: stage?.active_manifest_kind ?? null,
    dataCutoff:run.data_cutoff, processVersion:run.process_version,
    pilotageVersion:run.pilotage_contract_version,mode:run.canonical_mode,
    runStateVersion:run.state_version,
    stageStateVersion:stage?.state_version ?? null,
    blockers:asOperatorBlockers(stage?.blocker_summary),
    nextStepAuthorized: false,
    instruction:"Observed status only. Use get_orotitan_next_prompt for independent evidence verification.",
  };
}

/** D29a: OAuth returns only an RLS-scoped blocker/status observation.
 * This MUST NOT call privileged D2 readers or reconstruct a verified prompt.
 */
export async function getMcpNextPromptDelegated(
  runIdRaw: string, jwt: string, env: NodeJS.ProcessEnv,
) {
  const observed=await getMcpRunContextDelegated(runIdRaw,jwt,env);
  return delegatedPilotageStatus(observed);
}

/**
 * OAuth-only, least-privileged blocker worklist. The caller's independently
 * verified short-lived JWT is used with the publishable key under RLS.
 * No service-role fallback, no quarantine read and no D2 verification.
 * Returns a DIAGNOSTIC PLAN, never accepted evidence or a stage transition.
 */
export async function getMcpRecoveryWorklistDelegated(
  runIdRaw: string, jwt: string, env: NodeJS.ProcessEnv,
) {
  const observed=await getMcpRunContextDelegated(runIdRaw,jwt,env);
  if(observed.processVersion!=="3.1" || observed.pilotageVersion!=="3.1") {
    throw new McpBridgeError("RECOVERY_UNSUPPORTED_CONTRACT_VERSION");
  }
  if(!observed.registryStage || !Number.isSafeInteger(observed.stageStateVersion) ||
    !Number.isSafeInteger(observed.runStateVersion)) {
    throw new McpBridgeError("RECOVERY_STAGE_STATE_UNAVAILABLE");
  }
  const asRun:OperatorRun={
    runId:observed.runId,
    issuerName:observed.issuerName,
    ticker:observed.ticker,
    isin:observed.isin,
    status:observed.runStatus,
    stage:observed.registryStage,
    mode:observed.mode,
    cutoff:observed.dataCutoff,
    stateVersion:observed.runStateVersion,
    createdAt:"",
    processVersion:observed.processVersion,
    pilotageVersion:observed.pilotageVersion,
    stageLifecycle:observed.stageLifecycle,
    contractStatusCode:observed.contractStatusCode,
    handoffGateState:observed.handoffGateState,
    activeManifestKind:observed.activeManifestKind,
    blockers:observed.blockers,
    supportsV31Verifier:true,
  };
  const worklist=buildRecoveryWorklist(asRun);
  return {
    status:"WORKLIST_ONLY_NOT_VERIFIED" as const,
    trust:"REGISTRY_OBSERVATION_NOT_ARTIFACT_VERIFICATION" as const,
    runId:observed.runId,
    stage:observed.registryStage,
    dataCutoff:observed.dataCutoff,
    runStateVersion:observed.runStateVersion,
    stageStateVersion:observed.stageStateVersion,
    contractStatusCode:observed.contractStatusCode,
    handoffGateState:observed.handoffGateState,
    activeManifestKind:observed.activeManifestKind,
    tasks:worklist.tasks,
    mayAdvance:false as const,
    mayPublish:false as const,
    mayMutate:false as const,
    verifiedResearchPerformed:false as const,
    nextStepAuthorized:false as const,
    publicationAuthorized:false as const,
    mutationPerformed:false as const,
    guidance:"Public research leads may be investigated separately, but no task is closed until independently verified, version-bound immutable evidence is admitted. No uploads, transitions, or publication are authorized by this tool.",
  };
}

/**
 * Reuses D2 trusted immutable bytes verification and D1 admission.
 * Re-observes production project and checks current CAS every call.
 */
export async function getMcpNextPrompt(runIdRaw: string, env: NodeJS.ProcessEnv) {
  const runId=requireMcpRun(runIdRaw,env);
  const db=client();
  const run=await getRun(db,runId);
  if (run.process_version!=="3.1" || run.pilotage_contract_version!=="3.1") {
    return { runId, status:"UNSUPPORTED_VERSION", prompt:null,
      guidance:"Use the contract-version-specific historical Pilotage path. Never rebind a historical run." };
  }
  if (run.run_status!=="ACTIVE" && run.run_status!=="READY_TO_PUBLISH") {
    return { runId, status:"RUN_NOT_EXECUTABLE_UPSTREAM", prompt:null,
      guidance:"The Registry run is not active/admissible. Resolve upstream state before requesting a handoff." };
  }
  const {data:stages,error}=await db.from("orotitan_run_stages")
    .select("stage_code,state_version")
    .eq("run_id",runId).eq("stage_code",run.current_stage ?? "").maybeSingle();
  if(error || !stages) throw new McpBridgeError("CURRENT_STAGE_MISSING");
  const environment=await observeSupabaseEnvironment();
  const decision=await loadTrustedPilotageDecision({
    port:createServerPilotageReadPort(db),
    environment,
    runId,
    expectedRunStateVersion:run.state_version,
    expectedCurrentStageStateVersion:stages.state_version,
  });
  const preview=asOperatorDecision(decision);
  return {
    runId,
    status:preview.outcome,
    phase:preview.phase,
    reasonCode:preview.reasonCode,
    nextAction:preview.nextAction,
    prompt:preview.verifiedPrompt,
    runStateVersion:preview.runStateVersion,
    stageStateVersion:preview.stageStateVersion,
    stateFingerprintSha256:preview.stateFingerprintSha256,
    trust:preview.trust,
    publicationAuthorized:false,
    mutationPerformed:false,
    guidance:preview.verifiedPrompt===null ?
      "No verified executable prompt is available. Do not advance to the next phase." :
      "Use this prompt manually in ChatGPT; it does not start or finish any Registry stage.",
  };
}

export async function listMcpSubmissions(runIdRaw: string, env: NodeJS.ProcessEnv) {
  const runId=requireMcpRun(runIdRaw,env);
  const db=client();
  const run=await getRun(db,runId);
  if(!run.current_stage) throw new McpBridgeError("CURRENT_STAGE_MISSING");
  const prefix=intakePrefix(runId,run.current_stage);
  const {data,error}=await db.storage.from(MANUAL_INTAKE_BUCKET).list(prefix,{
    limit:MANUAL_INTAKE_MAX_FILES_PER_STAGE,
    sortBy:{column:"name",order:"asc"},
  });
  if(error) throw new McpBridgeError("QUARANTINE_LIST_UNAVAILABLE");
  return {
    runId,registryStage:run.current_stage,
    status:"QUARANTINE_ONLY_NOT_REGISTRY" as const,
    submissions:(data ?? []).map(item=>parseIntakeObject(item,run.current_stage!))
      .filter((item):item is NonNullable<typeof item> => item!==null),
    guidance:"Received does not mean verified, registered or admitted; never advance a stage from this list.",
  };
}

/**
 * Optional stage-bound text handoff. Does NOT register an OroTitan artifact.
 * Default disabled; the MCP tool is not advertised unless write flag is set.
 */
export async function submitMcpWorkProduct(
  input: {runId:string; expectedStage:string; expectedRunStateVersion:number;
    filename:string; content:string}, env:NodeJS.ProcessEnv,
) {
  if(!mcpWriteEnabled(env)) throw new McpBridgeError("MCP_WRITE_DISABLED");
  const runId=requireMcpRun(input.runId,env);
  if(!Number.isSafeInteger(input.expectedRunStateVersion) ||
    input.expectedRunStateVersion < 1 || input.content.length>100_000) {
    throw new McpBridgeError("SUBMISSION_ARGUMENTS_INVALID");
  }
  const db=client();
  const run=await getRun(db,runId);
  if(run.current_stage!==input.expectedStage ||
    run.state_version!==input.expectedRunStateVersion) {
    throw new McpBridgeError("RUN_CHANGED_REFRESH");
  }
  if(!MANUAL_INTAKE_ALLOWED_RUN_STATUSES.includes(
    run.run_status as typeof MANUAL_INTAKE_ALLOWED_RUN_STATUSES[number])) {
    throw new McpBridgeError("RUN_NOT_EDITABLE");
  }
  const file=validateIntakeBytes(input.filename,new TextEncoder().encode(input.content));
  const prefix=intakePrefix(runId,run.current_stage!);
  const objectPath=intakeObjectPath(runId,run.current_stage!,file);
  const {data:objects,error:listError}=await db.storage.from(MANUAL_INTAKE_BUCKET).list(prefix,{
    limit:MANUAL_INTAKE_MAX_FILES_PER_STAGE+1,sortBy:{column:"name",order:"asc"},
  });
  if(listError) throw new McpBridgeError("QUARANTINE_LIST_UNAVAILABLE");
  if(objects?.some(x=>x.name===objectPath.slice(prefix.length+1))){
    return {runId, filename:file.fileName, sha256:file.sha256,
      status:"RECEIVED_NOT_VERIFIED" as const, duplicate:true, authority:"QUARANTINE_ONLY"};
  }
  if((objects?.length ?? 0)>=MANUAL_INTAKE_MAX_FILES_PER_STAGE){
    throw new McpBridgeError("QUARANTINE_LIMIT_REACHED");
  }
  const {error:uploadError}=await db.storage.from(MANUAL_INTAKE_BUCKET).upload(objectPath,file.content,{
    contentType:file.contentType,upsert:false,cacheControl:"0",
  });
  if(uploadError)throw new McpBridgeError("QUARANTINE_UPLOAD_FAILED");
  return {
    runId,filename:file.fileName,sha256:file.sha256,sizeBytes:file.sizeBytes,
    status:"RECEIVED_NOT_VERIFIED" as const,duplicate:false,
    authority:"QUARANTINE_ONLY",
    guidance:"No Registry artifact, manifest, attestation, gate or stage was created.",
  };
}
