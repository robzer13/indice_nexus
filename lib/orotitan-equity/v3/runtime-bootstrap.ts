import { createHash } from "node:crypto";
import runtimeBootstrapJson from "../../../contracts/orotitan-equity/v3/runtime/OROTITAN_RUNTIME_BOOTSTRAP_V3.0.2.json";
import contractPinPackJson from "../../../contracts/orotitan-equity/v3/contract-pin-pack-v3/OROTITAN_CONTRACT_PIN_PACK_V3_0_1.json";

export const OROTITAN_RUNTIME_BOOTSTRAP_VERSION = "3.0.2" as const;
export const OROTITAN_VERSION = "3.0" as const;

export type RuntimeEnvironmentObservation = {
  projectRef: string;
  projectName: string;
  region: string;
  status: string;
};

export type ResolvedCompanyIdentity = {
  company: string;
  companyCommandName: string;
  issuerId: string;
  securityId: string;
  dossierId: string;
  currentSnapshotId: string | null;
  currentSnapshotContractVersion: string | null;
};

export type V3RunType = "INITIAL" | "REFRESH";

export type HistoricalParentObservation = {
  runId: string;
  issuerId: string;
  securityId: string;
  dossierId: string;
  dataCutoff: string;
  contractSetSha256: string;
  stateVersion: number;
  runStatus: string;
  currentStage: string | null;
  publishedAt: string | null;
  cancelledAt: string | null;
};

export type CreateRunRpcArgs = {
  p_creation_idempotency_key: string;
  p_issuer_id: string;
  p_entry_path: "IMPOSED_COMPANY";
  p_canonical_mode: "ANALYZE";
  p_run_type: V3RunType;
  p_data_cutoff: string;
  p_parent_run_id: string | null;
  p_baseline_snapshot_id: string | null;
  p_process_version: string;
  p_pilotage_contract_version: string;
  p_contract_pins: typeof contractPinPackJson.contract_pins;
  p_contract_set_sha256: string;
  p_request_fingerprint_sha256: string;
  p_expected_parent_state_version?: number;
  p_expected_parent_run_status?: string;
  p_expected_parent_current_stage?: string;
  p_expected_parent_contract_set_sha256?: string;
  p_expected_parent_security_id?: string;
  p_expected_parent_dossier_id?: string;
};

export type RunCreationPlan = {
  company: string;
  companyCommandName: string;
  issuerId: string;
  securityId: string;
  dossierId: string;
  runType: V3RunType;
  parentRunId: string | null;
  baselineSnapshotId: string | null;
  baselineContractVersion: string | null;
  dataCutoff: string;
  creationReason: "NORMAL" | "METHODOLOGY_REPLAY_SUCCESSOR";
  requiresIdentityBinding: boolean;
  rpc: "create_orotitan_run" | "create_orotitan_methodology_successor_run";
  rpcArgs: CreateRunRpcArgs;
};

export type PersistedRunObservation = {
  runId: string;
  stateVersion: number;
  runStatus: string;
  issuerId: string;
  securityId: string | null;
  dossierId: string | null;
  parentRunId: string | null;
  baselineSnapshotId: string | null;
  entryPath: string;
  canonicalMode: string;
  runType: string;
  dataCutoff: string;
  processVersion: string;
  pilotageContractVersion: string;
  contractSetSha256: string;
};

export type RunContextV3 = {
  format: "OROTITAN_RUN_CONTEXT_V3";
  contextVersion: "3.0.2";
  orotitanVersion: "3.0";
  runtimeBootstrapVersion: "3.0.2";
  runtimeBootstrapCanonicalSha256: string;
  contractSetSha256: string;
  productionProjectRef: string;
  company: string;
  companyCommandName: string;
  issuerId: string;
  securityId: string;
  dossierId: string;
  runId: string;
  runType: V3RunType;
  parentRunId: string | null;
  canonicalMode: "ANALYZE";
  entryPath: "IMPOSED_COMPANY";
  dataCutoff: string;
  baselineSnapshotId: string | null;
  baselineContractVersion: string | null;
  currentRegistryStage: "RESEARCH";
  executionPhase: "RESEARCH";
  publicationAuthorized: false;
};

export class RuntimeBootstrapError extends Error {
  readonly code: string;

  constructor(code: string, message: string) {
    super(`${code}: ${message}`);
    this.code = code;
    this.name = "RuntimeBootstrapError";
  }
}

type JsonValue = null | boolean | number | string | JsonValue[] | { [key: string]: JsonValue };

function canonicalJson(value: JsonValue): string {
  if (value === null || typeof value !== "object") return JSON.stringify(value);
  if (Array.isArray(value)) return `[${value.map(canonicalJson).join(",")}]`;
  const entries = Object.entries(value).sort(([a], [b]) => a.localeCompare(b));
  return `{${entries.map(([key, child]) => `${JSON.stringify(key)}:${canonicalJson(child)}`).join(",")}}`;
}

function sha256Text(value: string): string {
  return createHash("sha256").update(value, "utf8").digest("hex");
}

function assertNonEmpty(value: string, label: string): void {
  if (!value.trim()) throw new RuntimeBootstrapError("IDENTITY_INCOMPLETE", `${label} is required`);
}

function assertIsoDate(value: string): void {
  if (!/^\d{4}-\d{2}-\d{2}$/.test(value)) {
    throw new RuntimeBootstrapError("INVALID_DATA_CUTOFF", "DATA_CUTOFF must be YYYY-MM-DD");
  }
}

function contractSetSha256(
  pins: typeof contractPinPackJson.contract_pins,
): string {
  const lines = Object.entries(pins)
    .map(([logicalName, pin]) => `${logicalName}|${pin.version}|${pin.content_sha256}`)
    .sort();
  return sha256Text(`${lines.join("\n")}\n`);
}

export const runtimeBootstrap = runtimeBootstrapJson;
export const activeContractPinPack = contractPinPackJson;
export const runtimeBootstrapCanonicalSha256 = sha256Text(canonicalJson(runtimeBootstrapJson as JsonValue));
export const activeContractSetComputedSha256 = contractSetSha256(contractPinPackJson.contract_pins);

export function assertRuntimeBootstrapIntegrity(): void {
  if (
    runtimeBootstrap.format !== "OROTITAN_RUNTIME_BOOTSTRAP_V3" ||
    runtimeBootstrap.version !== OROTITAN_RUNTIME_BOOTSTRAP_VERSION ||
    runtimeBootstrap.orotitan_version !== OROTITAN_VERSION ||
    runtimeBootstrap.production_status !== "ACTIVE_FOR_NEW_RUNS"
  ) {
    throw new RuntimeBootstrapError("RUNTIME_BOOTSTRAP_INVALID", "bootstrap identity/status mismatch");
  }

  if (
    runtimeBootstrap.authority_boundary.contract_set_sha256 !== activeContractPinPack.contract_set_sha256 ||
    activeContractSetComputedSha256 !== activeContractPinPack.contract_set_sha256 ||
    runtimeBootstrap.authority_boundary.contract_pin_pack_version !== activeContractPinPack.version ||
    runtimeBootstrap.authority_boundary.contract_pin_pack_modified !== true ||
    runtimeBootstrap.authority_boundary.analytical_methodology_modified !== true ||
    runtimeBootstrap.authority_boundary.methodology_delta !== "DCF_TIMING_ONLY"
  ) {
    throw new RuntimeBootstrapError(
      "RUNTIME_AUTHORITY_MISMATCH",
      "runtime bootstrap does not reconcile byte-independent Contract Set V3 authority",
    );
  }

  if (
    activeContractPinPack.contract_pins.process.version !== "3.0" ||
    activeContractPinPack.contract_pins.pilotage.version !== "3.0" ||
    activeContractPinPack.contract_pins.research_stage.version !== "2.0" ||
    activeContractPinPack.contract_pins.deep_dive_stage.version !== "3.0" ||
    activeContractPinPack.contract_pins.integration_stage.version !== "3.0" ||
    activeContractPinPack.contract_pins.dcf_timing.version !== "1.0"
  ) {
    throw new RuntimeBootstrapError("RUNTIME_AUTHORITY_MISMATCH", "V3 stage authority composition is not exact");
  }

  if (Object.keys(activeContractPinPack.contract_pins).length !== 14) {
    throw new RuntimeBootstrapError("RUNTIME_AUTHORITY_MISMATCH", "V3.0.1 Contract Set must contain exactly 14 logical pins");
  }
}

export function assertProductionEnvironment(observed: RuntimeEnvironmentObservation): void {
  assertRuntimeBootstrapIntegrity();
  const expected = runtimeBootstrap.production_environment;
  if (
    observed.projectRef !== expected.project_ref ||
    observed.projectName !== expected.project_name ||
    observed.region !== expected.region ||
    observed.status !== expected.required_status
  ) {
    throw new RuntimeBootstrapError(
      expected.mismatch_code,
      `expected ${expected.project_name}/${expected.project_ref}/${expected.region}/${expected.required_status}; got ${observed.projectName}/${observed.projectRef}/${observed.region}/${observed.status}`,
    );
  }
}

export function determineRunType(identity: ResolvedCompanyIdentity): {
  runType: V3RunType;
  baselineSnapshotId: string | null;
  baselineContractVersion: string | null;
} {
  assertNonEmpty(identity.issuerId, "issuerId");
  assertNonEmpty(identity.securityId, "securityId");
  assertNonEmpty(identity.dossierId, "dossierId");

  if (identity.currentSnapshotId === null) {
    if (identity.currentSnapshotContractVersion !== null) {
      throw new RuntimeBootstrapError("BASELINE_STATE_MISMATCH", "snapshot contract version exists without current snapshot");
    }
    return { runType: "INITIAL", baselineSnapshotId: null, baselineContractVersion: null };
  }

  if (!identity.currentSnapshotContractVersion) {
    throw new RuntimeBootstrapError("BASELINE_STATE_MISMATCH", "current snapshot exists without contract version");
  }

  return {
    runType: "REFRESH",
    baselineSnapshotId: identity.currentSnapshotId,
    baselineContractVersion: identity.currentSnapshotContractVersion,
  };
}

function buildPlan(input: {
  environment: RuntimeEnvironmentObservation;
  identity: ResolvedCompanyIdentity;
  runType: V3RunType;
  parentRunId: string | null;
  baselineSnapshotId: string | null;
  baselineContractVersion: string | null;
  dataCutoff: string;
  creationReason: "NORMAL" | "METHODOLOGY_REPLAY_SUCCESSOR";
  parentObservation?: HistoricalParentObservation;
}): RunCreationPlan {
  assertProductionEnvironment(input.environment);
  assertIsoDate(input.dataCutoff);
  assertNonEmpty(input.identity.company, "company");
  assertNonEmpty(input.identity.companyCommandName, "companyCommandName");

  const descriptor = {
    runtime_bootstrap_version: OROTITAN_RUNTIME_BOOTSTRAP_VERSION,
    issuer_id: input.identity.issuerId,
    entry_path: runtimeBootstrap.run_defaults.entry_path,
    canonical_mode: runtimeBootstrap.run_defaults.canonical_mode,
    run_type: input.runType,
    data_cutoff: input.dataCutoff,
    parent_run_id: input.parentRunId,
    baseline_snapshot_id: input.baselineSnapshotId,
    process_version: activeContractPinPack.contract_pins.process.version,
    pilotage_contract_version: activeContractPinPack.contract_pins.pilotage.version,
    contract_set_sha256: activeContractPinPack.contract_set_sha256,
    creation_reason: input.creationReason,
    parent_expected_state_version: input.parentObservation?.stateVersion ?? null,
    parent_expected_run_status: input.parentObservation?.runStatus ?? null,
    parent_expected_current_stage: input.parentObservation?.currentStage ?? null,
    parent_expected_contract_set_sha256: input.parentObservation?.contractSetSha256 ?? null,
  } as const;
  const requestFingerprint = sha256Text(canonicalJson(descriptor as unknown as JsonValue));
  const creationIdempotencyKey = `orotitan:v3:create:${requestFingerprint}`;

  return {
    company: input.identity.company,
    companyCommandName: input.identity.companyCommandName,
    issuerId: input.identity.issuerId,
    securityId: input.identity.securityId,
    dossierId: input.identity.dossierId,
    runType: input.runType,
    parentRunId: input.parentRunId,
    baselineSnapshotId: input.baselineSnapshotId,
    baselineContractVersion: input.baselineContractVersion,
    dataCutoff: input.dataCutoff,
    creationReason: input.creationReason,
    requiresIdentityBinding: input.runType === "INITIAL",
    rpc: input.creationReason === "METHODOLOGY_REPLAY_SUCCESSOR"
      ? "create_orotitan_methodology_successor_run"
      : "create_orotitan_run",
    rpcArgs: {
      p_creation_idempotency_key: creationIdempotencyKey,
      p_issuer_id: input.identity.issuerId,
      p_entry_path: "IMPOSED_COMPANY",
      p_canonical_mode: "ANALYZE",
      p_run_type: input.runType,
      p_data_cutoff: input.dataCutoff,
      p_parent_run_id: input.parentRunId,
      p_baseline_snapshot_id: input.baselineSnapshotId,
      p_process_version: activeContractPinPack.contract_pins.process.version,
      p_pilotage_contract_version: activeContractPinPack.contract_pins.pilotage.version,
      p_contract_pins: activeContractPinPack.contract_pins,
      p_contract_set_sha256: activeContractPinPack.contract_set_sha256,
      p_request_fingerprint_sha256: requestFingerprint,
      ...(input.parentObservation
        ? {
            p_expected_parent_state_version: input.parentObservation.stateVersion,
            p_expected_parent_run_status: input.parentObservation.runStatus,
            p_expected_parent_current_stage: input.parentObservation.currentStage ?? "",
            p_expected_parent_contract_set_sha256: input.parentObservation.contractSetSha256,
            p_expected_parent_security_id: input.parentObservation.securityId,
            p_expected_parent_dossier_id: input.parentObservation.dossierId,
          }
        : {}),
    },
  };
}

export function buildRunCreationPlan(input: {
  environment: RuntimeEnvironmentObservation;
  identity: ResolvedCompanyIdentity;
  dataCutoff: string;
}): RunCreationPlan {
  const route = determineRunType(input.identity);
  return buildPlan({
    environment: input.environment,
    identity: input.identity,
    runType: route.runType,
    parentRunId: null,
    baselineSnapshotId: route.baselineSnapshotId,
    baselineContractVersion: route.baselineContractVersion,
    dataCutoff: input.dataCutoff,
    creationReason: "NORMAL",
  });
}

export function buildMethodologyReplaySuccessorPlan(input: {
  environment: RuntimeEnvironmentObservation;
  identity: ResolvedCompanyIdentity;
  parent: HistoricalParentObservation;
}): RunCreationPlan {
  assertProductionEnvironment(input.environment);
  assertIsoDate(input.parent.dataCutoff);

  if (
    input.identity.issuerId !== input.parent.issuerId ||
    input.identity.securityId !== input.parent.securityId ||
    input.identity.dossierId !== input.parent.dossierId
  ) {
    throw new RuntimeBootstrapError("SUCCESSOR_IDENTITY_MISMATCH", "successor identity must equal the historical parent identity");
  }
  if (input.identity.currentSnapshotId !== null || input.identity.currentSnapshotContractVersion !== null) {
    throw new RuntimeBootstrapError(
      "SUCCESSOR_BASELINE_MISMATCH",
      "pure INITIAL methodology replay requires no canonical baseline snapshot",
    );
  }
  if (!Number.isInteger(input.parent.stateVersion) || input.parent.stateVersion < 1) {
    throw new RuntimeBootstrapError("SUCCESSOR_PARENT_STATE_VERSION_MISMATCH", "parent state_version must be a positive integer");
  }
  if (input.parent.runStatus !== "ACTIVE") {
    throw new RuntimeBootstrapError("SUCCESSOR_PARENT_STATUS_MISMATCH", "methodology replay requires ACTIVE parent");
  }
  if (input.parent.currentStage !== "DEEP_DIVE") {
    throw new RuntimeBootstrapError("SUCCESSOR_PARENT_STAGE_MISMATCH", "methodology replay requires parent at DEEP_DIVE");
  }
  if (input.parent.publishedAt !== null || input.parent.cancelledAt !== null) {
    throw new RuntimeBootstrapError("SUCCESSOR_PARENT_TERMINAL", "published/cancelled parent is not admissible for this route");
  }
  if (input.parent.contractSetSha256 === activeContractPinPack.contract_set_sha256) {
    throw new RuntimeBootstrapError("SUCCESSOR_NOT_REQUIRED", "parent is already pinned to the active V3.0.1 Contract Set");
  }

  return buildPlan({
    environment: input.environment,
    identity: input.identity,
    runType: "INITIAL",
    parentRunId: input.parent.runId,
    baselineSnapshotId: null,
    baselineContractVersion: null,
    dataCutoff: input.parent.dataCutoff,
    creationReason: "METHODOLOGY_REPLAY_SUCCESSOR",
    parentObservation: input.parent,
  });
}

export function buildIdentityBindingRpcArgs(plan: RunCreationPlan, input: {
  runId: string;
  expectedStateVersion: number;
}): {
  rpc: "bind_orotitan_run_identity";
  args: {
    p_run_id: string;
    p_expected_state_version: number;
    p_security_id: string;
    p_dossier_id: string;
    p_upstream_discovery_artifact_id: null;
    p_upstream_discovery_artifact_version: null;
    p_idempotency_key: string;
    p_request_fingerprint_sha256: string;
  };
} {
  if (!plan.requiresIdentityBinding) {
    throw new RuntimeBootstrapError("IDENTITY_ALREADY_BOUND_BY_BASELINE", "REFRESH identity is derived from the baseline snapshot");
  }

  const descriptor = {
    run_id: input.runId,
    security_id: plan.securityId,
    dossier_id: plan.dossierId,
    upstream_discovery_artifact_id: null,
    upstream_discovery_artifact_version: null,
  };
  const fingerprint = sha256Text(canonicalJson(descriptor as JsonValue));

  return {
    rpc: "bind_orotitan_run_identity",
    args: {
      p_run_id: input.runId,
      p_expected_state_version: input.expectedStateVersion,
      p_security_id: plan.securityId,
      p_dossier_id: plan.dossierId,
      p_upstream_discovery_artifact_id: null,
      p_upstream_discovery_artifact_version: null,
      p_idempotency_key: `orotitan:v3:bind:${fingerprint}`,
      p_request_fingerprint_sha256: fingerprint,
    },
  };
}

export function buildRunContextV3(plan: RunCreationPlan, persisted: PersistedRunObservation): RunContextV3 {
  const mismatch = [
    persisted.issuerId !== plan.issuerId && "issuer_id",
    persisted.securityId !== plan.securityId && "security_id",
    persisted.dossierId !== plan.dossierId && "dossier_id",
    persisted.parentRunId !== plan.parentRunId && "parent_run_id",
    persisted.baselineSnapshotId !== plan.baselineSnapshotId && "baseline_snapshot_id",
    persisted.entryPath !== "IMPOSED_COMPANY" && "entry_path",
    persisted.canonicalMode !== "ANALYZE" && "canonical_mode",
    persisted.runType !== plan.runType && "run_type",
    persisted.dataCutoff !== plan.dataCutoff && "data_cutoff",
    persisted.processVersion !== activeContractPinPack.contract_pins.process.version && "process_version",
    persisted.pilotageContractVersion !== activeContractPinPack.contract_pins.pilotage.version && "pilotage_contract_version",
    persisted.contractSetSha256 !== activeContractPinPack.contract_set_sha256 && "contract_set_sha256",
  ].filter(Boolean) as string[];

  if (mismatch.length > 0) {
    throw new RuntimeBootstrapError("PERSISTED_RUN_MISMATCH", mismatch.join(", "));
  }
  if (!persisted.runId || persisted.stateVersion < 1) {
    throw new RuntimeBootstrapError("RUN_PERSISTENCE_FAIL", "persisted run identity/state is invalid");
  }

  return {
    format: "OROTITAN_RUN_CONTEXT_V3",
    contextVersion: "3.0.1",
    orotitanVersion: OROTITAN_VERSION,
    runtimeBootstrapVersion: OROTITAN_RUNTIME_BOOTSTRAP_VERSION,
    runtimeBootstrapCanonicalSha256,
    contractSetSha256: activeContractPinPack.contract_set_sha256,
    productionProjectRef: runtimeBootstrap.production_environment.project_ref,
    company: plan.company,
    companyCommandName: plan.companyCommandName,
    issuerId: plan.issuerId,
    securityId: plan.securityId,
    dossierId: plan.dossierId,
    runId: persisted.runId,
    runType: plan.runType,
    parentRunId: plan.parentRunId,
    canonicalMode: "ANALYZE",
    entryPath: "IMPOSED_COMPANY",
    dataCutoff: plan.dataCutoff,
    baselineSnapshotId: plan.baselineSnapshotId,
    baselineContractVersion: plan.baselineContractVersion,
    currentRegistryStage: "RESEARCH",
    executionPhase: "RESEARCH",
    publicationAuthorized: false,
  };
}

export function assertRunContextV3(context: RunContextV3): void {
  assertRuntimeBootstrapIntegrity();
  const mismatch = [
    context.format !== "OROTITAN_RUN_CONTEXT_V3" && "format",
    context.contextVersion !== "3.0.1" && "contextVersion",
    context.orotitanVersion !== OROTITAN_VERSION && "orotitanVersion",
    context.runtimeBootstrapVersion !== OROTITAN_RUNTIME_BOOTSTRAP_VERSION && "runtimeBootstrapVersion",
    context.runtimeBootstrapCanonicalSha256 !== runtimeBootstrapCanonicalSha256 && "runtimeBootstrapCanonicalSha256",
    context.contractSetSha256 !== activeContractPinPack.contract_set_sha256 && "contractSetSha256",
    context.productionProjectRef !== runtimeBootstrap.production_environment.project_ref && "productionProjectRef",
    context.publicationAuthorized !== false && "publicationAuthorized",
  ].filter(Boolean) as string[];
  if (mismatch.length > 0) throw new RuntimeBootstrapError("RUN_CONTEXT_MISMATCH", mismatch.join(", "));
}

export function handoffEnvelopeLines(): string[] {
  assertRuntimeBootstrapIntegrity();
  return [
    `OROTITAN_VERSION = ${OROTITAN_VERSION}`,
    `RUNTIME_BOOTSTRAP_VERSION = ${OROTITAN_RUNTIME_BOOTSTRAP_VERSION}`,
    `RUNTIME_BOOTSTRAP_SHA256 = ${runtimeBootstrapCanonicalSha256}`,
    `CONTRACT_SET_SHA256 = ${activeContractPinPack.contract_set_sha256}`,
    `PRODUCTION_PROJECT_REF = ${runtimeBootstrap.production_environment.project_ref}`,
  ];
}

export function buildResearchStartPrompt(context: RunContextV3): string {
  assertRunContextV3(context);
  const baseline = context.baselineSnapshotId ?? "NULL";
  const parent = context.parentRunId ?? "NULL";
  const researchPin = activeContractPinPack.contract_pins.research_stage;
  const successorCas = context.parentRunId
    ? "\nSUCCESSOR_PARENT_CAS = TRANSACTIONAL_REQUIRED\nRESEARCH_SCOPE = SAME_CUTOFF_NON_VALUATION_REVALIDATION_ONLY\nFIRST_ANALYTICAL_PHASE = VALUATION_AFTER_REVALIDATION"
    : "";
  return `OROTITAN V3 — START RESEARCH\n\n${handoffEnvelopeLines().join("\n")}\nCOMPANY = ${context.company}\nRUN_ID = ${context.runId}\nPARENT_RUN_ID = ${parent}\nCANONICAL_MODE = ${context.canonicalMode}\nRUN_TYPE = ${context.runType}\nREGISTRY_STAGE = RESEARCH\nEXECUTION_PHASE = RESEARCH\nDATA_CUTOFF = ${context.dataCutoff}\nBASELINE_SNAPSHOT_ID = ${baseline}\nEXPECTED_STAGE_CONTRACT = ${researchPin.name}\nEXPECTED_STAGE_CONTRACT_VERSION = ${researchPin.version}\nPUBLICATION_AUTHORIZED = NO${successorCas}\n\nBefore any external research, verify the exact production environment, persisted RUN_ID, parent lineage where present, identity, DATA_CUTOFF, baseline and V3 Contract Pin Pack against the Registry. WRONG_ENVIRONMENT or any version / ID / hash mismatch is a hard stop. Start RESEARCH only through the controlled Registry path. For a methodology-replay successor, Research is limited to exact same-cutoff revalidation and must not introduce post-cutoff evidence or new fundamental judgment. For an ordinary run, Research remains evidence production under the pinned Research Stage Contract. Do not perform Valuation or construct an economic-share-count bound in this bootstrap.\n\nDO NOT USE THIS BOOTSTRAP AS ANALYTICAL EVIDENCE.\nFAIL CLOSED ON VERSION / ID / STATUS / HASH MISMATCH.`;
}

export function renderPreflightCard(context: RunContextV3): string {
  assertRunContextV3(context);
  return `OROTITAN V3 PREFLIGHT\n\nAUTHORITY = PASS\nPRODUCTION DB = PASS\nIDENTITY = PASS\nLINEAGE = PASS\nBASELINE = PASS\nRUN TYPE = ${context.runType}\nRUN PERSISTENCE = PASS\n\nRUN_ID = ${context.runId}\nNEXT = RESEARCH`;
}
