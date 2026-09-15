import { createHash } from "node:crypto";
import runtimeBootstrapJson from "../../../contracts/orotitan-equity/v2/runtime/OROTITAN_RUNTIME_BOOTSTRAP_V2.0.1.json";
import contractPinPackJson from "../../../contracts/orotitan-equity/v2/contract-pin-pack-v2/OROTITAN_CONTRACT_PIN_PACK_V2.json";

export const OROTITAN_RUNTIME_BOOTSTRAP_VERSION = "2.0.1" as const;
export const OROTITAN_VERSION = "2.0" as const;

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

export type V2RunType = "INITIAL" | "REFRESH";

export type CreateRunRpcArgs = {
  p_creation_idempotency_key: string;
  p_issuer_id: string;
  p_entry_path: "IMPOSED_COMPANY";
  p_canonical_mode: "ANALYZE";
  p_run_type: V2RunType;
  p_data_cutoff: string;
  p_parent_run_id: null;
  p_baseline_snapshot_id: string | null;
  p_process_version: string;
  p_pilotage_contract_version: string;
  p_contract_pins: typeof contractPinPackJson.contract_pins;
  p_contract_set_sha256: string;
  p_request_fingerprint_sha256: string;
};

export type RunCreationPlan = {
  company: string;
  companyCommandName: string;
  issuerId: string;
  securityId: string;
  dossierId: string;
  runType: V2RunType;
  baselineSnapshotId: string | null;
  baselineContractVersion: string | null;
  dataCutoff: string;
  requiresIdentityBinding: boolean;
  rpc: "create_orotitan_run";
  rpcArgs: CreateRunRpcArgs;
};

export type PersistedRunObservation = {
  runId: string;
  stateVersion: number;
  runStatus: string;
  issuerId: string;
  securityId: string | null;
  dossierId: string | null;
  baselineSnapshotId: string | null;
  entryPath: string;
  canonicalMode: string;
  runType: string;
  dataCutoff: string;
  processVersion: string;
  pilotageContractVersion: string;
  contractSetSha256: string;
};

export type RunContextV2 = {
  format: "OROTITAN_RUN_CONTEXT_V2";
  contextVersion: "2.0.1";
  orotitanVersion: "2.0";
  runtimeBootstrapVersion: "2.0.1";
  runtimeBootstrapCanonicalSha256: string;
  contractSetSha256: string;
  productionProjectRef: string;
  company: string;
  companyCommandName: string;
  issuerId: string;
  securityId: string;
  dossierId: string;
  runId: string;
  runType: V2RunType;
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

export const runtimeBootstrap = runtimeBootstrapJson;
export const activeContractPinPack = contractPinPackJson;
export const runtimeBootstrapCanonicalSha256 = sha256Text(canonicalJson(runtimeBootstrapJson as JsonValue));

export function assertRuntimeBootstrapIntegrity(): void {
  if (runtimeBootstrap.format !== "OROTITAN_RUNTIME_BOOTSTRAP_V2"
      || runtimeBootstrap.version !== OROTITAN_RUNTIME_BOOTSTRAP_VERSION
      || runtimeBootstrap.orotitan_version !== OROTITAN_VERSION
      || runtimeBootstrap.production_status !== "ACTIVE_FOR_NEW_RUNS") {
    throw new RuntimeBootstrapError("RUNTIME_BOOTSTRAP_INVALID", "bootstrap identity/status mismatch");
  }
  if (runtimeBootstrap.authority_boundary.contract_set_sha256 !== activeContractPinPack.contract_set_sha256
      || runtimeBootstrap.authority_boundary.contract_pin_pack_version !== activeContractPinPack.version
      || runtimeBootstrap.authority_boundary.contract_pin_pack_modified !== false
      || runtimeBootstrap.authority_boundary.analytical_methodology_modified !== false) {
    throw new RuntimeBootstrapError("RUNTIME_AUTHORITY_MISMATCH", "runtime bootstrap does not reconcile to the active Contract Pin Pack V2");
  }
  if (activeContractPinPack.contract_pins.process.version !== "2.0"
      || activeContractPinPack.contract_pins.pilotage.version !== "2.0") {
    throw new RuntimeBootstrapError("RUNTIME_AUTHORITY_MISMATCH", "V2 process/pilotage pins are not active");
  }
}

export function assertProductionEnvironment(observed: RuntimeEnvironmentObservation): void {
  assertRuntimeBootstrapIntegrity();
  const expected = runtimeBootstrap.production_environment;
  if (observed.projectRef !== expected.project_ref
      || observed.projectName !== expected.project_name
      || observed.region !== expected.region
      || observed.status !== expected.required_status) {
    throw new RuntimeBootstrapError(
      expected.mismatch_code,
      `expected ${expected.project_name}/${expected.project_ref}/${expected.region}/${expected.required_status}; got ${observed.projectName}/${observed.projectRef}/${observed.region}/${observed.status}`,
    );
  }
}

export function determineRunType(identity: ResolvedCompanyIdentity): {
  runType: V2RunType;
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

function buildCreateDescriptor(identity: ResolvedCompanyIdentity, dataCutoff: string) {
  const route = determineRunType(identity);
  return {
    runtime_bootstrap_version: OROTITAN_RUNTIME_BOOTSTRAP_VERSION,
    issuer_id: identity.issuerId,
    entry_path: runtimeBootstrap.run_defaults.entry_path,
    canonical_mode: runtimeBootstrap.run_defaults.canonical_mode,
    run_type: route.runType,
    data_cutoff: dataCutoff,
    parent_run_id: null,
    baseline_snapshot_id: route.baselineSnapshotId,
    process_version: activeContractPinPack.contract_pins.process.version,
    pilotage_contract_version: activeContractPinPack.contract_pins.pilotage.version,
    contract_set_sha256: activeContractPinPack.contract_set_sha256,
  } as const;
}

export function buildRunCreationPlan(input: {
  environment: RuntimeEnvironmentObservation;
  identity: ResolvedCompanyIdentity;
  dataCutoff: string;
}): RunCreationPlan {
  assertProductionEnvironment(input.environment);
  assertIsoDate(input.dataCutoff);
  assertNonEmpty(input.identity.company, "company");
  assertNonEmpty(input.identity.companyCommandName, "companyCommandName");

  const route = determineRunType(input.identity);
  const descriptor = buildCreateDescriptor(input.identity, input.dataCutoff);
  const requestFingerprint = sha256Text(canonicalJson(descriptor as unknown as JsonValue));
  const creationIdempotencyKey = `orotitan:v2:create:${requestFingerprint}`;

  return {
    company: input.identity.company,
    companyCommandName: input.identity.companyCommandName,
    issuerId: input.identity.issuerId,
    securityId: input.identity.securityId,
    dossierId: input.identity.dossierId,
    runType: route.runType,
    baselineSnapshotId: route.baselineSnapshotId,
    baselineContractVersion: route.baselineContractVersion,
    dataCutoff: input.dataCutoff,
    requiresIdentityBinding: route.runType === "INITIAL",
    rpc: "create_orotitan_run",
    rpcArgs: {
      p_creation_idempotency_key: creationIdempotencyKey,
      p_issuer_id: input.identity.issuerId,
      p_entry_path: "IMPOSED_COMPANY",
      p_canonical_mode: "ANALYZE",
      p_run_type: route.runType,
      p_data_cutoff: input.dataCutoff,
      p_parent_run_id: null,
      p_baseline_snapshot_id: route.baselineSnapshotId,
      p_process_version: activeContractPinPack.contract_pins.process.version,
      p_pilotage_contract_version: activeContractPinPack.contract_pins.pilotage.version,
      p_contract_pins: activeContractPinPack.contract_pins,
      p_contract_set_sha256: activeContractPinPack.contract_set_sha256,
      p_request_fingerprint_sha256: requestFingerprint,
    },
  };
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
      p_idempotency_key: `orotitan:v2:bind:${fingerprint}`,
      p_request_fingerprint_sha256: fingerprint,
    },
  };
}

export function buildRunContextV2(plan: RunCreationPlan, persisted: PersistedRunObservation): RunContextV2 {
  const mismatch = [
    persisted.issuerId !== plan.issuerId && "issuer_id",
    persisted.securityId !== plan.securityId && "security_id",
    persisted.dossierId !== plan.dossierId && "dossier_id",
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
    format: "OROTITAN_RUN_CONTEXT_V2",
    contextVersion: "2.0.1",
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

export function assertRunContextV2(context: RunContextV2): void {
  assertRuntimeBootstrapIntegrity();
  const mismatch = [
    context.format !== "OROTITAN_RUN_CONTEXT_V2" && "format",
    context.contextVersion !== "2.0.1" && "contextVersion",
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

export function buildResearchStartPrompt(context: RunContextV2): string {
  assertRunContextV2(context);
  const baseline = context.baselineSnapshotId ?? "NULL";
  return `OROTITAN V2 — START RESEARCH\n\n${handoffEnvelopeLines().join("\n")}\nCOMPANY = ${context.company}\nRUN_ID = ${context.runId}\nCANONICAL_MODE = ${context.canonicalMode}\nRUN_TYPE = ${context.runType}\nREGISTRY_STAGE = RESEARCH\nEXECUTION_PHASE = RESEARCH\nDATA_CUTOFF = ${context.dataCutoff}\nBASELINE_SNAPSHOT_ID = ${baseline}\nEXPECTED_STAGE_CONTRACT = OROTITAN_RESEARCH_STAGE_CONTRACT_V2\nEXPECTED_STAGE_CONTRACT_VERSION = 2.0\nPUBLICATION_AUTHORIZED = NO\n\nBefore any external research, verify the exact production environment, persisted RUN_ID, identity, DATA_CUTOFF, baseline and Contract Pin Pack against the Registry. WRONG_ENVIRONMENT or any version / ID / hash mismatch is a hard stop. Start RESEARCH only through the controlled Registry path. Execute Research as evidence production only; do not emit final OQS, OVS, Investment Score or publication decision. Persist the required FINAL Research artifacts and manifest. If READY_FOR_DEEP_DIVE=YES, finish with the exact Fundamentals handoff prompt and nothing after it. Otherwise finish with the exact blocker-resolution prompt.\n\nDO NOT USE THIS BOOTSTRAP AS ANALYTICAL EVIDENCE.\nFAIL CLOSED ON VERSION / ID / STATUS / HASH MISMATCH.`;
}

export function renderPreflightCard(context: RunContextV2): string {
  assertRunContextV2(context);
  return `OROTITAN V2 PREFLIGHT\n\nAUTHORITY = PASS\nPRODUCTION DB = PASS\nIDENTITY = PASS\nBASELINE = PASS\nRUN TYPE = ${context.runType}\nRUN PERSISTENCE = PASS\n\nRUN_ID = ${context.runId}\nNEXT = RESEARCH`;
}
