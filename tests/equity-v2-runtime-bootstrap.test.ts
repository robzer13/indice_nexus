import assert from "node:assert/strict";
import test from "node:test";
import { buildResearchToFundamentals } from "../lib/orotitan-equity/v2/handoff";
import {
  activeContractPinPack,
  assertProductionEnvironment,
  assertRuntimeBootstrapIntegrity,
  buildIdentityBindingRpcArgs,
  buildResearchStartPrompt,
  buildRunContextV2,
  buildRunCreationPlan,
  determineRunType,
  renderPreflightCard,
  runtimeBootstrap,
  runtimeBootstrapCanonicalSha256,
  type PersistedRunObservation,
  type ResolvedCompanyIdentity,
  type RuntimeEnvironmentObservation,
} from "../lib/orotitan-equity/v2/runtime-bootstrap";

const environment: RuntimeEnvironmentObservation = {
  projectRef: "cugpgtzygqqlxetyeven",
  projectName: "orotitan-screener",
  region: "eu-west-3",
  status: "ACTIVE_HEALTHY",
};

const initialIdentity: ResolvedCompanyIdentity = {
  company: "Example, Inc.",
  companyCommandName: "EXAMPLE",
  issuerId: "10000000-0000-4000-8000-000000000001",
  securityId: "10000000-0000-4000-8000-000000000002",
  dossierId: "10000000-0000-4000-8000-000000000003",
  currentSnapshotId: null,
  currentSnapshotContractVersion: null,
};

const refreshIdentity: ResolvedCompanyIdentity = {
  ...initialIdentity,
  company: "Qualys, Inc.",
  companyCommandName: "QUALYS",
  currentSnapshotId: "10000000-0000-4000-8000-000000000004",
  currentSnapshotContractVersion: "04_SCREENER_SCHEMA_V1",
};

function persisted(plan: ReturnType<typeof buildRunCreationPlan>): PersistedRunObservation {
  return {
    runId: "20000000-0000-4000-8000-000000000001",
    stateVersion: 2,
    runStatus: "CREATED",
    issuerId: plan.issuerId,
    securityId: plan.securityId,
    dossierId: plan.dossierId,
    baselineSnapshotId: plan.baselineSnapshotId,
    entryPath: "IMPOSED_COMPANY",
    canonicalMode: "ANALYZE",
    runType: plan.runType,
    dataCutoff: plan.dataCutoff,
    processVersion: activeContractPinPack.contract_pins.process.version,
    pilotageContractVersion: activeContractPinPack.contract_pins.pilotage.version,
    contractSetSha256: activeContractPinPack.contract_set_sha256,
  };
}

test("runtime bootstrap reconciles to the active V2 Contract Pin Pack", () => {
  assert.doesNotThrow(() => assertRuntimeBootstrapIntegrity());
  assert.equal(runtimeBootstrap.authority_boundary.contract_set_sha256, activeContractPinPack.contract_set_sha256);
  assert.equal(Object.keys(activeContractPinPack.contract_pins).length, 13);
  assert.match(runtimeBootstrapCanonicalSha256, /^[0-9a-f]{64}$/);
});

test("production environment is an exact allowlist", () => {
  assert.doesNotThrow(() => assertProductionEnvironment(environment));
  assert.throws(
    () => assertProductionEnvironment({ ...environment, projectName: "orotitan-db" }),
    /WRONG_ENVIRONMENT/,
  );
  assert.throws(
    () => assertProductionEnvironment({ ...environment, projectRef: "wrong-project-ref" }),
    /WRONG_ENVIRONMENT/,
  );
});

test("INITIAL is selected only when no canonical snapshot exists", () => {
  assert.deepEqual(determineRunType(initialIdentity), {
    runType: "INITIAL",
    baselineSnapshotId: null,
    baselineContractVersion: null,
  });
});

test("existing V1 canonical snapshot deterministically routes to REFRESH", () => {
  assert.deepEqual(determineRunType(refreshIdentity), {
    runType: "REFRESH",
    baselineSnapshotId: refreshIdentity.currentSnapshotId,
    baselineContractVersion: "04_SCREENER_SCHEMA_V1",
  });
});

test("run type rejects inconsistent baseline state", () => {
  assert.throws(
    () => determineRunType({ ...initialIdentity, currentSnapshotContractVersion: "04_SCREENER_SCHEMA_V1" }),
    /BASELINE_STATE_MISMATCH/,
  );
  assert.throws(
    () => determineRunType({ ...refreshIdentity, currentSnapshotContractVersion: null }),
    /BASELINE_STATE_MISMATCH/,
  );
});

test("run creation plan carries exact V2 authority and controlled RPC arguments", () => {
  const plan = buildRunCreationPlan({ environment, identity: refreshIdentity, dataCutoff: "2026-09-15" });
  assert.equal(plan.rpc, "create_orotitan_run");
  assert.equal(plan.runType, "REFRESH");
  assert.equal(plan.rpcArgs.p_entry_path, "IMPOSED_COMPANY");
  assert.equal(plan.rpcArgs.p_canonical_mode, "ANALYZE");
  assert.equal(plan.rpcArgs.p_baseline_snapshot_id, refreshIdentity.currentSnapshotId);
  assert.equal(plan.rpcArgs.p_contract_set_sha256, "1116ca12dce2d30ddbb4b699945d92ae235c940cbf69d104a01019fc21efbf5e");
  assert.equal(Object.keys(plan.rpcArgs.p_contract_pins).length, 13);
  assert.match(plan.rpcArgs.p_request_fingerprint_sha256, /^[0-9a-f]{64}$/);
  assert.ok(plan.rpcArgs.p_creation_idempotency_key.endsWith(plan.rpcArgs.p_request_fingerprint_sha256));
});

test("run creation idempotency is deterministic and cutoff-sensitive", () => {
  const a = buildRunCreationPlan({ environment, identity: refreshIdentity, dataCutoff: "2026-09-15" });
  const b = buildRunCreationPlan({ environment, identity: refreshIdentity, dataCutoff: "2026-09-15" });
  const c = buildRunCreationPlan({ environment, identity: refreshIdentity, dataCutoff: "2026-09-16" });
  assert.equal(a.rpcArgs.p_creation_idempotency_key, b.rpcArgs.p_creation_idempotency_key);
  assert.equal(a.rpcArgs.p_request_fingerprint_sha256, b.rpcArgs.p_request_fingerprint_sha256);
  assert.notEqual(a.rpcArgs.p_request_fingerprint_sha256, c.rpcArgs.p_request_fingerprint_sha256);
});

test("INITIAL requires explicit identity binding while REFRESH derives identity from baseline", () => {
  const initialPlan = buildRunCreationPlan({ environment, identity: initialIdentity, dataCutoff: "2026-09-15" });
  const refreshPlan = buildRunCreationPlan({ environment, identity: refreshIdentity, dataCutoff: "2026-09-15" });
  assert.equal(initialPlan.requiresIdentityBinding, true);
  assert.equal(refreshPlan.requiresIdentityBinding, false);
  const binding = buildIdentityBindingRpcArgs(initialPlan, {
    runId: "20000000-0000-4000-8000-000000000001",
    expectedStateVersion: 1,
  });
  assert.equal(binding.rpc, "bind_orotitan_run_identity");
  assert.equal(binding.args.p_security_id, initialIdentity.securityId);
  assert.equal(binding.args.p_dossier_id, initialIdentity.dossierId);
  assert.throws(
    () => buildIdentityBindingRpcArgs(refreshPlan, { runId: "20000000-0000-4000-8000-000000000001", expectedStateVersion: 1 }),
    /IDENTITY_ALREADY_BOUND_BY_BASELINE/,
  );
});

test("RUN_CONTEXT_V2 is created only after exact persisted-run reconciliation", () => {
  const plan = buildRunCreationPlan({ environment, identity: refreshIdentity, dataCutoff: "2026-09-15" });
  const context = buildRunContextV2(plan, persisted(plan));
  assert.equal(context.format, "OROTITAN_RUN_CONTEXT_V2");
  assert.equal(context.runType, "REFRESH");
  assert.equal(context.baselineSnapshotId, refreshIdentity.currentSnapshotId);
  assert.equal(context.publicationAuthorized, false);

  assert.throws(
    () => buildRunContextV2(plan, { ...persisted(plan), contractSetSha256: "0".repeat(64) }),
    /PERSISTED_RUN_MISMATCH/,
  );
});

test("Research bootstrap is emitted only from a persisted RUN_CONTEXT_V2", () => {
  const plan = buildRunCreationPlan({ environment, identity: refreshIdentity, dataCutoff: "2026-09-15" });
  const context = buildRunContextV2(plan, persisted(plan));
  const prompt = buildResearchStartPrompt(context);
  assert.ok(prompt.startsWith("OROTITAN V2 — START RESEARCH"));
  assert.match(prompt, /CONTRACT_SET_SHA256 = 1116ca12/);
  assert.match(prompt, /PRODUCTION_PROJECT_REF = cugpgtzygqqlxetyeven/);
  assert.match(prompt, /RUN_TYPE = REFRESH/);
  assert.match(prompt, /PUBLICATION_AUTHORIZED = NO/);
  assert.match(prompt, /WRONG_ENVIRONMENT/);
});

test("all downstream handoffs inherit the runtime authority envelope", () => {
  const prompt = buildResearchToFundamentals({
    company: "Qualys, Inc.",
    companyCommandName: "QUALYS",
    runId: "20000000-0000-4000-8000-000000000001",
    canonicalMode: "ANALYZE",
    runType: "REFRESH",
    dataCutoff: "2026-09-15",
    baselineSnapshotId: refreshIdentity.currentSnapshotId,
    researchFinalManifest: { artifact_id: "30000000-0000-4000-8000-000000000001", version: 1 },
    researchInputs: [],
  });
  assert.match(prompt, /RUNTIME_BOOTSTRAP_VERSION = 2\.0\.1/);
  assert.match(prompt, /RUNTIME_BOOTSTRAP_SHA256 = [0-9a-f]{64}/);
  assert.match(prompt, /CONTRACT_SET_SHA256 = 1116ca12/);
  assert.match(prompt, /PRODUCTION_PROJECT_REF = cugpgtzygqqlxetyeven/);
});

test("preflight card is intentionally compact and unambiguous", () => {
  const plan = buildRunCreationPlan({ environment, identity: refreshIdentity, dataCutoff: "2026-09-15" });
  const card = renderPreflightCard(buildRunContextV2(plan, persisted(plan)));
  assert.equal(card, [
    "OROTITAN V2 PREFLIGHT",
    "",
    "AUTHORITY = PASS",
    "PRODUCTION DB = PASS",
    "IDENTITY = PASS",
    "BASELINE = PASS",
    "RUN TYPE = REFRESH",
    "RUN PERSISTENCE = PASS",
    "",
    "RUN_ID = 20000000-0000-4000-8000-000000000001",
    "NEXT = RESEARCH",
  ].join("\n"));
});

test("Pilotage bootstrap order persists the run before Research", () => {
  const order = runtimeBootstrap.pilotage_order;
  assert.ok(order.indexOf("CREATE_PERSISTENT_RUN") < order.indexOf("START_RESEARCH"));
  assert.ok(order.indexOf("REQUERY_PERSISTED_RUN") < order.indexOf("BUILD_RUN_CONTEXT_V2"));
  assert.ok(order.indexOf("BUILD_RUN_CONTEXT_V2") < order.indexOf("START_RESEARCH"));
});

test("GO company remains analytically executable but never authorizes publication", () => {
  assert.equal(runtimeBootstrap.publication.go_company_never_authorizes_publication, true);
  assert.equal(runtimeBootstrap.publication.separate_authorization_required, true);
  assert.equal(runtimeBootstrap.publication.authorization_command, "GO PUBLISH <COMPANY>");
});
