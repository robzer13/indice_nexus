import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import test from "node:test";
import {
  activeContractPinPack,
  activeContractSetComputedSha256,
  assertProductionEnvironment,
  assertRuntimeBootstrapIntegrity,
  buildMethodologyReplaySuccessorPlan,
  buildResearchStartPrompt,
  buildRunContextV3,
  buildRunCreationPlan,
  runtimeBootstrap,
  type PersistedRunObservation,
  type ResolvedCompanyIdentity,
  type RuntimeEnvironmentObservation,
} from "../lib/orotitan-equity/v3/runtime-bootstrap";
import {
  activeContractPinPack as v2ContractPinPack,
  assertRuntimeBootstrapIntegrity as assertV2RuntimeBootstrapIntegrity,
} from "../lib/orotitan-equity/v2/runtime-bootstrap";

const V3_HASH = "8d9596911b98d8c2125e5a0a19997f1620cc9034efc99bf8b5a763450bf9c0cf";
const V2_HASH = "1116ca12dce2d30ddbb4b699945d92ae235c940cbf69d104a01019fc21efbf5e";

const environment: RuntimeEnvironmentObservation = {
  projectRef: "cugpgtzygqqlxetyeven",
  projectName: "orotitan-screener",
  region: "eu-west-3",
  status: "ACTIVE_HEALTHY",
};

const identity: ResolvedCompanyIdentity = {
  company: "ASML Holding N.V.",
  companyCommandName: "ASML",
  issuerId: "60d786af-46de-4302-8b63-15a925bcbfc0",
  securityId: "9741ea88-3c30-42aa-aca9-72b5b6ba1389",
  dossierId: "813813c3-0efa-49d7-9c1b-6ec724205b1c",
  currentSnapshotId: null,
  currentSnapshotContractVersion: null,
};

const parent = {
  runId: "2fc2bb73-86d1-48b2-b9c3-e219b4757416",
  issuerId: identity.issuerId,
  securityId: identity.securityId,
  dossierId: identity.dossierId,
  dataCutoff: "2026-09-19",
  contractSetSha256: V2_HASH,
};

function persisted(plan: ReturnType<typeof buildMethodologyReplaySuccessorPlan>): PersistedRunObservation {
  return {
    runId: "20000000-0000-4000-8000-000000000003",
    stateVersion: 2,
    runStatus: "CREATED",
    issuerId: plan.issuerId,
    securityId: plan.securityId,
    dossierId: plan.dossierId,
    parentRunId: plan.parentRunId,
    baselineSnapshotId: plan.baselineSnapshotId,
    entryPath: "IMPOSED_COMPANY",
    canonicalMode: "ANALYZE",
    runType: plan.runType,
    dataCutoff: plan.dataCutoff,
    processVersion: activeContractPinPack.contract_pins.process.version,
    pilotageContractVersion: activeContractPinPack.contract_pins.pilotage.version,
    contractSetSha256: V3_HASH,
  };
}

test("V3 runtime reconciles exactly to the frozen 13-pin Contract Set", () => {
  assert.doesNotThrow(() => assertRuntimeBootstrapIntegrity());
  assert.equal(activeContractPinPack.contract_set_sha256, V3_HASH);
  assert.equal(activeContractSetComputedSha256, V3_HASH);
  assert.equal(runtimeBootstrap.authority_boundary.contract_set_sha256, V3_HASH);
  assert.equal(Object.keys(activeContractPinPack.contract_pins).length, 13);
});

test("all frozen V3 pin bytes still match their content SHA-256", () => {
  for (const pin of Object.values(activeContractPinPack.contract_pins)) {
    const bytes = readFileSync(new URL(`../${pin.locator.path}`, import.meta.url));
    const actual = createHash("sha256").update(bytes).digest("hex");
    assert.equal(actual, pin.content_sha256, pin.locator.path);
  }
});

test("V2 runtime and Contract Set remain unchanged for existing runs", () => {
  assert.doesNotThrow(() => assertV2RuntimeBootstrapIntegrity());
  assert.equal(v2ContractPinPack.contract_set_sha256, V2_HASH);
});

test("production environment remains exact allowlist", () => {
  assert.doesNotThrow(() => assertProductionEnvironment(environment));
  assert.throws(
    () => assertProductionEnvironment({ ...environment, projectRef: "wrong-project" }),
    /WRONG_ENVIRONMENT/,
  );
});

test("ordinary new runs now carry exact V3 authority", () => {
  const plan = buildRunCreationPlan({ environment, identity, dataCutoff: "2026-09-20" });
  assert.equal(plan.runType, "INITIAL");
  assert.equal(plan.parentRunId, null);
  assert.equal(plan.rpcArgs.p_contract_set_sha256, V3_HASH);
  assert.equal(plan.rpcArgs.p_process_version, "3.0");
  assert.equal(plan.rpcArgs.p_pilotage_contract_version, "3.0");
});

test("pure methodology replay successor preserves parent lineage and cutoff", () => {
  const plan = buildMethodologyReplaySuccessorPlan({ environment, identity, parent });
  assert.equal(plan.creationReason, "METHODOLOGY_REPLAY_SUCCESSOR");
  assert.equal(plan.runType, "INITIAL");
  assert.equal(plan.parentRunId, parent.runId);
  assert.equal(plan.baselineSnapshotId, null);
  assert.equal(plan.dataCutoff, "2026-09-19");
  assert.equal(plan.requiresIdentityBinding, true);
  assert.equal(plan.rpcArgs.p_parent_run_id, parent.runId);
  assert.equal(plan.rpcArgs.p_contract_set_sha256, V3_HASH);
});

test("successor admission fails closed on identity or baseline mismatch", () => {
  assert.throws(
    () => buildMethodologyReplaySuccessorPlan({
      environment,
      identity: { ...identity, dossierId: "813813c3-0efa-49d7-9c1b-6ec724205b1d" },
      parent,
    }),
    /SUCCESSOR_IDENTITY_MISMATCH/,
  );
  assert.throws(
    () => buildMethodologyReplaySuccessorPlan({
      environment,
      identity: { ...identity, currentSnapshotId: "10000000-0000-4000-8000-000000000004", currentSnapshotContractVersion: "04_SCREENER_SCHEMA_V2" },
      parent,
    }),
    /SUCCESSOR_BASELINE_MISMATCH/,
  );
});

test("successor is rejected when parent already uses active V3 authority", () => {
  assert.throws(
    () => buildMethodologyReplaySuccessorPlan({
      environment,
      identity,
      parent: { ...parent, contractSetSha256: V3_HASH },
    }),
    /SUCCESSOR_NOT_REQUIRED/,
  );
});

test("persisted run reconciliation enforces exact V3 Contract Set hash", () => {
  const plan = buildMethodologyReplaySuccessorPlan({ environment, identity, parent });
  assert.doesNotThrow(() => buildRunContextV3(plan, persisted(plan)));
  assert.throws(
    () => buildRunContextV3(plan, { ...persisted(plan), contractSetSha256: V2_HASH }),
    /PERSISTED_RUN_MISMATCH: contract_set_sha256/,
  );
});

test("Research replay bootstrap carries V3 runtime and pinned Research contract", () => {
  const plan = buildMethodologyReplaySuccessorPlan({ environment, identity, parent });
  const context = buildRunContextV3(plan, persisted(plan));
  const prompt = buildResearchStartPrompt(context);
  assert.ok(prompt.startsWith("OROTITAN V3 — START RESEARCH"));
  assert.match(prompt, /PARENT_RUN_ID = 2fc2bb73-86d1-48b2-b9c3-e219b4757416/);
  assert.match(prompt, /CONTRACT_SET_SHA256 = 8d959691/);
  assert.match(prompt, /EXPECTED_STAGE_CONTRACT = OROTITAN_RESEARCH_STAGE_CONTRACT_V2_FREEZE_V2\.0/);
  assert.match(prompt, /DATA_CUTOFF = 2026-09-19/);
  assert.match(prompt, /PUBLICATION_AUTHORIZED = NO/);
});
