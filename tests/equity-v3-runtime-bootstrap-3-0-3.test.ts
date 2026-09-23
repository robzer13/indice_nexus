import assert from "node:assert/strict";
import test from "node:test";

import {
  activeContractPinPack,
  activeContractSetComputedSha256,
  assertRuntimeBootstrapIntegrity,
  buildMethodologyReplaySuccessorPlan,
  buildRunCreationPlan,
  runtimeBootstrap,
  RuntimeBootstrapError,
  type HistoricalParentObservation,
  type ResolvedCompanyIdentity,
  type RuntimeEnvironmentObservation,
} from "../lib/orotitan-equity/v3/runtime-bootstrap-v3-0-3";

const environment: RuntimeEnvironmentObservation = {
  projectRef: "cugpgtzygqqlxetyeven",
  projectName: "orotitan-screener",
  region: "eu-west-3",
  status: "ACTIVE_HEALTHY",
};

const identity: ResolvedCompanyIdentity = {
  company: "Example S.A.",
  companyCommandName: "EXAMPLE",
  issuerId: "11111111-1111-4111-8111-111111111111",
  securityId: "22222222-2222-4222-8222-222222222222",
  dossierId: "33333333-3333-4333-8333-333333333333",
  currentSnapshotId: null,
  currentSnapshotContractVersion: null,
};

function activeParent(): HistoricalParentObservation {
  return {
    runId: "44444444-4444-4444-8444-444444444444",
    issuerId: identity.issuerId,
    securityId: identity.securityId,
    dossierId: identity.dossierId,
    dataCutoff: "2026-09-22",
    contractSetSha256: "8d9596911b98d8c2125e5a0a19997f1620cc9034efc99bf8b5a763450bf9c0cf",
    stateVersion: 7,
    runStatus: "ACTIVE",
    currentStage: "DEEP_DIVE",
    publishedAt: null,
    cancelledAt: null,
  };
}

function blockedTimingParent(): HistoricalParentObservation {
  return {
    ...activeParent(),
    contractSetSha256: "257c287357c19a5d47a42f140a1eb0377d48701b04b07e1e9e740646797c172c",
    stateVersion: 8,
    runStatus: "BLOCKED",
    deepDiveLifecycleStatus: "BLOCKED",
    deepDiveContractStatusCode: "VALUATION_BLOCKED_PINNED_TIMING_DENOMINATOR_DATE_CONFLICT",
    deepDiveBlockerCode: "V3_VALUATION_TIMING_DENOMINATOR_DATE_CONFLICT",
    deepDiveBlockerClassification: "PINNED_AUTHORITY_CONFLICT",
    deepDiveBlockerScope: "VALUATION_DCF_AND_DEPENDENT_OUTPUTS",
  };
}

function expectCode(fn: () => unknown, code: string): void {
  assert.throws(
    fn,
    (error: unknown) => error instanceof RuntimeBootstrapError && error.code === code,
    code,
  );
}

test("runtime V3.0.3 pins exact successor Contract Set and authority composition", () => {
  assertRuntimeBootstrapIntegrity();
  assert.equal(runtimeBootstrap.production_status, "ACTIVE_FOR_NEW_RUNS");
  assert.equal(runtimeBootstrap.version, "3.0.3");
  assert.equal(runtimeBootstrap.orotitan_version, "3.1");
  assert.equal(
    activeContractPinPack.contract_set_sha256,
    "3644e501909326af04d66730fa30b1ac3da0d82fb6b717202af6d948f3211fe2",
  );
  assert.equal(activeContractSetComputedSha256, activeContractPinPack.contract_set_sha256);
  assert.equal(Object.keys(activeContractPinPack.contract_pins).length, 15);
  assert.equal(
    activeContractPinPack.contract_pins.valuation_date_alignment.name,
    "OROTITAN_VALUATION_DATE_ALIGNMENT_AUTHORITY_V1_FREEZE_V1.0",
  );
  assert.equal(activeContractPinPack.contract_pins.dcf_timing.version, "1.0");
  assert.equal(activeContractPinPack.contract_pins.analysis_standard.version, "1.0");
});

test("ordinary new run uses successor Contract Set without publication authority", () => {
  const plan = buildRunCreationPlan({ environment, identity, dataCutoff: "2026-09-23" });
  assert.equal(plan.runType, "INITIAL");
  assert.equal(plan.rpc, "create_orotitan_run");
  assert.equal(plan.rpcArgs.p_contract_set_sha256, activeContractPinPack.contract_set_sha256);
  assert.equal(runtimeBootstrap.activation_scope.canonical_publication_authorized, false);
});

test("same-cutoff successor from ACTIVE prior Contract Set remains admitted", () => {
  const parent = activeParent();
  const plan = buildMethodologyReplaySuccessorPlan({ environment, identity, parent });
  assert.equal(plan.runType, "INITIAL");
  assert.equal(plan.parentRunId, parent.runId);
  assert.equal(plan.baselineSnapshotId, null);
  assert.equal(plan.dataCutoff, parent.dataCutoff);
  assert.equal(plan.rpc, "create_orotitan_methodology_successor_run");
  assert.equal(plan.rpcArgs.p_expected_parent_state_version, 7);
  assert.equal(plan.rpcArgs.p_expected_parent_run_status, "ACTIVE");
});

test("exact blocked timing-conflict parent is admitted without parent status mutation", () => {
  const parent = blockedTimingParent();
  const plan = buildMethodologyReplaySuccessorPlan({ environment, identity, parent });
  assert.equal(plan.rpc, "create_orotitan_methodology_successor_run");
  assert.equal(plan.rpcArgs.p_expected_parent_run_status, "BLOCKED");
  assert.equal(plan.rpcArgs.p_expected_parent_contract_set_sha256, parent.contractSetSha256);
  assert.equal(plan.dataCutoff, "2026-09-22");
  assert.equal(plan.parentRunId, parent.runId);
});

test("BLOCKED parent with any blocker mismatch remains rejected", () => {
  const parent = blockedTimingParent();
  parent.deepDiveBlockerCode = "SOME_OTHER_BLOCKER";
  expectCode(
    () => buildMethodologyReplaySuccessorPlan({ environment, identity, parent }),
    "SUCCESSOR_PARENT_STATUS_MISMATCH",
  );
});

test("BLOCKED parent under any other prior Contract Set remains rejected", () => {
  const parent = blockedTimingParent();
  parent.contractSetSha256 = "8d9596911b98d8c2125e5a0a19997f1620cc9034efc99bf8b5a763450bf9c0cf";
  expectCode(
    () => buildMethodologyReplaySuccessorPlan({ environment, identity, parent }),
    "SUCCESSOR_PARENT_STATUS_MISMATCH",
  );
});

test("parent already on successor Contract Set cannot spawn redundant replay", () => {
  const parent = activeParent();
  parent.contractSetSha256 = activeContractPinPack.contract_set_sha256;
  expectCode(
    () => buildMethodologyReplaySuccessorPlan({ environment, identity, parent }),
    "SUCCESSOR_NOT_REQUIRED",
  );
});

test("successor preserves identity, same cutoff and null baseline", () => {
  const parent = blockedTimingParent();
  const plan = buildMethodologyReplaySuccessorPlan({ environment, identity, parent });
  assert.equal(plan.issuerId, parent.issuerId);
  assert.equal(plan.securityId, parent.securityId);
  assert.equal(plan.dossierId, parent.dossierId);
  assert.equal(plan.dataCutoff, parent.dataCutoff);
  assert.equal(plan.baselineSnapshotId, null);
});
