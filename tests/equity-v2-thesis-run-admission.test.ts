import assert from "node:assert/strict";
import test from "node:test";
import {
  FROZEN_V2_CONTRACT_SET,
  evaluateThesisLengthRunPreflight,
  type ThesisLengthRunAdmissionInput,
} from "../lib/orotitan-equity/v2/thesis-length-run-admission";

function base(): ThesisLengthRunAdmissionInput {
  return {
    runStatus: "ACTIVE",
    currentStage: "INTEGRATION",
    runStateVersion: 11,
    contractSetSha256: FROZEN_V2_CONTRACT_SET,
    contractPinsUnchanged: true,
    deepDiveLifecycle: "COMPLETE",
    deepDiveManifestKind: "FINAL",
    deepDiveHandoffGate: "YES",
    deepDiveManifestHashVerified: true,
    fullThesisSourceHashVerified: true,
    certifiedThesisMatchesDeepDiveReport: true,
    integrationLifecycle: "IN_PROGRESS",
    integrationStageStateVersion: 1,
    integrationManifestKind: null,
    integrationArtifactCount: 0,
    integrationHandoffGate: "NOT_EVALUATED",
    canonicalSnapshotCount: 0,
    currentDossierSnapshotId: null,
    publicationAuthorizations: 0,
    publicationSuccessEvents: 0,
    approvalOfPatchScope: true,
    exactCompatibilitySchemaHashVerified: true,
    exactValidatorHashVerified: true,
    fullCiPassed: true,
    analyticalOutputMutationRequested: false,
    frozenContractMutationRequested: false,
  };
}

test("existing empty IN_PROGRESS Integration may pass readonly preflight but never grants mutation or publish", () => {
  const x = evaluateThesisLengthRunPreflight(base());
  assert.equal(x.readOnlyPreflightEligible, true);
  assert.equal(x.route, "EXISTING_INTEGRATION_EMPTY_IN_PROGRESS");
  assert.deepEqual(x.blockers, []);
  assert.equal(x.executionAuthorized, false);
  assert.equal(x.publicationAuthorized, false);
});

test("partial, paused or already-final Integration is not an empty preflight", () => {
  const changed = [
    { integrationLifecycle: "PAUSED" },
    { integrationStageStateVersion: 2 },
    { integrationManifestKind: "CHECKPOINT" },
    { integrationArtifactCount: 1 },
    { integrationHandoffGate: "YES" },
  ];
  for (const delta of changed) {
    const result = evaluateThesisLengthRunPreflight({ ...base(), ...delta });
    assert.equal(result.readOnlyPreflightEligible, false, JSON.stringify(delta));
    assert.ok(result.blockers.includes("INTEGRATION_NOT_EMPTY_PREFLIGHT"));
  }
});

test("old published and contract-divergent runs fail", () => {
  const cases = [
    { runStatus: "PUBLISHED" },
    { currentStage: "DEEP_DIVE" },
    { contractSetSha256: "1".repeat(64) },
    { contractPinsUnchanged: false },
    { frozenContractMutationRequested: true },
    { canonicalSnapshotCount: 1 },
    { currentDossierSnapshotId: "00000000-0000-4000-8000-000000000099" },
    { publicationAuthorizations: 1 },
    { publicationSuccessEvents: 1 },
  ];
  for (const changed of cases) {
    const result = evaluateThesisLengthRunPreflight({ ...base(), ...changed });
    assert.equal(result.readOnlyPreflightEligible, false, JSON.stringify(changed));
  }
});

test("no upstream hash, missing FINAL gate, unmatched certified sources or unapproved CI passes", () => {
  const cases = [
    { deepDiveLifecycle: "IN_PROGRESS" },
    { deepDiveManifestKind: "CHECKPOINT" },
    { deepDiveHandoffGate: "NO" },
    { deepDiveManifestHashVerified: false },
    { fullThesisSourceHashVerified: false },
    { certifiedThesisMatchesDeepDiveReport: false },
    { approvalOfPatchScope: false },
    { exactCompatibilitySchemaHashVerified: false },
    { exactValidatorHashVerified: false },
    { fullCiPassed: false },
    { analyticalOutputMutationRequested: true },
  ];
  for (const changed of cases) {
    const result = evaluateThesisLengthRunPreflight({ ...base(), ...changed });
    assert.equal(result.readOnlyPreflightEligible, false, JSON.stringify(changed));
  }
});

test("function is pure; no request mutability, no GO or Registry operations", () => {
  const input=base(),before=JSON.stringify(input);
  evaluateThesisLengthRunPreflight(input);
  assert.equal(JSON.stringify(input),before);
});
