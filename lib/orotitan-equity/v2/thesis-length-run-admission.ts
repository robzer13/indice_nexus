/**
 * V2.0.11 run-level PRE-FLIGHT only. Not an authorization to mutate Registry,
 * revise frozen pins, persist the canonical snapshot or publish.
 *
 * This adds a narrow eligible route for an Integration stage that has already
 * started but has produced NO Integration artifact and NO stage manifest.
 * Historical compatibility routes remain unchanged.
 */
export const FROZEN_V2_CONTRACT_SET =
  "1116ca12dce2d30ddbb4b699945d92ae235c940cbf69d104a01019fc21efbf5e";

export type ThesisLengthRunAdmissionInput = Readonly<{
  runStatus: string;
  currentStage: string;
  runStateVersion: number;
  contractSetSha256: string;
  contractPinsUnchanged: boolean;
  deepDiveLifecycle: string;
  deepDiveManifestKind: string | null;
  deepDiveHandoffGate: string;
  deepDiveManifestHashVerified: boolean;
  fullThesisSourceHashVerified: boolean;
  certifiedThesisMatchesDeepDiveReport: boolean;
  integrationLifecycle: string;
  integrationStageStateVersion: number;
  integrationManifestKind: string | null;
  integrationArtifactCount: number;
  integrationHandoffGate: string;
  canonicalSnapshotCount: number;
  currentDossierSnapshotId: string | null;
  publicationAuthorizations: number;
  publicationSuccessEvents: number;
  approvalOfPatchScope: boolean;
  exactCompatibilitySchemaHashVerified: boolean;
  exactValidatorHashVerified: boolean;
  fullCiPassed: boolean;
  analyticalOutputMutationRequested: boolean;
  frozenContractMutationRequested: boolean;
}>;
export type ThesisLengthRunAdmissionResult = Readonly<{
  readOnlyPreflightEligible: boolean;
  route: "EXISTING_INTEGRATION_EMPTY_IN_PROGRESS" | null;
  blockers: readonly string[];
  executionAuthorized: false;
  publicationAuthorized: false;
}>;

export function evaluateThesisLengthRunPreflight(
  input: ThesisLengthRunAdmissionInput,
): ThesisLengthRunAdmissionResult {
  const blockers: string[] = [];
  if (input.runStatus !== "ACTIVE" || input.currentStage !== "INTEGRATION" ||
      !Number.isSafeInteger(input.runStateVersion) || input.runStateVersion < 1) {
    blockers.push("RUN_NOT_ACTIVE_IN_INTEGRATION");
  }
  if (input.contractSetSha256 !== FROZEN_V2_CONTRACT_SET ||
      !input.contractPinsUnchanged || input.frozenContractMutationRequested) {
    blockers.push("CONTRACT_PIN_SET_CHANGE_FORBIDDEN");
  }
  if (input.deepDiveLifecycle !== "COMPLETE" ||
      input.deepDiveManifestKind !== "FINAL" ||
      input.deepDiveHandoffGate !== "YES" ||
      !input.deepDiveManifestHashVerified) {
    blockers.push("DEEP_DIVE_FINAL_NOT_ADMITTED");
  }
  if (!input.fullThesisSourceHashVerified ||
      !input.certifiedThesisMatchesDeepDiveReport) {
    blockers.push("SEALED_THESIS_PROVENANCE_NOT_VERIFIED");
  }
  if (input.integrationLifecycle !== "IN_PROGRESS" ||
      !Number.isSafeInteger(input.integrationStageStateVersion) ||
      input.integrationStageStateVersion !== 1 ||
      input.integrationManifestKind !== null ||
      input.integrationArtifactCount !== 0 ||
      input.integrationHandoffGate !== "NOT_EVALUATED") {
    blockers.push("INTEGRATION_NOT_EMPTY_PREFLIGHT");
  }
  if (input.canonicalSnapshotCount !== 0 || input.currentDossierSnapshotId !== null) {
    blockers.push("CANONICAL_SNAPSHOT_ALREADY_EXISTS");
  }
  if (input.publicationAuthorizations !== 0 || input.publicationSuccessEvents !== 0) {
    blockers.push("PUBLICATION_EVENT_ALREADY_EXISTS");
  }
  if (!input.approvalOfPatchScope || !input.exactCompatibilitySchemaHashVerified ||
      !input.exactValidatorHashVerified || !input.fullCiPassed) {
    blockers.push("COMPATIBILITY_AUTHORITY_NOT_ADMITTED");
  }
  if (input.analyticalOutputMutationRequested) {
    blockers.push("ANALYTICAL_REWRITE_FORBIDDEN");
  }
  return {
    readOnlyPreflightEligible: blockers.length === 0,
    route: blockers.length === 0 ? "EXISTING_INTEGRATION_EMPTY_IN_PROGRESS" : null,
    blockers,
    executionAuthorized: false,
    publicationAuthorized: false,
  };
}
