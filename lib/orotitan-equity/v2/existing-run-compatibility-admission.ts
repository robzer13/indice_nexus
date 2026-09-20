export const V2_FROZEN_CONTRACT_SET_SHA256 =
  "1116ca12dce2d30ddbb4b699945d92ae235c940cbf69d104a01019fc21efbf5e";

export const V2_COMPATIBILITY_SEMANTIC_BASE = {
  authorityName: "OROTITAN_CANONICAL_SEMANTIC_COMPATIBILITY_PATCH_V2.0.9",
  authorityVersion: "2.0.9",
  authoritySha256: "5caf10e16c895abb23ee73fa8d9f0da99664c0699d0a84d3a7f68b7a082946bc",
  schemaName: "04_SCREENER_SCHEMA_V1_COMPAT_V2.0.9",
  schemaVersion: "2.0.9",
  schemaSha256: "edda338d50d3efcd98c21f463f3c4990dc42a1d1d71de5cd38be61788c0dd84d",
  validatorSha256: "c119257c6201e4a5c13cfb9085e1479e5fa238d7300c1f1c3cd588f260fd017b",
} as const;

export type CompatibilitySemanticProjection = {
  targetField: string;
  sourceValue: string | number;
  targetValue: string | number;
};

export type IntegrationCheckpointState = {
  lifecycleStatus: string;
  activeManifestKind: string | null;
  activeManifestDurable: boolean;
  sameStageRevision: boolean;
};

export type ExistingRunCompatibilityAdmissionInput = {
  runStatus: string;
  currentStage: string | null;
  runContractSetSha256: string;
  contractPinsUnchanged: boolean;
  sourceArtifactsHashValid: boolean;
  authorityHashVerified: boolean;
  schemaHashVerified: boolean;
  validatorHashVerified: boolean;
  regressionsPassed: boolean;
  integrationHasAdmittedCanonicalSnapshot: boolean;
  readyToPublish: boolean;
  publicationAuthorizationExists: boolean;
  publicationEventCount: number;
  currentSnapshotStateCompatible: boolean;
  semanticProjections: CompatibilitySemanticProjection[];
  semanticLoss: string;
  analyticalArtifactMutationRequired: boolean;
  contractPinMutationRequired: boolean;
  analyticalContradiction: boolean;
  unrelatedBlocker: boolean;
  deepDiveLifecycleStatus: string;
  deepDiveActiveManifestKind: string | null;
  readyForIntegration: boolean;
  integrationStage: IntegrationCheckpointState | null;
  integrationArtifactCount: number;
};

export type CompatibilityAdmissionRoute =
  | "PRE_INTEGRATION_EXISTING_V2_RUN"
  | "EXISTING_INTEGRATION_CHECKPOINT";

export type ExistingRunCompatibilityAdmissionResult =
  | {
      admitted: true;
      route: CompatibilityAdmissionRoute;
      failures: [];
    }
  | {
      admitted: false;
      route: null;
      failures: string[];
    };

const authorizedSemanticRules = new Set([
  "l2_research_fundamentals.fundamental_states.roic_trend\u0000NOT_APPLICABLE",
  "l2_research_fundamentals.analytical_metrics.roiic\u0000NOT_INTERPRETABLE",
  "l2_research_fundamentals.analytical_metrics.standard_roic\u0000NOT_INTERPRETABLE",
  "l2_research_fundamentals.fundamental_states.roic_trend\u0000UNKNOWN",
  "l2_research_fundamentals.analytical_metrics.roic_ex_goodwill\u0000NOT_INTERPRETABLE",
  "l2_research_fundamentals.analytical_metrics.rd_adjusted_roic\u0000NOT_INTERPRETABLE",
]);

const RETURN_HORIZON_FIELD =
  "l3_investment_valuation.valuation.return_horizon";

function isAuthorizedSemanticProjection(
  projection: CompatibilitySemanticProjection,
): boolean {
  if (projection.targetField === RETURN_HORIZON_FIELD) {
    return (
      typeof projection.sourceValue === "number" &&
      typeof projection.targetValue === "number" &&
      Number.isFinite(projection.sourceValue) &&
      Number.isFinite(projection.targetValue) &&
      projection.sourceValue > 0 &&
      projection.targetValue > 0 &&
      projection.sourceValue === projection.targetValue
    );
  }

  if (
    typeof projection.sourceValue !== "string" ||
    typeof projection.targetValue !== "string"
  ) {
    return false;
  }

  const key = `${projection.targetField}\u0000${projection.targetValue}`;
  return authorizedSemanticRules.has(key);
}

function reject(failures: string[]): ExistingRunCompatibilityAdmissionResult {
  return { admitted: false, route: null, failures };
}

function validateCommon(
  input: ExistingRunCompatibilityAdmissionInput,
): string[] {
  const failures: string[] = [];

  if (input.runContractSetSha256 !== V2_FROZEN_CONTRACT_SET_SHA256) {
    failures.push("CONTRACT_SET_MISMATCH");
  }
  if (!input.contractPinsUnchanged) failures.push("CONTRACT_PINS_CHANGED");
  if (!input.sourceArtifactsHashValid) failures.push("SOURCE_ARTIFACT_HASH_INVALID");
  if (!input.authorityHashVerified) failures.push("COMPATIBILITY_AUTHORITY_HASH_UNVERIFIED");
  if (!input.schemaHashVerified) failures.push("COMPATIBILITY_SCHEMA_HASH_UNVERIFIED");
  if (!input.validatorHashVerified) failures.push("COMPATIBILITY_VALIDATOR_HASH_UNVERIFIED");
  if (!input.regressionsPassed) failures.push("COMPATIBILITY_REGRESSION_NOT_PASS_ALL");
  if (input.integrationHasAdmittedCanonicalSnapshot) failures.push("INTEGRATION_SNAPSHOT_ALREADY_ADMITTED");
  if (input.readyToPublish) failures.push("READY_TO_PUBLISH_ALREADY_YES");
  if (input.publicationAuthorizationExists) failures.push("PUBLICATION_AUTHORIZATION_EXISTS");
  if (input.publicationEventCount !== 0) failures.push("PUBLICATION_EVENT_EXISTS");
  if (!input.currentSnapshotStateCompatible) failures.push("CURRENT_SNAPSHOT_STATE_INCOMPATIBLE");
  if (input.semanticLoss !== "NONE") failures.push("SEMANTIC_LOSS_NOT_NONE");
  if (input.analyticalArtifactMutationRequired) failures.push("ANALYTICAL_ARTIFACT_MUTATION_REQUIRED");
  if (input.contractPinMutationRequired) failures.push("CONTRACT_PIN_MUTATION_REQUIRED");
  if (input.analyticalContradiction) failures.push("ANALYTICAL_CONTRADICTION");
  if (input.unrelatedBlocker) failures.push("UNRELATED_BLOCKER");

  if (input.semanticProjections.length === 0) {
    failures.push("NO_COMPATIBILITY_RULE");
  }

  for (const projection of input.semanticProjections) {
    if (projection.sourceValue !== projection.targetValue) {
      failures.push("SEMANTIC_COERCION_FORBIDDEN");
      continue;
    }
    if (!isAuthorizedSemanticProjection(projection)) {
      failures.push("SEMANTIC_RULE_NOT_AUTHORIZED");
    }
  }
  return [...new Set(failures)];
}

export function evaluateExistingRunCompatibilityAdmission(
  input: ExistingRunCompatibilityAdmissionInput,
): ExistingRunCompatibilityAdmissionResult {
  const commonFailures = validateCommon(input);
  if (commonFailures.length > 0) return reject(commonFailures);

  if (input.integrationStage !== null) {
    const checkpointLifecycleValid =
      input.integrationStage.lifecycleStatus === "BLOCKED" ||
      input.integrationStage.lifecycleStatus === "PAUSED";

    if (
      input.currentStage === "INTEGRATION" &&
      checkpointLifecycleValid &&
      input.integrationStage.activeManifestKind === "CHECKPOINT" &&
      input.integrationStage.activeManifestDurable &&
      input.integrationStage.sameStageRevision
    ) {
      return {
        admitted: true,
        route: "EXISTING_INTEGRATION_CHECKPOINT",
        failures: [],
      };
    }

    return reject(["INTEGRATION_STAGE_INCONSISTENT"]);
  }

  const preIntegrationFailures: string[] = [];
  if (input.runStatus !== "ACTIVE") preIntegrationFailures.push("RUN_NOT_ACTIVE");
  if (input.currentStage !== "DEEP_DIVE") preIntegrationFailures.push("CURRENT_STAGE_NOT_DEEP_DIVE");
  if (input.deepDiveLifecycleStatus !== "COMPLETE") {
    preIntegrationFailures.push("DEEP_DIVE_NOT_COMPLETE");
  }
  if (input.deepDiveActiveManifestKind !== "FINAL") {
    preIntegrationFailures.push("DEEP_DIVE_ACTIVE_MANIFEST_NOT_FINAL");
  }
  if (!input.readyForIntegration) preIntegrationFailures.push("READY_FOR_INTEGRATION_NOT_YES");
  if (input.integrationArtifactCount !== 0) {
    preIntegrationFailures.push("PRE_INTEGRATION_ARTIFACTS_ALREADY_EXIST");
  }

  if (preIntegrationFailures.length > 0) return reject(preIntegrationFailures);

  return {
    admitted: true,
    route: "PRE_INTEGRATION_EXISTING_V2_RUN",
    failures: [],
  };
}
