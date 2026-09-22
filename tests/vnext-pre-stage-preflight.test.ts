import assert from "node:assert/strict";
import test from "node:test";

import {
  assertPreStagePreflightPass,
  runPreStagePreflight,
  type PreflightScope,
  type PreStagePreflightRequest,
} from "../runtime/vnext/pre-stage-preflight";

const HASH_A = "a".repeat(64);
const HASH_B = "b".repeat(64);
const HASH_C = "c".repeat(64);
const RUN_ID = "run-gate9";
const ISSUER_ID = "issuer-gate9";
const SECURITY_ID = "security-gate9";
const DOSSIER_ID = "dossier-gate9";

function stageFor(scope: PreflightScope) {
  if (scope === "RESEARCH") return "RESEARCH" as const;
  if (scope === "INTEGRATION") return "INTEGRATION" as const;
  return "DEEP_DIVE" as const;
}

function contractFor(scope: PreflightScope) {
  const stage = stageFor(scope);
  if (stage === "RESEARCH") {
    return {
      name: "OROTITAN_RESEARCH_STAGE_CONTRACT_V1",
      version: "1.0",
      hash: HASH_A,
    };
  }
  if (stage === "INTEGRATION") {
    return {
      name: "OROTITAN_INTEGRATION_STAGE_CONTRACT_V1",
      version: "1.0",
      hash: HASH_B,
    };
  }
  return {
    name: "OROTITAN_DEEP_DIVE_STAGE_CONTRACT_V1",
    version: "1.0",
    hash: HASH_C,
  };
}

function upstreamFor(scope: PreflightScope) {
  if (scope === "DEEP_DIVE") {
    return {
      artifactId: "research-manifest",
      version: 3,
      runId: RUN_ID,
      stageCode: "RESEARCH" as const,
      contentSha256: HASH_A,
      handoffGateName: "READY_FOR_DEEP_DIVE" as const,
    };
  }

  if (scope === "INTEGRATION") {
    return {
      artifactId: "deep-dive-manifest",
      version: 4,
      runId: RUN_ID,
      stageCode: "DEEP_DIVE" as const,
      contentSha256: HASH_B,
      handoffGateName: "READY_FOR_INTEGRATION" as const,
    };
  }

  return null;
}

function artifactTypeFor(scope: PreflightScope): string {
  switch (scope) {
    case "DEEP_DIVE":
      return "ANALYSIS_INPUT_LOCK";
    case "FUNDAMENTALS":
      return "EVIDENCE_LEDGER";
    case "VALUATION":
      return "FUNDAMENTALS_LOCK";
    case "CERTIFICATION":
      return "VALUATION_OUTPUT";
    case "INTEGRATION":
      return "DEEP_DIVE_FINAL_OUTPUT";
    case "RESEARCH":
      return "SOURCE_ATTACHMENT";
  }
}

function validRequest(scope: PreflightScope): PreStagePreflightRequest {
  const stageCode = stageFor(scope);
  const contract = contractFor(scope);
  const upstream = upstreamFor(scope);
  const isResearch = scope === "RESEARCH";
  const isSubscope =
    scope === "FUNDAMENTALS" ||
    scope === "VALUATION" ||
    scope === "CERTIFICATION";

  const expectedIdentity = {
    issuerId: ISSUER_ID,
    securityId: isResearch ? null : SECURITY_ID,
    dossierId: DOSSIER_ID,
  };

  const expectedArtifacts = isResearch
    ? []
    : [
        {
          artifactId: "input-" + scope.toLowerCase(),
          version: 2,
          runId: RUN_ID,
          artifactType: artifactTypeFor(scope),
          contentSha256: HASH_C,
          expectedAuthorityState: "AUTHORITATIVE" as const,
        },
      ];

  return {
    scope,
    expectedRunId: RUN_ID,
    expectedRunStateVersion: 11,
    expectedStageStateVersion: 7,
    expectedIdentity,
    expectedCanonicalMode: "ANALYZE",
    expectedRunType: "INITIAL",
    expectedDataCutoff: "2026-09-21",
    expectedProcessVersion: "1.0",
    expectedPilotageContractVersion: "1.0.1",
    expectedContractSetSha256: HASH_A,
    expectedStageContractName: contract.name,
    expectedStageContractVersion: contract.version,
    expectedStageContractSha256: contract.hash,

    run: {
      runId: RUN_ID,
      runStatus: "ACTIVE",
      currentStage: stageCode,
      canonicalMode: "ANALYZE",
      runType: "INITIAL",
      dataCutoff: "2026-09-21",
      processVersion: "1.0",
      pilotageContractVersion: "1.0.1",
      contractSetSha256: HASH_A,
      stateVersion: 11,
      baselineSnapshotId: null,
      identity: { ...expectedIdentity },
    },

    stage: {
      runId: RUN_ID,
      stageCode,
      stageRevision: 1,
      stageContractName: contract.name,
      stageContractVersion: contract.version,
      stageContractSha256: contract.hash,
      lifecycleStatus: isSubscope ? "IN_PROGRESS" : "NOT_STARTED",
      handoffGateName:
        stageCode === "RESEARCH"
          ? "READY_FOR_DEEP_DIVE"
          : stageCode === "DEEP_DIVE"
            ? "READY_FOR_INTEGRATION"
            : "READY_TO_PUBLISH",
      handoffGateState: "NOT_EVALUATED",
      activeManifestArtifactId: null,
      activeManifestVersion: null,
      activeManifestKind: null,
      stateVersion: 7,
      blockerCount: 0,
    },

    expectedContracts: [
      {
        name: contract.name,
        version: contract.version,
        contentSha256: contract.hash,
      },
    ],
    resolvedContracts: [
      {
        name: contract.name,
        version: contract.version,
        contentSha256: contract.hash,
        durableLocatorAvailable: true,
      },
    ],

    expectedArtifacts,
    resolvedArtifacts: expectedArtifacts.map((artifact) => ({
      ...artifact,
      artifactStatus: "SEALED" as const,
      authorityState: artifact.expectedAuthorityState,
      availabilityState: "AVAILABLE" as const,
      durableLocatorAvailable: true,
    })),

    expectedUpstreamManifest: upstream,
    resolvedUpstreamManifest: upstream
      ? {
          ...upstream,
          artifactStatus: "SEALED",
          authorityState: "AUTHORITATIVE",
          availabilityState: "AVAILABLE",
          manifestKind: "FINAL",
          stageLifecycleStatus: "COMPLETE",
          handoffGateState: "YES",
          durableLocatorAvailable: true,
        }
      : null,

    priorAssuranceGatePassed: true,
    authorityStateCompatible: true,
    noBlockingExecutionDefect: true,
  };
}

function failureCodes(request: PreStagePreflightRequest): string[] {
  return runPreStagePreflight(request).failures.map((failure) => failure.code);
}

test("Gate 9 admits clean inputs for every required scope", () => {
  const scopes: PreflightScope[] = [
    "RESEARCH",
    "DEEP_DIVE",
    "FUNDAMENTALS",
    "VALUATION",
    "CERTIFICATION",
    "INTEGRATION",
  ];

  for (const scope of scopes) {
    const request = validRequest(scope);
    assert.deepEqual(runPreStagePreflight(request), {
      status: "PASS",
      scope,
      failures: [],
    });
    assert.doesNotThrow(() => assertPreStagePreflightPass(request));
  }
});

test("Gate 9 rejects stale run and stage concurrency versions", () => {
  const request = validRequest("DEEP_DIVE");
  request.expectedRunStateVersion = 10;
  request.expectedStageStateVersion = 6;

  const codes = failureCodes(request);
  assert.ok(codes.includes("RUN_STATE_VERSION_STALE"));
  assert.ok(codes.includes("STAGE_STATE_VERSION_STALE"));
});

test("Gate 9 rejects identity, cutoff and contract-set drift", () => {
  const request = validRequest("DEEP_DIVE");
  request.run.identity.securityId = "wrong-security";
  request.run.dataCutoff = "2026-09-22";
  request.run.contractSetSha256 = HASH_B;

  const codes = failureCodes(request);
  assert.ok(codes.includes("IDENTITY_SECURITY_MISMATCH"));
  assert.ok(codes.includes("DATA_CUTOFF_MISMATCH"));
  assert.ok(codes.includes("CONTRACT_SET_HASH_MISMATCH"));
});

test("Gate 9 pins the stage contract independently from resolved contract lookup", () => {
  const request = validRequest("INTEGRATION");
  request.stage.stageContractVersion = "wrong";
  request.stage.stageContractSha256 = HASH_C;

  const codes = failureCodes(request);
  assert.ok(codes.includes("STAGE_CONTRACT_VERSION_MISMATCH"));
  assert.ok(codes.includes("STAGE_CONTRACT_HASH_MISMATCH"));
});

test("Gate 9 rejects missing or stale contract locators", () => {
  const request = validRequest("RESEARCH");
  request.resolvedContracts[0] = {
    ...request.resolvedContracts[0],
    version: "0.9",
    durableLocatorAvailable: false,
  };

  const codes = failureCodes(request);
  assert.ok(codes.includes("CONTRACT_VERSION_MISMATCH"));
  assert.ok(codes.includes("CONTRACT_LOCATOR_MISSING"));
});

test("Gate 9 rejects a checkpoint or non-admitting upstream manifest", () => {
  const request = validRequest("DEEP_DIVE");
  assert.ok(request.resolvedUpstreamManifest);
  request.resolvedUpstreamManifest.manifestKind = "CHECKPOINT";
  request.resolvedUpstreamManifest.stageLifecycleStatus = "PAUSED";
  request.resolvedUpstreamManifest.handoffGateState = "NO";

  const codes = failureCodes(request);
  assert.ok(codes.includes("UPSTREAM_MANIFEST_NOT_FINAL"));
  assert.ok(codes.includes("UPSTREAM_STAGE_NOT_COMPLETE"));
  assert.ok(codes.includes("UPSTREAM_HANDOFF_NOT_YES"));
});

test("Gate 9 rejects an upstream manifest with stale bytes or locator", () => {
  const request = validRequest("INTEGRATION");
  assert.ok(request.resolvedUpstreamManifest);
  request.resolvedUpstreamManifest.contentSha256 = HASH_C;
  request.resolvedUpstreamManifest.durableLocatorAvailable = false;

  const codes = failureCodes(request);
  assert.ok(codes.includes("UPSTREAM_MANIFEST_HASH_MISMATCH"));
  assert.ok(codes.includes("UPSTREAM_MANIFEST_LOCATOR_MISSING"));
});

test("Gate 9 rejects stale, invalidated, unavailable or relocated artifacts", () => {
  const request = validRequest("VALUATION");
  request.resolvedArtifacts[0] = {
    ...request.resolvedArtifacts[0],
    version: 3,
    contentSha256: HASH_B,
    artifactStatus: "INVALIDATED",
    authorityState: "SUPERSEDED",
    availabilityState: "MISSING",
    durableLocatorAvailable: false,
  };

  const codes = failureCodes(request);
  assert.ok(codes.includes("ARTIFACT_VERSION_MISMATCH"));
  assert.ok(codes.includes("ARTIFACT_HASH_MISMATCH"));
  assert.ok(codes.includes("ARTIFACT_NOT_SEALED"));
  assert.ok(codes.includes("ARTIFACT_AUTHORITY_MISMATCH"));
  assert.ok(codes.includes("ARTIFACT_NOT_AVAILABLE"));
  assert.ok(codes.includes("ARTIFACT_LOCATOR_MISSING"));
});

test("Gate 9 requires a baseline snapshot for refresh or activation paths", () => {
  const refresh = validRequest("RESEARCH");
  refresh.expectedCanonicalMode = "REFRESH";
  refresh.expectedRunType = "REFRESH";
  refresh.run.canonicalMode = "REFRESH";
  refresh.run.runType = "REFRESH";

  assert.ok(failureCodes(refresh).includes("BASELINE_SNAPSHOT_REQUIRED"));

  refresh.run.baselineSnapshotId = "baseline-snapshot";
  assert.equal(runPreStagePreflight(refresh).status, "PASS");
});

test("Gate 9 fails closed when authority, prior assurance or blockers disagree", () => {
  const request = validRequest("CERTIFICATION");
  request.priorAssuranceGatePassed = false;
  request.authorityStateCompatible = false;
  request.noBlockingExecutionDefect = false;
  request.stage.blockerCount = 2;

  const codes = failureCodes(request);
  assert.ok(codes.includes("PRIOR_ASSURANCE_GATE_FAILED"));
  assert.ok(codes.includes("AUTHORITY_STATE_INCOMPATIBLE"));
  assert.ok(codes.includes("BLOCKING_EXECUTION_DEFECT"));
});

test("Gate 9 requires exact upstream admission for Deep Dive and Integration only", () => {
  const deepDive = validRequest("DEEP_DIVE");
  deepDive.expectedUpstreamManifest = null;
  deepDive.resolvedUpstreamManifest = null;
  assert.ok(failureCodes(deepDive).includes("UPSTREAM_MANIFEST_REQUIRED"));

  const research = validRequest("RESEARCH");
  research.expectedUpstreamManifest = {
    artifactId: "unexpected",
    version: 1,
    runId: RUN_ID,
    stageCode: "RESEARCH",
    contentSha256: HASH_A,
    handoffGateName: "READY_FOR_DEEP_DIVE",
  };
  assert.ok(failureCodes(research).includes("UPSTREAM_MANIFEST_UNEXPECTED"));
});

test("Gate 9 never admits terminal or ready-to-publish runs to expensive work", () => {
  for (const status of ["READY_TO_PUBLISH", "PUBLISHED", "CANCELLED"] as const) {
    const request = validRequest("VALUATION");
    request.run.runStatus = status;
    assert.ok(
      failureCodes(request).includes("RUN_TERMINAL_OR_UNSTARTABLE"),
    );
  }
});
