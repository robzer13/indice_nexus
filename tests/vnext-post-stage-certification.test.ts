import assert from "node:assert/strict";
import test from "node:test";

import {
  certifyPostStageBundle,
  executeCertifiedStageFinalization,
  type FinalizationArtifact,
  type PostFinalizeContext,
  type PostStageCertificationRequest,
  type PostStageCertificationStore,
} from "../runtime/vnext/post-stage-certification";

const RUN_ID = "run-gate10";
const PRIOR_RUN_ID = "run-gate10-prior";
const HASH_A = "a".repeat(64);
const HASH_B = "b".repeat(64);
const HASH_C = "c".repeat(64);
const HASH_D = "d".repeat(64);

function artifact(
  overrides: Partial<FinalizationArtifact> = {},
): FinalizationArtifact {
  return {
    artifactId: "artifact-input",
    version: 1,
    runId: RUN_ID,
    stageCode: "RESEARCH",
    artifactType: "SOURCE_ATTACHMENT",
    authorityClass: "AUTHORITATIVE_STAGE_OUTPUT",
    artifactStatus: "SEALED",
    authorityState: "AUTHORITATIVE",
    availabilityState: "AVAILABLE",
    contentSha256: HASH_A,
    retrievedContentSha256: HASH_A,
    mediaType: "application/json",
    sizeBytes: 100,
    durableLocatorAvailable: true,
    schemaValidationRequired: false,
    schemaValidationPassed: true,
    ...overrides,
  };
}

function baseContext(): PostFinalizeContext {
  return {
    run: {
      runId: RUN_ID,
      runStatus: "ACTIVE",
      issuerId: "issuer-gate10",
      securityId: "security-gate10",
      dossierId: "dossier-gate10",
      canonicalMode: "ANALYZE",
      runType: "INITIAL",
      dataCutoff: "2026-09-21",
      baselineSnapshotId: null,
      processVersion: "1.0",
      pilotageContractVersion: "1.0.1",
      contractSetSha256: HASH_D,
      stateVersion: 11,
    },
    stage: {
      runId: RUN_ID,
      stageCode: "RESEARCH",
      stageRevision: 1,
      stageContractName: "OROTITAN_RESEARCH_STAGE_CONTRACT_V1",
      stageContractVersion: "1.0",
      stageContractSha256: HASH_C,
      lifecycleStatus: "IN_PROGRESS",
      handoffGateName: "READY_FOR_DEEP_DIVE",
      handoffGateState: "NOT_EVALUATED",
      activeManifestArtifactId: null,
      activeManifestVersion: null,
      activeManifestKind: null,
      blockerCount: 0,
      stateVersion: 7,
    },
    registeredArtifacts: [
      artifact(),
    ],
  };
}

function validRequest(): PostStageCertificationRequest {
  const input = artifact();
  const output = artifact({
    artifactId: "artifact-output",
    artifactType: "EVIDENCE_LEDGER",
    contentSha256: HASH_B,
    retrievedContentSha256: HASH_B,
    sizeBytes: 222,
    schemaValidationRequired: true,
  });
  const manifestArtifact = artifact({
    artifactId: "manifest-research",
    artifactType: "RESEARCH_STAGE_MANIFEST",
    contentSha256: HASH_C,
    retrievedContentSha256: HASH_C,
    sizeBytes: 333,
    schemaValidationRequired: true,
  });

  return {
    runId: RUN_ID,
    stageCode: "RESEARCH",
    expectedRunStateVersion: 11,
    expectedStageStateVersion: 7,
    requiredOutputTypes: ["EVIDENCE_LEDGER"],
    requiredLineageEdges: [
      {
        childArtifactId: output.artifactId,
        childVersion: output.version,
        parentArtifactId: input.artifactId,
        parentVersion: input.version,
        relationType: "CONSUMES",
      },
    ],
    outputs: [output],
    resolvedInputs: [input],
    manifest: {
      artifact: manifestArtifact,
      body: {
        manifestSchemaVersion: "1.0.0",
        manifestId: manifestArtifact.artifactId,
        manifestKind: "FINAL",
        runId: RUN_ID,
        stage: "RESEARCH",
        stageRevision: 1,
        issuerId: "issuer-gate10",
        securityId: "security-gate10",
        canonicalMode: "ANALYZE",
        runType: "INITIAL",
        dataCutoff: "2026-09-21",
        baselineSnapshotId: null,
        processVersion: "1.0",
        pilotageContractVersion: "1.0.1",
        stageContract: {
          name: "OROTITAN_RESEARCH_STAGE_CONTRACT_V1",
          version: "1.0",
          contentSha256: HASH_C,
        },
        contractSetSha256: HASH_D,
        inputArtifacts: [
          {
            artifactId: input.artifactId,
            version: input.version,
            artifactType: input.artifactType,
            contentSha256: input.contentSha256,
          },
        ],
        outputArtifacts: [
          {
            artifactId: output.artifactId,
            version: output.version,
            artifactType: output.artifactType,
            contentSha256: output.contentSha256,
            authorityClass: output.authorityClass,
            mediaType: output.mediaType,
            sizeBytes: output.sizeBytes,
          },
        ],
        stageStatus: "COMPLETE",
        handoffGate: {
          name: "READY_FOR_DEEP_DIVE",
          state: "YES",
        },
        criticalBlockers: [],
        parentManifests: [],
      },
    },
    lineageEdges: [
      {
        childRunId: RUN_ID,
        childArtifactId: output.artifactId,
        childVersion: output.version,
        parentRunId: RUN_ID,
        parentArtifactId: input.artifactId,
        parentVersion: input.version,
        relationType: "CONSUMES",
      },
    ],
    selfAuditPassed: true,
    contractPinsVerified: true,
    forbiddenMutationCheckPassed: true,
  };
}

type BadFinalizeMode =
  | "NONE"
  | "NO_COMPLETE"
  | "WRONG_POINTER"
  | "DROP_OUTPUT"
  | "IMMUTABLE_DRIFT";

class InMemoryFinalizationStore implements PostStageCertificationStore {
  context: PostFinalizeContext;
  mutationCount = 0;
  badMode: BadFinalizeMode = "NONE";

  constructor(context = baseContext()) {
    this.context = structuredClone(context);
  }

  async readContext(): Promise<PostFinalizeContext> {
    return structuredClone(this.context);
  }

  async finalizeCertifiedStage(input: {
    runId: string;
    stageCode: "RESEARCH" | "DEEP_DIVE" | "INTEGRATION";
    expectedRunStateVersion: number;
    expectedStageStateVersion: number;
    manifestArtifactId: string;
    manifestVersion: number;
    handoffGateName: string;
    outputs: readonly FinalizationArtifact[];
    manifestArtifact: FinalizationArtifact;
    lineageEdges: readonly unknown[];
  }): Promise<void> {
    if (
      this.context.run.runId !== input.runId ||
      this.context.stage.runId !== input.runId ||
      this.context.stage.stageCode !== input.stageCode
    ) {
      throw new Error("FAKE_FINALIZE_IDENTITY_CONFLICT");
    }

    if (
      this.context.run.stateVersion !== input.expectedRunStateVersion ||
      this.context.stage.stateVersion !== input.expectedStageStateVersion
    ) {
      throw new Error("FAKE_FINALIZE_CONCURRENCY_CONFLICT");
    }

    this.mutationCount += 1;

    this.context.stage = {
      ...this.context.stage,
      lifecycleStatus:
        this.badMode === "NO_COMPLETE" ? "IN_PROGRESS" : "COMPLETE",
      handoffGateState: "YES",
      activeManifestArtifactId:
        this.badMode === "WRONG_POINTER"
          ? "wrong-manifest"
          : input.manifestArtifactId,
      activeManifestVersion: input.manifestVersion,
      activeManifestKind: "FINAL",
      stateVersion: this.context.stage.stateVersion + 1,
    };

    if (this.badMode === "IMMUTABLE_DRIFT") {
      this.context.run = {
        ...this.context.run,
        dataCutoff: "2026-09-22",
      };
    }

    const outputs =
      this.badMode === "DROP_OUTPUT" ? [] : [...input.outputs];

    this.context.registeredArtifacts = [
      ...this.context.registeredArtifacts,
      ...outputs,
      input.manifestArtifact,
    ];
  }
}

function failureCodes(
  request: PostStageCertificationRequest,
  context = baseContext(),
): string[] {
  return certifyPostStageBundle(request, context).failures.map(
    (failure) => failure.code,
  );
}

test("Gate 10 certifies and finalizes a valid stage bundle", async () => {
  const request = validRequest();
  const store = new InMemoryFinalizationStore();

  const result = await executeCertifiedStageFinalization(store, request);

  assert.equal(result.certification.status, "PASS");
  assert.equal(store.mutationCount, 1);
  assert.equal(result.after.stage.lifecycleStatus, "COMPLETE");
  assert.equal(result.after.stage.handoffGateState, "YES");
  assert.equal(
    result.after.stage.activeManifestArtifactId,
    "manifest-research",
  );
  assert.equal(result.after.stage.activeManifestKind, "FINAL");
  assert.equal(result.after.stage.stateVersion, 8);
});

test("Gate 10 refuses COMPLETE when a frozen required output is missing", async () => {
  const request = validRequest();
  request.requiredOutputTypes = ["EVIDENCE_LEDGER", "ANALYSIS_INPUT_LOCK"];
  const store = new InMemoryFinalizationStore();

  await assert.rejects(
    executeCertifiedStageFinalization(store, request),
    /REQUIRED_OUTPUT_MISSING/,
  );

  assert.equal(store.mutationCount, 0);
  assert.equal(store.context.stage.lifecycleStatus, "IN_PROGRESS");
});

test("Gate 10 rejects invalid schema, hash, locator and authority before mutation", async () => {
  const request = validRequest();
  request.outputs = [
    {
      ...request.outputs[0],
      authorityState: "SUPERSEDED",
      retrievedContentSha256: HASH_A,
      durableLocatorAvailable: false,
      schemaValidationPassed: false,
    },
  ];
  const store = new InMemoryFinalizationStore();

  const codes = failureCodes(request);
  assert.ok(codes.includes("OUTPUT_NOT_AUTHORITATIVE"));
  assert.ok(codes.includes("OUTPUT_HASH_MISMATCH"));
  assert.ok(codes.includes("OUTPUT_LOCATOR_MISSING"));
  assert.ok(codes.includes("OUTPUT_SCHEMA_INVALID"));

  await assert.rejects(
    executeCertifiedStageFinalization(store, request),
    /VNEXT_POST_STAGE_CERTIFICATION_FAIL/,
  );
  assert.equal(store.mutationCount, 0);
});

test("Gate 10 requires FINAL manifest and exact manifest/output correspondence", () => {
  const request = validRequest();
  request.manifest.body.manifestKind = "CHECKPOINT";
  request.manifest.body.outputArtifacts = [
    {
      ...request.manifest.body.outputArtifacts[0],
      contentSha256: HASH_A,
    },
  ];

  const codes = failureCodes(request);
  assert.ok(codes.includes("MANIFEST_NOT_FINAL"));
  assert.ok(codes.includes("MANIFEST_OUTPUT_MISMATCH"));
});

test("Gate 10 blocks manifest self-reference", () => {
  const request = validRequest();
  request.manifest.body.outputArtifacts = [
    ...request.manifest.body.outputArtifacts,
    {
      artifactId: request.manifest.artifact.artifactId,
      version: request.manifest.artifact.version,
      artifactType: request.manifest.artifact.artifactType,
      contentSha256: request.manifest.artifact.contentSha256,
      authorityClass: request.manifest.artifact.authorityClass,
      mediaType: request.manifest.artifact.mediaType,
      sizeBytes: request.manifest.artifact.sizeBytes,
    },
  ];

  assert.ok(
    failureCodes(request).includes("MANIFEST_SELF_REFERENCE"),
  );
});

test("Gate 10 rejects missing, duplicate, self or unresolved lineage", () => {
  const missing = validRequest();
  missing.lineageEdges = [];
  assert.ok(
    failureCodes(missing).includes("REQUIRED_LINEAGE_EDGE_MISSING"),
  );

  const duplicate = validRequest();
  duplicate.lineageEdges = [
    duplicate.lineageEdges[0],
    duplicate.lineageEdges[0],
  ];
  assert.ok(
    failureCodes(duplicate).includes("LINEAGE_DUPLICATE_EDGE"),
  );

  const self = validRequest();
  self.lineageEdges = [
    {
      childRunId: RUN_ID,
      childArtifactId: "artifact-output",
      childVersion: 1,
      parentRunId: RUN_ID,
      parentArtifactId: "artifact-output",
      parentVersion: 1,
      relationType: "DERIVED_FROM",
    },
  ];
  assert.ok(failureCodes(self).includes("LINEAGE_SELF_EDGE"));

  const unresolved = validRequest();
  unresolved.lineageEdges = [
    {
      childRunId: RUN_ID,
      childArtifactId: "artifact-output",
      childVersion: 1,
      parentRunId: RUN_ID,
      parentArtifactId: "missing-parent",
      parentVersion: 1,
      relationType: "DERIVED_FROM",
    },
  ];
  assert.ok(
    failureCodes(unresolved).includes("LINEAGE_ENDPOINT_UNRESOLVED"),
  );
});

test("Gate 10 supports explicit cross-run refresh lineage", () => {
  const request = validRequest();
  const prior = artifact({
    artifactId: "prior-baseline-artifact",
    runId: PRIOR_RUN_ID,
    contentSha256: HASH_D,
    retrievedContentSha256: HASH_D,
  });

  request.resolvedInputs = [...request.resolvedInputs, prior];
  request.requiredLineageEdges = [
    ...request.requiredLineageEdges,
    {
      childRunId: RUN_ID,
      childArtifactId: "artifact-output",
      childVersion: 1,
      parentRunId: PRIOR_RUN_ID,
      parentArtifactId: prior.artifactId,
      parentVersion: prior.version,
      relationType: "REVALIDATES",
    },
  ];
  request.lineageEdges = [
    ...request.lineageEdges,
    {
      childRunId: RUN_ID,
      childArtifactId: "artifact-output",
      childVersion: 1,
      parentRunId: PRIOR_RUN_ID,
      parentArtifactId: prior.artifactId,
      parentVersion: prior.version,
      relationType: "REVALIDATES",
    },
  ];

  assert.deepEqual(certifyPostStageBundle(request, baseContext()), {
    status: "PASS",
    failures: [],
  });
});

test("Gate 10 fresh reread catches incomplete finalization", async () => {
  const store = new InMemoryFinalizationStore();
  store.badMode = "NO_COMPLETE";

  await assert.rejects(
    executeCertifiedStageFinalization(store, validRequest()),
    /FRESH_REREAD_STAGE_NOT_COMPLETE/,
  );

  assert.equal(store.mutationCount, 1);
});

test("Gate 10 fresh reread catches wrong manifest pointer", async () => {
  const store = new InMemoryFinalizationStore();
  store.badMode = "WRONG_POINTER";

  await assert.rejects(
    executeCertifiedStageFinalization(store, validRequest()),
    /FRESH_REREAD_MANIFEST_POINTER_MISMATCH/,
  );
});

test("Gate 10 fresh reread requires all output registry rows", async () => {
  const store = new InMemoryFinalizationStore();
  store.badMode = "DROP_OUTPUT";

  await assert.rejects(
    executeCertifiedStageFinalization(store, validRequest()),
    /FRESH_REREAD_OUTPUT_REGISTRATION_MISSING/,
  );
});

test("Gate 10 fresh reread catches forbidden immutable drift", async () => {
  const store = new InMemoryFinalizationStore();
  store.badMode = "IMMUTABLE_DRIFT";

  await assert.rejects(
    executeCertifiedStageFinalization(store, validRequest()),
    /FRESH_REREAD_RUN_IMMUTABLE_DRIFT/,
  );
});

test("Gate 10 refuses finalization if self-audit, pins or mutation assurance fail", async () => {
  const request = validRequest();
  request.selfAuditPassed = false;
  request.contractPinsVerified = false;
  request.forbiddenMutationCheckPassed = false;

  const codes = failureCodes(request);
  assert.ok(codes.includes("SELF_AUDIT_FAILED"));
  assert.ok(codes.includes("CONTRACT_PINS_NOT_VERIFIED"));
  assert.ok(codes.includes("FORBIDDEN_MUTATION_CHECK_FAILED"));

  const store = new InMemoryFinalizationStore();
  await assert.rejects(
    executeCertifiedStageFinalization(store, request),
    /VNEXT_POST_STAGE_CERTIFICATION_FAIL/,
  );
  assert.equal(store.mutationCount, 0);
});
