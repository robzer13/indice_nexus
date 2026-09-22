import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import corpusJson from "../calibration/vnext/AUDIT_CORPUS_VNEXT_V1.json";

import {
  AUDIT_CORPUS_VNEXT_V1_ID,
  AUDIT_CORPUS_VNEXT_V1_SIZE,
  assertNoProductionMutation,
  assertProductionBaselineMatchesCorpus,
  assertValidAuditCorpus,
  assessGate17Exit,
  runVNextShadowComparison,
  type AuditCorpusManifest,
  type ProductionFingerprint,
  type ShadowDivergence,
  type ShadowRunnerDependencies,
  type VnextShadowExecutionInput,
} from "../runtime/vnext/shadow-runner";

const corpus = corpusJson as unknown as AuditCorpusManifest;

type TestV2State = {
  version: "V2";
  issuerId: string;
};

type TestVnextState = {
  version: "VNEXT";
  issuerId: string;
};

function makeDependencies(
  options: {
    failIssuerId?: string;
    mismatchedBaselineIssuerId?: string;
    zeroDivergences?: boolean;
    duplicateChangeForFirstTwo?: boolean;
    onExecutionInput?: (
      input: VnextShadowExecutionInput,
    ) => void;
    onBaselineLoad?: (issuerId: string) => void;
  } = {},
): ShadowRunnerDependencies<TestV2State, TestVnextState> {
  return {
    async executeVNext(input) {
      options.onExecutionInput?.(input);

      if (input.issuerId === options.failIssuerId) {
        throw new Error("SYNTHETIC_EXECUTOR_FAILURE");
      }

      return {
        executionId: `EXEC-${input.issuerId}`,
        state: {
          version: "VNEXT",
          issuerId: input.issuerId,
        },
        evidenceRefs: [input.sourceRunId],
        moduleRefs: ["SHADOW_TEST_MODULE"],
      };
    },

    async loadV2Baseline(member) {
      options.onBaselineLoad?.(member.issuer_id);

      return {
        snapshotId: member.v2_snapshot_id,
        payloadSha256:
          member.issuer_id === options.mismatchedBaselineIssuerId
            ? "0".repeat(64)
            : member.v2_payload_sha256,
        state: {
          version: "V2",
          issuerId: member.issuer_id,
        },
      };
    },

    async compare({ member, v2, vnext }) {
      if (options.zeroDivergences) {
        return [];
      }

      const firstTwo = corpus.members
        .slice(0, 2)
        .map((entry) => entry.issuer_id);

      const changeId =
        options.duplicateChangeForFirstTwo &&
        firstTwo.includes(member.issuer_id)
          ? "CHANGE-DUPLICATE"
          : `CHANGE-${member.issuer_id}`;

      return [
        {
          CHANGE_ID: changeId,
          V2_STATE: v2.state,
          VNEXT_STATE: vnext.state,
          CAUSE: "SYNTHETIC_RUNNER_ASSURANCE",
          MODULE: "SHADOW_TEST_MODULE",
          EVIDENCE: [member.v2_source_run_id],
          ECONOMIC_INTERPRETATION:
            "Synthetic comparison used only to validate Gate 17 runner mechanics.",
        },
      ];
    },
  };
}

test("Gate 17 corpus is exactly the pinned 33-member V1 corpus", () => {
  assert.doesNotThrow(() => assertValidAuditCorpus(corpus));
  assert.equal(corpus.format, AUDIT_CORPUS_VNEXT_V1_ID);
  assert.equal(
    corpus.expected_member_count,
    AUDIT_CORPUS_VNEXT_V1_SIZE,
  );
  assert.equal(corpus.members.length, 33);

  assert.equal(
    corpus.source.production_project_ref,
    "cugpgtzygqqlxetyeven",
  );
  assert.equal(
    corpus.source.vnext_shadow_project_ref,
    "awgsurdyvsyolcgpnygh",
  );

  assert.equal(
    new Set(corpus.members.map((member) => member.issuer_id)).size,
    33,
  );
  assert.equal(
    new Set(
      corpus.members.map((member) => member.v2_snapshot_id),
    ).size,
    33,
  );
  assert.equal(
    new Set(
      corpus.members.map((member) => member.v2_source_run_id),
    ).size,
    33,
  );
});

test("33/33 deterministic shadow execution completes with no dropped members", async () => {
  const executionInputs: VnextShadowExecutionInput[] = [];
  const events: string[] = [];

  const result = await runVNextShadowComparison(
    corpus,
    makeDependencies({
      onExecutionInput(input) {
        executionInputs.push(input);
        events.push(`vnext:${input.issuerId}`);
      },
      onBaselineLoad(issuerId) {
        events.push(`v2:${issuerId}`);
      },
    }),
  );

  assert.equal(result.status, "COMPLETE");
  assert.equal(result.attemptedMemberCount, 33);
  assert.equal(result.completedMemberCount, 33);
  assert.equal(result.failedMemberCount, 0);
  assert.equal(result.members.length, 33);

  for (const member of corpus.members) {
    const vnextIndex = events.indexOf(
      `vnext:${member.issuer_id}`,
    );
    const v2Index = events.indexOf(`v2:${member.issuer_id}`);

    assert.notEqual(vnextIndex, -1);
    assert.notEqual(v2Index, -1);
    assert.equal(vnextIndex < v2Index, true);
  }

  for (const input of executionInputs) {
    const keys = Object.keys(input);

    assert.equal(keys.includes("v2PayloadSha256"), false);
    assert.equal(keys.includes("v2SnapshotId"), false);
    assert.equal(keys.includes("v2State"), false);
    assert.equal(keys.includes("v2CanonicalPayload"), false);
  }
});

test("zero divergences is a valid completed comparison", async () => {
  const result = await runVNextShadowComparison(
    corpus,
    makeDependencies({ zeroDivergences: true }),
  );

  assert.equal(result.status, "COMPLETE");
  assert.equal(result.completedMemberCount, 33);

  for (const member of result.members) {
    assert.equal(member.status, "COMPLETE");

    if (member.status === "COMPLETE") {
      assert.equal(member.changeCount, 0);
      assert.deepEqual(member.changes, []);
    }
  }
});

test("one member failure does not prevent all 33 members from being attempted", async () => {
  const failingIssuerId = corpus.members[9].issuer_id;
  let executionAttempts = 0;
  let baselineLoads = 0;

  const result = await runVNextShadowComparison(
    corpus,
    makeDependencies({
      failIssuerId: failingIssuerId,
      onExecutionInput() {
        executionAttempts += 1;
      },
      onBaselineLoad() {
        baselineLoads += 1;
      },
    }),
  );

  assert.equal(executionAttempts, 33);
  assert.equal(baselineLoads, 32);
  assert.equal(result.attemptedMemberCount, 33);
  assert.equal(result.completedMemberCount, 32);
  assert.equal(result.failedMemberCount, 1);
  assert.equal(result.status, "FAILED");

  const failed = result.members.find(
    (member) => member.issuerId === failingIssuerId,
  );

  assert.equal(failed?.status, "FAILED");
});

test("pinned V2 baseline hash mismatch fails that member without substitution", async () => {
  const mismatchedIssuerId = corpus.members[4].issuer_id;

  const result = await runVNextShadowComparison(
    corpus,
    makeDependencies({
      mismatchedBaselineIssuerId: mismatchedIssuerId,
    }),
  );

  const failed = result.members.find(
    (member) => member.issuerId === mismatchedIssuerId,
  );

  assert.equal(failed?.status, "FAILED");

  if (failed?.status === "FAILED") {
    assert.equal(
      failed.errorCode,
      "VNEXT_SHADOW_V2_PAYLOAD_HASH_MISMATCH",
    );
  }

  assert.equal(result.attemptedMemberCount, 33);
  assert.equal(result.failedMemberCount, 1);
});

test("duplicate CHANGE_ID fails closed while the corpus continues", async () => {
  const result = await runVNextShadowComparison(
    corpus,
    makeDependencies({
      duplicateChangeForFirstTwo: true,
    }),
  );

  assert.equal(result.attemptedMemberCount, 33);
  assert.equal(result.failedMemberCount, 1);

  const second = result.members[1];

  assert.equal(second.status, "FAILED");

  if (second.status === "FAILED") {
    assert.equal(
      second.errorCode,
      "VNEXT_SHADOW_DUPLICATE_CHANGE_ID",
    );
  }
});

test("divergence record rejects fields outside the exact seven-field schema", async () => {
  const dependencies = makeDependencies();

  dependencies.compare = async ({ member, v2, vnext }) => {
    const malformed = {
      CHANGE_ID: `CHANGE-${member.issuer_id}`,
      V2_STATE: v2.state,
      VNEXT_STATE: vnext.state,
      CAUSE: "SYNTHETIC_RUNNER_ASSURANCE",
      MODULE: "SHADOW_TEST_MODULE",
      EVIDENCE: [member.v2_source_run_id],
      ECONOMIC_INTERPRETATION: "Synthetic test.",
      EXTRA_FIELD: "NOT_ALLOWED",
    } as ShadowDivergence;

    return [malformed];
  };

  const result = await runVNextShadowComparison(
    corpus,
    dependencies,
  );

  assert.equal(result.failedMemberCount, 33);

  for (const member of result.members) {
    assert.equal(member.status, "FAILED");

    if (member.status === "FAILED") {
      assert.equal(
        member.errorCode,
        "VNEXT_SHADOW_CHANGE_FIELD_SET_INVALID",
      );
    }
  }
});

test("production baseline must match the pinned pre-Gate-17 fingerprint", () => {
  const before = corpus.production_before_fingerprint;

  assert.doesNotThrow(() =>
    assertProductionBaselineMatchesCorpus(corpus, before),
  );

  const drifted: ProductionFingerprint = {
    ...before,
    orotitan_runs: before.orotitan_runs + 1,
  };

  assert.throws(
    () =>
      assertProductionBaselineMatchesCorpus(corpus, drifted),
    /VNEXT_SHADOW_PRODUCTION_BASELINE_DRIFT/,
  );
});

test("Gate 17 exit passes only for 33/33 plus unchanged production", async () => {
  const run = await runVNextShadowComparison(
    corpus,
    makeDependencies({ zeroDivergences: true }),
  );

  const before = corpus.production_before_fingerprint;
  const after: ProductionFingerprint = { ...before };

  assert.doesNotThrow(() =>
    assertNoProductionMutation(before, after),
  );

  const assessment = assessGate17Exit(
    corpus,
    run,
    before,
    after,
  );

  assert.deepEqual(assessment, {
    gate: 17,
    corpusId: "AUDIT_CORPUS_VNEXT_V1",
    status: "PASS",
    expectedMemberCount: 33,
    attemptedMemberCount: 33,
    completedMemberCount: 33,
    failedMemberCount: 0,
    productionUnchanged: true,
  });
});

test("any production fingerprint mutation makes Gate 17 exit fail", async () => {
  const run = await runVNextShadowComparison(
    corpus,
    makeDependencies({ zeroDivergences: true }),
  );

  const before = corpus.production_before_fingerprint;
  const after: ProductionFingerprint = {
    ...before,
    research_snapshots: before.research_snapshots + 1,
  };

  assert.throws(
    () => assertNoProductionMutation(before, after),
    /VNEXT_SHADOW_PRODUCTION_POLLUTION_DETECTED/,
  );

  const assessment = assessGate17Exit(
    corpus,
    run,
    before,
    after,
  );

  assert.equal(assessment.status, "FAIL");
  assert.equal(assessment.productionUnchanged, false);
});

test("Gate 17 runtime is provider-neutral and has no persistence or publication path", () => {
  const source = readFileSync(
    "runtime/vnext/shadow-runner.ts",
    "utf8",
  );

  const forbidden = [
    /createClient\s*\(/,
    /supabase/i,
    /process\.env/,
    /fetch\s*\(/,
    /\.(insert|update|upsert|delete)\s*\(/,
    /cugpgtzygqqlxetyeven/,
    /openai\.azure\.com/i,
    /AzureProvider/,
    /OpenAIProvider/,
    /AnthropicProvider/,
    /GeminiProvider/,
    /GO PUBLISH/,
    /current_snapshot_id/,
  ];

  for (const pattern of forbidden) {
    assert.equal(
      pattern.test(source),
      false,
      `unexpected Gate 17 runtime dependency: ${pattern}`,
    );
  }
});
