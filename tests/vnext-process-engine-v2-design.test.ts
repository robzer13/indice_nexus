import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import {
  BLOCK_CODES,
  blockCompletionEligibility,
  buildDependencyGraph,
  computeDownstreamClosure,
  computeExecutionFingerprint,
  decideRetry,
  deriveNextBlockAction,
  deriveSaveDisposition,
  planDependencyReopen,
  planRefresh,
  validateRequiredBlockDependencies,
  validateRequiredSectorOverlays,
  type AnalyticalBlockState,
  type BlockCode,
} from "../lib/orotitan-equity/post-c7/process-engine-v2";

function block(
  code: BlockCode,
  status: AnalyticalBlockState["status"] = "COMPLETE",
  upstreamBlockRefs: BlockCode[] = [],
): AnalyticalBlockState {
  return {
    block: code,
    status,
    upstreamBlockRefs,
    criticalUnresolvedGap: false,
    blockingConflict: false,
    sectorOverlays: [],
    materialRevalidationStatus: "NOT_REQUIRED",
    completionAuditPassed: true,
  };
}

function chain(): AnalyticalBlockState[] {
  return [
    block("BUSINESS_MODEL"),
    block("MOAT", "COMPLETE", ["BUSINESS_MODEL"]),
    block("RUNWAY", "COMPLETE", ["MOAT"]),
    block("VALUATION", "COMPLETE", ["RUNWAY"]),
    block("CROSS_BLOCK_RECONCILIATION", "COMPLETE", ["VALUATION"]),
  ];
}

test("Process Engine block vocabulary stays aligned with Data Contracts V2", () => {
  const schema = JSON.parse(
    readFileSync(
      new URL(
        "../contracts/orotitan-equity/post-c7/OROTITAN_ANALYTICAL_DATA_CONTRACTS_V2_SCHEMA_V0.1.json",
        import.meta.url,
      ),
      "utf8",
    ),
  ) as {
    $defs: { blockCode: { enum: string[] } };
  };

  assert.deepEqual([...BLOCK_CODES], schema.$defs.blockCode.enum);
});

test("dependency graph fails closed on cycles", () => {
  const blocks = [
    block("MOAT", "COMPLETE", ["RUNWAY"]),
    block("RUNWAY", "COMPLETE", ["MOAT"]),
  ];
  const result = buildDependencyGraph(blocks);
  assert.equal(result.ok, false);
  if (!result.ok) assert.match(result.errors.join("\n"), /cycle/);
});

test("dependency graph uses canonical block order for independent blocks", () => {
  const result = buildDependencyGraph([
    block("VALUATION"),
    block("BUSINESS_MODEL"),
    block("MOAT"),
  ]);
  assert.equal(result.ok, true);
  if (result.ok) {
    assert.deepEqual(result.topologicalOrder, [
      "BUSINESS_MODEL",
      "MOAT",
      "VALUATION",
    ]);
  }
});

test("downstream closure reopens only the affected dependency cone", () => {
  const blocks = [
    ...chain(),
    block("MANAGEMENT_GOVERNANCE"),
  ];
  const result = computeDownstreamClosure(["MOAT"], blocks);
  assert.equal(result.ok, true);
  if (result.ok) {
    assert.deepEqual(result.blocks, [
      "MOAT",
      "RUNWAY",
      "VALUATION",
      "CROSS_BLOCK_RECONCILIATION",
    ]);
    assert.equal(result.blocks.includes("MANAGEMENT_GOVERNANCE"), false);
  }
});

test("PRICE_ONLY_DELTA preserves fundamentals and forbids fundamental changes", () => {
  const blocks = chain();
  const plan = planRefresh("PRICE_ONLY_DELTA", [], blocks);
  assert.equal(plan.ok, true);
  if (plan.ok) {
    assert.deepEqual(plan.directReopenBlocks, ["VALUATION"]);
    assert.deepEqual(plan.staleBlocks, ["CROSS_BLOCK_RECONCILIATION"]);
    assert.deepEqual(plan.reopenBlocks, [
      "VALUATION",
      "CROSS_BLOCK_RECONCILIATION",
    ]);
    assert.equal(plan.preservedBlocks.includes("MOAT"), true);
    assert.equal(plan.oqsMayChange, false);
    assert.equal(plan.researchMode, "MINIMAL_REVALIDATION");
    assert.equal(plan.fundamentalsMode, "REVALIDATE_PRIOR_LOCK");
  }

  const invalid = planRefresh("PRICE_ONLY_DELTA", ["MOAT"], blocks);
  assert.equal(invalid.ok, false);
  if (!invalid.ok) assert.match(invalid.errors.join("\n"), /cannot declare fundamental block MOAT/);
});

test("ROUTINE_FUNDAMENTAL_DELTA reopens affected and downstream blocks only", () => {
  const blocks = [
    ...chain(),
    block("MANAGEMENT_GOVERNANCE"),
  ];
  const plan = planRefresh("ROUTINE_FUNDAMENTAL_DELTA", ["MOAT"], blocks);
  assert.equal(plan.ok, true);
  if (plan.ok) {
    assert.deepEqual(plan.directReopenBlocks, ["MOAT"]);
    assert.deepEqual(plan.staleBlocks, [
      "RUNWAY",
      "VALUATION",
      "CROSS_BLOCK_RECONCILIATION",
    ]);
    assert.deepEqual(plan.reopenBlocks, [
      "MOAT",
      "RUNWAY",
      "VALUATION",
      "CROSS_BLOCK_RECONCILIATION",
    ]);
    assert.deepEqual(plan.preservedBlocks, [
      "BUSINESS_MODEL",
      "MANAGEMENT_GOVERNANCE",
    ]);
    assert.equal(plan.researchMode, "TARGETED_DELTA");
    assert.equal(plan.fundamentalsMode, "REOPEN_AFFECTED_BLOCKS");
    assert.equal(plan.oqsMayChange, true);
  }
});

test("ROUTINE_FUNDAMENTAL_DELTA requires a fundamental origin", () => {
  const blocks = chain();
  const empty = planRefresh("ROUTINE_FUNDAMENTAL_DELTA", [], blocks);
  assert.equal(empty.ok, false);

  const valuationOnly = planRefresh(
    "ROUTINE_FUNDAMENTAL_DELTA",
    ["VALUATION"],
    blocks,
  );
  assert.equal(valuationOnly.ok, false);
});

test("FULL_REFRESH_REQUIRED reopens every analytical block", () => {
  const blocks = chain();
  const plan = planRefresh("FULL_REFRESH_REQUIRED", [], blocks);
  assert.equal(plan.ok, true);
  if (plan.ok) {
    assert.deepEqual(plan.directReopenBlocks, blocks.map((item) => item.block));
    assert.deepEqual(plan.staleBlocks, []);
    assert.deepEqual(plan.reopenBlocks, blocks.map((item) => item.block));
    assert.deepEqual(plan.preservedBlocks, []);
    assert.equal(plan.researchMode, "FULL");
    assert.equal(plan.fundamentalsMode, "FULL");
  }
});

test("authoritative dependency requirements prevent under-specified graphs", () => {
  const blocks = [
    block("BUSINESS_MODEL"),
    block("MOAT"),
  ];

  const invalid = validateRequiredBlockDependencies(
    [{ block: "MOAT", upstream: "BUSINESS_MODEL" }],
    blocks,
  );
  assert.equal(invalid.ok, false);
  if (!invalid.ok) {
    assert.match(invalid.errors.join("\n"), /missing required upstream dependency/);
  }

  const validBlocks = [
    block("BUSINESS_MODEL"),
    block("MOAT", "COMPLETE", ["BUSINESS_MODEL"]),
  ];
  assert.equal(
    validateRequiredBlockDependencies(
      [{ block: "MOAT", upstream: "BUSINESS_MODEL" }],
      validBlocks,
    ).ok,
    true,
  );
});

test("dependency reopen plan distinguishes direct reopen from stale downstream", () => {
  const result = planDependencyReopen(["MOAT"], chain(), [
    { block: "MOAT", upstream: "BUSINESS_MODEL" },
    { block: "RUNWAY", upstream: "MOAT" },
    { block: "VALUATION", upstream: "RUNWAY" },
  ]);
  assert.equal(result.ok, true);
  if (result.ok) {
    assert.deepEqual(result.plan.directReopenBlocks, ["MOAT"]);
    assert.deepEqual(result.plan.staleDownstreamBlocks, [
      "RUNWAY",
      "VALUATION",
      "CROSS_BLOCK_RECONCILIATION",
    ]);
  }
});

test("required sector overlay must exist exactly once and be APPLIED", () => {
  const blocks = [
    {
      ...block("RETURN_QUALITY"),
      sectorOverlays: [
        { overlay: "BANK_RETURN_ON_EQUITY", status: "APPLIED" as const },
      ],
    },
  ];

  assert.equal(
    validateRequiredSectorOverlays(
      [{ block: "RETURN_QUALITY", overlay: "BANK_RETURN_ON_EQUITY" }],
      blocks,
    ).ok,
    true,
  );

  const missing = validateRequiredSectorOverlays(
    [{ block: "RETURN_QUALITY", overlay: "BANK_CAPITAL_ADEQUACY" }],
    blocks,
  );
  assert.equal(missing.ok, false);

  const wrong = [
    {
      ...blocks[0],
      sectorOverlays: [
        {
          overlay: "BANK_RETURN_ON_EQUITY",
          status: "NOT_APPLICABLE" as const,
        },
      ],
    },
  ];
  const notApplied = validateRequiredSectorOverlays(
    [{ block: "RETURN_QUALITY", overlay: "BANK_RETURN_ON_EQUITY" }],
    wrong,
  );
  assert.equal(notApplied.ok, false);
});

test("block completion fails on material process blockers", () => {
  const base = block("MOAT", "CHECKPOINTED");

  for (const candidate of [
    { ...base, criticalUnresolvedGap: true },
    { ...base, blockingConflict: true },
    {
      ...base,
      sectorOverlays: [
        { overlay: "REQUIRED_TEST", status: "REQUIRED_MISSING" as const },
      ],
    },
    { ...base, materialRevalidationStatus: "PENDING" as const },
    { ...base, materialRevalidationStatus: "FAIL" as const },
    { ...base, completionAuditPassed: false },
  ]) {
    const result = blockCompletionEligibility(candidate, [candidate]);
    assert.equal(result.eligible, false);
  }
});

test("NOT_ASSESSABLE upstream is terminally resolved but CHECKPOINTED is not final", () => {
  const upstreamResolved = block("MOAT", "NOT_ASSESSABLE");
  const downstream = block("RUNWAY", "CHECKPOINTED", ["MOAT"]);

  assert.equal(
    blockCompletionEligibility(downstream, [
      upstreamResolved,
      downstream,
    ]).eligible,
    true,
  );

  const upstreamCheckpoint = block("MOAT", "CHECKPOINTED");
  assert.equal(
    blockCompletionEligibility(downstream, [
      upstreamCheckpoint,
      downstream,
    ]).eligible,
    false,
  );
});

test("next-action resolver surfaces a blocker instead of re-executing it", () => {
  const blocks = [
    block("BUSINESS_MODEL"),
    {
      ...block("MOAT", "BLOCKED", ["BUSINESS_MODEL"]),
      criticalUnresolvedGap: true,
    },
    block("RUNWAY", "READY", ["MOAT"]),
  ];

  const next = deriveNextBlockAction(blocks);
  assert.equal(next.action, "RESOLVE_BLOCKER");
  assert.equal(next.block, "MOAT");
});

test("current executable block is preserved before an unrelated later blocker", () => {
  const blocks = [
    block("BUSINESS_MODEL", "IN_PROGRESS"),
    {
      ...block("MANAGEMENT_GOVERNANCE", "BLOCKED"),
      criticalUnresolvedGap: true,
    },
  ];

  const next = deriveNextBlockAction(
    blocks,
    [],
    "BUSINESS_MODEL",
  );
  assert.equal(next.action, "EXECUTE_BLOCK");
  assert.equal(next.block, "BUSINESS_MODEL");
});

test("empty analytical state fails closed", () => {
  const next = deriveNextBlockAction([]);
  assert.equal(next.action, "FAIL_CLOSED");
  assert.equal(next.block, null);
});

test("checkpointed upstream may feed provisional downstream execution", () => {
  const blocks = [
    block("BUSINESS_MODEL", "CHECKPOINTED"),
    block("MOAT", "READY", ["BUSINESS_MODEL"]),
  ];
  const next = deriveNextBlockAction(blocks);
  assert.equal(next.action, "EXECUTE_BLOCK");
  assert.equal(next.block, "MOAT");
});

test("checkpointed block finalizes only when terminal dependencies are resolved", () => {
  const blocks = [
    block("BUSINESS_MODEL"),
    block("MOAT", "CHECKPOINTED", ["BUSINESS_MODEL"]),
  ];
  const next = deriveNextBlockAction(blocks);
  assert.equal(next.action, "FINALIZE_BLOCK");
  assert.equal(next.block, "MOAT");
});

test("all terminally resolved blocks require no further block action", () => {
  const next = deriveNextBlockAction([
    block("BUSINESS_MODEL", "COMPLETE"),
    block("MOAT", "NOT_ASSESSABLE", ["BUSINESS_MODEL"]),
  ]);
  assert.equal(next.action, "NO_BLOCK_ACTION");
  assert.equal(next.block, null);
});

const saveBase = {
  registryLifecycle: "IN_PROGRESS" as const,
  selfAuditPassed: true,
  schemaValidationPassed: true,
  identityVersionChecksPassed: true,
  analyticalReconciliationPassed: true,
  requiredArtifactsPresent: true,
  persistenceVerified: true,
  criticalBlockers: [] as string[],
  phaseGate: "YES" as const,
};

test("SAVE finalizes only eligible stage boundaries and never publishes", () => {
  const research = deriveSaveDisposition({
    ...saveBase,
    phase: "RESEARCH",
  });
  assert.equal(research.action, "FINALIZE");
  assert.equal(research.publishAuthorized, false);

  const certification = deriveSaveDisposition({
    ...saveBase,
    phase: "CERTIFICATION",
  });
  assert.equal(certification.action, "FINALIZE");
  assert.equal(certification.publishAuthorized, false);

  const integration = deriveSaveDisposition({
    ...saveBase,
    phase: "INTEGRATION",
  });
  assert.equal(integration.action, "FINALIZE");
  assert.equal(integration.publishAuthorized, false);

  const fundamentals = deriveSaveDisposition({
    ...saveBase,
    phase: "FUNDAMENTALS",
  });
  assert.equal(fundamentals.action, "CHECKPOINT");
  assert.equal(fundamentals.publishAuthorized, false);
});

test("SAVE lifecycle prevents finalization from NOT_STARTED or PAUSED state", () => {
  const notStarted = deriveSaveDisposition({
    ...saveBase,
    phase: "RESEARCH",
    registryLifecycle: "NOT_STARTED",
  });
  assert.equal(notStarted.action, "NOOP");

  const paused = deriveSaveDisposition({
    ...saveBase,
    phase: "RESEARCH",
    registryLifecycle: "PAUSED",
  });
  assert.equal(paused.action, "CHECKPOINT");

  const complete = deriveSaveDisposition({
    ...saveBase,
    phase: "RESEARCH",
    registryLifecycle: "COMPLETE",
  });
  assert.equal(complete.action, "NOOP");
});

test("SAVE gate NO checkpoints unless a real blocker exists", () => {
  const notReady = deriveSaveDisposition({
    ...saveBase,
    phase: "RESEARCH",
    phaseGate: "NO",
  });
  assert.equal(notReady.action, "CHECKPOINT");

  const blocked = deriveSaveDisposition({
    ...saveBase,
    phase: "RESEARCH",
    criticalBlockers: ["critical input missing"],
  });
  assert.equal(blocked.action, "BLOCK");
});

test("SAVE cannot finalize before durable prerequisites pass", () => {
  const result = deriveSaveDisposition({
    ...saveBase,
    phase: "RESEARCH",
    persistenceVerified: false,
  });
  assert.equal(result.action, "CHECKPOINT");
});

test("execution fingerprint is stable and changes with authoritative input", () => {
  const base = {
    runId: "11111111-1111-4111-8111-111111111111",
    block: "MOAT" as const,
    inputVersion: "7",
    evidenceSetHash: "a".repeat(64),
    methodVersion: "V2",
    outputSchemaVersion: "0.1",
  };

  const first = computeExecutionFingerprint(base);
  const second = computeExecutionFingerprint(base);
  assert.equal(first, second);
  assert.match(first, /^[a-f0-9]{64}$/);

  const changed = computeExecutionFingerprint({
    ...base,
    evidenceSetHash: "b".repeat(64),
  });
  assert.notEqual(first, changed);
});

test("unchanged fingerprint cannot loop without a justified retry condition", () => {
  const common = {
    priorFingerprint: "same",
    nextFingerprint: "same",
    retryBudgetExhausted: false,
    newEvidence: false,
    resolvedConflict: false,
    methodChanged: false,
    deterministicBugFixed: false,
    criticalUserInputAdded: false,
    forensicJustification: false,
  };

  assert.equal(decideRetry(common).allowed, false);
  assert.equal(
    decideRetry({ ...common, forensicJustification: true }).allowed,
    true,
  );
  assert.equal(
    decideRetry({
      ...common,
      forensicJustification: true,
      retryBudgetExhausted: true,
    }).allowed,
    false,
  );
});
