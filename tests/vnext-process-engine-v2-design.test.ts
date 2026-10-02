import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import Ajv2020 from "ajv/dist/2020";
import addFormats from "ajv-formats";

import {
  BLOCK_CODES,
  blockCompletionEligibility,
  buildDependencyGraph,
  computeDownstreamClosure,
  computeExecutionFingerprint,
  decideRetry,
  deriveNextBlockAction,
  deriveReopenTransitions,
  deriveSaveDisposition,
  evaluateMaterialRevalidationGate,
  planDependencyReopen,
  planRefresh,
  validateRequiredBlockDependencies,
  validateRequiredSectorOverlays,
  type AnalyticalBlockState,
  type AnalyticalExecutionStatus,
  type BlockCode,
  type BlockFreshness,
} from "../lib/orotitan-equity/post-c7/process-engine-v2";

function block(
  code: BlockCode,
  analyticalStatus: AnalyticalExecutionStatus = "LOCKED",
  upstreamBlockRefs: BlockCode[] = [],
  freshness: BlockFreshness = "CURRENT",
): AnalyticalBlockState {
  return {
    block: code,
    presence: "PRESENT",
    analyticalStatus,
    freshness,
    upstreamBlockRefs,
    criticalUnresolvedGap: false,
    blockingConflict: false,
    sectorOverlays: [],
    materialRevalidationStatus: "NOT_REQUIRED",
    completionAuditPassed: true,
  };
}

function notStarted(
  code: BlockCode,
  upstreamBlockRefs: BlockCode[] = [],
): AnalyticalBlockState {
  return {
    block: code,
    presence: "NOT_STARTED",
    analyticalStatus: null,
    freshness: "CURRENT",
    upstreamBlockRefs,
    criticalUnresolvedGap: false,
    blockingConflict: false,
    sectorOverlays: [],
    materialRevalidationStatus: "NOT_REQUIRED",
    completionAuditPassed: false,
  };
}

function chain(): AnalyticalBlockState[] {
  return [
    block("BUSINESS_MODEL"),
    block("MOAT", "LOCKED", ["BUSINESS_MODEL"]),
    block("RUNWAY", "LOCKED", ["MOAT"]),
    block("VALUATION", "LOCKED", ["RUNWAY"]),
    block("CROSS_BLOCK_RECONCILIATION", "LOCKED", ["VALUATION"]),
  ];
}

function loadDataCommon() {
  return JSON.parse(
    readFileSync(
      new URL(
        "../schemas/vnext/data-contracts/orotitan-analytical-common.schema.v0.1.json",
        import.meta.url,
      ),
      "utf8",
    ),
  );
}

function loadProcessSchema() {
  return JSON.parse(
    readFileSync(
      new URL(
        "../schemas/vnext/process-engine/process-state.schema.v0.1.json",
        import.meta.url,
      ),
      "utf8",
    ),
  );
}

test("Process Engine block vocabulary matches frozen Data Contracts V2", () => {
  const common = loadDataCommon();
  assert.deepEqual([...BLOCK_CODES], common.$defs.blockCode.enum);
});

test("analytical status stays frozen while process freshness is orthogonal", () => {
  const common = loadDataCommon();
  const process = loadProcessSchema();
  const row = process.properties.blocks.items.properties;

  assert.deepEqual(common.$defs.blockExecutionStatus.enum, [
    "INSUFFICIENT",
    "IN_PROGRESS",
    "PROVISIONALLY_STABLE",
    "LOCKED",
  ]);
  assert.deepEqual(row.freshness.enum, ["CURRENT", "REOPENED", "STALE"]);
  assert.equal(
    row.analytical_execution_status.anyOf[0].$ref,
    "../data-contracts/orotitan-analytical-common.v0.1.json#/$defs/blockExecutionStatus",
  );
  for (const forbidden of ["READY", "CHECKPOINTED", "COMPLETE", "BLOCKED", "REOPENED", "STALE"]) {
    assert.equal(common.$defs.blockExecutionStatus.enum.includes(forbidden), false);
  }
});

test("process-state schema compiles against frozen Data Contracts V2", () => {
  const ajv = new Ajv2020({ allErrors: true, strict: false });
  addFormats(ajv as unknown as Parameters<typeof addFormats>[0]);
  const common = loadDataCommon();
  const process = loadProcessSchema();
  ajv.addSchema(common);
  ajv.addSchema(process);
  assert.ok(ajv.getSchema(process.$id));
});

test("process-state schema accepts orthogonal valid state", () => {
  const ajv = new Ajv2020({ allErrors: true, strict: false });
  const common = loadDataCommon();
  const process = loadProcessSchema();
  ajv.addSchema(common);
  const validate = ajv.compile(process);
  const sha = "a".repeat(64);

  const valid = validate({
    context: {
      schema_name: "OROTITAN_PROCESS_ENGINE_V2_STATE",
      schema_version: "0.1.0",
      run_id: "RUN-1",
      stage_code: "DEEP_DIVE",
      stage_revision: 1,
      data_cutoff: "2026-10-02",
      contract_set_sha256: sha,
      generated_at: "2026-10-02T00:00:00Z",
    },
    process_state_version: "1",
    current_block: "MOAT",
    blocks: [
      {
        block_id: "MOAT",
        presence: "PRESENT",
        analytical_execution_status: "PROVISIONALLY_STABLE",
        freshness: "CURRENT",
        upstream_block_refs: [],
        critical_unresolved_gap: false,
        blocking_conflict: false,
        sector_overlays: [],
        material_revalidation_status: "NOT_REQUIRED",
        completion_audit_passed: true,
        last_execution_fingerprint: null,
        retry_count: 0,
      },
    ],
  });
  assert.equal(valid, true, JSON.stringify(validate.errors, null, 2));
});

test("dependency graph fails closed on impossible state and cycles", () => {
  const impossible = {
    ...notStarted("MOAT"),
    analyticalStatus: "LOCKED" as const,
  };
  const badState = buildDependencyGraph([impossible]);
  assert.equal(badState.ok, false);
  if (!badState.ok) assert.match(badState.errors.join("\n"), /NOT_STARTED/);

  const cycle = buildDependencyGraph([
    block("MOAT", "LOCKED", ["RUNWAY"]),
    block("RUNWAY", "LOCKED", ["MOAT"]),
  ]);
  assert.equal(cycle.ok, false);
  if (!cycle.ok) assert.match(cycle.errors.join("\n"), /cycle/);
});

test("dependency graph uses canonical order for independent blocks", () => {
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

test("downstream closure reopens only affected dependency cone", () => {
  const blocks = [...chain(), block("MANAGEMENT_GOVERNANCE")];
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

test("reopen plan keeps analytical status separate from freshness transition", () => {
  const result = planDependencyReopen(["MOAT"], chain(), [
    { block: "MOAT", upstream: "BUSINESS_MODEL" },
    { block: "RUNWAY", upstream: "MOAT" },
    { block: "VALUATION", upstream: "RUNWAY" },
  ]);
  assert.equal(result.ok, true);
  if (!result.ok) return;

  assert.deepEqual(result.plan.directReopenBlocks, ["MOAT"]);
  assert.deepEqual(result.plan.staleDownstreamBlocks, [
    "RUNWAY",
    "VALUATION",
    "CROSS_BLOCK_RECONCILIATION",
  ]);
  assert.deepEqual(deriveReopenTransitions(result.plan), [
    { block: "MOAT", freshness: "REOPENED", materialRevalidationRequired: true },
    { block: "RUNWAY", freshness: "STALE", materialRevalidationRequired: true },
    { block: "VALUATION", freshness: "STALE", materialRevalidationRequired: true },
    {
      block: "CROSS_BLOCK_RECONCILIATION",
      freshness: "STALE",
      materialRevalidationRequired: true,
    },
  ]);
});

test("authoritative dependency requirements prevent under-specified graphs", () => {
  const invalid = validateRequiredBlockDependencies(
    [{ block: "MOAT", upstream: "BUSINESS_MODEL" }],
    [block("BUSINESS_MODEL"), block("MOAT")],
  );
  assert.equal(invalid.ok, false);

  const valid = validateRequiredBlockDependencies(
    [{ block: "MOAT", upstream: "BUSINESS_MODEL" }],
    [block("BUSINESS_MODEL"), block("MOAT", "LOCKED", ["BUSINESS_MODEL"])],
  );
  assert.equal(valid.ok, true);
});

test("six-part material-change revalidation gate is deterministic", () => {
  const base = {
    required: true,
    reopenedMaterialEvidence: true,
    rootPrimarySourcesVerified: true,
    disconfirmingEvidenceSearched: true,
    bestAlternativeExplanationTested: true,
    affectedDownstreamBlocksReconciled: true,
    priorStateChangeReasonRecorded: true,
    hardFailureReasons: [] as string[],
  };

  assert.equal(evaluateMaterialRevalidationGate(base).status, "PASS");

  const pending = evaluateMaterialRevalidationGate({
    ...base,
    disconfirmingEvidenceSearched: false,
  });
  assert.equal(pending.status, "PENDING");
  assert.deepEqual(pending.missingChecks, ["SEARCH_DISCONFIRMING_EVIDENCE"]);

  const failed = evaluateMaterialRevalidationGate({
    ...base,
    hardFailureReasons: ["root source contradiction unresolved"],
  });
  assert.equal(failed.status, "FAIL");

  assert.equal(
    evaluateMaterialRevalidationGate({ ...base, required: false }).status,
    "NOT_REQUIRED",
  );
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

  assert.equal(
    validateRequiredSectorOverlays(
      [{ block: "RETURN_QUALITY", overlay: "BANK_CAPITAL_ADEQUACY" }],
      blocks,
    ).ok,
    false,
  );
});

test("block completion fails on every material process blocker", () => {
  const base = block("MOAT", "PROVISIONALLY_STABLE");

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
    { ...base, freshness: "STALE" as const },
  ]) {
    assert.equal(
      blockCompletionEligibility(candidate, [candidate]).eligible,
      false,
    );
  }
});

test("NOT_STARTED and INSUFFICIENT cannot jump to LOCKED", () => {
  const fresh = notStarted("MOAT");
  assert.equal(blockCompletionEligibility(fresh, [fresh]).eligible, false);

  const insufficient = block("MOAT", "INSUFFICIENT");
  assert.equal(
    blockCompletionEligibility(insufficient, [insufficient]).eligible,
    false,
  );
});

test("PROVISIONALLY_STABLE may feed downstream work but cannot satisfy terminal dependency lock", () => {
  const upstream = block("BUSINESS_MODEL", "PROVISIONALLY_STABLE");
  const downstream = notStarted("MOAT", ["BUSINESS_MODEL"]);

  const next = deriveNextBlockAction([upstream, downstream]);
  assert.equal(next.action, "EXECUTE_BLOCK");
  assert.equal(next.block, "MOAT");

  const downstreamStable = block(
    "MOAT",
    "PROVISIONALLY_STABLE",
    ["BUSINESS_MODEL"],
  );
  const completion = blockCompletionEligibility(downstreamStable, [
    upstream,
    downstreamStable,
  ]);
  assert.equal(completion.eligible, false);
});

test("PROVISIONALLY_STABLE block finalizes to analytical LOCKED only with terminal upstream", () => {
  const blocks = [
    block("BUSINESS_MODEL"),
    block("MOAT", "PROVISIONALLY_STABLE", ["BUSINESS_MODEL"]),
  ];
  const next = deriveNextBlockAction(blocks);
  assert.equal(next.action, "FINALIZE_BLOCK");
  assert.equal(next.block, "MOAT");
});

test("persisted LOCKED block fails closed if completion conditions are inconsistent", () => {
  const invalid = {
    ...block("MOAT"),
    completionAuditPassed: false,
  };
  const next = deriveNextBlockAction([invalid]);
  assert.equal(next.action, "FAIL_CLOSED");
  assert.match(next.reason, /LOCKED block MOAT violates completion conditions/);
});

test("reopened or stale block is executable without changing analytical status vocabulary", () => {
  for (const freshness of ["REOPENED", "STALE"] as const) {
    const reopened = block("MOAT", "LOCKED", [], freshness);
    const next = deriveNextBlockAction([reopened]);
    assert.equal(next.action, "EXECUTE_BLOCK");
    assert.equal(next.block, "MOAT");
  }
});

test("pending material revalidation returns block to execution work", () => {
  const value = {
    ...block("MOAT", "PROVISIONALLY_STABLE"),
    materialRevalidationStatus: "PENDING" as const,
  };
  const next = deriveNextBlockAction([value]);
  assert.equal(next.action, "EXECUTE_BLOCK");
  assert.equal(next.block, "MOAT");
});

test("next-action resolver surfaces blocker instead of repeating analysis", () => {
  const blocks = [
    block("BUSINESS_MODEL"),
    {
      ...block("MOAT", "INSUFFICIENT", ["BUSINESS_MODEL"]),
      criticalUnresolvedGap: true,
    },
    notStarted("RUNWAY", ["MOAT"]),
  ];
  const next = deriveNextBlockAction(blocks);
  assert.equal(next.action, "RESOLVE_BLOCKER");
  assert.equal(next.block, "MOAT");
});

test("current executable block is preserved before unrelated later blocker", () => {
  const blocks = [
    block("BUSINESS_MODEL", "IN_PROGRESS"),
    {
      ...block("MANAGEMENT_GOVERNANCE", "INSUFFICIENT"),
      criticalUnresolvedGap: true,
    },
  ];
  const next = deriveNextBlockAction(blocks, [], "BUSINESS_MODEL");
  assert.equal(next.action, "EXECUTE_BLOCK");
  assert.equal(next.block, "BUSINESS_MODEL");
});

test("all LOCKED/CURRENT blocks require no further block action", () => {
  const next = deriveNextBlockAction([
    block("BUSINESS_MODEL"),
    block("MOAT", "LOCKED", ["BUSINESS_MODEL"]),
  ]);
  assert.equal(next.action, "NO_BLOCK_ACTION");
  assert.equal(next.block, null);
});

test("next-action resolver fails closed when a required overlay targets an absent block", () => {
  const next = deriveNextBlockAction(
    [block("BUSINESS_MODEL")],
    [{ block: "RETURN_QUALITY", overlay: "BANK_RETURN_ON_EQUITY" }],
  );
  assert.equal(next.action, "FAIL_CLOSED");
  assert.match(next.reason, /references absent block RETURN_QUALITY/);
});

test("PRICE_ONLY_DELTA preserves fundamentals and forbids fundamental changes", () => {
  const blocks = chain();
  const plan = planRefresh("PRICE_ONLY_DELTA", [], blocks);
  assert.equal(plan.ok, true);
  if (plan.ok) {
    assert.deepEqual(plan.directReopenBlocks, ["VALUATION"]);
    assert.deepEqual(plan.staleBlocks, ["CROSS_BLOCK_RECONCILIATION"]);
    assert.equal(plan.preservedBlocks.includes("MOAT"), true);
    assert.equal(plan.oqsMayChange, false);
    assert.equal(plan.researchMode, "MINIMAL_REVALIDATION");
    assert.equal(plan.fundamentalsMode, "REVALIDATE_PRIOR_LOCK");
  }

  const invalid = planRefresh("PRICE_ONLY_DELTA", ["MOAT"], blocks);
  assert.equal(invalid.ok, false);
});

test("ROUTINE_FUNDAMENTAL_DELTA reopens affected dependency cone only", () => {
  const blocks = [...chain(), block("MANAGEMENT_GOVERNANCE")];
  const plan = planRefresh("ROUTINE_FUNDAMENTAL_DELTA", ["MOAT"], blocks);
  assert.equal(plan.ok, true);
  if (plan.ok) {
    assert.deepEqual(plan.directReopenBlocks, ["MOAT"]);
    assert.deepEqual(plan.staleBlocks, [
      "RUNWAY",
      "VALUATION",
      "CROSS_BLOCK_RECONCILIATION",
    ]);
    assert.deepEqual(plan.preservedBlocks, [
      "BUSINESS_MODEL",
      "MANAGEMENT_GOVERNANCE",
    ]);
    assert.equal(plan.oqsMayChange, true);
  }
});

test("ROUTINE_FUNDAMENTAL_DELTA requires a fundamental origin", () => {
  assert.equal(
    planRefresh("ROUTINE_FUNDAMENTAL_DELTA", [], chain()).ok,
    false,
  );
  assert.equal(
    planRefresh("ROUTINE_FUNDAMENTAL_DELTA", ["VALUATION"], chain()).ok,
    false,
  );
});

test("FULL_REFRESH_REQUIRED invalidates every previously executed analytical block", () => {
  const blocks = chain();
  const plan = planRefresh("FULL_REFRESH_REQUIRED", [], blocks);
  assert.equal(plan.ok, true);
  if (plan.ok) {
    assert.deepEqual(plan.directReopenBlocks, blocks.map((item) => item.block));
    assert.deepEqual(plan.preservedBlocks, []);
    assert.equal(plan.researchMode, "FULL");
    assert.equal(plan.fundamentalsMode, "FULL");
  }
});

test("FULL_REFRESH_REQUIRED leaves NOT_STARTED blocks current and executable", () => {
  const blocks = [block("BUSINESS_MODEL"), notStarted("MOAT")];
  const plan = planRefresh("FULL_REFRESH_REQUIRED", [], blocks);
  assert.equal(plan.ok, true);
  if (plan.ok) {
    assert.deepEqual(plan.directReopenBlocks, ["BUSINESS_MODEL"]);
    assert.deepEqual(plan.reopenBlocks, ["BUSINESS_MODEL"]);
    assert.deepEqual(plan.preservedBlocks, ["MOAT"]);
  }
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

test("SAVE disposition separates checkpoint/final sealing from analytical block status", () => {
  assert.equal(
    deriveSaveDisposition({ ...saveBase, phase: "RESEARCH" }).action,
    "FINALIZE",
  );
  assert.equal(
    deriveSaveDisposition({ ...saveBase, phase: "CERTIFICATION" }).action,
    "FINALIZE",
  );
  assert.equal(
    deriveSaveDisposition({ ...saveBase, phase: "INTEGRATION" }).action,
    "FINALIZE",
  );
  assert.equal(
    deriveSaveDisposition({ ...saveBase, phase: "FUNDAMENTALS" }).action,
    "CHECKPOINT",
  );
  assert.equal(
    deriveSaveDisposition({ ...saveBase, phase: "VALUATION" }).action,
    "CHECKPOINT",
  );
});

test("SAVE never publishes and obeys registry lifecycle", () => {
  for (const phase of ["RESEARCH", "CERTIFICATION", "INTEGRATION"] as const) {
    assert.equal(
      deriveSaveDisposition({ ...saveBase, phase }).publishAuthorized,
      false,
    );
  }
  assert.equal(
    deriveSaveDisposition({
      ...saveBase,
      phase: "RESEARCH",
      registryLifecycle: "NOT_STARTED",
    }).action,
    "NOOP",
  );
  assert.equal(
    deriveSaveDisposition({
      ...saveBase,
      phase: "RESEARCH",
      registryLifecycle: "PAUSED",
    }).action,
    "CHECKPOINT",
  );
  assert.equal(
    deriveSaveDisposition({
      ...saveBase,
      phase: "RESEARCH",
      registryLifecycle: "BLOCKED",
    }).action,
    "BLOCK",
  );
  assert.equal(
    deriveSaveDisposition({
      ...saveBase,
      phase: "RESEARCH",
      registryLifecycle: "COMPLETE",
    }).action,
    "NOOP",
  );
});

test("SAVE incomplete durability or phase gate NO checkpoints instead of false finalization", () => {
  assert.equal(
    deriveSaveDisposition({
      ...saveBase,
      phase: "RESEARCH",
      persistenceVerified: false,
    }).action,
    "CHECKPOINT",
  );
  assert.equal(
    deriveSaveDisposition({
      ...saveBase,
      phase: "RESEARCH",
      phaseGate: "NO",
    }).action,
    "CHECKPOINT",
  );
});

test("execution fingerprint is stable and authoritative input changes it", () => {
  const base = {
    runId: "11111111-1111-4111-8111-111111111111",
    block: "MOAT" as const,
    inputVersion: "7",
    evidenceSetHash: "a".repeat(64),
    methodVersion: "V2",
    outputSchemaVersion: "0.1",
  };
  const first = computeExecutionFingerprint(base);
  assert.equal(first, computeExecutionFingerprint(base));
  assert.match(first, /^[a-f0-9]{64}$/);
  assert.notEqual(
    first,
    computeExecutionFingerprint({ ...base, evidenceSetHash: "b".repeat(64) }),
  );
});

test("unchanged fingerprint cannot loop without explicit justified retry", () => {
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
  assert.equal(
    decideRetry({
      ...common,
      nextFingerprint: "changed",
      retryBudgetExhausted: true,
    }).allowed,
    false,
  );
});
