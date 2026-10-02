import { createHash } from "node:crypto";
import { v2RefreshRoute, type RefreshClass } from "../v2/deep-dive-runtime";

export const BLOCK_CODES = [
  "BUSINESS_MODEL",
  "ECONOMIC_QUALITY",
  "INDUSTRY_STRUCTURE",
  "TECHNOLOGY",
  "CYCLICALITY",
  "MOAT",
  "RUNWAY",
  "RETURN_QUALITY",
  "FCF_FORENSIC",
  "CAPITAL_ALLOCATION",
  "MANAGEMENT_GOVERNANCE",
  "OUTSIDE_VIEW",
  "RISK_RESILIENCE",
  "RED_TEAM",
  "VALUATION",
  "CROSS_BLOCK_RECONCILIATION",
] as const;

export type BlockCode = (typeof BLOCK_CODES)[number];

export type AnalyticalExecutionStatus =
  | "INSUFFICIENT"
  | "IN_PROGRESS"
  | "PROVISIONALLY_STABLE"
  | "LOCKED";

export type BlockPresence = "NOT_STARTED" | "PRESENT";

export type BlockFreshness = "CURRENT" | "REOPENED" | "STALE";

export type OverlayStatus =
  | "APPLIED"
  | "NOT_APPLICABLE"
  | "REQUIRED_MISSING"
  | "CONFLICTED";

export type MaterialRevalidationStatus =
  | "NOT_REQUIRED"
  | "PENDING"
  | "PASS"
  | "FAIL";

export type MaterialRevalidationInput = {
  required: boolean;
  reopenedMaterialEvidence: boolean;
  rootPrimarySourcesVerified: boolean;
  disconfirmingEvidenceSearched: boolean;
  bestAlternativeExplanationTested: boolean;
  affectedDownstreamBlocksReconciled: boolean;
  priorStateChangeReasonRecorded: boolean;
  hardFailureReasons: string[];
};

export type MaterialRevalidationDecision = {
  status: MaterialRevalidationStatus;
  missingChecks: string[];
  failureReasons: string[];
};

export type RegistryLifecycle =
  | "NOT_STARTED"
  | "IN_PROGRESS"
  | "PAUSED"
  | "BLOCKED"
  | "COMPLETE";

export type ExecutionPhase =
  | "RESEARCH"
  | "FUNDAMENTALS"
  | "VALUATION"
  | "CERTIFICATION"
  | "INTEGRATION";

export type SectorOverlayState = {
  overlay: string;
  status: OverlayStatus;
};

export type AnalyticalBlockState = {
  block: BlockCode;
  presence: BlockPresence;
  analyticalStatus: AnalyticalExecutionStatus | null;
  freshness: BlockFreshness;
  upstreamBlockRefs: BlockCode[];
  criticalUnresolvedGap: boolean;
  blockingConflict: boolean;
  sectorOverlays: SectorOverlayState[];
  materialRevalidationStatus: MaterialRevalidationStatus;
  completionAuditPassed: boolean;
};

export type RequiredSectorOverlay = {
  block: BlockCode;
  overlay: string;
};

export type RequiredBlockDependency = {
  block: BlockCode;
  upstream: BlockCode;
};

export type DependencyReopenPlan = {
  directReopenBlocks: BlockCode[];
  staleDownstreamBlocks: BlockCode[];
  allAffectedBlocks: BlockCode[];
};

export type BlockFreshnessTransition = {
  block: BlockCode;
  freshness: Extract<BlockFreshness, "REOPENED" | "STALE">;
  materialRevalidationRequired: true;
};

export type DependencyGraphResult =
  | {
      ok: true;
      reverse: Map<BlockCode, Set<BlockCode>>;
      topologicalOrder: BlockCode[];
    }
  | {
      ok: false;
      errors: string[];
    };

export type RefreshPlan =
  | {
      ok: true;
      refreshClass: RefreshClass;
      researchMode: "MINIMAL_REVALIDATION" | "TARGETED_DELTA" | "FULL";
      fundamentalsMode:
        | "REVALIDATE_PRIOR_LOCK"
        | "REOPEN_AFFECTED_BLOCKS"
        | "FULL";
      directReopenBlocks: BlockCode[];
      staleBlocks: BlockCode[];
      reopenBlocks: BlockCode[];
      preservedBlocks: BlockCode[];
      oqsMayChange: boolean;
      requiresValuation: true;
      requiresCertification: true;
      requiresIntegration: true;
    }
  | {
      ok: false;
      errors: string[];
    };

export type SaveEligibilityInput = {
  phase: ExecutionPhase;
  registryLifecycle: RegistryLifecycle;
  selfAuditPassed: boolean;
  schemaValidationPassed: boolean;
  identityVersionChecksPassed: boolean;
  analyticalReconciliationPassed: boolean;
  requiredArtifactsPresent: boolean;
  persistenceVerified: boolean;
  criticalBlockers: string[];
  phaseGate: "NOT_EVALUATED" | "NO" | "YES";
};

export type SaveDisposition =
  | {
      action: "CHECKPOINT";
      reason: string;
      publishAuthorized: false;
    }
  | {
      action: "FINALIZE";
      reason: string;
      publishAuthorized: false;
    }
  | {
      action: "BLOCK";
      reason: string;
      publishAuthorized: false;
    }
  | {
      action: "NOOP";
      reason: string;
      publishAuthorized: false;
    };

export type FingerprintInput = {
  runId: string;
  block: BlockCode;
  inputVersion: string;
  evidenceSetHash: string;
  methodVersion: string;
  outputSchemaVersion: string;
};

export type RetryDecision =
  | { allowed: true; reason: string }
  | { allowed: false; reason: string };

export type NextBlockAction =
  | {
      action: "FAIL_CLOSED";
      block: null;
      reason: string;
    }
  | {
      action: "RESOLVE_BLOCKER";
      block: BlockCode;
      reason: string;
    }
  | {
      action: "EXECUTE_BLOCK";
      block: BlockCode;
      reason: string;
    }
  | {
      action: "FINALIZE_BLOCK";
      block: BlockCode;
      reason: string;
    }
  | {
      action: "NO_BLOCK_ACTION";
      block: null;
      reason: string;
    };

const FUNDAMENTAL_BLOCKS = new Set<BlockCode>([
  "BUSINESS_MODEL",
  "ECONOMIC_QUALITY",
  "INDUSTRY_STRUCTURE",
  "TECHNOLOGY",
  "CYCLICALITY",
  "MOAT",
  "RUNWAY",
  "RETURN_QUALITY",
  "FCF_FORENSIC",
  "CAPITAL_ALLOCATION",
  "MANAGEMENT_GOVERNANCE",
  "OUTSIDE_VIEW",
  "RISK_RESILIENCE",
  "RED_TEAM",
]);

const BLOCK_ORDER = new Map<BlockCode, number>(
  BLOCK_CODES.map((code, index) => [code, index]),
);

function orderOf(code: BlockCode): number {
  return BLOCK_ORDER.get(code) ?? Number.MAX_SAFE_INTEGER;
}

function uniqueInCanonicalOrder(values: Iterable<BlockCode>): BlockCode[] {
  return [...new Set(values)].sort((left, right) => orderOf(left) - orderOf(right));
}

function isTerminallyResolved(block: AnalyticalBlockState): boolean {
  return (
    block.presence === "PRESENT" &&
    block.analyticalStatus === "LOCKED" &&
    block.freshness === "CURRENT"
  );
}

function isProvisionallyUsable(block: AnalyticalBlockState): boolean {
  return (
    block.presence === "PRESENT" &&
    block.freshness === "CURRENT" &&
    (block.analyticalStatus === "PROVISIONALLY_STABLE" ||
      block.analyticalStatus === "LOCKED") &&
    !block.criticalUnresolvedGap &&
    !block.blockingConflict &&
    block.materialRevalidationStatus !== "PENDING" &&
    block.materialRevalidationStatus !== "FAIL"
  );
}

function validateBlockStateShape(block: AnalyticalBlockState): string[] {
  const errors: string[] = [];

  if (block.presence === "NOT_STARTED") {
    if (block.analyticalStatus !== null) {
      errors.push("NOT_STARTED block must not carry an analytical execution status");
    }
    if (block.freshness !== "CURRENT") {
      errors.push("NOT_STARTED block cannot be REOPENED or STALE");
    }
  } else if (block.analyticalStatus === null) {
    errors.push("PRESENT block requires an analytical execution status");
  }

  if (
    block.analyticalStatus === "LOCKED" &&
    block.materialRevalidationStatus === "PENDING"
  ) {
    errors.push("LOCKED block cannot retain pending material revalidation");
  }

  return errors;
}

export function evaluateMaterialRevalidationGate(
  input: MaterialRevalidationInput,
): MaterialRevalidationDecision {
  if (!input.required) {
    return { status: "NOT_REQUIRED", missingChecks: [], failureReasons: [] };
  }

  if (input.hardFailureReasons.length > 0) {
    return {
      status: "FAIL",
      missingChecks: [],
      failureReasons: [...input.hardFailureReasons],
    };
  }

  const checks = [
    ["REOPEN_MATERIAL_EVIDENCE", input.reopenedMaterialEvidence],
    ["VERIFY_ROOT_PRIMARY_SOURCES", input.rootPrimarySourcesVerified],
    ["SEARCH_DISCONFIRMING_EVIDENCE", input.disconfirmingEvidenceSearched],
    ["TEST_BEST_ALTERNATIVE_EXPLANATION", input.bestAlternativeExplanationTested],
    ["RECONCILE_AFFECTED_DOWNSTREAM_BLOCKS", input.affectedDownstreamBlocksReconciled],
    ["RECORD_PRIOR_STATE_CHANGE_REASON", input.priorStateChangeReasonRecorded],
  ] as const;

  const missingChecks = checks
    .filter(([, passed]) => !passed)
    .map(([name]) => name);

  return missingChecks.length === 0
    ? { status: "PASS", missingChecks: [], failureReasons: [] }
    : { status: "PENDING", missingChecks, failureReasons: [] };
}

function blockHasOverlayBlocker(block: AnalyticalBlockState): boolean {
  return block.sectorOverlays.some(
    (overlay) =>
      overlay.status === "REQUIRED_MISSING" || overlay.status === "CONFLICTED",
  );
}

export function buildDependencyGraph(
  blocks: AnalyticalBlockState[],
): DependencyGraphResult {
  const errors: string[] = [];
  const byCode = new Map<BlockCode, AnalyticalBlockState>();

  for (const block of blocks) {
    if (byCode.has(block.block)) errors.push(`duplicate block ${block.block}`);
    for (const error of validateBlockStateShape(block)) {
      errors.push(`block ${block.block}: ${error}`);
    }
    byCode.set(block.block, block);
  }

  const reverse = new Map<BlockCode, Set<BlockCode>>();
  for (const code of byCode.keys()) reverse.set(code, new Set());

  for (const block of blocks) {
    for (const upstream of block.upstreamBlockRefs) {
      if (upstream === block.block) {
        errors.push(`block ${block.block} cannot depend on itself`);
        continue;
      }
      if (!byCode.has(upstream)) {
        errors.push(
          `block ${block.block} references unknown upstream block ${upstream}`,
        );
        continue;
      }
      reverse.get(upstream)?.add(block.block);
    }
  }

  if (errors.length > 0) return { ok: false, errors };

  const indegree = new Map<BlockCode, number>();
  for (const block of blocks) {
    indegree.set(block.block, new Set(block.upstreamBlockRefs).size);
  }

  const queue = uniqueInCanonicalOrder(
    [...indegree.entries()]
      .filter(([, degree]) => degree === 0)
      .map(([code]) => code),
  );

  const order: BlockCode[] = [];

  while (queue.length > 0) {
    const current = queue.shift() as BlockCode;
    order.push(current);

    const downstream = uniqueInCanonicalOrder(reverse.get(current) ?? []);
    for (const child of downstream) {
      const next = (indegree.get(child) ?? 0) - 1;
      indegree.set(child, next);
      if (next === 0) {
        queue.push(child);
        queue.sort((left, right) => orderOf(left) - orderOf(right));
      }
    }
  }

  if (order.length !== blocks.length) {
    return {
      ok: false,
      errors: ["dependency graph contains a cycle or unresolved dependency"],
    };
  }

  return { ok: true, reverse, topologicalOrder: order };
}

export function computeDownstreamClosure(
  changedBlocks: BlockCode[],
  blocks: AnalyticalBlockState[],
): { ok: true; blocks: BlockCode[] } | { ok: false; errors: string[] } {
  const graph = buildDependencyGraph(blocks);
  if (!graph.ok) return graph;

  const known = new Set(blocks.map((block) => block.block));
  const unknown = changedBlocks
    .filter((block) => !known.has(block))
    .map(
      (block) =>
        `changed block ${block} is absent from current analytical package`,
    );

  if (unknown.length > 0) return { ok: false, errors: unknown };

  const visited = new Set<BlockCode>();
  const queue = [...changedBlocks];

  while (queue.length > 0) {
    const current = queue.shift() as BlockCode;
    if (visited.has(current)) continue;
    visited.add(current);

    for (const downstream of graph.reverse.get(current) ?? []) {
      if (!visited.has(downstream)) queue.push(downstream);
    }
  }

  return { ok: true, blocks: uniqueInCanonicalOrder(visited) };
}


export function validateRequiredBlockDependencies(
  requirements: RequiredBlockDependency[],
  blocks: AnalyticalBlockState[],
): { ok: true } | { ok: false; errors: string[] } {
  const errors: string[] = [];
  const byCode = new Map(blocks.map((block) => [block.block, block]));
  const seen = new Set<string>();

  for (const requirement of requirements) {
    const key = `${requirement.block}|${requirement.upstream}`;
    if (seen.has(key)) {
      errors.push(
        `duplicate required dependency ${requirement.block} <- ${requirement.upstream}`,
      );
      continue;
    }
    seen.add(key);

    if (requirement.block === requirement.upstream) {
      errors.push(
        `required dependency ${requirement.block} cannot reference itself`,
      );
      continue;
    }

    const block = byCode.get(requirement.block);
    if (!block) {
      errors.push(
        `required dependency references absent block ${requirement.block}`,
      );
      continue;
    }

    if (!byCode.has(requirement.upstream)) {
      errors.push(
        `required dependency for ${requirement.block} references absent upstream ${requirement.upstream}`,
      );
      continue;
    }

    if (!block.upstreamBlockRefs.includes(requirement.upstream)) {
      errors.push(
        `block ${requirement.block} is missing required upstream dependency ${requirement.upstream}`,
      );
    }
  }

  return errors.length === 0 ? { ok: true } : { ok: false, errors };
}

export function planDependencyReopen(
  changedBlocks: BlockCode[],
  blocks: AnalyticalBlockState[],
  requiredDependencies: RequiredBlockDependency[] = [],
):
  | { ok: true; plan: DependencyReopenPlan }
  | { ok: false; errors: string[] } {
  const dependencyValidation = validateRequiredBlockDependencies(
    requiredDependencies,
    blocks,
  );
  if (!dependencyValidation.ok) return dependencyValidation;

  const closure = computeDownstreamClosure(changedBlocks, blocks);
  if (!closure.ok) return closure;

  const direct = uniqueInCanonicalOrder(changedBlocks);
  const directSet = new Set(direct);
  const stale = closure.blocks.filter((block) => !directSet.has(block));

  return {
    ok: true,
    plan: {
      directReopenBlocks: direct,
      staleDownstreamBlocks: uniqueInCanonicalOrder(stale),
      allAffectedBlocks: closure.blocks,
    },
  };
}

export function deriveReopenTransitions(
  plan: DependencyReopenPlan,
): BlockFreshnessTransition[] {
  const direct = new Set(plan.directReopenBlocks);
  return plan.allAffectedBlocks.map((block) => ({
    block,
    freshness: direct.has(block) ? "REOPENED" : "STALE",
    materialRevalidationRequired: true,
  }));
}

export function validateRequiredSectorOverlays(
  requirements: RequiredSectorOverlay[],
  blocks: AnalyticalBlockState[],
): { ok: true } | { ok: false; errors: string[] } {
  const errors: string[] = [];
  const byCode = new Map(blocks.map((block) => [block.block, block]));

  const seen = new Set<string>();
  for (const requirement of requirements) {
    const key = `${requirement.block}|${requirement.overlay}`;
    if (seen.has(key)) {
      errors.push(
        `duplicate required sector overlay ${requirement.overlay} for ${requirement.block}`,
      );
      continue;
    }
    seen.add(key);

    const block = byCode.get(requirement.block);
    if (!block) {
      errors.push(
        `required sector overlay ${requirement.overlay} references absent block ${requirement.block}`,
      );
      continue;
    }

    const matches = block.sectorOverlays.filter(
      (overlay) => overlay.overlay === requirement.overlay,
    );

    if (matches.length !== 1) {
      errors.push(
        `block ${requirement.block} must contain exactly one required overlay ${requirement.overlay}`,
      );
      continue;
    }

    if (matches[0].status !== "APPLIED") {
      errors.push(
        `required overlay ${requirement.overlay} for ${requirement.block} is ${matches[0].status}, expected APPLIED`,
      );
    }
  }

  return errors.length === 0 ? { ok: true } : { ok: false, errors };
}

export function blockCompletionEligibility(
  block: AnalyticalBlockState,
  allBlocks: AnalyticalBlockState[],
  requiredOverlays: RequiredSectorOverlay[] = [],
  requiredDependencies: RequiredBlockDependency[] = [],
): { eligible: true } | { eligible: false; reasons: string[] } {
  const reasons: string[] = [];

  const shapeErrors = validateBlockStateShape(block);
  reasons.push(...shapeErrors);

  if (block.presence !== "PRESENT") reasons.push("block has not been executed");
  if (block.freshness !== "CURRENT") {
    reasons.push(`block freshness is ${block.freshness}`);
  }
  if (
    block.analyticalStatus !== "PROVISIONALLY_STABLE" &&
    block.analyticalStatus !== "LOCKED"
  ) {
    reasons.push(
      `analytical execution status is ${block.analyticalStatus ?? "NONE"}`,
    );
  }
  if (block.criticalUnresolvedGap) reasons.push("critical unresolved gap");
  if (block.blockingConflict) reasons.push("open/blocking conflict");
  if (blockHasOverlayBlocker(block)) reasons.push("sector overlay blocker");
  if (!block.completionAuditPassed) reasons.push("completion self-audit not passed");

  if (block.materialRevalidationStatus === "PENDING") {
    reasons.push("material change revalidation pending");
  }
  if (block.materialRevalidationStatus === "FAIL") {
    reasons.push("material change revalidation failed");
  }

  const overlayValidation = validateRequiredSectorOverlays(
    requiredOverlays.filter((requirement) => requirement.block === block.block),
    allBlocks,
  );
  if (!overlayValidation.ok) reasons.push(...overlayValidation.errors);

  const dependencyValidation = validateRequiredBlockDependencies(
    requiredDependencies.filter(
      (requirement) => requirement.block === block.block,
    ),
    allBlocks,
  );
  if (!dependencyValidation.ok) reasons.push(...dependencyValidation.errors);

  const byCode = new Map(allBlocks.map((item) => [item.block, item]));
  for (const upstream of block.upstreamBlockRefs) {
    const parent = byCode.get(upstream);
    if (!parent) {
      reasons.push(`upstream block ${upstream} missing`);
      continue;
    }

    if (!isTerminallyResolved(parent)) {
      reasons.push(
        `upstream block ${upstream} is not LOCKED/CURRENT (status=${parent.analyticalStatus ?? "NONE"}, freshness=${parent.freshness})`,
      );
    }
  }

  return reasons.length === 0
    ? { eligible: true }
    : { eligible: false, reasons: [...new Set(reasons)] };
}

export function planRefresh(
  refreshClass: RefreshClass,
  changedBlocks: BlockCode[],
  blocks: AnalyticalBlockState[],
  requiredDependencies: RequiredBlockDependency[] = [],
): RefreshPlan {
  const graph = buildDependencyGraph(blocks);
  if (!graph.ok) return graph;

  const dependencyValidation = validateRequiredBlockDependencies(
    requiredDependencies,
    blocks,
  );
  if (!dependencyValidation.ok) return dependencyValidation;

  const currentCodes = new Set(blocks.map((block) => block.block));
  const base = v2RefreshRoute(refreshClass);

  if (refreshClass === "PRICE_ONLY_DELTA") {
    const invalidChanges = changedBlocks.filter((block) =>
      FUNDAMENTAL_BLOCKS.has(block),
    );
    if (invalidChanges.length > 0) {
      return {
        ok: false,
        errors: invalidChanges.map(
          (block) =>
            `PRICE_ONLY_DELTA cannot declare fundamental block ${block} as changed`,
        ),
      };
    }

    const reopen = uniqueInCanonicalOrder(
      ["VALUATION", "CROSS_BLOCK_RECONCILIATION"].filter(
        (block): block is BlockCode => currentCodes.has(block as BlockCode),
      ),
    );

    const directReopenBlocks: BlockCode[] = reopen.filter(
      (block) => block === "VALUATION",
    );
    const staleBlocks = reopen.filter(
      (block) => !directReopenBlocks.includes(block),
    );

    return {
      ok: true,
      refreshClass,
      researchMode: base.research,
      fundamentalsMode: base.fundamentals,
      directReopenBlocks,
      staleBlocks,
      reopenBlocks: reopen,
      preservedBlocks: uniqueInCanonicalOrder(
        blocks
          .map((block) => block.block)
          .filter((block) => !reopen.includes(block)),
      ),
      oqsMayChange: false,
      requiresValuation: true,
      requiresCertification: true,
      requiresIntegration: true,
    };
  }

  if (refreshClass === "FULL_REFRESH_REQUIRED") {
    const allBlocks = uniqueInCanonicalOrder(
      blocks.map((block) => block.block),
    );
    return {
      ok: true,
      refreshClass,
      researchMode: base.research,
      fundamentalsMode: base.fundamentals,
      directReopenBlocks: allBlocks,
      staleBlocks: [],
      reopenBlocks: allBlocks,
      preservedBlocks: [],
      oqsMayChange: true,
      requiresValuation: true,
      requiresCertification: true,
      requiresIntegration: true,
    };
  }

  if (changedBlocks.length === 0) {
    return {
      ok: false,
      errors: [
        "ROUTINE_FUNDAMENTAL_DELTA requires at least one materially changed fundamental block",
      ],
    };
  }

  const invalidChanges = changedBlocks.filter(
    (block) => !FUNDAMENTAL_BLOCKS.has(block),
  );
  if (invalidChanges.length > 0) {
    return {
      ok: false,
      errors: invalidChanges.map(
        (block) =>
          `ROUTINE_FUNDAMENTAL_DELTA cannot originate from non-fundamental block ${block}`,
      ),
    };
  }

  const dependencyReopen = planDependencyReopen(
    changedBlocks,
    blocks,
    requiredDependencies,
  );
  if (!dependencyReopen.ok) return dependencyReopen;

  const reopen = new Set<BlockCode>(
    dependencyReopen.plan.allAffectedBlocks,
  );
  if (currentCodes.has("VALUATION")) reopen.add("VALUATION");
  if (currentCodes.has("CROSS_BLOCK_RECONCILIATION")) {
    reopen.add("CROSS_BLOCK_RECONCILIATION");
  }

  const reopenBlocks = uniqueInCanonicalOrder(reopen);
  const directReopenBlocks = uniqueInCanonicalOrder(
    dependencyReopen.plan.directReopenBlocks,
  );
  const directSet = new Set(directReopenBlocks);
  const staleBlocks = reopenBlocks.filter((block) => !directSet.has(block));

  return {
    ok: true,
    refreshClass,
    researchMode: base.research,
    fundamentalsMode: base.fundamentals,
    directReopenBlocks,
    staleBlocks,
    reopenBlocks,
    preservedBlocks: uniqueInCanonicalOrder(
      blocks
        .map((block) => block.block)
        .filter((block) => !reopen.has(block)),
    ),
    oqsMayChange: true,
    requiresValuation: true,
    requiresCertification: true,
    requiresIntegration: true,
  };
}

export function deriveSaveDisposition(
  input: SaveEligibilityInput,
): SaveDisposition {
  if (input.registryLifecycle === "NOT_STARTED") {
    return {
      action: "NOOP",
      reason: "stage is not started; SAVE requires an active stage",
      publishAuthorized: false,
    };
  }

  if (input.registryLifecycle === "COMPLETE") {
    return {
      action: "NOOP",
      reason: "stage already COMPLETE; explicit reopen required before new analytical save",
      publishAuthorized: false,
    };
  }

  if (
    input.registryLifecycle === "BLOCKED" ||
    input.criticalBlockers.length > 0
  ) {
    return {
      action: "BLOCK",
      reason:
        input.criticalBlockers.length > 0
          ? `critical blockers: ${input.criticalBlockers.join("; ")}`
          : "registry lifecycle is BLOCKED",
      publishAuthorized: false,
    };
  }

  if (input.registryLifecycle === "PAUSED") {
    return {
      action: "CHECKPOINT",
      reason: "stage is PAUSED; resume before terminal finalization",
      publishAuthorized: false,
    };
  }

  const durableReady =
    input.selfAuditPassed &&
    input.schemaValidationPassed &&
    input.identityVersionChecksPassed &&
    input.analyticalReconciliationPassed &&
    input.requiredArtifactsPresent &&
    input.persistenceVerified;

  if (!durableReady) {
    return {
      action: "CHECKPOINT",
      reason: "durable finalization prerequisites are not all satisfied",
      publishAuthorized: false,
    };
  }

  if (input.phaseGate !== "YES") {
    return {
      action: "CHECKPOINT",
      reason:
        input.phaseGate === "NO"
          ? "phase completion gate is NO"
          : "phase completion gate is not yet evaluated",
      publishAuthorized: false,
    };
  }

  if (input.phase === "RESEARCH") {
    return {
      action: "FINALIZE",
      reason: "Research completion gate and durable prerequisites are satisfied",
      publishAuthorized: false,
    };
  }

  if (input.phase === "CERTIFICATION") {
    return {
      action: "FINALIZE",
      reason: "Deep Dive Certification completion gate and durable prerequisites are satisfied",
      publishAuthorized: false,
    };
  }

  if (input.phase === "INTEGRATION") {
    return {
      action: "FINALIZE",
      reason:
        "Integration READY_TO_PUBLISH gate is satisfied; publication remains separately authorized",
      publishAuthorized: false,
    };
  }

  return {
    action: "CHECKPOINT",
    reason: `${input.phase} is an intermediate Deep Dive phase`,
    publishAuthorized: false,
  };
}

export function computeExecutionFingerprint(input: FingerprintInput): string {
  const canonical = JSON.stringify([
    input.runId,
    input.block,
    input.inputVersion,
    input.evidenceSetHash,
    input.methodVersion,
    input.outputSchemaVersion,
  ]);
  return createHash("sha256").update(canonical, "utf8").digest("hex");
}

export function decideRetry(input: {
  priorFingerprint: string | null;
  nextFingerprint: string;
  retryBudgetExhausted: boolean;
  newEvidence: boolean;
  resolvedConflict: boolean;
  methodChanged: boolean;
  deterministicBugFixed: boolean;
  criticalUserInputAdded: boolean;
  forensicJustification: boolean;
}): RetryDecision {
  if (input.priorFingerprint === null) {
    return { allowed: true, reason: "first execution" };
  }

  if (input.priorFingerprint !== input.nextFingerprint) {
    return { allowed: true, reason: "execution fingerprint changed" };
  }

  if (input.retryBudgetExhausted) {
    return {
      allowed: false,
      reason: "retry budget exhausted for unchanged execution fingerprint",
    };
  }

  const justifications = [
    ["new evidence", input.newEvidence],
    ["resolved conflict", input.resolvedConflict],
    ["method change", input.methodChanged],
    ["deterministic bug fix", input.deterministicBugFixed],
    ["critical user input", input.criticalUserInputAdded],
    ["explicit forensic reason", input.forensicJustification],
  ] as const;

  const reason = justifications.find(([, active]) => active)?.[0];
  if (reason) return { allowed: true, reason };

  return {
    allowed: false,
    reason: "same execution fingerprint with no new information or justified retry condition",
  };
}

function firstInTopologicalOrder(
  codes: Iterable<BlockCode>,
  order: BlockCode[],
): BlockCode | null {
  const candidates = new Set(codes);
  return order.find((code) => candidates.has(code)) ?? null;
}

export function deriveNextBlockAction(
  blocks: AnalyticalBlockState[],
  requiredOverlays: RequiredSectorOverlay[] = [],
  currentBlock: BlockCode | null = null,
  requiredDependencies: RequiredBlockDependency[] = [],
): NextBlockAction {
  if (blocks.length === 0) {
    return {
      action: "FAIL_CLOSED",
      block: null,
      reason: "analytical block state is empty",
    };
  }

  const graph = buildDependencyGraph(blocks);
  if (!graph.ok) {
    return {
      action: "FAIL_CLOSED",
      block: null,
      reason: graph.errors.join("; "),
    };
  }

  const dependencyValidation = validateRequiredBlockDependencies(
    requiredDependencies,
    blocks,
  );
  if (!dependencyValidation.ok) {
    return {
      action: "FAIL_CLOSED",
      block: null,
      reason: dependencyValidation.errors.join("; "),
    };
  }

  const byCode = new Map(blocks.map((block) => [block.block, block]));

  for (const code of graph.topologicalOrder) {
    const locked = byCode.get(code);
    if (
      !locked ||
      locked.presence !== "PRESENT" ||
      locked.analyticalStatus !== "LOCKED" ||
      locked.freshness !== "CURRENT"
    ) {
      continue;
    }

    const eligibility = blockCompletionEligibility(
      locked,
      blocks,
      requiredOverlays,
      requiredDependencies,
    );
    if (!eligibility.eligible) {
      return {
        action: "FAIL_CLOSED",
        block: null,
        reason: `LOCKED block ${code} violates completion conditions: ${eligibility.reasons.join("; ")}`,
      };
    }
  }

  const upstreamProvisionallyUsable = (block: AnalyticalBlockState): boolean =>
    block.upstreamBlockRefs.every((upstream) => {
      const parent = byCode.get(upstream);
      return Boolean(parent && isProvisionallyUsable(parent));
    });

  const hasMaterialBlocker = (block: AnalyticalBlockState): boolean =>
    block.criticalUnresolvedGap ||
    block.blockingConflict ||
    blockHasOverlayBlocker(block) ||
    block.materialRevalidationStatus === "FAIL";

  const isExecutable = (block: AnalyticalBlockState): boolean =>
    block.presence === "NOT_STARTED" ||
    block.freshness === "REOPENED" ||
    block.freshness === "STALE" ||
    block.analyticalStatus === "INSUFFICIENT" ||
    block.analyticalStatus === "IN_PROGRESS" ||
    block.materialRevalidationStatus === "PENDING";

  if (currentBlock) {
    const current = byCode.get(currentBlock);
    if (current && upstreamProvisionallyUsable(current)) {
      if (hasMaterialBlocker(current)) {
        return {
          action: "RESOLVE_BLOCKER",
          block: current.block,
          reason: "current block has a material process blocker",
        };
      }
      if (isExecutable(current)) {
        return {
          action: "EXECUTE_BLOCK",
          block: current.block,
          reason:
            "current block remains executable with satisfied upstream dependencies",
        };
      }
    }
  }

  const blocked = blocks.filter(
    (block) => upstreamProvisionallyUsable(block) && hasMaterialBlocker(block),
  );

  if (blocked.length > 0) {
    const code = firstInTopologicalOrder(
      blocked.map((block) => block.block),
      graph.topologicalOrder,
    ) as BlockCode;
    return {
      action: "RESOLVE_BLOCKER",
      block: code,
      reason: "block has a material process blocker",
    };
  }

  const executable = blocks.filter(
    (block) => upstreamProvisionallyUsable(block) && isExecutable(block),
  );

  if (executable.length > 0) {
    const code = firstInTopologicalOrder(
      executable.map((block) => block.block),
      graph.topologicalOrder,
    ) as BlockCode;
    return {
      action: "EXECUTE_BLOCK",
      block: code,
      reason: "next topological executable block with usable upstream state",
    };
  }

  for (const code of graph.topologicalOrder) {
    const block = byCode.get(code);
    if (
      !block ||
      block.presence !== "PRESENT" ||
      block.analyticalStatus !== "PROVISIONALLY_STABLE" ||
      block.freshness !== "CURRENT"
    ) {
      continue;
    }

    const eligibility = blockCompletionEligibility(
      block,
      blocks,
      requiredOverlays,
      requiredDependencies,
    );
    if (eligibility.eligible) {
      return {
        action: "FINALIZE_BLOCK",
        block: code,
        reason:
          "provisionally stable block satisfies analytical LOCKED transition conditions",
      };
    }
  }

  if (blocks.every((block) => isTerminallyResolved(block))) {
    return {
      action: "NO_BLOCK_ACTION",
      block: null,
      reason: "all analytical blocks are LOCKED and current",
    };
  }

  return {
    action: "RESOLVE_BLOCKER",
    block:
      firstInTopologicalOrder(
        blocks
          .filter((block) => !isTerminallyResolved(block))
          .map((block) => block.block),
        graph.topologicalOrder,
      ) ?? blocks[0]?.block ?? "BUSINESS_MODEL",
    reason:
      "no block is currently executable; resolve dependency, revalidation, freshness, or completion blocker",
  };
}
